"""Inference with the deployment models in models/final/ (written by scripts/m8/train_final.py).

Two input paths:

  dataset session   python -m m8.inference --session K D
      Uses the cached IR-112 DINOv2 tokens (cache/emb/ir_112), telemetry (cache/tel) and 60 fps
      event features (cache/events.npz). Runs every model: V*, Alert-calibrated V* (the
      reference is the first 240 s of the subject's Alert drive), events LR, calibrated V* +
      events fusion and M8 (optionally with a simulated camera blackout).

  uploaded video    python -m m8.inference --video drive.mp4 [--ref alert_drive.mp4]
      IR face video at any frame rate. The face is found with OpenCV's Haar detector (the
      dataset path uses UL-DD's own landmarks, so crops differ slightly), encoded at 2 fps with
      the same frozen DINOv2-S/14-reg, and scored with V*. With --ref, the calibrated V* also
      runs (reference = the first 240 s of the reference video). Events and M8 need the
      dataset's 60 fps landmarks / telemetry and are not available on this path.

Output: one row per 60 s window (15 s stride) with class probabilities per model, written as CSV
when --out is given. Models trained on all 16 video subjects: scoring a UL-DD session with them
is in-sample, so the demo's dataset path is a visualisation, not an evaluation (CV numbers live
in results/m8/).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from m8.config import CACHE, CLASS_NAMES, EPOCH_SEC, ROOT, SESSION_SEC, STRIDE_SEC, WINDOW_SEC
from m8.heads import MeanPoolHead
from m8.vigil import Vigil

MODELS = ROOT / "models" / "final"
TAG = "ir_112"
REF_S = 240


def _tokens(tok):
    """(n, 6, 384) DINOv2 tokens -> (n, 768) cls_mean, as m8.data._video_tokens."""
    tok = np.asarray(tok, np.float32)
    return np.concatenate([tok[:, 0], tok[:, 1]], 1)


def window_starts(duration_s):
    return np.arange(0, max(duration_s - WINDOW_SEC, 0) + 1e-6, STRIDE_SEC).astype(int)


def window_frames(t, starts):
    """Frame indices per window: t in [start, start + 60)."""
    lo, hi = np.searchsorted(t, starts), np.searchsorted(t, starts + WINDOW_SEC)
    return [np.arange(a, b) for a, b in zip(lo, hi)]


def clock_features(starts):
    ep = np.minimum((starts + WINDOW_SEC / 2) // EPOCH_SEC, SESSION_SEC // EPOCH_SEC - 1)
    t, e = starts / SESSION_SEC, ep / 9
    return np.stack([t, e, t ** 2, e ** 2, t * e], 1).astype(np.float32), ep.astype(int)


def _softmax(lp):
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


class Predictor:
    def __init__(self, models_dir=MODELS, device="cpu"):
        self.device = device
        self.heads = {}
        for name in ["vstar", "vstar_calA"]:
            ck = torch.load(models_dir / f"{name}.pt", map_location="cpu", weights_only=False)
            nets = []
            for st in ck["states"]:
                m = MeanPoolHead(ck["mu"].numel(), ck["cfg"]["n_classes"])
                m.load_state_dict(st)
                nets.append(m.eval())
            self.heads[name] = dict(nets=nets, mu=ck["mu"].numpy(), sd=ck["sd"].numpy())
        ck = torch.load(models_dir / "m8.pt", map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        self.m8 = dict(ck=ck, nets=[])
        for st in ck["states"]:
            m = Vigil(ck["d_v"], ck["d_t"], 3, ck["mode"], d=cfg["d"], drop=cfg["drop"], gate_bias=cfg["gate_bias"])
            m.load_state_dict(st)
            self.m8["nets"].append(m.eval())
        self.events = joblib.load(models_dir / "events.joblib")

    # ------------------------------------------------------------------ vision heads
    def head_probs(self, name, feats, frames):
        """feats (n, 768) raw (or reference-subtracted) tokens; frames: list of index arrays."""
        h = self.heads[name]
        x = (feats - h["mu"]) / h["sd"]
        pooled = np.stack([x[f].mean(0) if len(f) else np.zeros(x.shape[1], np.float32) for f in frames])
        mask = torch.ones(len(frames), 1, dtype=torch.bool)
        xb = torch.from_numpy(pooled.astype(np.float32))[:, None]
        with torch.no_grad():
            p = np.mean([torch.softmax(m(xb, mask)[0], -1).numpy() for m in h["nets"]], 0)
        p[[len(f) == 0 for f in frames]] = np.nan
        return p

    # ------------------------------------------------------------------ M8
    def m8_probs(self, feats, face_missing, frames, tel, has_tel, clock, blackout_s=0, seed=0):
        ck = self.m8["ck"]
        x = (feats - ck["vision_mu"][0].numpy()) / ck["vision_sd"][0].numpy()
        rng = np.random.default_rng(seed)
        pv, u = [], []
        for f in frames:
            T = len(f)
            keep = np.ones(T, bool)
            if blackout_s and T:
                L = min(T, int(round(blackout_s * T / WINDOW_SEC)))
                s = int(rng.integers(0, T - L + 1))
                keep[s:s + L] = False
            g = f[keep]
            pv.append(x[g].mean(0) if len(g) else np.zeros(x.shape[1], np.float32))
            of = len(g) / max(T, 1) if T else 0.0
            u.append([of, face_missing[g].mean() if len(g) else 0.0, float(len(g) == 0)])
        tel_n = (tel - ck["tel_mu"].numpy()) / ck["tel_sd"].numpy() * has_tel
        clk = (clock - ck["clock_mu"].numpy()) / ck["clock_sd"].numpy()
        pt = np.concatenate([tel_n, clk], 1).astype(np.float32)
        args = [torch.from_numpy(np.asarray(a, np.float32)) for a in (pv, pt, np.ones(len(frames)), u)]
        with torch.no_grad():
            outs = [m(*args) for m in self.m8["nets"]]
        p = np.mean([torch.softmax(z, -1).numpy() for z, _ in outs], 0)
        gate = np.mean([g.squeeze(-1).numpy() for _, g in outs], 0)
        return p, gate

    # ------------------------------------------------------------------ dataset path
    def predict_session(self, subj, sess, blackout_s=0):
        from m8.windows import load_windows
        w = load_windows()
        win = w[(w.subject == subj) & (w.session == sess)].reset_index()
        emb = np.load(CACHE / "emb" / TAG / f"{subj}_{sess}.npz")
        feats, t, fm = _tokens(emb["tok"]), emb["t"], emb["q"][:, 0]
        starts = win.start_s.values
        frames = window_frames(t, starts)
        out = win[["subject", "session", "start_s", "epoch_idx", "kss", "y3", "y_b2"]].copy()
        self._put(out, "vstar", self.head_probs("vstar", feats, frames))
        ref_f = CACHE / "emb" / TAG / f"{subj}_A.npz"
        if ref_f.exists():
            r = np.load(ref_f)
            ref = _tokens(r["tok"])[r["t"] < REF_S].mean(0, keepdims=True)
            self._put(out, "calA", self.head_probs("vstar_calA", feats - ref, frames))
        ev = np.load(CACHE / "events.npz")["X"][win["index"].values]
        self._put(out, "events", self.events["model"].predict_proba(ev))
        if "p_calA_0" in out:
            lp = (np.log(self._get(out, "calA") + 1e-9) + np.log(self._get(out, "events") + 1e-9)) / 2
            self._put(out, "fusion", _softmax(lp))
        tel = np.load(CACHE / "tel" / f"{subj}_{sess}.npz")["x"]
        tel_w = np.stack([tel[int(s * 4):int(s * 4) + 240].mean(0) for s in starts])
        clock, _ = clock_features(starts)
        p, g = self.m8_probs(feats, fm, frames, tel_w, win.avail_tel.values[:, None].astype(np.float32), clock,
                             blackout_s)
        self._put(out, "m8", p)
        out["m8_gate"] = g
        return out

    # ------------------------------------------------------------------ upload path
    def predict_video(self, path, ref_path=None, fps=2, max_s=None, progress=None):
        feats, t, fm = encode_video(path, fps, max_s, progress)
        starts = window_starts(t[-1] + 1 / fps if len(t) else 0)
        frames = window_frames(t, starts)
        out = pd.DataFrame(dict(start_s=starts))
        out["epoch_idx"] = clock_features(starts)[1]
        self._put(out, "vstar", self.head_probs("vstar", feats, frames))
        if ref_path:
            rf, rt, _ = encode_video(ref_path, fps, REF_S, progress)
            ref = rf[rt < REF_S].mean(0, keepdims=True)
            self._put(out, "calA", self.head_probs("vstar_calA", feats - ref, frames))
        out["face_missing"] = [fm[f].mean() if len(f) else 1.0 for f in frames]
        return out

    @staticmethod
    def _put(df, name, p):
        for k in range(p.shape[1]):
            df[f"p_{name}_{k}"] = p[:, k]
        df[f"pred_{name}"] = np.where(np.isnan(p).any(1), -1, np.nan_to_num(p).argmax(1))

    @staticmethod
    def _get(df, name):
        return df[[c for c in df.columns if c.startswith(f"p_{name}_")]].values


def encode_video(path, fps=2, max_s=None, progress=None):
    """Haar face crop -> 112 px -> DINOv2 tokens. Returns (feats (n, 768), t (n,), face_missing (n,))."""
    import cv2

    from m8.extract import HAAR, encode, load_encoder, square_crop
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_s:
        n = min(n, int(max_s * src_fps))
    step = max(1, int(round(src_fps / fps)))
    crops, t, miss, recent = [], [], [], []
    for f in range(n):
        if not cap.grab():
            break
        if f % step:
            continue
        ok, im = cap.retrieve()
        if not ok:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        d = HAAR.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
        if len(d):
            x, y, bw, bh = max(d, key=lambda r: r[2] * r[3])
            # Haar box -> UL-DD landmark-box framing (median over 268 frames of 6 sessions:
            # side = 1.18 x Haar width, centre 0.04 width lower), median of the last 9 detections
            recent = (recent + [(x + bw / 2, y + bh / 2 + 0.04 * bw, 1.18 * bw)])[-9:]
        box = tuple(np.median(recent, 0)) if recent else (im.shape[1] / 2, im.shape[0] / 2, min(im.shape[:2]))
        crops.append(square_crop(im, *box, 112)[..., ::-1])
        t.append(f / src_fps)
        miss.append(float(not len(d)))
        if progress and len(t) % 50 == 0:
            progress(f / max(n, 1))
    if not crops:
        raise ValueError(f"no frames decoded from {path}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = encode(load_encoder(112, device), np.stack(crops), device)
    return _tokens(tok), np.array(t, np.float32), np.array(miss, np.float32)


def summarize_session(df):
    """Window-level macro-F1 per model (in-sample for UL-DD sessions; for display only)."""
    from sklearn.metrics import f1_score
    out = {}
    for c in df.columns:
        if c.startswith("pred_") and "y3" in df:
            m = df[c] >= 0
            out[c[5:]] = float(f1_score(df.y3[m], df[c][m], average="macro", labels=[0, 1, 2], zero_division=0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", nargs=2, metavar=("SUBJ", "SESS"))
    g.add_argument("--video", type=Path)
    ap.add_argument("--ref", type=Path, help="reference (alert) video for the calibrated model")
    ap.add_argument("--blackout", type=float, default=0, help="simulated camera blackout per window (s), M8 only")
    ap.add_argument("--max-s", type=float, help="only the first N seconds of the video")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    pr = Predictor()
    if args.session:
        df = pr.predict_session(*args.session, blackout_s=args.blackout)
        print("in-sample window macro-F1 (display only):", {k: round(v, 3) for k, v in summarize_session(df).items()})
    else:
        df = pr.predict_video(args.video, args.ref, max_s=args.max_s)
    cols = [c for c in df.columns if c.startswith("pred_")]
    for c in cols:
        print(f"{c[5:]:7s} predicted class share:",
              {CLASS_NAMES[k]: round(float((df[c] == k).mean()), 2) for k in range(3)})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
