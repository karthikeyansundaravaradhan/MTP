"""Frozen DINOv2-S/14-reg feature extraction for the video streams (E2-E5, E9 in plan_14day).

Per sampled frame we keep 6 tokens x 384: [CLS, mean-patch, 2x2 pooled patch grid].
Output per session: cache/emb/<stream>_<px>/<S>_<sess>.npz with
  tok (n, 6, 384) float16, frame (n,) int32 60fps frame index, t (n,) float32 seconds,
  q   (n, k) float32 per-frame quality features (see QUALITY_COLS).

Streams:
  ir   IR face crop from the dataset's own FL landmarks (FL frame n == IR frame n-1)
  l3d  L3D face crop from an OpenCV Haar detector (missing detections forward-fill the last
       box; before the first detection the session-median box is used)
  r3d  same for R3D
  pose side-view full frame, padded to square
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from m8.config import CACHE, FEAT_ROOT, NO_VIDEO, VIDEO_ROOT
from m8.windows import load_windows

ENCODER = "vit_small_patch14_reg4_dinov2.lvd142m"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
VIDEO_NAME = {"ir": "IR", "l3d": "L3D", "r3d": "R3D", "pose": "Pose", "ireye": "IR"}
QUALITY_COLS = ["face_missing", "box_side", "crop_mean", "crop_std"]
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


ENCODER_TAGS = {ENCODER: "", "vit_small_patch16_dinov3.lvd1689m": "_dinov3"}


def load_encoder(px, device):
    import timm
    m = timm.create_model(ENCODER, pretrained=True, num_classes=0, img_size=px)
    return m.eval().to(device).half()


@torch.no_grad()
def encode(model, crops, device, bs=64):
    out = []
    for i in range(0, len(crops), bs):
        x = (crops[i:i + bs].astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device).half()
        tok = model.forward_features(x)
        cls = tok[:, 0]
        patch = tok[:, model.num_prefix_tokens:]
        g = int(round(patch.shape[1] ** 0.5))
        pg = patch.transpose(1, 2).reshape(len(x), -1, g, g)
        if g % 2:  # MPS adaptive pooling needs an even grid (DINOv3 /16 at 112 px is 7x7)
            pg = pg.float().cpu()
        grid = F.adaptive_avg_pool2d(pg, 2).to(device).half()
        out.append(torch.cat([cls[:, None], patch.mean(1, keepdim=True),
                              grid.flatten(2).transpose(1, 2)], 1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def square_crop(img, cx, cy, side, px, border=cv2.BORDER_REPLICATE):
    h, w = img.shape[:2]
    half = side / 2
    x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=0)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(img[y0:y1, x0:x1], (px, px), interpolation=cv2.INTER_AREA)


def fl_boxes(subj, sess, frames):
    """Face box (cx, cy, side) per requested 0-indexed frame from FL landmarks; NaN if missing."""
    fl = pd.read_csv(FEAT_ROOT / subj / sess / f"{subj}_FL_{sess}.csv")
    fl = fl.drop_duplicates(subset=fl.columns[0], keep="first")
    fidx = fl.iloc[:, 0].values.astype(np.int64) - 1
    pts = fl.iloc[:, 1:].values.astype(np.float32).reshape(len(fl), -1, 2)
    lo, hi = pts.min(1), pts.max(1)
    cx, cy = (lo[:, 0] + hi[:, 0]) / 2, (lo[:, 1] + hi[:, 1]) / 2
    side = 1.5 * np.maximum(hi[:, 0] - lo[:, 0], hi[:, 1] - lo[:, 1])
    cy = cy - 0.08 * side
    lut = np.full((max(fidx.max(), frames.max()) + 1, 3), np.nan, np.float32)
    lut[fidx] = np.stack([cx, cy, side], 1)
    return lut[frames]


def fl_eye_boxes(subj, sess, frames):
    """R2-2: both-eye square box per 0-indexed frame: centre = mean of eye landmarks 36-47,
    side = 1.1 x outer-eye-corner distance (36-45). NaN if the frame has no landmarks."""
    fl = pd.read_csv(FEAT_ROOT / subj / sess / f"{subj}_FL_{sess}.csv")
    fl = fl.drop_duplicates(subset=fl.columns[0], keep="first")
    fidx = fl.iloc[:, 0].values.astype(np.int64) - 1
    pts = fl.iloc[:, 1:].values.astype(np.float32).reshape(len(fl), -1, 2)
    c = pts[:, 36:48].mean(1)
    side = 1.1 * np.linalg.norm(pts[:, 45] - pts[:, 36], axis=1)
    lut = np.full((max(fidx.max(), frames.max()) + 1, 3), np.nan, np.float32)
    lut[fidx] = np.stack([c[:, 0], c[:, 1], side], 1)
    return lut[frames]


def sample_eye_crops(subj, sess, path, step, px, n_limit=None):
    """Decode and crop on the fly (10 fps full frames would not fit in memory)."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_limit:
        n = min(n, n_limit)
    want = np.arange(0, n, step)
    boxes = fl_eye_boxes(subj, sess, want)
    missing = np.isnan(boxes[:, 0])
    med = np.nanmedian(boxes, 0)
    idx, crops, q, last, j = [], [], [], med, 0
    for f in range(n):
        if not cap.grab():
            break
        if f % step == 0:
            ok, im = cap.retrieve()
            if ok:
                b = boxes[j] if not missing[j] else last
                last = b
                cr = square_crop(im, b[0], b[1], b[2], px)
                idx.append(f)
                crops.append(cr[..., ::-1])
                q.append([missing[j], b[2], cr.mean(), cr.std()])
            j += 1
    return np.array(idx, np.int32), np.stack(crops), np.array(q, np.float32)


def sample_frames(path, step, n_limit=None):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_limit:
        n = min(n, n_limit)
    idx, imgs = [], []
    for f in range(n):
        if not cap.grab():
            break
        if f % step == 0:
            ok, im = cap.retrieve()
            if ok:
                idx.append(f)
                imgs.append(im)
    return np.array(idx, np.int32), imgs


def crops_for(stream, subj, sess, idx, imgs, px):
    n = len(imgs)
    q = np.zeros((n, len(QUALITY_COLS)), np.float32)
    if stream == "pose":
        out = []
        for im in imgs:
            h, w = im.shape[:2]
            out.append(square_crop(im, w / 2, h / 2, max(h, w), px, cv2.BORDER_CONSTANT))
        boxes = None
    else:
        if stream == "ir":
            boxes = fl_boxes(subj, sess, idx)
        else:
            boxes = np.full((n, 3), np.nan, np.float32)
            for i, im in enumerate(imgs):
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                d = HAAR.detectMultiScale(g, 1.1, 5, minSize=(80, 80))
                if len(d):
                    x, y, w, h = max(d, key=lambda r: r[2] * r[3])
                    boxes[i] = [x + w / 2, y + h / 2, 1.25 * w]
        missing = np.isnan(boxes[:, 0])
        q[:, 0] = missing
        med = np.nanmedian(boxes, 0) if (~missing).any() else np.array(
            [imgs[0].shape[1] / 2, imgs[0].shape[0] / 2, min(imgs[0].shape[:2])], np.float32)
        last = med
        out = []
        for i, im in enumerate(imgs):
            b = boxes[i] if not missing[i] else last
            last = b if not missing[i] else last
            q[i, 1] = b[2]
            out.append(square_crop(im, b[0], b[1], b[2], px))
    out = np.stack(out)
    q[:, 2] = out.mean((1, 2, 3))
    q[:, 3] = out.std((1, 2, 3))
    return out[..., ::-1].copy(), q


def run_stream(stream, px, fps, device, sessions=None, n_limit=None):
    tag = f"{stream}_{px}{ENCODER_TAGS[ENCODER]}"
    out_dir = CACHE / ("emb_smoke" if n_limit else "emb") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_encoder(px, device)
    w = load_windows()
    todo = sessions or sorted({(r.subject, r.session) for r in w.itertuples() if r.subject not in NO_VIDEO})
    step = 60 // fps
    for subj, sess in todo:
        dst = out_dir / f"{subj}_{sess}.npz"
        if dst.exists() and not n_limit:
            continue
        path = VIDEO_ROOT / subj / sess / f"{subj}_{VIDEO_NAME[stream]}_{sess}.mp4"
        if not path.exists():
            print(f"[{tag}] missing {path.name}", flush=True)
            continue
        t0 = time.time()
        if stream == "ireye":
            idx, crops, q = sample_eye_crops(subj, sess, path, step, px, n_limit)
            t1 = t2 = time.time()
        else:
            idx, imgs = sample_frames(path, step, n_limit)
            t1 = time.time()
            crops, q = crops_for(stream, subj, sess, idx, imgs, px)
            t2 = time.time()
        tok = encode(model, crops, device)
        tmp = dst.with_name(dst.stem + ".tmp.npz")
        np.savez(tmp, tok=tok, frame=idx, t=(idx / 60.0).astype(np.float32), q=q,
                 quality_cols=np.array(QUALITY_COLS), encoder=ENCODER)
        tmp.replace(dst)
        print(f"[{tag}] {subj}_{sess}: {len(idx)} frames, miss {q[:,0].mean():.1%} | decode {t1-t0:.0f}s "
              f"crop {t2-t1:.0f}s encode {time.time()-t2:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=list(VIDEO_NAME), required=True)
    ap.add_argument("--px", type=int, required=True)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--session", nargs=2, metavar=("SUBJ", "SESS"))
    ap.add_argument("--limit-frames", type=int)
    ap.add_argument("--encoder", default=ENCODER, choices=list(ENCODER_TAGS),
                    help="R2-12: frozen encoder; non-default ones get a cache-tag suffix")
    args = ap.parse_args()
    globals()["ENCODER"] = args.encoder
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sessions = [tuple(args.session)] if args.session else None
    run_stream(args.stream, args.px, args.fps, device, sessions, args.limit_frames)


if __name__ == "__main__":
    main()
