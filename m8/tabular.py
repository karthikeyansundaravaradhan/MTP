"""Non-video streams on the canonical clock (E7, E8 in plan_14day).

E7  face/body geometry at 2 Hz (all 35 sessions): per 0.5 s bin of 30 video frames
    geo  : EAR L/R, MAR, PERCLOS@0.20/0.25, blink count, yaw/pitch/roll proxies,
           inter-ocular px, face-missing fraction
    fau  : 30 AU means
    fl   : 68 landmarks, centred on the face and scaled by inter-ocular distance (136)
    pl   : 11 upper-body MediaPipe joints, centred on shoulders, scaled by shoulder width (33)
E8  vehicle at 4 Hz (sessions with telemetry): pitch, roll, speed, rpm, gear, heading rate,
    |heading rate|, heading-rate std, d(speed), plus grip LGP/RGP/asymmetry/zero-mask
    (grip exists even where telemetry is missing).
Output: cache/tab/<S>_<sess>.npz and cache/tel/<S>_<sess>.npz with t (seconds), x, names.
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from m8.config import CACHE, CSV_ROOT, FEAT_ROOT, SESSION_FRAMES
from m8.windows import load_windows

BIN = 30
N_BINS = SESSION_FRAMES // BIN
PL_JOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
GEO_NAMES = ["ear_l", "ear_r", "mar", "perclos20", "perclos25", "blinks", "yaw", "pitch", "roll",
             "iod_px", "face_missing"]
TEL_NAMES = ["pitch", "roll", "speed", "rpm", "gear", "hrate", "abs_hrate", "hrate_std", "dspeed",
             "lgp", "rgp", "grip_asym", "grip_zero", "tel_missing"]


def _dist(a, b):
    return np.linalg.norm(a - b, axis=-1)


def _ear(p):
    return (_dist(p[:, 1], p[:, 5]) + _dist(p[:, 2], p[:, 4])) / (2 * _dist(p[:, 0], p[:, 3]) + 1e-6)


def _bin_mean(values, frame, valid=None):
    """Mean of per-frame values within each 30-frame bin; NaN where a bin has no valid frame."""
    b = frame // BIN
    keep = (b >= 0) & (b < N_BINS) if valid is None else (b >= 0) & (b < N_BINS) & valid
    out = np.zeros((N_BINS,) + values.shape[1:])
    cnt = np.bincount(b[keep], minlength=N_BINS).astype(float)
    np.add.at(out, b[keep], values[keep])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / cnt.reshape((-1,) + (1,) * (values.ndim - 1))
    return out, cnt


def _ffill(x):
    x = pd.DataFrame(x.reshape(len(x), -1)).ffill().bfill().fillna(0).values
    return x


def face_session(subj, sess):
    d = FEAT_ROOT / subj / sess
    fl = pd.read_csv(d / f"{subj}_FL_{sess}.csv").drop_duplicates(subset="Frame_Number")
    frame = fl.Frame_Number.values.astype(np.int64) - 1
    p = fl.iloc[:, 1:].values.astype(np.float32).reshape(len(fl), 68, 2)
    iod = _dist(p[:, 36], p[:, 45])
    ear_l, ear_r = _ear(p[:, 36:42]), _ear(p[:, 42:48])
    ear = (ear_l + ear_r) / 2
    mar = (_dist(p[:, 61], p[:, 67]) + _dist(p[:, 63], p[:, 65])) / (2 * _dist(p[:, 60], p[:, 64]) + 1e-6)
    eye_mid = (p[:, 36] + p[:, 45]) / 2
    yaw = (p[:, 30, 0] - eye_mid[:, 0]) / (iod + 1e-6)
    pitch = _dist(p[:, 30], eye_mid) / (_dist(p[:, 8], p[:, 30]) + 1e-6)
    roll = np.degrees(np.arctan2(p[:, 45, 1] - p[:, 36, 1], p[:, 45, 0] - p[:, 36, 0]))
    closed = ear < 0.2
    onset = np.zeros(len(ear), bool)
    contiguous = np.r_[False, np.diff(frame) == 1]
    onset[1:] = closed[1:] & ~closed[:-1] & contiguous[1:]
    per_frame = np.stack([ear_l, ear_r, mar, closed, ear < 0.25, onset, yaw, pitch, roll, iod], 1)
    geo, cnt = _bin_mean(per_frame.astype(np.float64), frame)
    geo[:, 5] *= cnt
    missing = 1 - np.clip(cnt / BIN, 0, 1)

    centre = p.mean(1, keepdims=True)
    fl_norm = ((p - centre) / (iod[:, None, None] + 1e-6)).reshape(len(p), -1)
    flb, _ = _bin_mean(fl_norm.astype(np.float64), frame)

    fau = pd.read_csv(d / f"{subj}_FAU_{sess}.csv").drop_duplicates(subset="Frame")
    fau_v = np.nan_to_num(fau.iloc[:, 1:].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    faub, _ = _bin_mean(fau_v, fau.Frame.values.astype(np.int64) - 1)

    pl = pd.read_csv(d / f"{subj}_PL_{sess}.csv").drop_duplicates(subset="frame")
    j = pl.iloc[:, 1:].values.astype(np.float32).reshape(len(pl), 33, 3)[:, PL_JOINTS]
    sh_mid = (j[:, 3:4] + j[:, 4:5]) / 2
    sh_w = _dist(j[:, 3, :2], j[:, 4, :2])[:, None, None] + 1e-6
    plb, plc = _bin_mean(((j - sh_mid) / sh_w).reshape(len(j), -1).astype(np.float64),
                         pl.frame.values.astype(np.int64))

    x = np.concatenate([_ffill(geo), missing[:, None], _ffill(faub), _ffill(flb), _ffill(plb)], 1)
    names = GEO_NAMES + [f"fau_{c}" for c in fau.columns[1:]] + [f"fl{i}" for i in range(136)] + \
        [f"pl{j}_{c}" for j in PL_JOINTS for c in "xyz"]
    np.savez(CACHE / "tab" / f"{subj}_{sess}.npz", t=(np.arange(N_BINS) * BIN / 60.0).astype(np.float32),
             x=x.astype(np.float32), names=np.array(names), groups=np.array(
                 [len(GEO_NAMES), 30, 136, len(PL_JOINTS) * 3]))
    return subj, sess, float(missing.mean())


def vehicle_session(subj, sess):
    n = SESSION_FRAMES // 15
    t = np.arange(n) / 4.0
    x = np.zeros((n, len(TEL_NAMES)), np.float32)
    tp = CSV_ROOT / subj / sess / f"{subj}_Telemetry_{sess}.csv"
    if tp.exists() and subj != "A":
        tel = pd.read_csv(tp).iloc[:SESSION_FRAMES]
        hd = np.degrees(np.unwrap(np.radians(tel["heading[deg]"].values)))
        hrate = np.gradient(hd) * 60.0
        sp = tel["speed[m/s]"].values
        cols = [tel["pitch[deg]"].values, tel["roll[deg]"].values, sp, tel["rpm"].values,
                tel["gear"].values, hrate, np.abs(hrate)]
        m = len(tel) // 15
        blocks = np.stack(cols, 1)[: m * 15].reshape(m, 15, -1)
        x[:m, :7] = blocks.mean(1)
        x[:m, 7] = hrate[: m * 15].reshape(m, 15).std(1)
        x[:m, 8] = np.gradient(x[:m, 2]) * 4.0
        x[m:, 13] = 1
    else:
        x[:, 13] = 1
    g = []
    for side in ("LGP", "RGP"):
        gp = CSV_ROOT / subj / sess / f"{subj}_{side}_{sess}.csv"
        v = pd.read_csv(gp, header=None).iloc[:, 0].values.astype(np.float32) if gp.exists() else np.zeros(1)
        g.append(np.interp(t, np.arange(len(v)) / 3.0, v))
    x[:, 9], x[:, 10] = g
    x[:, 11] = (g[0] - g[1]) / (g[0] + g[1] + 1e-3)
    x[:, 12] = ((g[0] <= 0.01) | (g[1] <= 0.01)).astype(np.float32)
    np.savez(CACHE / "tel" / f"{subj}_{sess}.npz", t=t.astype(np.float32), x=x, names=np.array(TEL_NAMES))
    return subj, sess, float(x[:, 13].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    (CACHE / "tab").mkdir(exist_ok=True)
    (CACHE / "tel").mkdir(exist_ok=True)
    sessions = sorted({(r.subject, r.session) for r in load_windows().itertuples()})
    for s, se, miss in Parallel(n_jobs=args.jobs)(delayed(vehicle_session)(s, se) for s, se in sessions):
        print(f"[tel] {s}_{se}: tel-missing {miss:.1%}", flush=True)
    for s, se, miss in Parallel(n_jobs=args.jobs)(delayed(face_session)(s, se) for s, se in sessions):
        print(f"[tab] {s}_{se}: face-missing {miss:.1%}", flush=True)


if __name__ == "__main__":
    main()
