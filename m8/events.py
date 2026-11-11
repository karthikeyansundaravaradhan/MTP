"""Round-2 R2-1 / R2-4: event-level ocular, mouth, head and posture dynamics at the native 60 fps.

Every earlier model saw <= 2 fps, so blink dynamics were never available. Per canonical window
(frames in [start_s, start_s+60)):

ocular (FL, 60 fps): closure c = 1 - EAR / EAR_open, with EAR_open the session's 95th-percentile
    EAR over its first 240 s (start-of-drive calibration, deployable). Blink = run of c > 0.5.
    blink_rate, blink_dur_mean, blink_dur_p90, long_closures (> 0.5 s, micro-sleep proxy),
    perclos70, perclos80, perclos_abs (EAR < 0.2), close_vel, open_vel, avr (amplitude /
    peak closing velocity), ibi_cv, ear_mean, ear_std
mouth/head (FL): yawns (MAR > 0.6 for > 1 s), mar_mean, pitch_std, pitch_vel_rms, nods
    (downward pitch-proxy velocity peaks), face_missing
posture (PL, 60 fps, shoulder-normalised): motion energy (nose, shoulders, wrists), stillness
    fraction, head-drop (nose height above shoulders: mean, min, std), slouch (shoulder-nose dist)

Output: cache/events.npz with X (W, F), names, groups {ocular, mouth_head, posture} aligned to
load_windows() rows.
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from m8.config import CACHE, FEAT_ROOT, FPS
from m8.windows import load_windows

OCULAR = ["blink_rate", "blink_dur_mean", "blink_dur_p90", "long_closures", "perclos70", "perclos80",
          "perclos_abs", "close_vel", "open_vel", "avr", "ibi_cv", "ear_mean", "ear_std"]
MOUTH_HEAD = ["yawns", "mar_mean", "pitch_std", "pitch_vel_rms", "nods", "face_missing"]
POSTURE = ["motion_nose", "motion_shoulders", "motion_wrists", "stillness", "head_height_mean",
           "head_height_min", "head_height_std", "slouch"]
NAMES = OCULAR + MOUTH_HEAD + POSTURE
MAX_GAP = 3  # frames; larger gaps in landmark rows break an event


def _dist(a, b):
    return np.linalg.norm(a - b, axis=-1)


def _runs(mask):
    """(start, end) index pairs of True runs in a 1-D bool array (end exclusive)."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _dense(frame, values, n):
    """Scatter per-row values onto a dense frame grid; short gaps linearly filled, long gaps NaN."""
    out = np.full((n,) + values.shape[1:], np.nan)
    ok = (frame >= 0) & (frame < n)
    out[frame[ok]] = values[ok]
    s = pd.DataFrame(out.reshape(n, -1)).interpolate(limit=MAX_GAP, limit_area="inside").values
    return s.reshape(out.shape)


def session_frames(subj, sess, n_frames):
    d = FEAT_ROOT / subj / sess
    fl = pd.read_csv(d / f"{subj}_FL_{sess}.csv").drop_duplicates(subset="Frame_Number")
    p = fl.iloc[:, 1:].values.astype(np.float32).reshape(len(fl), 68, 2)
    frame = fl.Frame_Number.values.astype(np.int64) - 1
    ear = lambda q: (_dist(q[:, 1], q[:, 5]) + _dist(q[:, 2], q[:, 4])) / (2 * _dist(q[:, 0], q[:, 3]) + 1e-6)  # noqa: E731
    e = (ear(p[:, 36:42]) + ear(p[:, 42:48])) / 2
    mar = (_dist(p[:, 61], p[:, 67]) + _dist(p[:, 63], p[:, 65])) / (2 * _dist(p[:, 60], p[:, 64]) + 1e-6)
    eye_mid = (p[:, 36] + p[:, 45]) / 2
    pitch = _dist(p[:, 30], eye_mid) / (_dist(p[:, 8], p[:, 30]) + 1e-6)
    face = _dense(frame, np.stack([e, mar, pitch], 1), n_frames)

    pl = pd.read_csv(d / f"{subj}_PL_{sess}.csv").drop_duplicates(subset="frame")
    j = pl.iloc[:, 1:].values.astype(np.float32).reshape(len(pl), 33, 3)[:, :, :2]
    sh_mid = (j[:, 11] + j[:, 12]) / 2
    sh_w = _dist(j[:, 11], j[:, 12])[:, None] + 1e-6
    rel = (j - sh_mid[:, None]) / sh_w[:, None]
    pose = _dense(pl.frame.values.astype(np.int64),
                  np.concatenate([rel[:, [0, 11, 12, 15, 16]].reshape(len(rel), -1),
                                  (sh_mid / sh_w)], 1), n_frames)
    return face, pose


def window_features(face, pose, start_f, ear_open):
    sl = slice(start_f, start_f + 60 * FPS)
    e, mar, pitch = face[sl, 0], face[sl, 1], face[sl, 2]
    valid = ~np.isnan(e)
    f = dict.fromkeys(NAMES, 0.0)
    f["face_missing"] = 1 - valid.mean()
    if valid.sum() > 10 * FPS:
        c = 1 - e / ear_open
        closed = np.nan_to_num(c, nan=0) > 0.5
        st, en = _runs(closed)
        keep = (en - st) >= 3  # >= 3 frames (50 ms); shorter runs are landmark flicker
        st, en = st[keep], en[keep]
        dur = (en - st) / FPS
        f["blink_rate"] = len(st) / (valid.mean() * 1.0)  # per valid minute
        f["blink_dur_mean"] = dur.mean() if len(dur) else 0
        f["blink_dur_p90"] = np.percentile(dur, 90) if len(dur) else 0
        f["long_closures"] = float((dur > 0.5).sum())
        f["perclos70"] = np.nanmean(c > 0.7)
        f["perclos80"] = np.nanmean(c > 0.8)
        f["perclos_abs"] = np.nanmean(e < 0.2)
        de = np.gradient(np.nan_to_num(e, nan=np.nanmedian(e))) * FPS
        cv, ov, avr = [], [], []
        for a, b in zip(st, en):
            lo, hi = max(a - 6, 0), min(b + 6, len(e))
            cv.append(-de[lo:b].min() if b > lo else 0)
            ov.append(de[a:hi].max() if hi > a else 0)
            amp = np.nanmax(c[a:b]) if b > a else 0
            avr.append(amp / (cv[-1] + 1e-6))
        f["close_vel"] = np.mean(cv) if cv else 0
        f["open_vel"] = np.mean(ov) if ov else 0
        f["avr"] = np.mean(avr) if avr else 0
        ibi = np.diff(st) / FPS
        f["ibi_cv"] = ibi.std() / (ibi.mean() + 1e-6) if len(ibi) > 2 else 0
        f["ear_mean"], f["ear_std"] = np.nanmean(e), np.nanstd(e)
        ys, ye = _runs(np.nan_to_num(mar, nan=0) > 0.6)
        f["yawns"] = float(((ye - ys) > FPS).sum())
        f["mar_mean"] = np.nanmean(mar)
        pv = np.gradient(np.nan_to_num(pitch, nan=np.nanmedian(pitch))) * FPS
        f["pitch_std"], f["pitch_vel_rms"] = np.nanstd(pitch), np.sqrt(np.mean(pv ** 2))
        thr = 4 * np.median(np.abs(pv)) + 1e-6
        ns, ne = _runs(pv < -thr)
        f["nods"] = float(((ne - ns) >= 6).sum())  # sustained >= 100 ms downward head motion
    q = pose[sl]
    if (~np.isnan(q[:, 0])).sum() > 10 * FPS:
        q = pd.DataFrame(q).ffill().bfill().values
        v = np.abs(np.diff(q, axis=0)) * FPS
        nose_v, sh_v, wr_v = v[:, 0:2].sum(1), v[:, 2:6].sum(1), v[:, 6:10].sum(1)
        f["motion_nose"], f["motion_shoulders"], f["motion_wrists"] = nose_v.mean(), sh_v.mean(), wr_v.mean()
        f["stillness"] = float((nose_v + sh_v < np.percentile(nose_v + sh_v, 20) * 1.5).mean())
        h = -q[:, 1]  # nose y relative to shoulder mid (image y grows downward)
        f["head_height_mean"], f["head_height_min"], f["head_height_std"] = h.mean(), h.min(), h.std()
        f["slouch"] = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2).mean()
    return [f[n] for n in NAMES]


def one_session(subj, sess, rows, starts, n_frames):
    face, pose = session_frames(subj, sess, n_frames)
    ref = face[:240 * FPS, 0]
    ear_open = np.nanpercentile(ref, 95) if np.isfinite(ref).sum() > 100 else np.nanpercentile(face[:, 0], 95)
    return rows, np.array([window_features(face, pose, int(s * FPS), ear_open) for s in starts], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    w = load_windows()
    n_frames = int(w.end_f.max()) + 60 * FPS
    tasks = [(s, se, idx, w.start_s.values[idx]) for (s, se), idx in w.groupby(["subject", "session"]).indices.items()]
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one_session)(s, se, idx, st, n_frames) for s, se, idx, st in tasks)
    X = np.zeros((len(w), len(NAMES)), np.float32)
    for rows, x in res:
        X[rows] = x
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez(CACHE / "events.npz", X=X, names=np.array(NAMES),
             groups=np.array([len(OCULAR), len(MOUTH_HEAD), len(POSTURE)]))
    print("saved", X.shape, "non-zero frac per feature:", dict(zip(NAMES, np.round((X != 0).mean(0), 2))))


if __name__ == "__main__":
    main()
