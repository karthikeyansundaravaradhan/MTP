"""Canonical window table shared by every M8-phase model.

Windows live on the true 60 fps video clock (frame f <-> f/60 s), not on CSV row index,
so all streams (video, FL/FAU by their Frame column, telemetry/grip by row) align.
Label = KSS epoch containing the window centre, matching Kavipriya's ULDDProcessor rule.
"""
import json

import cv2
import numpy as np
import pandas as pd

from m8.config import (
    CACHE, CSV_ROOT, EPOCH_SEC, FEAT_ROOT, FOLDS, FPS, LABELS_CSV, N_INNER, NO_TELEMETRY,
    NO_VIDEO, SESSION_SEC, STRIDE_SEC, SUBJECTS, VIDEO_ROOT, WINDOW_SEC, kss_to_b2,
    kss_to_y3, outer_fold_of,
)

WINDOWS_CSV = CACHE / "windows.csv"
INNER_JSON = CACHE / "inner_folds.json"


def load_kss():
    kss = {}
    for line in LABELS_CSV.read_text().strip().splitlines():
        parts = line.strip().split(",")
        subj, name = parts[0].split("_", 1)
        kss[(subj, {"Alert": "A", "Drowsy": "D"}[name])] = [float(v) for v in parts[1:]]
    return kss


def build_windows():
    rows = []
    n_win = (SESSION_SEC - WINDOW_SEC) // STRIDE_SEC + 1
    for (subj, sess), epochs in sorted(load_kss().items()):
        ir = VIDEO_ROOT / subj / sess / f"{subj}_IR_{sess}.mp4"
        has_video = ir.exists()
        n_video = int(cv2.VideoCapture(str(ir)).get(cv2.CAP_PROP_FRAME_COUNT)) if has_video else 0
        has_tel = (CSV_ROOT / subj / sess / f"{subj}_Telemetry_{sess}.csv").exists()
        has_fl = (FEAT_ROOT / subj / sess / f"{subj}_FL_{sess}.csv").exists()
        for w in range(n_win):
            start_s = w * STRIDE_SEC
            centre_s = start_s + WINDOW_SEC / 2
            ep = min(int(centre_s // EPOCH_SEC), len(epochs) - 1)
            k = epochs[ep]
            rows.append(dict(
                subject=subj, session=sess, win_idx=w,
                start_f=start_s * FPS, end_f=(start_s + WINDOW_SEC) * FPS,
                start_s=start_s, epoch_idx=ep, kss=k, y3=kss_to_y3(k), y_b2=kss_to_b2(k),
                fold=outer_fold_of(subj),
                avail_video=int(has_video and subj not in NO_VIDEO
                                and (start_s + WINDOW_SEC) * FPS <= n_video),
                video_frames=n_video,
                avail_tel=int(has_tel and subj not in NO_TELEMETRY),
                avail_fl=int(has_fl),
            ))
    return pd.DataFrame(rows)


def assign_inner_folds(df):
    """Per outer fold, split its training subjects into N_INNER subject-wise inner folds,
    greedily balancing Alert epochs among video subjects so every inner fold can score Alert."""
    ep = df.drop_duplicates(["subject", "session", "epoch_idx"])
    alert = ep[ep.y3 == 0].groupby("subject").size().reindex(SUBJECTS, fill_value=0)
    out = {}
    for k, test in enumerate(FOLDS):
        train = [s for s in SUBJECTS if s not in test]
        vid = sorted([s for s in train if s not in NO_VIDEO], key=lambda s: (-alert[s], s))
        load = [[0, 0] for _ in range(N_INNER)]
        assign = {}
        for s in vid:
            j = min(range(N_INNER), key=lambda i: (load[i][0], load[i][1], i))
            assign[s] = j
            load[j][0] += int(alert[s])
            load[j][1] += 1
        for s in [s for s in train if s in NO_VIDEO]:
            j = min(range(N_INNER), key=lambda i: (load[i][1], i))
            assign[s] = j
            load[j][1] += 1
        out[str(k)] = assign
    return out


def load_windows():
    return pd.read_csv(WINDOWS_CSV, dtype={"subject": str, "session": str})


def load_inner():
    return {int(k): v for k, v in json.loads(INNER_JSON.read_text()).items()}


def main():
    CACHE.mkdir(exist_ok=True)
    df = build_windows()
    df.to_csv(WINDOWS_CSV, index=False)
    inner = assign_inner_folds(df)
    INNER_JSON.write_text(json.dumps(inner, indent=1))

    print(f"windows: {len(df)} rows, {df.groupby(['subject','session']).ngroups} sessions, "
          f"{df.win_idx.max()+1} windows/session")
    v = df[df.avail_video == 1]
    print(f"video windows: {len(v)}  tel windows: {df.avail_tel.sum()}  fl windows: {df.avail_fl.sum()}")
    print("y3 counts (video subjects) per outer fold:")
    for k in range(len(FOLDS)):
        print(f"  fold {k}: {np.bincount(v[v.fold == k].y3, minlength=3).tolist()}")
    ep = df.drop_duplicates(["subject", "session", "epoch_idx"])
    for k, assign in inner.items():
        parts = []
        for j in range(N_INNER):
            subs = sorted(s for s, i in assign.items() if i == j)
            e = ep[ep.subject.isin(subs) & ep.subject.isin(set(v.subject))]
            parts.append(f"{''.join(subs)}:{np.bincount(e.y3, minlength=3).tolist()}")
        print(f"  outer {k} inner folds (video-subject epoch y3): " + "  ".join(parts))


if __name__ == "__main__":
    main()
