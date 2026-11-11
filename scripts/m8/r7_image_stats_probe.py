"""R7: image-statistics shortcut probe on the 1 fps IR frames.

Per window: mean over its frames of [grey mean/std, per-channel mean/std, 8-bin per-channel
histograms]. Logistic regression, subject-independent folds. Two targets:
  - y3 (drowsiness, P0)          -> how far pure lighting/colour statistics get
  - session A vs D (held-out subjects) -> the cross-subject session-identity probe
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.config import CACHE, KAVI, RESULTS  # noqa: E402
from m8.evaluate import metrics, summarize  # noqa: E402
from m8.windows import load_windows  # noqa: E402

FEAT_NPY = CACHE / "r7_frame_stats.npz"


def frame_stats(path):
    im = cv2.imread(path)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    f = [g.mean(), g.std()]
    for c in range(3):
        ch = im[..., c]
        f += [ch.mean(), ch.std()]
        f += list(np.histogram(ch, bins=8, range=(0, 256))[0] / ch.size)
    return np.array(f, np.float32)


def main():
    meta = pd.read_csv(KAVI / "datasets" / "yolo_frames_regen" / "metadata.csv",
                       dtype={"subject": str, "session": str})
    if FEAT_NPY.exists():
        feats = np.load(FEAT_NPY)["feats"]
    else:
        feats = np.stack(Parallel(n_jobs=8, batch_size=512)(delayed(frame_stats)(p) for p in meta.path))
        np.savez(FEAT_NPY, feats=feats)

    w = load_windows()
    w = w[w.avail_video == 1].reset_index(drop=True)
    X = np.zeros((len(w), feats.shape[1]), np.float32)
    keep = np.zeros(len(w), bool)
    for (s, se), g in meta.groupby(["subject", "session"]):
        t, F = g.timestamp_s.values, feats[g.index.values]
        for i in np.where((w.subject == s) & (w.session == se))[0]:
            m = (t >= w.start_s[i]) & (t < w.start_s[i] + 60)
            if m.any():
                X[i], keep[i] = F[m].mean(0), True
    w, X = w[keep].reset_index(drop=True), X[keep]

    out = {}
    for target, nc in [("y3", 3), ("session", 2)]:
        y = w.y3.values if target == "y3" else (w.session.values == "D").astype(int)
        logp = np.zeros((len(w), nc))
        for k in sorted(w.fold.unique()):
            tr, te = (w.fold != k).values, (w.fold == k).values
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000,
                                                                     class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            logp[te] = clf.predict_log_proba(X[te])
        if target == "y3":
            res = summarize(w, logp)
            print(f"R7 image-stats -> y3: fold-mean F1 {res['fold_mean_f1']:.3f}±{res['fold_std_f1']:.3f} "
                  f"pooled {res['pooled']['macro_f1']:.3f}")
        else:
            res = metrics(y, logp.argmax(1), 2)
            print(f"R7 image-stats -> session A/D on held-out subjects: acc {res['acc']:.3f} "
                  f"bal-acc {res['bal_acc']:.3f}")
        out[target] = res
    (RESULTS / "R7_image_stats_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
