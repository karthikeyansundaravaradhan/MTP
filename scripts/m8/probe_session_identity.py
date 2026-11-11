"""Shortcut probe: can frozen DINOv2 window embeddings tell the Alert session from the Drowsy
session of an unseen subject? Linear probe on window-mean [CLS, patch-mean] tokens, outer folds.

High session accuracy with low y3 F1 would mean the embedding carries session-level appearance
(lighting, clothing, camera drift) a head could latch onto. R7 (raw image statistics) gave 0.581.
Session D has higher KSS by design, so the raw probe is partly label signal; the control repeats it
on windows with KSS 5-6 only (both sessions present), where session should be near chance.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.config import RESULTS  # noqa: E402
from m8.data import video_bank  # noqa: E402
from m8.windows import load_windows  # noqa: E402


def window_means(bank):
    f = bank.feats.float().numpy()
    out = np.zeros((len(bank.index), f.shape[1]), np.float32)
    ok = (bank.index >= 0).any(1)
    for r in np.where(ok)[0]:
        idx = bank.index[r][bank.index[r] >= 0]
        out[r] = f[idx].mean(0)
    return out, ok


def main():
    w = load_windows()
    out = {}
    for tag in ["ir_168", "ir_112", "l3d_168", "r3d_168", "pose_224"]:
        X, ok = window_means(video_bank(w, tag))
        rows = np.where(ok & (w.avail_video.values == 1))[0]
        sess = (w.session.values == "D").astype(int)
        res = {}
        matched = rows[(w.kss.values[rows] >= 5) & (w.kss.values[rows] <= 6)]
        for target, y, rr in [("session", sess, rows), ("session_kss5_6", sess, matched), ("y3", w.y3.values, rows)]:
            pred = np.zeros(len(w), int)
            for k in range(5):
                tr, te = rr[w.fold.values[rr] != k], rr[w.fold.values[rr] == k]
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000))
                pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
            res[target] = dict(bal_acc=balanced_accuracy_score(y[rr], pred[rr]),
                               macro_f1=f1_score(y[rr], pred[rr], average="macro"), n=len(rr))
        out[tag] = res
        print(f"{tag:9s} session bal-acc {res['session']['bal_acc']:.3f} (KSS 5-6 only "
              f"{res['session_kss5_6']['bal_acc']:.3f}) | y3 linear-probe macro-F1 "
              f"{res['y3']['macro_f1']:.3f}", flush=True)
    (RESULTS / "probe_session_identity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
