"""R6: class-balanced logistic regression on per-window summaries of the 2 Hz face/pose table
(geo + FAU + PL; raw landmarks left out) and the 4 Hz vehicle table.

Per window: mean, std, p10, p90 of each column. C picked per outer fold by inner subject-wise
CV (macro-F1). Feature sets: face, tel, face+tel. Cohorts: video (comparable with C1-C4) and all.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.config import CACHE, N_INNER, RESULTS  # noqa: E402
from m8.evaluate import summarize  # noqa: E402
from m8.train import cohort_rows  # noqa: E402
from m8.windows import load_inner, load_windows  # noqa: E402

CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def window_stats(w, kind, cols):
    rate = {"tab": 2, "tel": 4}[kind]
    out = np.zeros((len(w), 4 * len(cols)), np.float32)
    for (s, se), rows in w.groupby(["subject", "session"]).indices.items():
        x = np.load(CACHE / kind / f"{s}_{se}.npz")["x"][:, cols]
        for r in rows:
            seg = x[int(w.start_s.values[r] * rate):int((w.start_s.values[r] + 60) * rate)]
            out[r] = np.concatenate([seg.mean(0), seg.std(0), *np.percentile(seg, [10, 90], 0)])
    return out


def fit(X, y, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(X, y)


def run(w, X, rows, inner):
    y = w.y3.values
    lp = np.zeros((len(w), 3), np.float32)
    chosen = {}
    for k in range(5):
        tr, te = rows[w.fold.values[rows] != k], rows[w.fold.values[rows] == k]
        inner_of = np.array([inner[k][s] for s in w.subject.values[tr]])
        sc = {}
        for C in CS:
            pred = np.zeros(len(tr), int)
            for j in range(N_INNER):
                m = inner_of == j
                if m.any():
                    pred[m] = fit(X[tr[~m]], y[tr[~m]], C).predict(X[tr[m]])
            sc[C] = f1_score(y[tr], pred, average="macro")
        chosen[k] = max(sc, key=sc.get)
        lp[te] = fit(X[tr], y[tr], chosen[k]).predict_log_proba(X[te])
    win = w.iloc[rows].reset_index(drop=True)
    return summarize(win, lp[rows]), chosen


def main():
    w, inner = load_windows(), load_inner()
    names = list(np.load(CACHE / "tab" / "A_A.npz")["names"])
    face_cols = list(range(11 + 30)) + list(range(11 + 30 + 136, len(names)))  # geo+FAU+PL
    face = window_stats(w, "tab", face_cols)
    tel = window_stats(w, "tel", list(range(14)))
    sets = {"face": face, "tel": tel, "face_tel": np.concatenate([face, tel], 1)}
    out = {}
    for cohort in ["video", "all"]:
        for fs, X in sets.items():
            rows = cohort_rows(w, cohort)
            if fs == "tel":
                rows = rows[w.avail_tel.values[rows] == 1]
            res, chosen = run(w, X, rows, inner)
            key = f"{fs}_{cohort}"
            out[key] = dict(summary=res, C_per_fold=chosen, n_rows=len(rows))
            print(f"R6 {key:14s} n={len(rows)} fold-mean F1 {res['fold_mean_f1']:.3f}±{res['fold_std_f1']:.3f} "
                  f"pooled {res['pooled']['macro_f1']:.3f} CI{np.round(res['pooled_f1_ci95'], 3).tolist()} "
                  f"per-fold {[round(m['macro_f1'], 3) for m in res['per_fold'].values()]} C {chosen}", flush=True)
    (RESULTS / "R6_tabular.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
