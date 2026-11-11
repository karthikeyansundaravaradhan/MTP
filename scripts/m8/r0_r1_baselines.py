"""R0 (majority / stratified random) and R1 (clock-only) baselines on the P0 window set.

R1  : logistic regression on time-in-session only (window start, epoch index).
R1b : R1 + session label (A/D), an upper bound on what the session/clock shortcut can reach.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.config import RESULTS  # noqa: E402
from m8.evaluate import summarize  # noqa: E402
from m8.windows import load_windows  # noqa: E402


def feats(df, with_session):
    t = (df.start_s.values / 2400.0)[:, None]
    x = PolynomialFeatures(2, include_bias=False).fit_transform(np.hstack([t, df.epoch_idx.values[:, None] / 9]))
    if with_session:
        x = np.hstack([x, (df.session.values == "D").astype(float)[:, None]])
    return x


def run(name, w, y_col, n_classes, fit_predict):
    logp = np.zeros((len(w), n_classes))
    for k in sorted(w.fold.unique()):
        tr, te = w.fold != k, w.fold == k
        logp[te.values] = fit_predict(w[tr], w[te])
    res = summarize(w, logp, y_col=y_col, n_classes=n_classes)
    print(f"{name:14s} fold-mean F1 {res['fold_mean_f1']:.3f}±{res['fold_std_f1']:.3f}  "
          f"pooled F1 {res['pooled']['macro_f1']:.3f} CI{res['pooled_f1_ci95']}")
    return res


def main():
    w = load_windows()
    w = w[w.avail_video == 1].reset_index(drop=True)
    out = {}
    for y_col, nc in [("y3", 3), ("y_b2", 2)]:
        def majority(tr, te):
            lp = np.full((len(te), nc), -10.0)
            lp[:, np.bincount(tr[y_col], minlength=nc).argmax()] = 0
            return lp

        def strat(tr, te, _seed=[0]):
            rng = np.random.default_rng(_seed[0])
            pri = np.bincount(tr[y_col], minlength=nc) / len(tr)
            lp = np.full((len(te), nc), -10.0)
            lp[np.arange(len(te)), rng.choice(nc, len(te), p=pri)] = 0
            return lp

        def clock(with_session):
            def f(tr, te):
                m = LogisticRegression(max_iter=2000, class_weight="balanced")
                m.fit(feats(tr, with_session), tr[y_col])
                return m.predict_log_proba(feats(te, with_session))
            return f

        out[y_col] = {
            "R0_majority": run(f"R0 majority {y_col}", w, y_col, nc, majority),
            "R0_stratified": run(f"R0 strat {y_col}", w, y_col, nc, strat),
            "R1_clock": run(f"R1 clock {y_col}", w, y_col, nc, clock(False)),
            "R1b_clock_session": run(f"R1b clock+sess {y_col}", w, y_col, nc, clock(True)),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "R0_R1_baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
