"""Round-2 R2-1 / R2-4: evaluate 60 fps event features (cache/events.npz).

Per feature group (ocular, mouth_head, posture, all): class-balanced logistic regression, C chosen
per outer fold by inner subject-wise CV (as R6), under the 5-fold protocol and LOSO (V1).
Also: matched-KSS session-shortcut audit (V8), and a pre-declared equal-weight late fusion with
V* (IR-112 mean-pool test log-probs). Video cohort, so numbers compare with C1-C5.
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
from m8.config import CACHE, RESULTS  # noqa: E402
from m8.evaluate import paired_permutation_pooled, summarize  # noqa: E402
from m8.windows import load_windows  # noqa: E402

CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def fit(X, y, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(X, y)


def cv_logp(X, y, groups_outer, rows, groups_inner_fn):
    """Outer split by groups_outer[rows]; C chosen on a subject-wise inner split of the train part."""
    lp = np.full((len(y), 3), np.nan, np.float32)
    for k in np.unique(groups_outer[rows]):
        tr, te = rows[groups_outer[rows] != k], rows[groups_outer[rows] == k]
        inner = groups_inner_fn(tr)
        sc = {}
        for C in CS:
            pred = np.zeros(len(tr), int)
            for j in np.unique(inner):
                m = inner == j
                pred[m] = fit(X[tr[~m]], y[tr[~m]], C).predict(X[tr[m]])
            sc[C] = f1_score(y[tr], pred, average="macro")
        lp[te] = fit(X[tr], y[tr], max(sc, key=sc.get)).predict_log_proba(X[te])
    return lp


def main():
    w = load_windows()
    d = np.load(CACHE / "events.npz")
    X_all, g = d["X"], d["groups"]
    cut = np.cumsum(np.r_[0, g])
    sets = {"ocular": slice(cut[0], cut[1]), "mouth_head": slice(cut[1], cut[2]),
            "posture": slice(cut[2], cut[3]), "all": slice(0, cut[3])}
    vstar = np.load(RESULTS / "runs" / "ir_112_mean_cls_mean_video_y3" / "preds.npz")
    v_lp = np.full((len(w), 3), np.nan, np.float32)
    v_lp[vstar["rows"]] = np.log(np.exp(vstar["test_lp"]).mean(0) + 1e-9)
    rows = vstar["rows"]
    y, subj = w.y3.values, w.subject.values
    subs = np.unique(subj[rows])
    sub_idx = {s: i for i, s in enumerate(subs)}
    inner5 = lambda tr: np.array([sub_idx[s] % 4 for s in subj[tr]])  # noqa: E731
    innerL = lambda tr: np.array([sub_idx[s] % 5 for s in subj[tr]])  # noqa: E731
    loso_group = np.array([sub_idx.get(s, -1) for s in subj])
    win = w.iloc[rows].reset_index(drop=True)
    sess = (w.session.values == "D").astype(int)
    out = {}
    vs = summarize(win, v_lp[rows])
    print(f"V* IR-112 mean           5-fold {vs['fold_mean_f1']:.3f}  pooled {vs['pooled']['macro_f1']:.3f}")
    for name, sl in sets.items():
        X = X_all[:, sl]
        lp5 = cv_logp(X, y, w.fold.values, rows, inner5)
        lpL = cv_logp(X, y, loso_group, rows, innerL)
        s5, sL = summarize(win, lp5[rows]), summarize(win, lpL[rows])
        fused = (lp5 + v_lp) / 2
        sf = summarize(win, fused[rows])
        t = paired_permutation_pooled(subj[rows], y[rows], fused[rows].argmax(1), v_lp[rows].argmax(1))
        matched = rows[(w.kss.values[rows] >= 5) & (w.kss.values[rows] <= 6)]
        sp = np.zeros(len(w), int)
        for k in range(5):
            tr, te = matched[w.fold.values[matched] != k], matched[w.fold.values[matched] == k]
            sp[te] = fit(X[tr], sess[tr], 0.1).predict(X[te])
        audit = balanced_accuracy_score(sess[matched], sp[matched])
        out[name] = dict(fivefold=s5, loso_pooled_f1=sL["pooled"]["macro_f1"], fused_with_vstar=sf,
                         fused_vs_vstar=t, session_audit_kss5_6=audit)
        print(f"{name:11s} 5-fold {s5['fold_mean_f1']:.3f} pooled {s5['pooled']['macro_f1']:.3f} | LOSO pooled "
              f"{sL['pooled']['macro_f1']:.3f} | +V* equal: 5-fold {sf['fold_mean_f1']:.3f} pooled "
              f"{sf['pooled']['macro_f1']:.3f} (Δ vs V* {t['diff_pooled_f1']:+.3f}, p={t['p']:.3f}) | "
              f"session audit {audit:.3f}", flush=True)
    (RESULTS / "R2_events.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
