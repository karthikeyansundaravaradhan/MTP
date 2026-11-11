"""Item 1 (+ item 4): pre-declared equal-weight fusion of Alert-calibrated V* (R2-7) with the 27
60 fps event features (R2-1/R2-4), under 5-fold and LOSO, for y3 and B2.

Rule fixed before looking at any fused number: log p = (log p_calV* + log p_events) / 2, then
argmax. No weights are fitted. Event-LR log-probs are also saved in the harness format
(results/m8/runs/events_lr_<y>[_loso]/preds.npz) so compare_runs.py / final_metrics.py can load them.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8.config import CACHE, RESULTS  # noqa: E402
from m8.evaluate import paired_permutation_pooled, summarize  # noqa: E402
from m8.windows import load_windows  # noqa: E402
from r2_events_eval import CS, fit  # noqa: E402

RUNS = {  # (y, protocol) -> (V* run, calibrated V* run)
    ("y3", "5fold"): ("ir_112_mean_cls_mean_video_y3", "ir_112_mean_cls_mean_video_y3_ref240A"),
    ("y3", "loso"): ("ir_112_mean_cls_mean_video_y3_loso", "ir_112_mean_cls_mean_video_y3_ref240A_loso"),
    ("y_b2", "5fold"): ("ir_112_mean_cls_mean_video_y_b2", "ir_112_mean_cls_mean_video_y_b2_ref240A"),
    ("y_b2", "loso"): ("ir_112_mean_cls_mean_video_y_b2_loso", "ir_112_mean_cls_mean_video_y_b2_ref240A_loso"),
}


def run_lp(name, n):
    p = RESULTS / "runs" / name / "preds.npz"
    if not p.exists():
        return None
    d = np.load(p)
    out = np.full((n, d["test_lp"].shape[-1]), np.nan, np.float32)
    out[d["rows"]] = np.log(np.exp(d["test_lp"]).mean(0) + 1e-9)
    return out


def events_lp(X, y, outer, rows, nc, n_inner):
    """Outer split by `outer`; C picked per outer fold by subject-wise inner CV on its train part."""
    lp = np.full((len(y), nc), np.nan, np.float32)
    for k in np.unique(outer[rows]):
        tr, te = rows[outer[rows] != k], rows[outer[rows] == k]
        inner = outer[tr] % n_inner if n_inner else None
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
    X = np.load(CACHE / "events.npz")["X"]
    subj = w.subject.values
    out = {}
    for (y_col, proto), (v_name, c_name) in RUNS.items():
        nc = 3 if y_col == "y3" else 2
        v_lp, c_lp = run_lp(v_name, len(w)), run_lp(c_name, len(w))
        if c_lp is None:
            print(f"skip {y_col}/{proto}: {c_name} not run yet")
            continue
        rows = np.where(~np.isnan(c_lp[:, 0]))[0]
        y = w[y_col].values
        if proto == "5fold":
            outer = w.fold.values
            subs = sorted(set(subj[rows]))
            inner_fold = np.array([subs.index(s) % 4 if s in subs else -1 for s in subj])
            # inner split = subject index mod 4 inside each outer-train set (as r2_events_eval)
            e_lp = np.full((len(w), nc), np.nan, np.float32)
            for k in range(5):
                tr, te = rows[outer[rows] != k], rows[outer[rows] == k]
                sc = {}
                for C in CS:
                    pred = np.zeros(len(tr), int)
                    for j in range(4):
                        m = inner_fold[tr] == j
                        pred[m] = fit(X[tr[~m]], y[tr[~m]], C).predict(X[tr[m]])
                    sc[C] = f1_score(y[tr], pred, average="macro")
                e_lp[te] = fit(X[tr], y[tr], max(sc, key=sc.get)).predict_log_proba(X[te])
        else:
            subs = sorted(set(subj[rows]))
            outer = np.array([subs.index(s) if s in subs else -1 for s in subj])
            e_lp = events_lp(X, y, outer, rows, nc, n_inner=5)
        tag = f"events_lr_{y_col}" + ("_loso" if proto == "loso" else "")
        (RESULTS / "runs" / tag).mkdir(parents=True, exist_ok=True)
        np.savez(RESULTS / "runs" / tag / "preds.npz", rows=rows, test_lp=e_lp[rows][None])
        fused = (c_lp + e_lp) / 2
        tag_f = f"fusion_calA_events_{y_col}" + ("_loso" if proto == "loso" else "")
        (RESULTS / "runs" / tag_f).mkdir(parents=True, exist_ok=True)
        np.savez(RESULTS / "runs" / tag_f / "preds.npz", rows=rows, test_lp=fused[rows][None])
        win = w.iloc[rows].reset_index(drop=True)
        if proto == "loso":
            win["fold"] = outer[rows]
        res = {}
        for nm, lp in [("vstar", v_lp), ("calA", c_lp), ("events", e_lp), ("fusion", fused)]:
            if lp is None:
                continue
            s = summarize(win, lp[rows], y_col, nc)
            res[nm] = dict(pooled_f1=s["pooled"]["macro_f1"], ci=s["pooled_f1_ci95"], fold_mean=s["fold_mean_f1"],
                           qwk=s["pooled"]["qwk"], per_class=s["pooled"]["per_class_f1"])
        for ref in ["calA", "vstar", "events"]:
            ref_lp = {"calA": c_lp, "vstar": v_lp, "events": e_lp}[ref]
            if ref_lp is None:
                continue
            t = paired_permutation_pooled(subj[rows], y[rows], fused[rows].argmax(1), ref_lp[rows].argmax(1), n_classes=nc)
            res[f"fusion_vs_{ref}"] = t
        out[f"{y_col}_{proto}"] = res
        line = " | ".join(f"{k} {v['pooled_f1']:.3f}" for k, v in res.items() if "pooled_f1" in v)
        tests = " ".join(f"{k[10:]} Δ{v['diff_pooled_f1']:+.3f} p={v['p']:.3f}" for k, v in res.items() if k.startswith("fusion_vs"))
        print(f"{y_col:5s} {proto:5s} pooled: {line} || fusion vs {tests}", flush=True)
    (RESULTS / "fusion_calA_events.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
