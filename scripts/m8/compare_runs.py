"""Paired comparison of window-level predictions from different runs on the same windows.

Run specs:
  <harness run name>        results/m8/runs/<name>/preds.npz (seed ensemble)
  frames:<run>[+<run>...]   per-frame runs (R4 seeds), averaged then window-aggregated (e.g. R4s)
  R1                        clock-only logistic regression, refitted here

Reports fold-mean / pooled macro-F1 and two subject-level paired permutation tests against the
reference (first spec): pooled macro-F1 (primary) and mean per-subject balanced accuracy.

  python scripts/m8/compare_runs.py frames:R4_seed0+R4_seed1+R4_seed2 R1 ir_112_mean_cls_mean_video_y3 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8.config import RESULTS  # noqa: E402
from m8.evaluate import (frames_to_windows, paired_permutation, paired_permutation_pooled,  # noqa: E402
                         summarize)
from m8.windows import load_windows  # noqa: E402
from r0_r1_baselines import feats  # noqa: E402
from score_frame_run import load_run  # noqa: E402


def window_logp(spec, w):
    """(len(w), C) log-probs with NaN where the run has no prediction."""
    out = np.full((len(w), 3), np.nan, np.float32)
    if spec == "R1":
        for k in range(5):
            tr, te = w.fold.values != k, w.fold.values == k
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats(w[tr], False), w.y3[tr])
            out[te] = m.predict_log_proba(feats(w[te], False))
        return out
    if spec.startswith("frames:"):
        runs = [load_run(RESULTS / "runs" / r, "probs") for r in spec[7:].split("+")]
        subj, sess, t = runs[0][:3]
        win, lp = frames_to_windows(subj, sess, t, np.mean([r[4] for r in runs], 0), w)
        pos = w.reset_index().merge(win[["subject", "session", "win_idx"]], on=["subject", "session", "win_idx"])
        out[pos["index"].values] = lp
        return out
    d = np.load(RESULTS / "runs" / spec / "preds.npz")
    out[d["rows"]] = np.log(np.exp(d["test_lp"]).mean(0) + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="first spec is the reference")
    ap.add_argument("--out", default="compare")
    args = ap.parse_args()
    w = load_windows()
    lps = {s: window_logp(s, w) for s in args.specs}
    rows = np.where(w.avail_video.values == 1)[0]
    for lp in lps.values():
        rows = rows[~np.isnan(lp[rows, 0])]
    win = w.iloc[rows].reset_index(drop=True)
    y, subj = win.y3.values, win.subject.values
    ref = args.specs[0]
    res = {}
    for s, lp in lps.items():
        summ = summarize(win, lp[rows])
        r = dict(fold_mean_f1=summ["fold_mean_f1"], pooled_f1=summ["pooled"]["macro_f1"],
                 pooled_ci=summ["pooled_f1_ci95"], per_class_f1=summ["pooled"]["per_class_f1"])
        if s != ref:
            r["vs_ref_pooled"] = paired_permutation_pooled(subj, y, lp[rows].argmax(1), lps[ref][rows].argmax(1))
            r["vs_ref_balacc"] = paired_permutation(summ["per_subject_balacc"],
                                                    summarize(win, lps[ref][rows])["per_subject_balacc"])
        res[s] = r
        line = f"{s[:44]:44s} fold-mean {r['fold_mean_f1']:.3f} pooled {r['pooled_f1']:.3f} " \
               f"CI{np.round(r['pooled_ci'], 3).tolist()}"
        if s != ref:
            line += f" | vs ref: Δpooled {r['vs_ref_pooled']['diff_pooled_f1']:+.3f} p={r['vs_ref_pooled']['p']:.3f}" \
                    f" | Δbalacc {r['vs_ref_balacc']['mean_diff']:+.3f} p={r['vs_ref_balacc']['p']:.3f}"
        print(line, flush=True)
    print(f"n windows {len(rows)}, subjects {len(np.unique(subj))}, reference {ref}")
    (RESULTS / f"{args.out}.json").write_text(json.dumps(dict(reference=ref, n_windows=int(len(rows)), runs=res),
                                                         indent=1, default=float))


if __name__ == "__main__":
    main()
