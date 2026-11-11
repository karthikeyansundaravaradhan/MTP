"""Score a per-frame run (R4a / R4 seeds): frame-level, window-level (P0), window + causal EMA.

EMA beta is chosen on the run's inner-validation predictions when they exist (R4 retrain);
otherwise the grid is reported and no beta is claimed as selected.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.config import RESULTS  # noqa: E402
from m8.evaluate import causal_ema, frames_to_windows, metrics, summarize  # noqa: E402
from m8.windows import load_windows  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

BETAS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]


def load_run(run_dir, key):
    parts = [np.load(f, allow_pickle=True) for f in sorted(run_dir.glob("fold_*.npz"))]
    cat = lambda name: np.concatenate([p[name] for p in parts])  # noqa: E731
    return cat("subject"), cat("session"), cat("timestamp_s"), cat("y3"), cat(key)


def _mean_probs(files, key):
    """Average `key` probabilities over seed runs; frames must line up exactly."""
    ds = [np.load(f, allow_pickle=True) for f in files]
    for d in ds[1:]:
        assert (d["subject"] == ds[0]["subject"]).all() and np.allclose(d["timestamp_s"], ds[0]["timestamp_s"])
    return ds[0], np.mean([d[key] for d in ds], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+", help="one run, or several seed runs to average (seed ensemble)")
    ap.add_argument("--key", default="probs")
    ap.add_argument("--name", help="output name when ensembling several runs")
    args = ap.parse_args()
    run_dirs = [RESULTS / "runs" / r for r in args.run]
    run_dir = run_dirs[0]
    args.run = args.name or args.run[0]
    w = load_windows()
    subj, sess, t, y, p = load_run(run_dir, args.key)
    if len(run_dirs) > 1:
        runs = [load_run(r, args.key) for r in run_dirs]
        for r in runs[1:]:
            assert (r[0] == subj).all() and (r[1] == sess).all() and np.allclose(r[2], t)
        p = np.mean([r[4] for r in runs], 0)
    out = {"frame_level": metrics(y, p.argmax(1))}

    win, lp = frames_to_windows(subj, sess, t, p, w)
    out["window"] = summarize(win, lp)
    out["window_ema_grid"] = {str(b): summarize(win, causal_ema(win, lp, b))["fold_mean_f1"] for b in BETAS}

    # Beta per outer fold k, from fold k's own inner-val subjects (all outer-train for k), using
    # only odd-second frames: even-second frames were used to pick best.pt.
    beta_sel = None
    if (run_dir / "innerval").exists():
        beta_sel, lp_sel = {}, lp.copy()
        for f in sorted((run_dir / "innerval").glob("fold_*.npz")):
            k = int(f.stem.split("_")[1])
            d, vp = _mean_probs([r / "innerval" / f.name for r in run_dirs], "probs")
            odd = np.round(d["timestamp_s"]).astype(int) % 2 == 1
            vw, vlp = frames_to_windows(d["subject"][odd], d["session"][odd], d["timestamp_s"][odd],
                                        vp[odd], w)
            scores = {b: f1_score(vw.y3, causal_ema(vw, vlp, b).argmax(1), average="macro",
                                  labels=[0, 1, 2], zero_division=0) for b in BETAS}
            beta_sel[k] = max(scores, key=scores.get)
            m = (win.fold == k).values
            lp_sel[m] = causal_ema(win[m].reset_index(drop=True), lp[m], beta_sel[k])
        out["beta_selected_per_fold"] = {str(k): b for k, b in beta_sel.items()}
        out["window_ema_selected"] = summarize(win, lp_sel)

    folds = sorted(set(win.fold))
    print(f"{args.run} [{args.key}] folds {folds}")
    print(f"  frame-level   acc {out['frame_level']['acc']:.3f}  macro-F1 {out['frame_level']['macro_f1']:.3f}")
    print(f"  window (P0)   fold-mean F1 {out['window']['fold_mean_f1']:.3f}±{out['window']['fold_std_f1']:.3f}"
          f"  pooled {out['window']['pooled']['macro_f1']:.3f} CI{np.round(out['window']['pooled_f1_ci95'],3).tolist()}")
    print(f"  per-fold F1   {[round(m['macro_f1'],3) for m in out['window']['per_fold'].values()]}")
    print(f"  EMA grid      {out['window_ema_grid']}")
    if beta_sel is not None:
        print(f"  EMA beta per fold {beta_sel} (inner-val) fold-mean F1 "
              f"{out['window_ema_selected']['fold_mean_f1']:.3f}")
    (RESULTS / f"{args.run}_{args.key}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
