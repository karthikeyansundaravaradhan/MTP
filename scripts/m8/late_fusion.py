"""C6/C8-style late fusion of finished harness runs: weighted sum of per-stream log-probs.

Per outer fold k, weights on the simplex (step 0.1) are chosen on fold k's inner OOF logits
(outer-train subjects only) by macro-F1, then applied to the seed-ensembled test logits.
An equal-weight average is reported alongside.

  python scripts/m8/late_fusion.py ir_112_mean_cls_mean_video_y3 pose_224_mil_cls_mean_video_y3 --name F_ir_pose
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.config import RESULTS  # noqa: E402
from m8.evaluate import summarize  # noqa: E402
from m8.windows import load_windows  # noqa: E402


def load(name, n):
    d = np.load(RESULTS / "runs" / name / "preds.npz")
    test = np.full((n, d["test_lp"].shape[2]), np.nan, np.float32)
    test[d["rows"]] = np.log(np.exp(d["test_lp"]).mean(0) + 1e-9)
    oof = np.full((n, 5, d["oof_lp"].shape[2]), np.nan, np.float32)
    oof[d["rows"]] = d["oof_lp"]
    return test, oof


def simplex(m, step=0.1):
    k = int(round(1 / step))
    return [np.array(c) / k for c in itertools.product(range(k + 1), repeat=m) if sum(c) == k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    w = load_windows()
    tests, oofs = zip(*(load(r, len(w)) for r in args.runs))
    rows = np.where(w.avail_video.values == 1)[0]
    for t in tests:
        rows = rows[~np.isnan(t[rows, 0])]
    win = w.iloc[rows].reset_index(drop=True)
    y = win.y3.values
    T = np.stack([t[rows] for t in tests])          # (S, R, C)
    O = np.stack([o[rows] for o in oofs])           # (S, R, 5, C)
    grid = simplex(len(args.runs))
    fused, chosen = np.zeros_like(T[0]), {}
    for k in range(5):
        tr = (win.fold.values != k) & ~np.isnan(O[:, :, k, 0]).any(0)
        sc = [f1_score(y[tr], np.tensordot(g, O[:, tr, k], 1).argmax(1), average="macro") for g in grid]
        g = grid[int(np.argmax(sc))]
        chosen[k] = g.round(2).tolist()
        te = win.fold.values == k
        fused[te] = np.tensordot(g, T[:, te], 1)
    res = dict(runs=args.runs, weights_per_fold={str(k): v for k, v in chosen.items()},
               learned=summarize(win, fused), equal=summarize(win, T.mean(0)),
               singles={r: summarize(win, T[i])["fold_mean_f1"] for i, r in enumerate(args.runs)})
    for tag in ["learned", "equal"]:
        s = res[tag]
        print(f"[{args.name}] {tag:7s} fold-mean {s['fold_mean_f1']:.3f}±{s['fold_std_f1']:.3f} pooled "
              f"{s['pooled']['macro_f1']:.3f} CI{np.round(s['pooled_f1_ci95'], 3).tolist()} per-fold "
              f"{[round(m['macro_f1'], 3) for m in s['per_fold'].values()]}")
    print(f"[{args.name}] weights {chosen} | singles {[round(v, 3) for v in res['singles'].values()]}")
    run_dir = RESULTS / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "fused.npz", rows=rows, lp=fused)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
