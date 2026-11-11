"""Subject-DEPENDENT splits: how much does seeing the test driver's own face in training help?

The V* model (IR-112 DINOv2 mean-pool head, 3 seeds) is retrained under three splits of the video
cohort, each with a no-vision shortcut reference fitted on the same split:

  random  windows of every session scattered over 5 folds. Leaky: neighbouring windows share 75%
          of their frames (60 s windows, 15 s stride), so this is an upper bound, not a result.
  epoch   whole 4-min KSS epochs held out (2 per session per fold), and training windows within
          60 s of any test window of the same session are purged, so no frame is shared.
  drive   test on one drive (A or D) of a subject; train on everything else, including that
          subject's other drive. The realistic "known driver, new drive" case. One fold per session.

Shortcut reference ("who + which drive + when"): LR on subject one-hot, session flag and the clock
features. It sees no video at all, so any score it reaches is identity/session memorisation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8.config import RESULTS  # noqa: E402
from m8.data import video_bank  # noqa: E402
from m8.evaluate import summarize  # noqa: E402
from m8.train import DEVICE, cohort_rows, default_cfg, fit_predict  # noqa: E402
from m8.windows import load_windows  # noqa: E402
from r0_r1_baselines import feats  # noqa: E402


def split_folds(w, rows, split, seed=0):
    """fold id per row of w (-1 outside rows); list of (train_rows, test_rows)."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(w), -1)
    sess_groups = w.iloc[rows].groupby(["subject", "session"]).indices
    if split == "random":
        for idx in sess_groups.values():
            r = rows[idx]
            fold[r] = rng.permutation(np.arange(len(r)) % 5)
    elif split == "epoch":
        for idx in sess_groups.values():
            r = rows[idx]
            off = rng.integers(5)
            fold[r] = (w.epoch_idx.values[r] + off) % 5
    elif split == "drive":
        for i, idx in enumerate(sess_groups.values()):
            fold[rows[idx]] = i
    pairs = []
    for k in sorted(set(fold[rows])):
        te = rows[fold[rows] == k]
        tr = rows[fold[rows] != k]
        if split == "epoch":  # purge training windows sharing frames with a test window
            key = w.subject.to_numpy(str) + w.session.to_numpy(str)
            keep = np.ones(len(tr), bool)
            for s in np.unique(key[te]):
                t_te = w.start_s.values[te[key[te] == s]]
                m = key[tr] == s
                d = np.abs(w.start_s.values[tr[m]][:, None] - t_te[None]).min(1)
                keep[np.where(m)[0]] = d >= 60
            tr = tr[keep]
        if split == "drive" and len(set(w.y3.values[tr])) < 3:
            continue
        pairs.append((tr, te))
    return fold, pairs


def shortcut_X(w):
    subs = sorted(w.subject.unique())
    onehot = (w.subject.to_numpy(str)[:, None] == np.array(subs)[None]).astype(float)
    return np.hstack([onehot, feats(w, True)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["random", "epoch", "drive"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    t0 = time.time()
    w = load_windows()
    bank = video_bank(w, "ir_112").to(DEVICE)
    rows = cohort_rows(w, "video")
    rows = rows[(bank.index[rows] >= 0).any(1)]
    cfg = default_cfg(head="mean", inner=False, seeds=args.seeds)
    fold, pairs = split_folds(w, rows, args.split)
    lp = np.full((len(args.seeds), len(w), 3), np.nan, np.float32)
    lp_sc = np.full((len(w), 3), np.nan, np.float32)
    Xs = shortcut_X(w)
    n_purged = []
    for i, (tr, te) in enumerate(pairs):
        for si, seed in enumerate(args.seeds):
            lp[si, te], _, _ = fit_predict([bank], w, tr, te, cfg, seed)
        m = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(Xs[tr], w.y3.values[tr])
        lp_sc[te] = m.predict_log_proba(Xs[te])
        n_purged.append(int(len(rows) - len(te) - len(tr)))
        print(f"[SD-{args.split}] fold {i + 1}/{len(pairs)} done ({time.time() - t0:.0f}s)", flush=True)
    done = rows[~np.isnan(lp[0, rows, 0])]
    wf = w.copy()
    wf["fold"] = fold
    win = wf.iloc[done].reset_index(drop=True)
    ens = np.log(np.exp(lp[:, done]).mean(0) + 1e-9)
    s_v, s_sc = summarize(win, ens), summarize(win, lp_sc[done])
    out = dict(split=args.split, n_folds=len(pairs), n_windows=int(len(done)),
               mean_purged_train_windows=float(np.mean(n_purged)), vstar=s_v, shortcut=s_sc,
               minutes=(time.time() - t0) / 60)
    print(f"[SD-{args.split}] V* pooled {s_v['pooled']['macro_f1']:.3f} CI{np.round(s_v['pooled_f1_ci95'], 3).tolist()}"
          f" fold-mean {s_v['fold_mean_f1']:.3f} | shortcut (who+drive+time, no video) pooled "
          f"{s_sc['pooled']['macro_f1']:.3f} | windows {len(done)} | purged/fold {np.mean(n_purged):.0f}", flush=True)
    run_dir = RESULTS / "runs" / f"SD_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "preds.npz", rows=done, test_lp=lp[:, done], shortcut_lp=lp_sc[done], fold=fold[done])
    (RESULTS / f"SD_{args.split}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
