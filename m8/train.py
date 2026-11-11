"""Subject-independent CV for heads on cached features.

Outer: the 5 fixed folds. For each outer fold k:
  - test logits from models trained on all outer-train windows (one per seed),
  - inner OOF logits on outer-train windows from 4 subject-wise inner folds (seed 0),
    used to pick smoothing beta (and later gates/stacking) without touching fold k.
Normalisation statistics are always fitted on the rows being trained on.

Views: make_banks may return a list of bank-lists (e.g. [[L3D], [R3D]]) -- the same feature
space seen by different cameras. Each training batch then uses one random view, and predictions
average probabilities over views (two-view training + test-time averaging).
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from m8.config import N_INNER, RESULTS
from m8.evaluate import causal_ema, causal_hmm, summarize
from m8.heads import HEADS
from m8.windows import load_inner, load_windows

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BETAS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]
STAYS = [0.9, 0.95, 0.98, 0.99]


def smooth(win_df, logp, kind, param, prior):
    if kind == "ema":
        return causal_ema(win_df, logp, param)
    return causal_hmm(win_df, logp, param, prior)


def default_cfg(**kw):
    cfg = dict(head="mil", d=64, drop=0.3, epochs=30, bs=64, lr=1e-3, wd=0.05,
               frame_drop=0.15, noise=0.05, seeds=[0, 1, 2], inner=True, y_col="y3",
               n_classes=3, cohort="video")
    cfg.update(kw)
    return cfg


def _batch(banks, rows, train, cfg):
    xs, masks = zip(*(b.gather(rows) for b in banks))
    x, mask = torch.cat(xs, -1), masks[0]
    for m in masks[1:]:
        mask = mask & m
    x, mask = x.to(DEVICE), mask.to(DEVICE)
    if train:
        if cfg["frame_drop"] > 0:
            mask = mask & (torch.rand(mask.shape, device=DEVICE) > cfg["frame_drop"])
        if cfg["noise"] > 0:
            x = x + cfg["noise"] * torch.randn_like(x)
        empty = mask.sum(1) == 0
        mask[empty, 0] = True
    return x, mask


def _views(banks):
    return banks if isinstance(banks[0], (list, tuple)) else [banks]


def _ordinal_logp(logits, nc):
    """Cumulative logits P(y > j), j < nc-1 (first nc-1 outputs) -> monotone class log-probs."""
    s = torch.sigmoid(logits[:, :nc - 1])
    s = torch.cummin(s, dim=1).values
    p = torch.cat([1 - s[:, :1], s[:, :-1] - s[:, 1:], s[:, -1:]], 1).clamp(min=1e-6)
    return torch.log(p / p.sum(1, keepdim=True))


def _loss(logits, yb, weight, cfg):
    if cfg.get("loss", "ce") == "ordinal":
        nc = cfg["n_classes"]
        tgt = (yb[:, None] > torch.arange(nc - 1, device=yb.device)[None]).float()
        bce = F.binary_cross_entropy_with_logits(logits[:, :nc - 1], tgt, reduction="none")
        return (bce.sum(1) * weight[yb]).sum() / weight[yb].sum()
    return F.cross_entropy(logits, yb, weight=weight)


def fit_predict(banks, w, train_rows, pred_rows, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    views = _views(banks)
    for v in views:
        for b in v:
            b.set_norm(train_rows)
    y = torch.from_numpy(w[cfg["y_col"]].values).long()
    d_in = sum(b.dim for b in views[0])
    model = HEADS[cfg["head"]](d_in, cfg["n_classes"], d=cfg["d"], drop=cfg["drop"]).to(DEVICE)
    counts = np.bincount(y[train_rows].numpy(), minlength=cfg["n_classes"]).astype(float)
    weight = torch.tensor(counts.sum() / (cfg["n_classes"] * np.maximum(counts, 1)), dtype=torch.float32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["epochs"] * int(np.ceil(len(train_rows) / cfg["bs"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(cfg["epochs"]):
        perm = rng.permutation(train_rows)
        for i in range(0, len(perm), cfg["bs"]):
            rows = perm[i:i + cfg["bs"]]
            v = views[rng.integers(len(views))] if len(views) > 1 else views[0]
            x, mask = _batch(v, rows, True, cfg)
            logits, _ = model(x, mask)
            loss = _loss(logits, y[rows].to(DEVICE), weight, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out, att = [], []
    with torch.no_grad():
        for i in range(0, len(pred_rows), 256):
            pr, a = 0, None
            for v in views:
                x, mask = _batch(v, pred_rows[i:i + 256], False, cfg)
                lg, av = model(x, mask)
                lp_v = _ordinal_logp(lg, cfg["n_classes"]) if cfg.get("loss", "ce") == "ordinal" \
                    else F.log_softmax(lg, -1)
                pr = pr + lp_v.exp() / len(views)
                a = av if a is None else a
            out.append(torch.log(pr + 1e-9).cpu().numpy())
            att.append(a.cpu().numpy())
    return np.concatenate(out), np.concatenate(att), model


def cohort_rows(w, cohort):
    if cohort == "video":
        return np.where(w.avail_video.values == 1)[0]
    if cohort == "all":
        return np.arange(len(w))
    if cohort == "tel":
        return np.where(w.avail_tel.values == 1)[0]
    raise ValueError(cohort)


def run_cv(name, make_banks, cfg, log_wandb=True, windows=None):
    """make_banks(windows) -> list[Bank]. Saves preds + summary under results/m8/."""
    t0 = time.time()
    w = load_windows() if windows is None else windows
    banks = make_banks(w)
    views = [[b.to(DEVICE) for b in v] for v in _views(banks)]
    banks = views if len(views) > 1 else views[0]
    inner = load_inner()
    rows_all = cohort_rows(w, cfg["cohort"])
    for b in (b for v in views for b in v):
        ok = (b.index[rows_all] >= 0).any(1)
        rows_all = rows_all[ok]
    if cfg.get("protocol", "5fold") == "loso":  # V1: one outer fold per subject, no inner CV
        assert not cfg["inner"], "LOSO runs without inner CV"
        subs = sorted(set(w.subject.values[rows_all]))
        w = w.copy()
        w["fold"] = w.subject.map({s: i for i, s in enumerate(subs)}).fillna(-1).astype(int)
    folds = sorted(set(w.fold.values[rows_all]))
    nc = cfg["n_classes"]
    test_lp = np.zeros((len(cfg["seeds"]), len(w), nc), np.float32)
    oof_lp = np.full((len(w), 5, nc), np.nan, np.float32)
    att = np.zeros((len(w), views[0][0].index.shape[1]), np.float32)
    for k in folds:
        tr = rows_all[w.fold.values[rows_all] != k]
        te = rows_all[w.fold.values[rows_all] == k]
        for si, seed in enumerate(cfg["seeds"]):
            lp, a, _ = fit_predict(banks, w, tr, te, cfg, seed)
            test_lp[si, te] = lp
            if si == 0:
                att[te, :a.shape[1]] = a
        if cfg["inner"]:
            inner_of = np.array([inner[k][s] for s in w.subject.values[tr]])
            for j in range(N_INNER):
                itr, iva = tr[inner_of != j], tr[inner_of == j]
                if len(iva) == 0:
                    continue
                lp, _, _ = fit_predict(banks, w, itr, iva, cfg, 0)
                oof_lp[iva, k] = lp
        print(f"[{name}] outer fold {k} done ({time.time()-t0:.0f}s)", flush=True)

    res = score(name, w, rows_all, test_lp, oof_lp, cfg)
    res["minutes"] = (time.time() - t0) / 60
    run_dir = RESULTS / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "preds.npz", rows=rows_all, test_lp=test_lp[:, rows_all], oof_lp=oof_lp[rows_all],
             att=att[rows_all])
    (RESULTS / f"{name}.json").write_text(json.dumps(dict(cfg=cfg, **res), indent=1))
    if log_wandb:
        _wandb(name, cfg, res)
    return res


def score(name, w, rows, test_lp, oof_lp, cfg):
    win = w.iloc[rows].reset_index(drop=True)
    y_col, nc = cfg["y_col"], cfg["n_classes"]
    ens = np.log(np.exp(test_lp[:, rows]).mean(0) + 1e-9)
    per_seed = [summarize(win, test_lp[s, rows], y_col, nc)["fold_mean_f1"] for s in range(len(test_lp))]
    out = dict(raw=summarize(win, ens, y_col, nc), per_seed_fold_mean_f1=per_seed)
    if cfg["inner"]:
        from sklearn.metrics import f1_score
        betas, sm = {}, ens.copy()
        for k in range(5):
            tr_mask = (win.fold.values != k) & ~np.isnan(oof_lp[rows, k, 0])
            ow = win[tr_mask].reset_index(drop=True)
            olp = oof_lp[rows][tr_mask, k]
            sc = {b: f1_score(ow[y_col], causal_ema(ow, olp, b).argmax(1), average="macro",
                              labels=list(range(nc)), zero_division=0) for b in BETAS}
            betas[k] = max(sc, key=sc.get)
            m = win.fold.values == k
            sm[m] = causal_ema(win[m].reset_index(drop=True), ens[m], betas[k])
        out["ema"] = summarize(win, sm, y_col, nc)
        out["beta_per_fold"] = {str(k): v for k, v in betas.items()}
        # Smoother family (none / EMA / HMM) chosen per fold on that fold's inner OOF.
        cands = [("ema", b) for b in BETAS] + [("hmm", s) for s in STAYS]
        chosen, sm2 = {}, ens.copy()
        for k in range(5):
            tr_mask = (win.fold.values != k) & ~np.isnan(oof_lp[rows, k, 0])
            ow = win[tr_mask].reset_index(drop=True)
            olp = oof_lp[rows][tr_mask, k]
            prior = np.bincount(win[y_col].values[win.fold.values != k], minlength=nc) / (win.fold.values != k).sum()
            sc = {c: f1_score(ow[y_col], smooth(ow, olp, *c, prior).argmax(1), average="macro",
                              labels=list(range(nc)), zero_division=0) for c in cands}
            chosen[k] = max(sc, key=sc.get)
            m = win.fold.values == k
            sm2[m] = smooth(win[m].reset_index(drop=True), ens[m], *chosen[k], prior)
        out["smooth"] = summarize(win, sm2, y_col, nc)
        out["smooth_per_fold"] = {str(k): list(v) for k, v in chosen.items()}
        out["hmm_grid"] = {str(s): summarize(win, smooth(win, ens, "hmm", s, np.bincount(
            win[y_col].values, minlength=nc) / len(win)), y_col, nc)["fold_mean_f1"] for s in STAYS}
        oof_mask = ~np.isnan(oof_lp[rows, :, 0])
        out["inner_oof_f1_mean"] = float(np.mean([
            f1_score(win[y_col].values[oof_mask[:, k]], oof_lp[rows][oof_mask[:, k], k].argmax(1),
                     average="macro", labels=list(range(nc)), zero_division=0) for k in range(5)]))
    r, e = out["raw"], out.get("ema")
    print(f"[{name}] raw fold-mean F1 {r['fold_mean_f1']:.3f}±{r['fold_std_f1']:.3f} pooled "
          f"{r['pooled']['macro_f1']:.3f} CI{np.round(r['pooled_f1_ci95'], 3).tolist()} | per-fold "
          f"{[round(m['macro_f1'], 3) for m in r['per_fold'].values()]}"
          + (f" | EMA {e['fold_mean_f1']:.3f} (betas {out['beta_per_fold']}) | inner-OOF "
             f"{out['inner_oof_f1_mean']:.3f} | smooth {out['smooth']['fold_mean_f1']:.3f} "
             f"{out['smooth_per_fold']}" if e else ""), flush=True)
    return out


def rescore(name):
    """Re-run score() from a finished run's saved predictions (no retraining)."""
    cfg = json.loads((RESULTS / f"{name}.json").read_text())["cfg"]
    old = json.loads((RESULTS / f"{name}.json").read_text())
    d = np.load(RESULTS / "runs" / name / "preds.npz")
    w, rows = load_windows(), d["rows"]
    test_lp = np.zeros((d["test_lp"].shape[0], len(w), d["test_lp"].shape[2]), np.float32)
    test_lp[:, rows] = d["test_lp"]
    oof_lp = np.full((len(w), 5, d["oof_lp"].shape[2]), np.nan, np.float32)
    oof_lp[rows] = d["oof_lp"]
    res = score(name, w, rows, test_lp, oof_lp, cfg)
    res["minutes"] = old.get("minutes")
    (RESULTS / f"{name}.json").write_text(json.dumps(dict(cfg=cfg, **res), indent=1))
    return res


def _wandb(name, cfg, res):
    try:
        import wandb
        run = wandb.init(entity="m25ai2065-iitj", project="drowsiness_project", name=name,
                         group=name.split("_")[0], config=cfg, reinit=True)
        flat = {"fold_mean_f1": res["raw"]["fold_mean_f1"], "fold_std_f1": res["raw"]["fold_std_f1"],
                "pooled_f1": res["raw"]["pooled"]["macro_f1"], "pooled_acc": res["raw"]["pooled"]["acc"]}
        if "ema" in res:
            flat.update(ema_fold_mean_f1=res["ema"]["fold_mean_f1"], inner_oof_f1=res["inner_oof_f1_mean"])
        for k, m in res["raw"]["per_fold"].items():
            flat[f"fold{k}_f1"] = m["macro_f1"]
        run.summary.update(flat)
        run.finish()
    except Exception as e:  # logging must never kill a run
        print(f"[wandb] skipped: {e}", flush=True)
