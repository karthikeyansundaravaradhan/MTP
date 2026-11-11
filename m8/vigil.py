"""C9: gated residual telemetry correction (VIGIL gate) on top of the frozen vision core.

    z = z_v + g * delta_t,     g = sigmoid(w . u + b),  b = -3 at init,  delta_t zero-init

z_v   : mean-pool head over each vision stream (concatenated), i.e. the V* core.
delta : MLP on the window-mean telemetry (or clock features for the C9-clock control).
u     : vision quality -- observed-frame fraction, face-missing rate of observed IR frames,
        no-frame flag -- recomputed after any blackout, so the gate can learn to open when
        vision fails.

Modes (all trained with the same ModDrop blackouts, so comparisons are fair -- critique #4):
  vision : z = z_v                 (vision-only comparator)
  gate   : z = z_v + g * delta     (M8 candidate)
  late   : z = z_v + delta         (ungated additive fusion)
  clock  : gate, with delta fed time-in-session instead of telemetry (C9-clock control)
  moe    : z = (1-g) z_v + g z_t   availability mixture of two full experts (aux CE on each);
           unlike the residual gate, the telemetry expert can only act through g, and g = 0
           when telemetry is absent
  moe_clock : moe with the clock expert in place of telemetry
  moe_telclock : moe whose fallback expert sees telemetry + clock (both available in a car);
           the clock is always present, so g is never forced to 0 -- missing/dropped telemetry
           only zeroes the telemetry part. Chosen by the user for M8 after C9c (disclosed).

Test conditions: a contiguous vision blackout of L seconds at a random position in every test
window, L in {0, 30, 45, 60}; the same positions for every mode and seed.
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from m8.config import RESULTS
from m8.data import series_bank, video_bank
from m8.evaluate import summarize
from m8.train import DEVICE, cohort_rows
from m8.windows import load_windows

CONDITIONS = [0, 30, 45, 60]


def default_cfg(**kw):
    cfg = dict(vision=["ir_112"], epochs=30, bs=64, lr=1e-3, wd=0.05, d=64, drop=0.3,
               frame_drop=0.15, noise=0.05, p_black=0.5, p_full=0.25, p_tel_drop=0.1,
               gate_bias=-3.0, aux_w=0.5, gate_lr_mult=1.0, seeds=[0, 1, 2], modes=["vision", "gate", "late", "clock"],
               y_col="y3", n_classes=3)
    cfg.update(kw)
    return cfg


class Vigil(nn.Module):
    def __init__(self, d_v, d_t, n_classes, mode, d=64, drop=0.3, gate_bias=-3.0, d_u=3):
        super().__init__()
        self.mode = mode
        self.v = nn.Sequential(nn.LayerNorm(d_v), nn.Dropout(0.2), nn.Linear(d_v, n_classes))
        self.t = nn.Sequential(nn.LayerNorm(d_t), nn.Linear(d_t, d), nn.GELU(), nn.Dropout(drop),
                               nn.Linear(d, n_classes))
        if not mode.startswith("moe"):
            nn.init.zeros_(self.t[-1].weight)
            nn.init.zeros_(self.t[-1].bias)
        self.g = nn.Linear(d_u, 1)
        nn.init.zeros_(self.g.weight)
        nn.init.constant_(self.g.bias, gate_bias)

    def forward(self, pv, pt, has_t, u, aux=False):
        z_v = self.v(pv)
        g = torch.sigmoid(self.g(u))
        if self.mode.startswith("moe"):
            g = g * has_t.unsqueeze(-1)
            z_t = self.t(pt)
            z = (1 - g) * z_v + g * z_t
            return (z, g, [z_v, z_t]) if aux else (z, g)
        if self.mode == "vision":
            return z_v, torch.zeros_like(g)
        delta = self.t(pt) * has_t.unsqueeze(-1)
        if self.mode == "late":
            return z_v + delta, torch.ones_like(g)
        return z_v + g * delta, g


class Data:
    """Vision banks + telemetry bank + clock features, gathered per batch on the device."""

    def __init__(self, w, vision):
        self.w = w
        self.vb = [video_bank(w, tag).to(DEVICE) for tag in vision]
        self.fps = [b.index.shape[1] / 60 for b in self.vb]
        self.tb = series_bank(w, "tel").to(DEVICE)
        t = w.start_s.values / 2400
        e = w.epoch_idx.values / 9
        self.clock = torch.tensor(np.stack([t, e, t ** 2, e ** 2, t * e], 1), dtype=torch.float32, device=DEVICE)
        self.has_tel = torch.tensor(w.avail_tel.values, dtype=torch.float32, device=DEVICE)
        fm = [torch.tensor(b.quality[..., 0], device=DEVICE) if b.quality.shape[-1] else None for b in self.vb]
        self.face_missing = fm

    def set_norm(self, rows):
        for b in self.vb + [self.tb]:
            b.set_norm(rows)
        c = self.clock[torch.as_tensor(rows, device=DEVICE)]
        self.c_mu, self.c_sd = c.mean(0), c.std(0) + 1e-6

    def batch(self, rows, mode, cfg, train, black_s=None, black_pos=None, rng=None):
        """black_s / black_pos: per-row blackout length (s) and start (fraction of the free span)."""
        n = len(rows)
        if train:
            black_s = np.zeros(n)
            hit = rng.random(n) < cfg["p_black"]
            full = rng.random(n) < cfg["p_full"]
            black_s[hit] = np.where(full[hit], 60.0, rng.uniform(15, 60, hit.sum()))
            black_pos = rng.random(n)
        pooled, obs_frac, fmiss = [], [], []
        for b, fps, fm in zip(self.vb, self.fps, self.face_missing):
            x, mask = b.gather(rows)
            T = mask.shape[1]
            if black_s is not None:
                L = np.round(np.asarray(black_s) * fps).astype(int)
                start = np.floor(np.asarray(black_pos) * (T - L + 1)).astype(int)
                ar = np.arange(T)[None]
                keep = ~((ar >= start[:, None]) & (ar < (start + L)[:, None]))
                mask = mask & torch.as_tensor(keep, device=DEVICE)
            if train and cfg["frame_drop"] > 0:
                mask = mask & (torch.rand(mask.shape, device=DEVICE) > cfg["frame_drop"])
            if train and cfg["noise"] > 0:
                x = x + cfg["noise"] * torch.randn_like(x)
            m = mask.unsqueeze(-1).float()
            pooled.append((x * m).sum(1) / m.sum(1).clamp(min=1))
            obs_frac.append(mask.float().mean(1))
            if fm is not None:
                f = fm[torch.as_tensor(rows, device=DEVICE)]
                fmiss.append((f * mask).sum(1) / mask.sum(1).clamp(min=1))
        pv = torch.cat(pooled, -1)
        of = torch.stack(obs_frac, -1).mean(-1)
        fmr = torch.stack(fmiss, -1).mean(-1) if fmiss else torch.zeros_like(of)
        u = torch.stack([of, fmr, (of == 0).float()], -1)
        r = torch.as_tensor(rows, device=DEVICE)
        if mode == "moe_telclock":
            xt, mt = self.tb.gather(rows)
            m = mt.unsqueeze(-1).float()
            tel = (xt * m).sum(1) / m.sum(1).clamp(min=1)
            avail = self.has_tel[r].clone()
            if train and cfg["p_tel_drop"] > 0:
                avail = avail * (torch.rand(n, device=DEVICE) > cfg["p_tel_drop"]).float()
            pt = torch.cat([tel * avail.unsqueeze(-1), (self.clock[r] - self.c_mu) / self.c_sd], -1)
            return pv, pt, torch.ones(n, device=DEVICE), u
        if mode in ("clock", "moe_clock"):
            pt, has_t = (self.clock[r] - self.c_mu) / self.c_sd, torch.ones(n, device=DEVICE)
        else:
            xt, mt = self.tb.gather(rows)
            m = mt.unsqueeze(-1).float()
            pt, has_t = (xt * m).sum(1) / m.sum(1).clamp(min=1), self.has_tel[r].clone()
            if train and cfg["p_tel_drop"] > 0:
                has_t = has_t * (torch.rand(n, device=DEVICE) > cfg["p_tel_drop"]).float()
        return pv, pt, has_t, u


def fit_predict(data, w, tr, te, mode, cfg, seed, test_conditions, return_model=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data.set_norm(tr)
    y_col, nc = cfg.get("y_col", "y3"), cfg.get("n_classes", 3)
    y = torch.from_numpy(w[y_col].values).long()
    d_t = {"clock": data.clock.shape[1], "moe_clock": data.clock.shape[1],
           "moe_telclock": data.tb.dim + data.clock.shape[1]}.get(mode, data.tb.dim)
    d_v = sum(b.dim for b in data.vb)
    model = Vigil(d_v, d_t, nc, mode, d=cfg["d"], drop=cfg["drop"], gate_bias=cfg["gate_bias"]).to(DEVICE)
    counts = np.bincount(w[y_col].values[tr], minlength=nc).astype(float)
    weight = torch.tensor(counts.sum() / (nc * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)
    gate_p = list(model.g.parameters())
    rest = [q for n, q in model.named_parameters() if not n.startswith("g.")]
    lr_g = cfg["lr"] * cfg["gate_lr_mult"]
    opt = torch.optim.AdamW([dict(params=rest, lr=cfg["lr"], weight_decay=cfg["wd"]),
                             dict(params=gate_p, lr=lr_g,
                                  weight_decay=cfg["wd"] if cfg["gate_lr_mult"] == 1 else 0.0)])
    steps = cfg["epochs"] * int(np.ceil(len(tr) / cfg["bs"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [cfg["lr"], lr_g], total_steps=steps, pct_start=0.1)
    model.train()
    for _ in range(cfg["epochs"]):
        perm = rng.permutation(tr)
        for i in range(0, len(perm), cfg["bs"]):
            rows = perm[i:i + cfg["bs"]]
            yb = y[rows].to(DEVICE)
            batch = data.batch(rows, mode, cfg, True, rng=rng)
            if mode.startswith("moe"):
                logits, _, (z_v, z_t) = model(*batch, aux=True)
                has_t = batch[2] > 0
                loss = F.cross_entropy(logits, yb, weight=weight) + cfg["aux_w"] * F.cross_entropy(z_v, yb, weight=weight)
                if has_t.any():
                    loss = loss + cfg["aux_w"] * F.cross_entropy(z_t[has_t], yb[has_t], weight=weight)
            else:
                logits, _ = model(*batch)
                loss = F.cross_entropy(logits, yb, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for L, pos in test_conditions.items():
            lps, gs = [], []
            for i in range(0, len(te), 512):
                sl = slice(i, i + 512)
                lg, g = model(*data.batch(te[sl], mode, cfg, False, black_s=np.full(len(te[sl]), float(L)),
                                          black_pos=pos[sl]))
                lps.append(F.log_softmax(lg, -1).cpu().numpy())
                gs.append(g.squeeze(-1).cpu().numpy())
            out[L] = (np.concatenate(lps), np.concatenate(gs))
    return (out, model) if return_model else out


def run(name, cfg):
    t0 = time.time()
    w = load_windows()
    data = Data(w, cfg["vision"])
    rows_all = cohort_rows(w, "video")
    for b in data.vb:
        rows_all = rows_all[(b.index[rows_all] >= 0).any(1)]
    S, M = len(cfg["seeds"]), len(cfg["modes"])
    y_col, nc = cfg.get("y_col", "y3"), cfg.get("n_classes", 3)
    lp = np.full((M, len(CONDITIONS), S, len(w), nc), np.nan, np.float32)
    gate = np.full((M, len(CONDITIONS), len(w)), np.nan, np.float32)
    for k in range(5):
        tr = rows_all[w.fold.values[rows_all] != k]
        te = rows_all[w.fold.values[rows_all] == k]
        pos_rng = np.random.default_rng(1000 + k)
        conds = {L: pos_rng.random(len(te)) for L in CONDITIONS}
        for mi, mode in enumerate(cfg["modes"]):
            for si, seed in enumerate(cfg["seeds"]):
                res = fit_predict(data, w, tr, te, mode, cfg, seed, conds)
                for ci, L in enumerate(CONDITIONS):
                    lp[mi, ci, si, te] = res[L][0]
                    if si == 0:
                        gate[mi, ci, te] = res[L][1]
        print(f"[{name}] outer fold {k} done ({time.time() - t0:.0f}s)", flush=True)

    win = w.iloc[rows_all].reset_index(drop=True)
    summ = {}
    for mi, mode in enumerate(cfg["modes"]):
        for ci, L in enumerate(CONDITIONS):
            ens = np.log(np.exp(lp[mi, ci][:, rows_all]).mean(0) + 1e-9)
            s = summarize(win, ens, y_col, nc)
            s["gate_mean"] = float(np.nanmean(gate[mi, ci, rows_all]))
            summ[f"{mode}@{L}"] = s
            print(f"[{name}] {mode:6s} blackout {L:2d}s  fold-mean {s['fold_mean_f1']:.3f}±{s['fold_std_f1']:.3f} "
                  f"pooled {s['pooled']['macro_f1']:.3f} CI{np.round(s['pooled_f1_ci95'], 3).tolist()} "
                  f"gate {s['gate_mean']:.3f}", flush=True)
    run_dir = RESULTS / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "preds.npz", rows=rows_all, lp=lp[..., rows_all, :], gate=gate[..., rows_all],
             modes=np.array(cfg["modes"]), conditions=np.array(CONDITIONS))
    (RESULTS / f"{name}.json").write_text(json.dumps(dict(cfg=cfg, minutes=(time.time() - t0) / 60,
                                                          summary=summ), indent=1, default=float))
    return summ
