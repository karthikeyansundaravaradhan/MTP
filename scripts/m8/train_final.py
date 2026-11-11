"""Deployment models: refit the selected configurations on ALL 16 video subjects (no held-out data).

Writes models/final/<name>.pt (torch) or .joblib (sklearn) with everything inference needs:
config, per-seed state dicts and the normalisation statistics fitted on the training rows.
Numbers for the report always come from the CV runs; these files are for inference.py / Gradio.

  vstar        IR-112 DINOv2 mean-pool head (V*), 3 seeds
  vstar_calA   V* on Alert-session-referenced tokens (R2-7; needs the driver's Alert-drive reference)
  m8           VIGIL MoE, fallback expert = telemetry + clock (C9d config), 3 seeds
  events       27 event features -> standardised LR, C chosen by subject-wise 4-fold CV
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8.config import CACHE, RESULTS, ROOT  # noqa: E402
from m8.data import video_bank  # noqa: E402
from m8.train import DEVICE, cohort_rows, default_cfg, fit_predict  # noqa: E402
from m8 import vigil  # noqa: E402
from m8.windows import load_windows  # noqa: E402
from r2_events_eval import CS, fit  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

OUT = ROOT / "models" / "final"


def cpu(t):
    return None if t is None else t.detach().cpu()


def head_model(w, rows, name, **bank_kw):
    bank = video_bank(w, "ir_112", **bank_kw).to(DEVICE)
    rows = rows[(bank.index[rows] >= 0).any(1)]
    cfg = default_cfg(head="mean", inner=False)
    states = []
    for seed in cfg["seeds"]:
        _, _, model = fit_predict([bank], w, rows, rows[:1], cfg, seed)
        states.append({k: v.cpu() for k, v in model.state_dict().items()})
    torch.save(dict(kind="head", cfg=cfg, stream="ir_112", tokens="cls_mean", bank_kw=bank_kw,
                    states=states, mu=cpu(bank.mu), sd=cpu(bank.sd), n_train_windows=len(rows),
                    subjects=sorted(set(w.subject.values[rows]))), OUT / f"{name}.pt")
    print(f"saved {name}: {len(rows)} windows", flush=True)


def m8_model(w, rows):
    cfg = json.loads((RESULTS / "C9d_ir112_moe_telclock.json").read_text())["cfg"]
    data = vigil.Data(w, cfg["vision"])
    for b in data.vb:
        rows = rows[(b.index[rows] >= 0).any(1)]
    states = []
    for seed in cfg["seeds"]:
        _, model = vigil.fit_predict(data, w, rows, rows[:1], "moe_telclock", cfg, seed, {}, return_model=True)
        states.append({k: v.cpu() for k, v in model.state_dict().items()})
    torch.save(dict(kind="vigil", mode="moe_telclock", cfg=cfg, states=states,
                    vision_mu=[cpu(b.mu) for b in data.vb], vision_sd=[cpu(b.sd) for b in data.vb],
                    tel_mu=cpu(data.tb.mu), tel_sd=cpu(data.tb.sd), clock_mu=cpu(data.c_mu),
                    clock_sd=cpu(data.c_sd), d_v=sum(b.dim for b in data.vb), d_t=data.tb.dim + data.clock.shape[1],
                    n_train_windows=len(rows)), OUT / "m8.pt")
    print(f"saved m8: {len(rows)} windows", flush=True)


def events_model(w, rows):
    d = np.load(CACHE / "events.npz")
    X, y, subj = d["X"], w.y3.values, w.subject.values
    subs = sorted(set(subj[rows]))
    inner = np.array([subs.index(s) % 4 for s in subj[rows]])
    sc = {}
    for C in CS:
        pred = np.zeros(len(rows), int)
        for j in range(4):
            m = inner == j
            pred[m] = fit(X[rows[~m]], y[rows[~m]], C).predict(X[rows[m]])
        sc[C] = f1_score(y[rows], pred, average="macro")
    C = max(sc, key=sc.get)
    joblib.dump(dict(kind="events", model=fit(X[rows], y[rows], C), C=C, names=list(d["names"]),
                     inner_f1=sc), OUT / "events.joblib")
    print(f"saved events: C={C}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = load_windows()
    rows = cohort_rows(w, "video")
    events_model(w, rows)
    head_model(w, rows, "vstar")
    head_model(w, rows, "vstar_calA", ref_s=240, ref_session="A")
    m8_model(w, rows)


if __name__ == "__main__":
    main()
