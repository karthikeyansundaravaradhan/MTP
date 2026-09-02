"""
Backfill already-completed M1-M4 training runs into wandb.

These models were trained before wandb was wired up. Their full per-epoch
history (accuracy/loss/val_accuracy/val_loss/learning_rate) and final test
metrics were saved to results_regen/*.json at training time, so this replays
that saved history into wandb as if it had been logged live -- one run per
(model, fold), tagged so they're distinguishable from live future runs.

Run with the ddd-repo conda env (wandb installed there):
    /opt/anaconda3/envs/ddd-repo/bin/python scripts/wandb_backfill.py
"""
import json
from pathlib import Path

import wandb

REPO = Path("/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system")
RESULTS_DIR = REPO / "results" / "reports_regen"

ENTITY = "m25ai2065-iitj"
PROJECT = "drowsiness_project"

MODEL_CONFIG = {
    "M1": {"modality": "FAU", "architecture": "BiLSTM(128)->BiLSTM(64)->Dense(64)->Dense(3)"},
    "M2": {"modality": "Telemetry", "architecture": "LSTM(64)->LSTM(32)->Dense(32)->Dense(3)"},
    "M3": {"modality": "FAU+Telemetry", "architecture": "Cross-modal attention fusion"},
    "M4": {"modality": "10 real-time features (EAR/MAR/PERCLOS/head-pose/brow/jaw)", "architecture": "BiLSTM(64)->BiLSTM(32)->Dense(32)->Dense(3)"},
}

for model_name, cfg in MODEL_CONFIG.items():
    results_path = RESULTS_DIR / f"{model_name}_results.json"
    if not results_path.exists():
        print(f"[skip] {model_name}: no results file at {results_path}")
        continue

    data = json.load(open(results_path))
    print(f"\n=== Backfilling {model_name} ({len(data['folds'])} folds) ===")

    for fold in data["folds"]:
        fold_idx = fold["fold"]
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=f"{model_name}_fold{fold_idx}_backfill",
            job_type="backfill",
            tags=[model_name, f"fold{fold_idx}", "reproduction", "backfilled"],
            config={
                "model": model_name,
                "fold": fold_idx,
                "window_sec": 60,
                "target_hz": 4,
                **cfg,
            },
            reinit=True,
        )

        history = fold["history"]
        n_epochs = len(history["accuracy"])
        for e in range(n_epochs):
            run.log({
                "epoch": e,
                "train/accuracy": history["accuracy"][e],
                "train/loss": history["loss"][e],
                "val/accuracy": history["val_accuracy"][e],
                "val/loss": history["val_loss"][e],
                "learning_rate": history["learning_rate"][e],
            })

        # Final held-out test metrics as run summary
        m = fold["metrics"]
        run.summary["test_accuracy"] = m["accuracy"]
        run.summary["test_macro_f1"] = m["macro_f1"]
        for i, cn in enumerate(["Alert", "LowVigilant", "Drowsy"]):
            run.summary[f"test_f1_{cn}"] = m["per_class_f1"][i]
        run.summary["train_time_s"] = fold["train_time_s"]

        run.finish()
        print(f"  fold {fold_idx}: logged {n_epochs} epochs, "
              f"test_acc={m['accuracy']:.4f} test_macro_f1={m['macro_f1']:.4f}")

    # One extra summary-only run per model with the aggregate mean/std
    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name=f"{model_name}_summary",
        job_type="summary",
        tags=[model_name, "reproduction", "aggregate"],
        config={"model": model_name, **cfg},
        reinit=True,
    )
    run.summary["mean_acc"] = data["mean_acc"]
    run.summary["std_acc"] = data["std_acc"]
    run.summary["mean_f1"] = data["mean_f1"]
    run.summary["std_f1"] = data["std_f1"]
    run.finish()
    print(f"  {model_name} summary: acc={data['mean_acc']:.4f}+-{data['std_acc']:.4f} "
          f"f1={data['mean_f1']:.4f}+-{data['std_f1']:.4f}")

print("\nDone.")
