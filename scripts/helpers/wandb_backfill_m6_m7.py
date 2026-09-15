"""
Backfill M6 (lite + full) per-fold results into wandb, and log the missing
aggregate summary run for M7 (M7's individual folds were already logged live
during training via scripts/m7_train.py -- only its cross-fold summary is missing).

M6 has no wandb integration in src/models/m6_train.py at all, so both variants'
5 folds each need full backfill from their saved results/reports/M6_{variant}_results.json
(per-epoch train_loss/train_acc/val_loss/val_acc history + final test metrics,
per-class F1 computed from the saved y_true/y_pred since M6's fold dicts don't
store it directly the way M1-4's did). Follows the same run-naming/tagging
convention as scripts/wandb_backfill.py (per-fold "_backfill" runs + one
"_summary" aggregate run per model).
"""
import json
from pathlib import Path

import wandb
from sklearn.metrics import f1_score

REPO = Path("/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system")
ENTITY = "m25ai2065-iitj"
PROJECT = "drowsiness_project"
CLASS_NAMES = ["Alert", "LowVigilant", "Drowsy"]


def backfill_m6(variant: str):
    model_name = f"M6_{variant}"
    results_path = REPO / "results" / "reports" / f"{model_name}_results.json"
    if not results_path.exists():
        print(f"[skip] {model_name}: no results file at {results_path}")
        return

    data = json.load(open(results_path))
    print(f"\n=== Backfilling {model_name} ({len(data['folds'])} folds) ===")

    arch = ("Visual BiLSTM(128) + CAN BiLSTM(64), concat, dense" if variant == "lite"
            else "Visual Transformer + CAN BiLSTM(64), bidirectional cross-attention, dense")

    for fold in data["folds"]:
        fold_idx = fold["fold"]
        run = wandb.init(
            entity=ENTITY, project=PROJECT,
            name=f"{model_name}_fold{fold_idx}_backfill",
            job_type="backfill",
            tags=[model_name, f"fold{fold_idx}", "m6-alignment-fix", "backfilled"],
            config={
                "model": model_name, "fold": fold_idx, "variant": variant,
                "modality": "60fps-aligned visual embeddings + CAN telemetry",
                "architecture": arch,
                "test_subjects": fold.get("test_subjects"),
            },
            reinit=True,
        )

        history = fold["history"]
        n_epochs = len(history["train_acc"])
        for e in range(n_epochs):
            run.log({
                "epoch": e,
                "train/accuracy": history["train_acc"][e],
                "train/loss": history["train_loss"][e],
                "val/accuracy": history["val_acc"][e],
                "val/loss": history["val_loss"][e],
            })

        y_true, y_pred = fold["y_true"], fold["y_pred"]
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

        run.summary["test_accuracy"] = fold["accuracy"]
        run.summary["test_macro_f1"] = fold["macro_f1"]
        for i, cn in enumerate(CLASS_NAMES):
            run.summary[f"test_f1_{cn}"] = float(per_class_f1[i])
        run.summary["train_time_s"] = fold["train_time_s"]
        run.finish()
        print(f"  fold {fold_idx}: logged {n_epochs} epochs, "
              f"test_acc={fold['accuracy']:.4f} test_macro_f1={fold['macro_f1']:.4f}")

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name=f"{model_name}_summary",
        job_type="summary",
        tags=[model_name, "m6-alignment-fix", "aggregate"],
        config={"model": model_name, "variant": variant, "architecture": arch},
        reinit=True,
    )
    run.summary["mean_acc"] = data["mean_acc"]
    run.summary["std_acc"] = data["std_acc"]
    run.summary["mean_f1"] = data["mean_f1"]
    run.summary["std_f1"] = data["std_f1"]
    run.finish()
    print(f"  {model_name} summary: acc={data['mean_acc']:.4f}+-{data['std_acc']:.4f} "
          f"f1={data['mean_f1']:.4f}+-{data['std_f1']:.4f}")


def backfill_m7_summary():
    results_path = REPO / "results" / "reports_regen" / "M7_results.json"
    if not results_path.exists():
        print(f"[skip] M7: no results file at {results_path}")
        return
    data = json.load(open(results_path))
    print(f"\n=== Logging M7 aggregate summary (folds already logged live) ===")

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name="M7_summary",
        job_type="summary",
        tags=["M7", "new-build", "aggregate"],
        config={
            "model": "M7",
            "architecture": "EfficientNet-B0 (frozen) -> BiLSTM(256, bidir) -> attention pooling -> MLP(256,128,3)",
        },
        reinit=True,
    )
    run.summary["mean_acc"] = data["mean_acc"]
    run.summary["std_acc"] = data["std_acc"]
    run.summary["mean_f1"] = data["mean_f1"]
    run.summary["std_f1"] = data["std_f1"]
    run.finish()
    print(f"  M7 summary: acc={data['mean_acc']:.4f}+-{data['std_acc']:.4f} "
          f"f1={data['mean_f1']:.4f}+-{data['std_f1']:.4f}")


if __name__ == "__main__":
    backfill_m6("lite")
    backfill_m6("full")
    backfill_m7_summary()
    print("\nDone.")
