"""
M7 — Temporal Infrared Video Model (built from scratch this session).

Architecture per Kavipriya's thesis spec (report_summary.md §3.5.7 / repo_understanding.md §3):
    16 sampled IR frames/window -> EfficientNet-B0 (frozen, precomputed by
    scripts/m7_extract_features.py) -> BiLSTM(256, bidirectional) -> attention pooling
    -> MLP(512 -> 256 -> 128 -> 3)

No src/models/m7_*.py exists anywhere in Kavipriya's repo history (confirmed in
docs/repo_understanding.md) -- this is new code, not a port of anything she wrote, unlike
scripts/regen_and_train.py / m4_train.py / m5_train.py which reuse her architecture.py/train.py.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

REPO = Path("/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system")
FEAT_DIR = REPO / "datasets" / "processed" / "uldd_m7_vis"
CKPT_DIR = REPO / "models" / "checkpoints_regen"
RESULTS_DIR = REPO / "results" / "reports_regen"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_NAMES = ["Alert", "LowVigilant", "Drowsy"]
WANDB_ENTITY = "m25ai2065-iitj"
WANDB_PROJECT = "drowsiness_project"

T_VIS, D_IN, D_LSTM = 16, 1280, 256


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):  # x: (B, T, D)
        w = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * w).sum(dim=1), w.squeeze(-1)  # (B, D), (B, T)


class M7Model(nn.Module):
    def __init__(self, d_in=D_IN, d_lstm=D_LSTM, n_classes=3, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_lstm, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling(d_lstm * 2)
        self.classifier = nn.Sequential(
            nn.Linear(d_lstm * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):  # x: (B, 16, 1280)
        h, _ = self.lstm(x)          # (B, 16, 512)
        pooled, attn_w = self.pool(h)  # (B, 512)
        return self.classifier(pooled), attn_w


def train_fold(fold_idx, epochs=100, batch_size=32, lr=1e-3, es_patience=15):
    d = np.load(FEAT_DIR / f"fold_{fold_idx}.npz")
    X_train, y_train = d["X_vis_train"], d["y_train"]
    X_test, y_test = d["X_vis_test"], d["y_test"]

    # Held-out validation split from train (subject-agnostic here, same convention as
    # Kavipriya's train.py validation_split=0.15 -- see repo_understanding.md caveat on this)
    n_val = int(len(X_train) * 0.15)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X_train))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    cw = compute_class_weight("balanced", classes=np.arange(3), y=y_tr)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    print(f"  [Fold {fold_idx}] train={len(X_tr)} val={len(X_val)} test={len(X_test)} "
          f"class_weights={cw.round(3).tolist()}")

    model = M7Model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)
    loss_fn = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.1)

    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, name=f"M7_fold{fold_idx}", job_type="train",
        tags=["M7", f"fold{fold_idx}", "efficientnet-b0-bilstm", "new-build"],
        config={"model": "M7", "fold": fold_idx, "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "es_patience": es_patience, "n_train": len(X_tr), "n_val": len(X_val),
                "n_test": len(X_test)},
        reinit=True,
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=DEVICE)

    best_val_acc, best_state, bad_epochs = -1.0, None, 0
    t0 = time.time()
    n = len(X_tr_t)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            bidx = perm[i:i + batch_size]
            xb = X_tr_t[bidx].to(DEVICE)
            yb = y_tr_t[bidx].to(DEVICE)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(bidx)
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val_t)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()
            val_loss = loss_fn(val_logits, y_val_t).item()
        sched.step(val_acc)

        run.log({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc,
                  "val_loss": val_loss, "lr": opt.param_groups[0]["lr"]})

        if val_acc > best_val_acc:
            best_val_acc, bad_epochs = val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= es_patience:
                print(f"    early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    model.load_state_dict(best_state)
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        test_logits, _ = model(X_test_t)
        y_pred = test_logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)

    ckpt_path = CKPT_DIR / f"M7_fold{fold_idx}.pt"
    torch.save(model.state_dict(), ckpt_path)
    run.summary["test_accuracy"] = acc
    run.summary["test_macro_f1"] = macro_f1
    run.finish()

    print(f"  [Fold {fold_idx}] acc={acc:.4f}  macro_f1={macro_f1:.4f}  time={train_time:.0f}s")
    return {
        "fold": fold_idx, "accuracy": acc, "macro_f1": macro_f1,
        "per_class_f1": per_class_f1, "report": report,
        "y_true": y_test.tolist(), "y_pred": y_pred.tolist(),
        "train_time_s": train_time, "checkpoint": str(ckpt_path),
    }


def main():
    print("=" * 60)
    print("  Cross-Validation: M7  (5 folds)")
    print("=" * 60)
    results = []
    for k in range(5):
        results.append(train_fold(k))

    acc = [r["accuracy"] for r in results]
    f1 = [r["macro_f1"] for r in results]
    summary = {
        "model": "M7", "folds": results,
        "mean_acc": float(np.mean(acc)), "std_acc": float(np.std(acc)),
        "mean_f1": float(np.mean(f1)), "std_f1": float(np.std(f1)),
    }
    with open(RESULTS_DIR / "M7_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "-" * 60)
    print(f"  M7 Summary:")
    print(f"  Accuracy  : {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
    print(f"  Macro F1  : {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
