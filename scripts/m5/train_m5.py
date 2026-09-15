"""
M5 — YOLOv8n-cls 5-fold training, replicating
driver-drowsiness-detection-system/notebooks/09-uldd-m5-yolo-cls.ipynb cells 9-13
as a headless script (so it can run detached in the background), reading frames
from datasets/yolo_frames_regen/ (produced by m4_m5_extract.py) instead of her
original datasets/yolo_frames/.

Same hyperparameters as her notebook: YOLOv8n-cls, 20 epochs, batch 32, imgsz 224,
lr0=1e-3, lrf=0.01, dropout=0.3, patience=15.
"""
import os, shutil, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import wandb

WANDB_ENTITY = "m25ai2065-iitj"
WANDB_PROJECT = "drowsiness_project"

REPO = Path("/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system")
FRAME_DIR = REPO / 'datasets' / 'yolo_frames_regen'
YOLO_DIR = REPO / 'datasets' / 'yolo_cls_regen'
CKPT_DIR = REPO / 'models' / 'checkpoints_regen'
REPORT_DIR = REPO / 'results' / 'reports_regen'
for d in [YOLO_DIR, CKPT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SUBJECTS = list('ABCDEFGHIJKLMNOPQRS')
NO_VIDEO = {'B', 'I', 'M'}
CLASS_NAMES = ['Alert', 'LowVigilant', 'Drowsy']
FOLDS = [list('ABCD'), list('EFGH'), list('IJK'), list('LMNO'), list('PQRS')]

YOLO_EPOCHS = 20
YOLO_BATCH = 32
YOLO_IMGSZ = 224
YOLO_BASE = 'yolov8n-cls.pt'
MODEL_NAME = 'M5'

from ultralytics import YOLO

meta_df = pd.read_csv(FRAME_DIR / 'metadata.csv')
print(f'Loaded metadata: {len(meta_df)} frames, subjects={sorted(meta_df["subject"].unique())}')

all_fold_results = []

for fold_idx, test_subjects in enumerate(FOLDS):
    print(f'\n{"="*60}\n  Fold {fold_idx}  |  test subjects = {test_subjects}\n{"="*60}')

    test_subs = [s for s in test_subjects if s not in NO_VIDEO]
    train_subs = [s for s in SUBJECTS if s not in NO_VIDEO and s not in test_subjects]

    test_df = meta_df[meta_df['subject'].isin(test_subs)]
    train_df = meta_df[meta_df['subject'].isin(train_subs)]
    print(f'  Train: {len(train_df):,} frames from {sorted(train_subs)}')
    print(f'  Test : {len(test_df):,} frames from {sorted(test_subs)}')

    if len(test_df) == 0 or len(train_df) == 0:
        print('  Insufficient data — skipping fold')
        continue

    fold_dir = YOLO_DIR / f'fold_{fold_idx}'
    for split in ['train', 'val']:
        for cn in CLASS_NAMES:
            (fold_dir / split / cn).mkdir(parents=True, exist_ok=True)

    def _link_frames(df, split):
        linked = 0
        for _, row in df.iterrows():
            src = Path(row['path'])
            if not src.exists():
                continue
            dst = fold_dir / split / row['class_name'] / f"{row['subject']}_{row['session']}_{src.name}"
            if not dst.exists():
                try:
                    os.link(str(src), str(dst))
                except OSError:
                    shutil.copy2(str(src), str(dst))
            linked += 1
        return linked

    existing_train = sum(1 for _ in (fold_dir / 'train').rglob('*.jpg')) if (fold_dir / 'train').exists() else 0
    if existing_train > 100:
        print(f'  Fold {fold_idx} dataset already exists ({existing_train} train frames)')
    else:
        n_tr = _link_frames(train_df, 'train')
        n_te = _link_frames(test_df, 'val')
        print(f'  Linked: {n_tr} train + {n_te} val frames')

    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT,
        name=f"M5_fold{fold_idx}", job_type="train",
        tags=["M5", f"fold{fold_idx}", "yolov8n-cls", "reproduction"],
        config={
            "model": "M5", "fold": fold_idx, "architecture": "YOLOv8n-cls",
            "epochs": YOLO_EPOCHS, "batch": YOLO_BATCH, "imgsz": YOLO_IMGSZ,
            "lr0": 1e-3, "lrf": 0.01, "dropout": 0.3, "patience": 15,
            "n_train": len(train_df), "n_test": len(test_df),
        },
        reinit=True,
    )

    def _on_fit_epoch_end(trainer):
        metrics = {k: float(v) for k, v in trainer.metrics.items()
                   if isinstance(v, (int, float))}
        metrics["epoch"] = int(trainer.epoch)
        run.log(metrics)

    best_pt = CKPT_DIR / 'yolo_cls' / f'M5_fold{fold_idx}' / 'weights' / 'best.pt'
    if best_pt.exists():
        print(f'  best.pt already exists — loading, skipping training')
        model = YOLO(str(best_pt))
    else:
        print(f'  Training YOLOv8n-cls (epochs={YOLO_EPOCHS}) ...')
        model = YOLO(YOLO_BASE)
        model.add_callback('on_fit_epoch_end', _on_fit_epoch_end)
        model.train(
            data=str(fold_dir), epochs=YOLO_EPOCHS, imgsz=YOLO_IMGSZ, batch=YOLO_BATCH,
            project=str(CKPT_DIR / 'yolo_cls'), name=f'M5_fold{fold_idx}', exist_ok=True,
            verbose=True, patience=15, lr0=1e-3, lrf=0.01, dropout=0.3,
            device='mps', workers=8,
        )
        if best_pt.exists():
            model = YOLO(str(best_pt))
            print(f'  Loaded best weights from {best_pt.name}')

    yolo_names = model.names
    yolo_to_ours = {yi: CLASS_NAMES.index(yn) for yi, yn in yolo_names.items()}
    print(f'  Class mapping: {yolo_names} -> {yolo_to_ours}')

    y_true, y_pred = [], []
    for cn_idx, cn in enumerate(CLASS_NAMES):
        class_dir = fold_dir / 'val' / cn
        if not class_dir.exists():
            continue
        img_list = sorted(class_dir.glob('*.jpg'))
        if not img_list:
            continue
        preds = model.predict(source=str(class_dir), verbose=False, batch=YOLO_BATCH, device='mps')
        for p in preds:
            y_true.append(cn_idx)
            y_pred.append(yolo_to_ours[int(p.probs.top1)])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f'  [Fold {fold_idx}] acc={acc:.4f}  macro_f1={f1:.4f}')

    run.summary["test_accuracy"] = acc
    run.summary["test_macro_f1"] = f1
    run.finish()

    all_fold_results.append({
        'fold': fold_idx, 'accuracy': acc, 'macro_f1': f1,
        'y_true': y_true, 'y_pred': y_pred,
        'test_subjects': test_subs, 'n_train': len(train_df), 'n_test': len(test_df),
    })

    if best_pt.exists():
        dst_pt = CKPT_DIR / f'M5_fold{fold_idx}.pt'
        shutil.copy2(str(best_pt), str(dst_pt))
        print(f'  Saved -> {dst_pt}')

accs = [f['accuracy'] for f in all_fold_results]
f1s = [f['macro_f1'] for f in all_fold_results]
results = {
    'model_name': MODEL_NAME,
    'mean_acc': float(np.mean(accs)), 'std_acc': float(np.std(accs)),
    'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s)),
    'folds': all_fold_results,
}

def _to_ser(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _to_ser(v) for k, v in o.items()}
    if isinstance(o, list): return [_to_ser(i) for i in o]
    return o

with open(REPORT_DIR / f'{MODEL_NAME}_results.json', 'w') as f:
    json.dump(_to_ser(results), f, indent=2)

summary_run = wandb.init(
    entity=WANDB_ENTITY, project=WANDB_PROJECT,
    name="M5_summary", job_type="summary",
    tags=["M5", "reproduction", "aggregate"],
    config={"model": "M5", "architecture": "YOLOv8n-cls"},
    reinit=True,
)
summary_run.summary["mean_acc"] = results["mean_acc"]
summary_run.summary["std_acc"] = results["std_acc"]
summary_run.summary["mean_f1"] = results["mean_f1"]
summary_run.summary["std_f1"] = results["std_f1"]
summary_run.finish()

print(f'\n{"="*60}\n=== M5 YOLOv8-cls Results ===')
print(f'Accuracy : {results["mean_acc"]:.4f} +- {results["std_acc"]:.4f}')
print(f'Macro-F1 : {results["mean_f1"]:.4f} +- {results["std_f1"]:.4f}')
print(f'{"="*60}')
