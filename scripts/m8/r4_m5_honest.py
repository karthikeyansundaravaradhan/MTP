"""R4: M5 (YOLOv8n-cls on 1 fps IR frames) with honest checkpoint selection.

--mode lastpt   R4a: score the existing M5 last.pt per fold (no retraining).
--mode retrain  R4:  retrain with best.pt chosen on inner-fold-0 *training* subjects
                (never the outer test subjects), 0.5 fps train/val frames, fixed 10 epochs.

Both modes save per-frame test probabilities to results/m8/runs/<run>/fold_k.npz
(subject, session, timestamp_s, y3, probs) for window-level scoring by m8.evaluate.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.config import CACHE, CLASS_NAMES, FOLDS, KAVI, NO_VIDEO, RESULTS  # noqa: E402
from m8.windows import load_inner  # noqa: E402

FRAME_META = KAVI / "datasets" / "yolo_frames_regen" / "metadata.csv"
OLD_CKPT = KAVI / "models" / "checkpoints_regen" / "yolo_cls"
YOLO_BASE = KAVI / "yolov8n-cls.pt"


def predict(model, df, batch=128):
    """Return (N,3) probabilities in CLASS_NAMES order."""
    order = [list(model.names.values()).index(c) for c in CLASS_NAMES]
    out = []
    paths = df.path.tolist()
    for i in range(0, len(paths), batch):
        for r in model.predict(paths[i:i + batch], device="mps", verbose=False, imgsz=224):
            out.append(r.probs.data.cpu().numpy()[order])
    return np.stack(out).astype(np.float32)


def save(run_dir, k, df, probs_by_name):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / f"fold_{k}.npz", subject=np.asarray(df.subject.tolist(), dtype="U2"),
             session=np.asarray(df.session.tolist(), dtype="U1"), timestamp_s=df.timestamp_s.values,
             y3=df["class"].values.astype(np.int64), **probs_by_name)


def link_split(df, root, split):
    for cn in CLASS_NAMES:
        (root / split / cn).mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        dst = root / split / r.class_name / f"{r.subject}_{r.session}_{Path(r.path).name}"
        if not dst.exists():
            os.link(r.path, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lastpt", "retrain"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    args = ap.parse_args()

    from ultralytics import YOLO
    meta = pd.read_csv(FRAME_META, dtype={"subject": str, "session": str})
    inner = load_inner()

    if args.mode == "lastpt":
        run_dir = RESULTS / "runs" / "R4a"
        for k in args.folds:
            test = meta[meta.subject.isin(FOLDS[k])]
            model = YOLO(str(OLD_CKPT / f"M5_fold{k}" / "weights" / "last.pt"))
            pl = predict(model, test)
            pb = predict(YOLO(str(OLD_CKPT / f"M5_fold{k}" / "weights" / "best.pt")), test)
            save(run_dir, k, test, {"probs": pl, "probs_best_testselected": pb})
            print(f"[R4a] fold {k}: {len(test)} frames scored", flush=True)
        return

    import wandb
    for seed in args.seeds:
        run_dir = RESULTS / "runs" / f"R4_seed{seed}"
        for k in args.folds:
            if (run_dir / f"fold_{k}.npz").exists():
                print(f"[R4] seed {seed} fold {k} done already, skipping", flush=True)
                continue
            val_subs = sorted(s for s, i in inner[k].items() if i == 0 and s not in NO_VIDEO)
            test = meta[meta.subject.isin(FOLDS[k])]
            rest = meta[~meta.subject.isin(FOLDS[k])]
            half = rest[rest.timestamp_s.round().astype(int) % 2 == 0]
            tr, va = half[~half.subject.isin(val_subs)], half[half.subject.isin(val_subs)]
            root = CACHE / "yolo_r4" / f"fold_{k}"
            link_split(tr, root, "train")
            link_split(va, root, "val")
            print(f"[R4] seed {seed} fold {k}: train {len(tr)} / val {len(va)} ({''.join(val_subs)}) "
                  f"/ test {len(test)}", flush=True)

            run = wandb.init(entity="m25ai2065-iitj", project="drowsiness_project",
                             name=f"R4_m5honest_s{seed}_fold{k}", group="R4_m5honest",
                             config=dict(fold=k, seed=seed, epochs=args.epochs, fps=0.5,
                                         optimizer="SGD", lr0=0.01, momentum=0.9,
                                         val_subjects=val_subs), reinit=True)
            t0 = time.time()
            model = YOLO(str(YOLO_BASE))
            # M5 ran optimizer=auto with >10k iterations, which Ultralytics resolves to SGD(0.01, 0.9);
            # pinned here because R4's shorter schedule would make auto pick AdamW instead.
            model.train(data=str(root), epochs=args.epochs, imgsz=224, batch=32,
                        optimizer="SGD", lr0=0.01, momentum=0.9,
                        lrf=0.01, dropout=0.3, patience=args.epochs, seed=seed,
                        deterministic=False, device="mps", workers=4,
                        project=str(CACHE / "yolo_r4_runs"), name=f"s{seed}_fold{k}",
                        exist_ok=True, verbose=False, plots=False)
            wdir = CACHE / "yolo_r4_runs" / f"s{seed}_fold{k}" / "weights"
            val_full = rest[rest.subject.isin(val_subs)]
            save(run_dir / "innerval", k, val_full, {"probs": predict(YOLO(str(wdir / "best.pt")), val_full)})
            pb = predict(YOLO(str(wdir / "best.pt")), test)
            pl = predict(YOLO(str(wdir / "last.pt")), test)
            save(run_dir, k, test, {"probs": pb, "probs_last": pl})
            acc = float((pb.argmax(1) == test["class"].values).mean())
            run.summary.update(dict(test_frame_acc=acc, train_minutes=(time.time() - t0) / 60))
            run.finish()
            print(f"[R4] seed {seed} fold {k}: frame acc {acc:.3f}, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    print("[R4] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
