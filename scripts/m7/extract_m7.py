"""
M7 feature extraction — builds the EfficientNet-B0 visual feature cache Kavipriya's
thesis describes (§3.5.7 / ARCHITECTURE_VERIFICATION.md) but never actually coded.

Never committed by Kavipriya: no src/models/m7_*.py exists in her repo (confirmed in
docs/repo_understanding.md §3/§5 step 7). This project writes it from scratch, using her
already-prepared window metadata (datasets/processed/uldd_m7/fold_{0..4}.npz — same windows,
same 3-class labels, just re-partitioned per fold) as the source of truth for which 60-second
windows exist and their KSS labels.

Per window: sample 16 frames uniformly across the window's 60fps frame range (start_4hz*15,
end_4hz*15), run each through a frozen ImageNet-pretrained EfficientNet-B0, keep the
1280-d pooled feature. Subjects with no video (B, I, M) are skipped, same convention as M5/M6.

Output: datasets/processed/uldd_m7_vis/
    features_cache.npz   -- one row per (subject,session,win_idx), shape (16,1280) each
    fold_{0..4}.npz       -- X_vis_train/test (N,16,1280), y_train/test, per-fold, ready
                             for cross_validate-style training
"""
import sys, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision

REPO = Path("/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system")
VIDEO_ROOT = Path("/Users/karthikeyansundaravaradhan/Desktop/Masters Project/DriverDrowsinessDataset/Video_Data/Video_Data")
FOLD_DIR = REPO / "datasets" / "processed" / "uldd_m7"
OUT_DIR = REPO / "datasets" / "processed" / "uldd_m7_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NO_VIDEO = {"B", "I", "M"}
T_VIS = 16
IMG_SIZE = 224
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_global_window_list():
    """Union of all windows across folds (same 5367 windows, just re-partitioned per fold)."""
    d = np.load(FOLD_DIR / "fold_0.npz", allow_pickle=True)
    windows = {}
    for split in ("train", "test"):
        subj = d[f"mm_subject_{split}"]
        sess = d[f"mm_session_{split}"]
        win = d[f"mm_win_idx_{split}"]
        s4 = d[f"mm_start_4hz_{split}"]
        e4 = d[f"mm_end_4hz_{split}"]
        y = d[f"mm_y_{split}"]
        for i in range(len(subj)):
            key = (str(subj[i]), str(sess[i]), int(win[i]))
            windows[key] = (int(s4[i]), int(e4[i]), int(y[i]))
    return windows


def load_effnet_b0():
    m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = torch.nn.Identity()  # -> 1280-d pooled feature
    m.eval()
    return m.to(DEVICE)


@torch.no_grad()
def embed_frames(model, frames_bgr):
    """frames_bgr: list of HxWx3 uint8 BGR arrays -> (N, 1280) np.float32."""
    batch = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(cv2.resize(f, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        arr = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(arr)
    x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().to(DEVICE)
    feat = model(x)
    return feat.cpu().numpy().astype(np.float32)


def main():
    windows = build_global_window_list()
    print(f"Global windows: {len(windows)}")

    # Group by session so each video is opened once, not once per window.
    by_session = defaultdict(list)
    for (subj, sess, win), (s4, e4, y) in windows.items():
        if subj in NO_VIDEO:
            continue
        by_session[(subj, sess)].append((win, s4, e4, y))

    n_sessions = len(by_session)
    print(f"Sessions with video: {n_sessions} (skipping subjects {NO_VIDEO})")

    model = load_effnet_b0()
    print(f"EfficientNet-B0 loaded on {DEVICE}")

    cache = {}  # (subj,sess,win) -> (16,1280) float32
    t0 = time.time()
    for si, ((subj, sess), wins) in enumerate(sorted(by_session.items())):
        video_path = VIDEO_ROOT / subj / sess / f"{subj}_IR_{sess}.mp4"
        if not video_path.exists():
            print(f"  [skip] {subj}/{sess}: video not found at {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [skip] {subj}/{sess}: cannot open video")
            continue

        ts = time.time()
        for win, s4, e4, y in wins:
            start_60, end_60 = s4 * 15, e4 * 15
            frame_ids = np.linspace(start_60, end_60 - 1, T_VIS).astype(int)
            frames = []
            for fid in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                ret, frame = cap.read()
                if not ret:
                    frame = frames[-1] if frames else np.zeros((220, 240, 3), dtype=np.uint8)
                frames.append(frame)
            feat = embed_frames(model, frames)  # (16, 1280)
            cache[(subj, sess, win)] = feat
        cap.release()
        print(f"  [{si+1}/{n_sessions}] {subj}/{sess}: {len(wins)} windows -> "
              f"{time.time()-ts:.1f}s")

    print(f"\nTotal extraction time: {time.time()-t0:.1f}s, {len(cache)} windows cached")

    # Save flat cache (for inspection / reuse)
    keys = list(cache.keys())
    feats = np.stack([cache[k] for k in keys])
    np.savez_compressed(
        OUT_DIR / "features_cache.npz",
        keys=np.array([f"{s}|{se}|{w}" for s, se, w in keys], dtype="U16"),
        features=feats,
    )
    print(f"Saved global cache: {feats.shape} -> {OUT_DIR / 'features_cache.npz'}")

    # Per-fold assembly, skipping windows we couldn't extract (NO_VIDEO subjects)
    for k in range(5):
        d = np.load(FOLD_DIR / f"fold_{k}.npz", allow_pickle=True)
        fold_out = {}
        for split in ("train", "test"):
            subj = d[f"mm_subject_{split}"]
            sess = d[f"mm_session_{split}"]
            win = d[f"mm_win_idx_{split}"]
            y = d[f"mm_y_{split}"]
            X, Y = [], []
            for i in range(len(subj)):
                key = (str(subj[i]), str(sess[i]), int(win[i]))
                if key in cache:
                    X.append(cache[key])
                    Y.append(int(y[i]))
            fold_out[f"X_vis_{split}"] = np.stack(X).astype(np.float32)
            fold_out[f"y_{split}"] = np.array(Y, dtype=np.int64)
        np.savez_compressed(OUT_DIR / f"fold_{k}.npz", **fold_out)
        print(f"  fold_{k}.npz: train={fold_out['X_vis_train'].shape} "
              f"test={fold_out['X_vis_test'].shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
