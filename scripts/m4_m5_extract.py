"""
Combined M4 + M5 raw-video extraction for UL-DD.

Single decode pass per IR video:
  - every 15th frame (4 Hz)  -> M4 10-feature vector (MediaPipe FaceMesh, reusing
    the exact EAR/MAR/head-pose/brow/jaw formulas from
    driver-drowsiness-detection-system/src/face_detection.py, unmodified)
  - every 60th frame (1 Hz)  -> M5 224x224 JPEG, same convention as her
    notebooks/09-uldd-m5-yolo-cls.ipynb cell 3 (frame extraction), just with
    corrected local paths instead of her Windows C:/Users/raka1005/... paths.

Outputs (new directories, nothing in her repo touched):
  datasets/yolo_frames_regen/{subject}_{session}/{class_name}/frame_NNNNN.jpg
  datasets/yolo_frames_regen/metadata.csv
  datasets/processed/uldd_m4/fold_{0..4}.npz   (240-step / 60-stride windows @ 4Hz,
                                                 same fold split as M1/M2/M3/M5)

Usage:
  python m4_m5_extract.py               # full run, all subjects/sessions
  python m4_m5_extract.py --test 1800   # quick smoke test: only first N frames/video
"""
import sys, os, time, argparse, json
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd

REPO = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
DATASET_ROOT = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/DriverDrowsinessDataset"
sys.path.insert(0, f"{REPO}/src")

import face_detection as fd  # noqa: E402  (reuse her _ear/_mar/index constants, unmodified)
import mediapipe as mp

# ── Constants (mirrors ULDDProcessor / the M5 notebook exactly) ────────────────
SUBJECTS = list('ABCDEFGHIJKLMNOPQRS')
NO_VIDEO = {'B', 'I', 'M'}
AWAKE_ONLY = {'C', 'F', 'L'}
SESSION_MAP = {'Alert': 'A', 'Drowsy': 'D'}
VIDEO_TYPE = 'IR'
CLASS_NAMES = ['Alert', 'LowVigilant', 'Drowsy']
KSS_THRESH = (3, 6)

ORIG_HZ = 60
M4_TARGET_HZ = 4
M4_SKIP = ORIG_HZ // M4_TARGET_HZ          # 15
M5_SAMPLE_FPS = 1.0
M5_SKIP = ORIG_HZ                          # 60
IMG_SIZE = 224
EPOCH_SEC = 240

WINDOW_SEC, STRIDE_SEC = 60, 15
WIN = WINDOW_SEC * M4_TARGET_HZ            # 240
STRIDE = STRIDE_SEC * M4_TARGET_HZ         # 60

FOLDS = [list('ABCD'), list('EFGH'), list('IJK'), list('LMNO'), list('PQRS')]

VIDEO_ROOT = Path(DATASET_ROOT) / 'Video_Data' / 'Video_Data'
LABELS_CSV = Path(DATASET_ROOT) / 'Labels.csv'
FRAME_DIR = Path(REPO) / 'datasets' / 'yolo_frames_regen'
M4_OUT_DIR = Path(REPO) / 'datasets' / 'processed' / 'uldd_m4'
for d in [FRAME_DIR, M4_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── FeatureExtractor constants, reused verbatim from face_detection.py ────────
EAR_CLOSED_THRESH = fd.FeatureExtractor.EAR_CLOSED_THRESH
PERCLOS_WINDOW_S = fd.FeatureExtractor.PERCLOS_WINDOW_S
BLINK_WINDOW_S = fd.FeatureExtractor.BLINK_WINDOW_S


def kss_to_class(kss: float) -> int:
    if kss <= KSS_THRESH[0]:
        return 0
    if kss <= KSS_THRESH[1]:
        return 1
    return 2


def load_labels(csv_path):
    labels = defaultdict(dict)
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            tag = parts[0]
            kss = [float(x) for x in parts[1:]]
            subj, sess_name = tag.split('_')
            sess_folder = SESSION_MAP[sess_name]
            labels[subj][sess_folder] = [kss_to_class(k) for k in kss]
    return dict(labels)


def compute_m4_features(landmarks, frame_shape, ear_hist, blink_ts, prev_closed, t_sec):
    """Adapted from fd.FeatureExtractor.extract() — identical formulas, but takes
    an explicit video timestamp `t_sec` instead of calling time.time(), so the
    PERCLOS/blink-rate rolling windows are computed against video time rather
    than wall-clock time (there is no live wall clock in offline extraction)."""
    lm = landmarks
    ear_l = fd._ear(lm, fd._LE)
    ear_r = fd._ear(lm, fd._RE)
    mar = fd._mar(lm)
    avg_ear = (ear_l + ear_r) / 2.0

    ear_hist.append((t_sec, avg_ear))
    cutoff_p = t_sec - PERCLOS_WINDOW_S
    recent = [e for t, e in ear_hist if t >= cutoff_p]
    perclos = float(np.mean([1 if e < EAR_CLOSED_THRESH else 0 for e in recent])) if recent else 0.0

    closed = avg_ear < EAR_CLOSED_THRESH
    if closed and not prev_closed[0]:
        blink_ts.append(t_sec)
    prev_closed[0] = closed
    cutoff_b = t_sec - BLINK_WINDOW_S
    blinks = sum(1 for t in blink_ts if t >= cutoff_b)
    blink_rate = blinks / (BLINK_WINDOW_S / 60.0)

    fh, fw = frame_shape[0], frame_shape[1]
    pts_3d = np.float32([[0, 0, 0], [0, -63, -13], [-43, 32, -26], [43, 32, -26]])
    pts_2d = np.float32([
        [lm[1].x * fw, lm[1].y * fh],
        [lm[152].x * fw, lm[152].y * fh],
        [lm[33].x * fw, lm[33].y * fh],
        [lm[263].x * fw, lm[263].y * fh],
    ])
    cam = np.float32([[fw, 0, fw / 2], [0, fw, fh / 2], [0, 0, 1]])
    try:
        _, rvec, _ = cv2.solvePnP(pts_3d, pts_2d, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP)
        rmat, _ = cv2.Rodrigues(rvec)
        angles = cv2.decomposeProjectionMatrix(np.hstack([rmat, np.zeros((3, 1))]))[6].flatten()
        head_pitch, head_roll, head_yaw = float(angles[0]), float(angles[1]), float(angles[2])
    except cv2.error:
        head_pitch = head_roll = head_yaw = 0.0

    rb = abs(lm[fd._RB_TOP].y - lm[fd._RB_BOT].y) * fh
    lb = abs(lm[fd._LB_TOP].y - lm[fd._LB_BOT].y) * fh
    brow_raise = ((rb + lb) / 2.0) / (fh + 1e-6)

    nose_y = lm[fd._NOSE].y * fh
    chin_y = lm[fd._CHIN].y * fh
    jaw_drop = (chin_y - nose_y) / (fh + 1e-6)

    return np.array([ear_l, ear_r, mar, perclos, blink_rate,
                      head_pitch, head_roll, head_yaw, brow_raise, jaw_drop], dtype=np.float32)


def process_video(video_path, subject, session, epoch_labels, face_mesh, test_limit=None):
    """One decode pass -> (m4_feature_array (T,10), m5_records list)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  x cannot open {video_path}')
        return None, []
    fps = cap.get(cv2.CAP_PROP_FPS) or ORIG_HZ

    out_dir = FRAME_DIR / f'{subject}_{session}'
    for cn in CLASS_NAMES:
        (out_dir / cn).mkdir(parents=True, exist_ok=True)

    ear_hist = deque(maxlen=int(PERCLOS_WINDOW_S * M4_TARGET_HZ) + 2)
    blink_ts = deque(maxlen=int(BLINK_WINDOW_S * M4_TARGET_HZ) + 2)
    prev_closed = [False]
    last_valid = None

    m4_feats = []
    m5_records = []
    frame_idx = 0
    saved5 = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if test_limit and frame_idx >= test_limit:
            break

        if frame_idx % M4_SKIP == 0:
            t_sec = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                feat = compute_m4_features(lm, frame.shape, ear_hist, blink_ts, prev_closed, t_sec)
                last_valid = feat
            else:
                feat = last_valid if last_valid is not None else np.zeros(10, dtype=np.float32)
            m4_feats.append(feat)

        if frame_idx % M5_SKIP == 0:
            timestamp_sec = frame_idx / fps
            epoch_idx = min(int(timestamp_sec / EPOCH_SEC), len(epoch_labels) - 1)
            cls = epoch_labels[epoch_idx]
            cls_name = CLASS_NAMES[cls]
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            fname = f'frame_{saved5:05d}.jpg'
            save_path = out_dir / cls_name / fname
            cv2.imwrite(str(save_path), resized)
            m5_records.append({
                'subject': subject, 'session': session, 'frame_idx': frame_idx,
                'timestamp_s': round(timestamp_sec, 2), 'epoch': epoch_idx,
                'class': cls, 'class_name': cls_name, 'path': str(save_path),
            })
            saved5 += 1

        frame_idx += 1

    cap.release()
    m4_arr = np.stack(m4_feats).astype(np.float32) if m4_feats else np.zeros((0, 10), dtype=np.float32)
    return m4_arr, m5_records


def sliding_windows(feat_4hz, epoch_classes):
    n = feat_4hz.shape[0]
    starts = range(0, n - WIN + 1, STRIDE)
    feats, labels = [], []
    for s in starts:
        e = s + WIN
        feats.append(feat_4hz[s:e])
        centre_sec = (s + WIN // 2) / M4_TARGET_HZ
        epoch_idx = min(int(centre_sec // EPOCH_SEC), len(epoch_classes) - 1)
        labels.append(epoch_classes[epoch_idx])
    if not feats:
        return np.zeros((0, WIN, 10), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.stack(feats).astype(np.float32), np.array(labels, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', type=int, default=None, help='limit frames/video for a quick smoke test')
    args = ap.parse_args()

    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    kss_labels = load_labels(LABELS_CSV)
    session_results = []  # list of dict(subject, session, m4_feats, epoch_labels)
    all_m5_meta = []

    t0 = time.time()
    for subject in SUBJECTS:
        if subject in NO_VIDEO or subject not in kss_labels:
            print(f'[{subject}] no video / no labels — skip')
            continue
        for session_name, session_folder in SESSION_MAP.items():
            if session_folder not in kss_labels[subject]:
                continue
            video_path = VIDEO_ROOT / subject / session_folder / f'{subject}_{VIDEO_TYPE}_{session_folder}.mp4'
            if not video_path.exists():
                print(f'[{subject}/{session_folder}] video not found — skip')
                continue
            print(f'[{subject}/{session_folder}] processing {video_path.name} ...', flush=True)
            tS = time.time()
            m4_arr, m5_recs = process_video(
                video_path, subject, session_folder, kss_labels[subject][session_folder],
                face_mesh, test_limit=args.test,
            )
            print(f'  -> m4 samples={len(m4_arr)}  m5 frames={len(m5_recs)}  '
                  f'({time.time()-tS:.1f}s)', flush=True)
            session_results.append({
                'subject': subject, 'session': session_folder,
                'm4': m4_arr, 'epoch_labels': kss_labels[subject][session_folder],
            })
            all_m5_meta.extend(m5_recs)

    print(f'\nTotal extraction time: {time.time()-t0:.1f}s')

    # ── Save M5 metadata ────────────────────────────────────────────────────
    meta_df = pd.DataFrame(all_m5_meta)
    meta_df.to_csv(FRAME_DIR / 'metadata.csv', index=False)
    print(f'Saved M5 metadata: {len(meta_df)} rows -> {FRAME_DIR / "metadata.csv"}')

    if args.test:
        print('Smoke test mode — skipping M4 fold assembly.')
        return

    # ── Build M4 windows per session ────────────────────────────────────────
    windowed = []
    for r in session_results:
        fw, lb = sliding_windows(r['m4'], r['epoch_labels'])
        if len(lb) == 0:
            continue
        windowed.append({'subject': r['subject'], 'session': r['session'], 'X': fw, 'y': lb})

    # ── 5-fold assembly (same FOLDS split, scaler fit on train only) ────────
    for k, test_subjs in enumerate(FOLDS):
        test_set = frozenset(test_subjs)
        train_res = [r for r in windowed if r['subject'] not in test_set]
        test_res = [r for r in windowed if r['subject'] in test_set]
        if not train_res or not test_res:
            print(f'  Fold {k}: insufficient data, skipping')
            continue

        X_tr = np.concatenate([r['X'] for r in train_res], axis=0)
        X_te = np.concatenate([r['X'] for r in test_res], axis=0)
        y_tr = np.concatenate([r['y'] for r in train_res], axis=0)
        y_te = np.concatenate([r['y'] for r in test_res], axis=0)

        flat = X_tr.reshape(-1, X_tr.shape[-1])
        mu, sig = flat.mean(axis=0), flat.std(axis=0) + 1e-8
        X_tr = ((X_tr - mu) / sig).astype(np.float32)
        X_te = ((X_te - mu) / sig).astype(np.float32)

        out = {
            'fold': k, 'test_subjects': np.array(test_subjs, dtype='U1'),
            'X_m4_train': X_tr, 'X_m4_test': X_te,
            'y_train': y_tr, 'y_test': y_te,
            'm4_mean': mu, 'm4_std': sig,
        }
        path = M4_OUT_DIR / f'fold_{k}.npz'
        np.savez_compressed(str(path), **out)
        print(f'  Saved fold_{k}.npz  train={len(y_tr)} test={len(y_te)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
