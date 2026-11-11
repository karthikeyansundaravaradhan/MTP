# UL-DD local grounding and MPS benchmark report

Everything below was measured on this machine on 2026-11-05. No project files were modified and nothing was committed. The only environment change was installing `timm` into `ddd-repo` (details in §4). Scratch scripts and raw outputs are in `/private/tmp/claude-501/-Users-karthikeyansundaravaradhan-Desktop-Masters-Project/a9c96569-4853-4a70-ae28-8f7816e31960/scratchpad/` (`bench_models.py/.json`, `bench_extra.py`, `bench_decode.py/.json`, `labels.py`, `inspect_npz.py`, `frames_E_D.jpg`).

## Critical findings (read first)

1. **The "stereo depth" videos are not depth maps.** I grabbed a frame from `E_D` at index 100000 from each stream. `L3D` and `R3D` are ordinary 3-channel colour frames (440×370) from the left and right cameras of a stereo rig: two more frontal face/upper-body views. Getting depth would need a stereo-matching step that nobody has run. The "IR" video (240×220) is also a 3-channel colour frontal view (red shirt clearly visible, channels not equal). `Pose` (620×470) is a side view showing the full body, hands on the wheel, and the face in profile. Any "depth" ablation should be renamed to a stereo-view or second-frontal-view ablation, unless stereo disparity is computed first.
2. **The machine was in Low Power Mode on battery during all benchmarks** (`pmset`: `lowpowermode 1`, "drawing from Battery Power"). GPU throughput varied by up to 2.1× between runs of the same config: ViT-S/14 at 224 px fp16 gave 88.6 fps, then 185.0 fps; MobileNetV3 fp16 bs64 gave 44.1, then 705.4. Treat every number below as a lower bound. Plug in and turn off Low Power Mode before any real extraction.
3. **The existing 60 fps M6 embeddings leak test subjects for folds 1–4.** `models/embeddings_uldd/` was produced by the M5 fold-0 backbone (`scripts/m6/train_m6.sh`: `--ckpt models/checkpoints_regen/M5_fold0.pt`). That backbone was trained on subjects E–S with KSS labels, so for folds 1–4 the features were learned on those folds' own test subjects. Also, the `cls_idx` field in those files is the session label (A=0, D=1), not KSS. These caches should not be reused for new models unless they are regenerated per fold, or replaced by a frozen generic encoder (which has no leakage).
4. **Alert is extremely scarce in some folds once B, I, M are dropped.** Video-subject 3-class Alert epochs per fold are 4, 21, 3, 9 and 34 (folds 0–4). That explains the Alert F1 of 0.000 in M7 folds 0–2 and M5 fold 2. M5's pooled per-class F1 (which I recomputed from `results/reports_regen/M5_results.json`) is **Alert 0.258 / LowVig 0.565 / Drowsy 0.620**, and 59.6% of Alert frames are predicted as LowVig.
5. **There is a strong session-identity and time-on-task confound.** Mean KSS rises steadily within a session (Alert sessions 2.53 → 4.55 across epochs 0–9; Drowsy sessions 6.19 → 8.28). 51.6% of Alert-session epochs are not 3-class "Alert". A model can partly score by recognising which session or how far into the session a frame comes from. This should be discussed in the report.
6. **60 fps is highly redundant.** Cosine similarity between adjacent frames in the cached embeddings is 0.9996. 1–4 fps is sufficient for per-frame encoders.

## 1. What already exists and what was already planned

- **`docs/models_summary.md`**: per-model write-ups and tables for M1–M7. M5 (YOLOv8n-cls on 1 fps IR frames) is best at acc 0.541±0.080 and macro-F1 0.447±0.071. Every fusion model (M3, M6 Lite/Full) and the temporal model M7 (16 frames of EfficientNet-B0 features → BiLSTM + attention, 0.355) came out below it. After the alignment fix, M6 got worse: Lite 0.303, Full 0.292.
- **`docs/literature_review.md`** (covers video + telemetry only):
  - Recommended target architecture: STFTransNet (*Sensors* 2025), with the second visual stream swapped for a telemetry encoder, telemetry as the cross-attention query, a dilated TCN for the temporal stage, and a modality gate.
  - Direction 1: gated asymmetric fusion with confidence-weighted blending.
  - Direction 2: an ordinal or contrastive auxiliary loss aimed at the Alert/LowVig boundary.
  - Direction 3: a mixture of modality-specialist experts with routing that handles missing modalities.
- **`model_research/README.md`** (branch `model_research`):
  - A research plan targeting 0.70–0.80 macro-F1, explicitly flagged as likely out of reach.
  - Fusion ranking: gated/residual correction on a frozen visual baseline first, then cross-attention with telemetry as query, then a Multimodal Bottleneck Transformer, then late fusion or mixture-of-experts; naive concatenation is not worth retrying.
  - Encoders under consideration: ViT/DeiT-S/T, frozen VideoMAE/TimeSformer, two-stream optical flow, TCN or 1D-CNN for telemetry.
  - Missing-modality plan: modality dropout.
  - Data-efficiency levers: cross-dataset pretraining, self-supervised learning, ordinal loss.
  - A 6-phase plan with honesty checkpoints.
  - Scope there was IR plus telemetry only. Pose, stereo, biometrics and grip were excluded, which is narrower than the user's current request.
- **Top of `docs/story_so_far.md`**: latest entry 2026-09-19 (wrote `models_summary.md`). Earlier entries cover the research-plan start (09-15), the scripts reorganisation into `scripts/m1..m7` plus `common`/`helpers`/`archive`, and the M6 alignment fix (09-08, three bugs fixed; results got worse).
- **Never used by any model so far**: Pose video, L3D/R3D, PL (33 MediaPipe joints), grip pressure, biometrics, telemetry heading.

## 2. Reusable cached artifacts

Base path: `/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system/`. Project-root `models/`, `datasets/` and `results/` do not exist. Free disk is 54 GiB of 228 GiB.

| Path | Size | Contents (shapes verified with the ddd-repo Python) |
|---|---|---|
| `models/checkpoints_regen/M5_fold{0-4}.pt` | 2.8 MB each | Regenerated M5 YOLOv8n-cls per fold (fold-correct, no leakage). Ultralytics run dirs (args, results.csv, confusion matrices) are in `models/checkpoints_regen/yolo_cls/M5_fold*/` (36 MB). |
| `models/checkpoints/M5_fold{0-4}.pt` | 2.8–8.4 MB | Kavipriya's original M5 checkpoints, plus her M1–M3 `.keras` and M6 lite/full `.pt` files. |
| `models/checkpoints_regen/M{1..4}_fold*.keras`, `M7_fold*.pt` (13 MB each) | 161 MB in total | Our reproduced baselines. |
| `models/embeddings_uldd/{29 sessions}.npz` | 3.6 GB (117–131 MB each) | `embedding (144000, 256) float32` (every 60 fps frame), `frame_idx (144000,) int32`, `cls_idx (144000,) int8` (session A/D). Built with the fold-0 backbone, so it leaks for folds 1–4 (see critical finding 3). Loading all 29 sessions at fp32 takes about 4.3 GB of RAM. |
| `models/embeddings/{29}.npz` | 62 MB | Old 1 fps version: `embedding (2400, 256)`, `frame_idx`, `cls_idx`, `files` (Windows paths). |
| `datasets/yolo_frames_regen/{S_sess}/{class}/frame_NNNNN.jpg` | 1.0 GB | **69,307 IR frames at 1 fps, 224×224 JPEG** (about 12 KB each). Hardlinked into `datasets/yolo_cls_regen/fold_{0-4}/{train,val}/{class}` (346,540 links, 54 MB apparent). Can be reused directly to fine-tune a ViT on IR. |
| `datasets/processed/ul_dd_regen/fold_k.npz` | 25 MB each | `X_fau_{train,test} (N, 240, 30)` with N = 4294 train / 1073 test in fold 0; `mm_fau_*` and `mm_tele_* (N, 240, 5)`, where fold 0 has 4294 train and 778 test because A has no telemetry; `y_*`; `fau_mean/std`, `tele_mean/std`; `test_subjects`. |
| `datasets/processed/uldd_m4/fold_k.npz` | 9 MB each | `X_m4_{train,test} (N, 240, 10)` with fold 0 at 3765 / 767; `y_*`; `m4_mean/std`. |
| `datasets/processed/uldd_m7/fold_k.npz` | 10.3 MB each | FAU windows plus clean window metadata: `mm_subject_*`, `mm_session_*`, `mm_win_idx_*`, `mm_start_4hz_*`, `mm_end_4hz_*`. This is the best index for aligning new video embeddings to the standard 60 s / 15 s windows (60 fps frame = 4 Hz index × 15). |
| `datasets/processed/uldd_m7_vis/features_cache.npz` + `fold_k.npz` | 1.9 GB (336 MB each) | `features (4429, 16, 1280)` (EfficientNet-B0, 16 frames per window), `keys 'S|sess|win'`; folds hold `X_vis_{train,test}` and `y_*` (fold 0: 3669 / 760). |
| `datasets/processed/ul_dd`, `ul_dd1` | 119 MB each | Older duplicates of `ul_dd_regen`. |
| `results/reports_regen/M{1,2,3,4,5,7}_results.json` | — | Per-fold `y_true` / `y_pred`, so per-class re-analysis is possible. M6 results are in `results/reports/M6_{lite,full}_results.json`. |

Raw data notes:
- `Video_Data` is 7.2 GB in total. `E_A` file sizes: IR 30.7 MB, L3D 46.1 MB, R3D 49.1 MB, Pose 96.2 MB. All are h264, 60.0 fps, exactly 144,000 frames (2400.0 s). There are 29 sessions with video.
- `Extracted_Features` is 13 GB. For `E_A`, FAU and FL have 140,583 rows: about 2.4% of frames have no detected face. PL has all 144,000 rows. PL's header lists all x columns, then all y, then all z, but the values look interleaved as x,y,z triplets (0.297, 0.199, −0.396). Check this before using PL.
- Telemetry columns: three raw timestamps, heading, pitch, roll, speed, rpm, gear. Grip pressure (LGP/RGP) is a single headerless column.

## 3. Label analysis (`DriverDrowsinessDataset/Labels.csv`)

`Labels.csv` has 35 sessions (19 Alert + 16 Drowsy; only C, F and L lack a Drowsy session), each with 10 epochs of 4 minutes, so **350 epochs in total and 290 epochs for the 16 subjects with video**. Fractional KSS values appear: 4.5 (×1), 6.5 (×1), 7.5 (×3).

The 60 s / 15 s windowing gives about 157 windows per session, roughly 16 per epoch. Windows are therefore highly correlated: the effective sample size is the 290–350 epochs, not the ~4.4–5.4k windows.

KSS histogram (all 350 epochs):

| KSS | 1 | 2 | 3 | 4 | 4.5 | 5 | 6 | 6.5 | 7 | 7.5 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Epochs | 21 | 30 | 41 | 40 | 1 | 36 | 49 | 1 | 54 | 3 | 56 | 18 |

Notable subjects:
- **Alert sessions that are mostly LowVig:** A_A (KSS 4–6), G_A (4–6), L_A (4–6), K_A, N_A, C_A.
- **Alert sessions that reach Drowsy:** J_A (up to 7) and D_A (up to 7).
- **Drowsy sessions that are half LowVig:** G_D (6–7) and O_D.

Class counts per fold, **video subjects only** (A / C / D / E / F / G / H / J / K / L / N / O / P / Q / R / S):

| Scheme | Totals | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| (a) 3-class Alert / LowVig / Drowsy | 71 / 112 / 107 (minority 24.5%) | 4/28/18 | 21/28/21 | 3/16/21 | 9/26/15 | 34/14/32 |
| (b1) KSS ≤6 vs ≥7 | 184 / 106 | 32/18 | 50/20 | 19/21 | 35/15 | 48/32 |
| (b2) KSS ≤5 vs ≥6 | 140 / 150 (48.3%) | 23/27 | 37/33 | 14/26 | 23/27 | 43/37 |
| (b3) ≤3 vs ≥7, middle dropped | 71 / 106 (177 epochs) | 4/18 | 21/20 | 3/21 | 9/15 | 34/32 |
| (b4) ≤4 vs ≥7, KSS 5–6 dropped | 105 / 106 (211 epochs, 49.8%) | 13/18 | 30/20 | 7/21 | 17/15 | 38/32 |
| Session label A vs D | 160 / 130 | 30/20 | 40/30 | 20/20 | 30/20 | 40/40 |
| (c) Ordinal KSS, mean±std | 290 epochs, 12 levels | 5.82±1.71 (range 3–9) | 4.96±2.27 | 6.28±1.71 (range 3–9) | 5.30±1.81 (range 1–8) | 4.96±2.83 |

With all 19 subjects, 3-class totals are 92 / 126 / 132, and per fold: 6/38/26, 21/28/21, 13/16/31, 18/30/22, 34/14/32. B2 totals are 169 / 181.

Takeaways:
- **B2 (≤5 vs ≥6) is the most balanced scheme in every fold** and avoids the near-empty Alert class in folds 0 and 2.
- **B4** is balanced and drops the ambiguous middle band, but it changes the task, so report it as a secondary protocol.
- **Ordinal regression** uses all 12 levels. Report MAE plus 3-class macro-F1 after thresholding, so results stay comparable with M1–M7.

## 4. ddd-repo environment

Hardware: Apple M4, 10 CPU cores, 10-core GPU (Metal 4), 16 GiB RAM (17,179,869,184 bytes).

- **Present:** torch 2.13.0 (MPS available and built), torchvision 0.28.0, opencv 4.11.0 (FFMPEG and AVFoundation backends), ultralytics 8.4.135, scikit-learn 1.9.0, numpy 2.4.6, pandas 3.0.5, tensorflow 2.21.0, wandb 0.29.0, PIL 12.3.0.
- **Missing:** transformers, decord, av, gradio, einops, accelerate, lightning, open_clip, torchaudio, imageio, mediapipe (correctly kept out of this env). There is no system ffmpeg or ffprobe.
- **Installed now:** `timm 1.0.31`. The dry run showed it would only *add* packages, with no upgrades to anything already installed: hf-xet 1.8.0, httpcore2 2.16.0, httpx2 2.16.0, huggingface_hub 2.4.0, safetensors 0.8.0, truststore 0.10.5.
- **Model weights:** pretrained weights downloaded from the HF hub with no problems (unauthenticated) into `~/.cache/huggingface/hub`.

## 5. Benchmarks

Model throughput is images per second, timed after warmup in ~12 s timed loops (8 s in the reruns), under Low Power Mode on battery. All encoders loaded real pretrained weights.

### 5.1 Frozen inference at 224 px

| Model (timm) | Params | Dim | fp32 bs32 | fp32 bs64 | fp16 bs32 | fp16 bs64 | MPS driver mem |
|---|---|---|---|---|---|---|---|
| vit_small_patch14_dinov2 (img_size 224, 256 tokens) | 21.6 M | 384 | 61.5 | 63.7 | 83.5 | 88.6 (rerun **185.0**) | 1.2 GB |
| vit_small_patch16_224.augreg_in21k | 21.7 M | 384 | 89.4 | 96.8 | 133.0 | 130.8 (rerun **272.2**) | 1.2 GB |
| vit_base_patch14_dinov2 @224 | 85.7 M | 768 | 20.4 | 20.5 | 26.8 | 27.4 | 1.45 GB |
| vit_base_patch16_224.augreg_in21k | 85.8 M | 768 | 27.8 | 28.5 | 36.8 | 37.9 | 1.46 GB |
| vit_tiny_patch16_224 | 5.5 M | 192 | 238.6 | 199.5 | 387.7 | 376.3 | 1.1 GB |
| efficientnet_b0 | 4.0 M | 1280 | 161.1 | 161.1 | 206.2 | 205.2 | 1.1–2.2 GB |
| mobilenetv3_large_100 | 4.2 M | 1280 | 306.9 | 304.9 | 405.5 | 44.1 (throttle outlier; rerun **705.4**) | 1.13 GB |

### 5.2 Lower resolution, fine-tuning and temporal heads

| Config | Throughput | MPS alloc / driver |
|---|---|---|
| DINOv2 ViT-S/14 fp16 bs64 @112 px (64 tokens) | **711.6** img/s | 0.05 / 1.16 GB |
| DINOv2 ViT-S/14 fp16 bs64 @168 px (144 tokens) | 328.0 img/s | 0.06 / 1.16 GB |
| DINOv2 ViT-B/14 fp16 bs64 @112 px | 225.9 img/s | — |
| YOLOv8n-cls (M5 checkpoint) fp32 bs64 @224 | 1986.4 img/s | 0.08 / 1.13 GB |
| ViT-S/16 full fine-tune fp32 bs32 @224 (forward + backward + AdamW) | 53.2 img/s | 1.88 / **5.53 GB** |
| DINOv2 ViT-S/14 full fine-tune fp32 bs32 @112 | 157.9 img/s | 0.84 / 3.37 GB |
| EfficientNet-B0 full fine-tune fp32 bs32 @224 | 68.3 img/s | 0.11 / 3.53 GB |
| Temporal Transformer (4 layers, d=256) over 120 tokens × 384 dims, train bs64 | **532 windows/s** | 0.06 / 1.16 GB |

The IR face is natively 240×220, so 112–168 px is a reasonable input size, and dropping to 112 px gives ViT-S an 8× speedup. The temporal head is so cheap that a full 5-fold CV on cached embeddings takes minutes.

### 5.3 OpenCV decode, one session (`E/A`)

| Stream | Resolution | Sequential read, every frame | grab()-skip to 2 fps: source fps / output fps | Seek to 2 fps (output fps) | Seek to 4 fps (output fps) |
|---|---|---|---|---|---|
| IR | 240×220 | **21,642 fps** (whole session in 6.6 s) | 48,046 / 1,602 | 297.5 | 282.7 |
| L3D | 440×370 | 7,776 fps | 24,475 / 816 | 166.4 | 161.7 |
| R3D | 440×370 | 7,962 fps | 25,724 / 858 | 163.5 | 162.8 |
| Pose | 620×470 | 4,606 fps | 15,102 / 504 | 90.0 | 88.3 |

**Use a sequential `grab()` and only `retrieve()` the frames you need. Do not seek.** Seeking is 5–10× slower. Decoding one full session with grab-skip takes 3.0 s (IR), 5.9 s (L3D/R3D) or 9.5 s (Pose). All 29 sessions take about **1.5 min for IR, 2.9 min for L3D (same again for R3D), and 4.6 min for Pose**, whatever the target fps. Decoding is never the bottleneck.

### 5.4 Extrapolated time to embed every session

Frame counts per modality: 29 sessions × 2400 s × target fps = 69,600 (1 fps), 139,200 (2 fps), 278,400 (4 fps).

Times are encoder-only, for **one** modality (IR, L3D or Pose). Where two values appear, they come from the throttled sweep and the faster rerun. To get the total for IR + L3D + Pose, multiply by 3 (and add about 10 min of decoding in total).

| Encoder | 1 fps | 2 fps | 4 fps |
|---|---|---|---|
| DINOv2 ViT-S/14 @224 fp16 | 0.22 / 0.10 h | 0.44 / 0.21 h | 0.87 / 0.42 h |
| ViT-S/16 @224 fp16 | 0.15 / 0.07 h | 0.30 / 0.14 h | 0.59 / 0.28 h |
| DINOv2 ViT-S/14 @168 fp16 | 0.06 h | 0.12 h | 0.24 h |
| DINOv2 ViT-S/14 @112 fp16 | 0.03 h | 0.05 h | 0.11 h |
| DINOv2 ViT-B/14 @224 fp16 | 0.71 h | 1.41 h | 2.82 h |
| DINOv2 ViT-B/14 @112 fp16 | 0.09 h | 0.17 h | 0.34 h |
| ViT-B/16 @224 fp16 | 0.51 h | 1.02 h | 2.04 h |
| ViT-Ti/16 @224 fp16 | 0.05 h | 0.10 h | 0.21 h |
| EfficientNet-B0 @224 fp16 | 0.09 h | 0.19 h | 0.38 h |
| MobileNetV3-L @224 fp16 | 0.05 / 0.03 h | 0.10 / 0.05 h | 0.19 / 0.11 h |

Examples:
- **DINOv2 ViT-S/14 at 224 px, 2 fps, for IR + L3D + Pose: about 0.6–1.3 h in total.**
- ViT-B/14 at 224 px, 4 fps, for all three streams: about 8.5 h. That needs an overnight run, or 112–168 px instead.
- Full fine-tuning is far more expensive. ViT-S/16 at 224 px on about 55k training frames (1 fps IR, 4/5 of the data) takes about 17 min per epoch per fold, so 5 folds × 5 epochs is about 7 h. DINOv2-S at 112 px takes about 6 min per epoch per fold. Under the deadline, **frozen encoders with cached embeddings and small trainable heads** are the realistic main path. A fine-tuned ViT-S on the IR stream fits as a single ablation.

Embedding storage at fp16, per modality:
- 384-d CLS tokens: 53 MB (1 fps), 107 MB (2 fps), 214 MB (4 fps). 768-d doubles these.
- Full patch tokens (ViT-S at 224 px, 2 fps) would be about 27 GB per modality. Pool them to a 4×4 grid instead, which is about 1.7 GB.

## 6. Peak memory on 16 GB unified RAM

- At measurement time macOS reported 37% free memory. In practice about 9–10 GB is usable without swapping; VS Code and the Claude processes take about 4%.
- **Frozen fp16 inference** uses 1.1–1.5 GB of MPS driver memory for every model tested, including ViT-B. Batch size 64 is safe, and larger batches gave no throughput gain (bs32 and bs64 are within a few percent).
- **Full fine-tuning:**
  - ViT-S/16 at 224 px, bs32, fp32: 5.5 GB driver memory. It fits, but leave headroom and avoid running alongside TensorFlow.
  - DINOv2-S at 112 px: 3.4 GB.
  - EfficientNet-B0: 3.5 GB.
  - ViT-B full fine-tuning was not measured. By parameter count it should be about 3–4× ViT-S, which risks swapping. Use frozen features, LoRA or partial fine-tuning, or Colab instead.
- **Data in host RAM:**
  - Loading all 29 sessions of `embeddings_uldd` at fp32 takes 4.3 GB, and each `uldd_m7_vis` fold file holds 336 MB of arrays.
  - Store new caches as per-session fp16 `.npy` files (memory-mappable; npz cannot be memory-mapped) at 1–4 fps, not 60 fps.
  - Keep DataLoader workers at 4 or fewer, so worker processes do not duplicate large arrays.
- **TensorFlow and PyTorch together:** don't import both in the same training process (TF reserves memory). Run Keras baselines and torch models as separate jobs.
- **Disk:** 54 GiB free. Per-modality CLS caches are under 0.5 GB each, so there is no storage constraint unless patch tokens are cached unpooled.