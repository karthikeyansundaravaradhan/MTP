# Project Timeline: From M1 to the Final Model

How the modelling and the evaluation procedure evolved over the project, model by model, with the
signals each model used and why. Numbers are 3-class macro-F1 unless stated otherwise. Detail
lives in [`models_summary.md`](models_summary.md) (M1–M7), [`m8_models.md`](m8_models.md) (final
phase) and [`../report/report.md`](../report/report.md).

---

## 0. The dataset: UL-DD

**What it is.** UL-DD (Bodaghi et al., 2025, University of Louisiana at Lafayette):
- 19 drivers (A–S), aged 25–40, drove an American Truck Simulator rig (three monitors, wheel,
  pedals) twice for 40 minutes: once **Alert** (session A) and once **Drowsy** (session D).
- Drivers C, F and L have only the Alert session. That gives 35 sessions and about 1,400 minutes.
- Drivers B, I and M have no released video (consent). Their extracted landmark features exist.
- The **video cohort** used for all final models is therefore 16 drivers and 29 sessions.

**Signals recorded:**

| Family | Signal | Rate | Used in this project? |
|---|---|---|---|
| Video | Frontal **IR** face video, 240×220 | 60 fps | Yes: M4, M5, M6, M7, all final models |
| | L3D / R3D (ZED 2 stereo; ordinary colour views, not depth maps) | 60 fps | Final-phase ablation only |
| | Side-view **Pose** RGB camera, 620×470 | 60 fps | Final-phase ablation only |
| Extracted features | **FL**: Dlib 68-point facial landmarks (computed on IR) | 60 fps | Final events model, face crops |
| | **FAU**: 30 facial action units (from FL) | 60 fps | M1, M3 |
| | **PL**: MediaPipe 33-joint body pose (from the Pose video) | 60 fps | Final events model (posture) |
| Behavioural | **Telemetry**: heading, pitch, roll, speed, rpm, gear (missing for A) | 60 Hz | M2, M3, M6, VIGIL fallback |
| | **Grip pressure**, left and right | 3 Hz | VIGIL fallback |
| Biometric | HR, BVP, EDA, TEMP, ACC, IBI (Empatica E4); SpO2, pulse rate | 0.5–64 Hz | **No**, by design: the target is a camera-based system |

**Labels.**
- One Karolinska Sleepiness Scale (KSS, 1–9) self-rating per 4-minute epoch, 10 per drive.
- An expert re-labelling of 30% of the video agrees with the self-reports: κ = 0.88 on 3 classes.
- **y3 (primary):** Alert (KSS ≤ 3), Low-vigilant (4–6), Drowsy (≥ 7). This is the dataset paper's
  binning.
- **B2 (secondary, final phase only):** KSS ≤ 5 vs ≥ 6, the most balanced binary cut.

**Properties that shaped every decision:**
- **Tiny label budget.** The video cohort has only about **290 labelled epochs**. Thousands of
  60 s windows are overlapping views of these labels, not independent samples.
- **Alert is scarce in some folds:** 4, 21, 3, 9 and 34 Alert epochs per fold.
- **KSS rises within every drive:**
  - Alert drives go from 2.5 to 4.6 on average; Drowsy drives from 6.2 to 8.3.
  - So *time-on-task* is a strong confound.
- **67% of KSS variance lies between the Alert and Drowsy sessions**, 16% between drivers and only
  18% within a session. So *which drive* a window comes from is a second confound.
- **The dataset paper's baselines** reach 83.8–88.0% accuracy (SVM / RF early fusion) with
  stratified 5-fold CV. That split is not subject-independent. An independent study holding out
  drivers fell to 30–34%.

---

## 1. Phase 1 (2026-08-30 → 09-19): sequence models M1–M7

### 1.1 Evaluation procedure in Phase 1

| Aspect | Phase 1 setting |
|---|---|
| Task | 3-class y3 |
| Windows | 60 s long, 15 s stride, resampled to 4 Hz (240 time steps), **indexed by FAU row**; each window takes the KSS of its epoch |
| Outer split | **Subject-independent 5-fold**: fold 0 = A B C D, 1 = E F G H, 2 = I J K, 3 = L M N O, 4 = P Q R S. Every window of a test driver is unseen in training |
| Cohort | Varied by model: M1 all 19 drivers; M2 / M3 drivers with FAU + telemetry (A, B, I, M excluded); M4–M7 the video drivers (B, I, M excluded) |
| Validation | M1–M4: the last 15% of each fold's training windows (`validation_split = 0.15`), *not* split by subject. M7: a random 15% of training windows. M6: fixed 50 epochs, no early stopping. **M5: the fold's test drivers were the validation set** |
| Model selection | Early stopping and checkpointing on validation accuracy (val loss diverged from epoch 1 on these folds). Class weights recomputed per fold. Label smoothing 0.1 |
| Seeds | None set, so runs are reproducible only "in the same ballpark" |
| Metrics | Accuracy and **macro-F1**, reported as **fold-mean ± std**; per-class F1 and confusion matrices |

**Why macro-F1.** Class balance differs strongly between folds, and Alert is rare in several of
them. Accuracy rewards predicting the majority class (Low-vigilant). Macro-F1 weights the three
classes equally, so a model that never predicts Alert is penalised. Macro-F1 stayed the primary
metric for the whole project.

### 1.2 Model by model

| Date | Model | Signals used | Why those signals | Architecture | Fold-mean macro-F1 (acc) |
|---|---|---|---|---|---|
| 08-31 | **M1** | 30 FAU at 4 Hz | Facial action units encode eye closure, yawning and brow movement, the classic visual cues. Available for all 19 drivers even without video | BiLSTM 128 → BiLSTM 64 → Dense 64 → 3 (336 k params) | **0.357** (0.390) |
| 08-31 | **M2** | 5 telemetry channels (pitch, roll, speed, rpm, gear) at 4 Hz | Driving behaviour degrades with sleepiness; a non-camera signal | LSTM 64 → LSTM 32 → Dense 32 → 3; unidirectional because telemetry is causal (45 k params) | 0.259 (0.324) |
| 08-31 | **M3** | FAU + telemetry | Tests whether face and vehicle signals are complementary | FAU BiLSTM + telemetry LSTM, bidirectional cross-modal attention, concatenate → Dense (750 k params) | 0.322 (0.352) |
| 09-01 | **M4** | 10 geometric features from MediaPipe FaceMesh on IR video at 4 Hz: EAR left/right, MAR, PERCLOS, blink rate, head pitch/roll/yaw, brow raise, jaw drop | Interpretable, cheap, real-time cues an in-car system could compute live; extraction pipeline built for UL-DD | BiLSTM 64 → BiLSTM 32 → Dense 32 → 3 (82 k params) | 0.275 (0.353) |
| 09-04 | **M5** | Raw IR frames at 1 fps, 224 px | Let a CNN learn its own visual features; IR is robust to cabin lighting | YOLOv8n-cls, ImageNet-initialised, fine-tuned 20 epochs; per-frame, no temporal model | 0.447 (0.541) *as first measured; see 2.1* |
| 09-08 | **M6 Lite / Full** | 16 frame embeddings per window from the M5 backbone + telemetry | Add temporal modelling over frames and fuse with driving behaviour | Lite: visual BiLSTM 128 + telemetry BiLSTM 64, mean-pool, concatenate. Full: visual transformer + telemetry BiLSTM with bidirectional cross-attention | 0.303 / 0.292 |
| 09-08 | **M7** | 16 IR frames per 60 s window | A true temporal video model with pretrained frame features | Frozen EfficientNet-B0 (1280-d per frame) → BiLSTM 256 → attention pooling → Dense 256 → 128 → 3 | 0.355 (0.452) |

### 1.3 What Phase 1 taught us

- **Fusion never beat the better single stream.** M3 < M1, and M6 < M5. Telemetry alone (M2) was
  the weakest signal.
- **Temporal sequence models added capacity, not signal.** M1, M4 and M7 all landed around
  0.28–0.36.
- **Alert was the hardest class.** M7 scored F1 = 0 on Alert in folds 0–2. Errors were mostly
  Alert ↔ Low-vigilant; Alert ↔ Drowsy errors were rare.
- **M5 looked best by a wide margin (0.447)**, which pointed the next phase towards vision.
- **2026-09-15:** a research plan targeted 0.70–0.80 with video + telemetry fusion. It flagged
  that target as probably unreachable under a strict subject-independent protocol. That flag
  turned out to be right.

---

## 2. Phase 2 (2026-11-05): rebuilding the evaluation protocol

Before building new models, we audited our own evaluation. Four problems were found, and they
changed the protocol.

### 2.1 Problems found in the Phase 1 procedure

1. **M5's checkpoint was chosen on the test drivers.** The training script linked each fold's test
   subjects as the Ultralytics validation split. Ultralytics picks `best.pt` and early-stops on that
   split. Re-scored: `best.pt` 0.449 fold-mean, `last.pt` 0.363.
2. **M6's frame embeddings leaked.** They were extracted once, with the fold-0 M5 backbone (trained
   on drivers E–S), and reused for every fold. So folds 1–4 used features trained on their own test
   drivers.
3. **Window drift.** FAU misses about 2.4% of rows, so FAU-indexed windows drift against the video.
4. **No controls.** Nothing measured what a model with no sensor at all could score.

### 2.2 The final evaluation procedure

| Aspect | Final setting | Why |
|---|---|---|
| Windows | **Canonical window table**: 60 s windows, 15 s stride, on the true 60 fps video clock; 157 per 40-minute session; 4,532 with full video; label = KSS epoch at the window centre | One shared table for every model; no drift |
| Cohort | Fixed: the 16 video drivers | Every model is scored on exactly the same windows |
| Outer protocol **P0** (primary) | Same subject-independent 5-fold split as Phase 1 | Continuity with M1–M7 |
| Outer protocol **V1** (co-primary) | **Leave-one-subject-out (LOSO)**, 16 folds | More training drivers per fold; checks that a result is not an artefact of one fold grouping |
| Protocol **P4** | P0 or V1, plus the first 240 s of the test driver's Alert drive, **unlabelled**, at test time | Models a realistic enrolment drive for calibration |
| Validation | Each outer fold's training drivers are split into **4 subject-wise inner folds**, balanced for Alert epochs. *All* selection (checkpoints, smoothing, regularisation C, EMA β) uses these. Test drivers are never seen | Removes all test leakage |
| Seeds | 3 seeds per model; test probabilities averaged | Reduces run-to-run noise |
| Primary metric | **Pooled macro-F1**: all test windows from all folds concatenated, with a **95% CI from a bootstrap over drivers** | Fold-mean is unstable when some folds have 3–4 Alert epochs; resampling drivers respects the real unit of independence |
| Secondary metrics | Fold-mean macro-F1; **epoch-level F1** (probabilities averaged within each 4-minute epoch, matching the label resolution); **QWK** (quadratic-weighted κ: rewards getting the order right, Alert < Low-vig < Drowsy); per-class confusion; for B2, ROC-AUC | Each answers a different question about the same predictions |
| Controls | R0 floors; **R1 clock** (time-in-drive only); R1b clock + session label (shortcut upper bound); R6 tabular statistics; R7 image-statistics probe; **session-identity probe** (predict A vs D from features at matched KSS 5–6) | Every model must beat the clock and must not just recognise the recording |
| Statistics | Four hypotheses fixed **before** the final evaluation (H1–H4). **Exact subject-level paired permutation test** on pooled macro-F1 (2¹⁶ sign flips over 16 drivers), with **Holm correction** | Small-n-correct test; no cherry-picking |
| Robustness tests | Contiguous camera **blackouts** of 0, 30, 45 and 60 s inside each test window, at the same random position for every model and seed | Simulates a real camera failure; random frame dropping barely degrades vision |
| Confound diagnostics | **Drive-AUC** (separate the Drowsy-drive windows from the Alert-drive windows) vs **within-drive AUC** (order windows inside one drive); partial Spearman ρ with KSS given time; known-driver splits | Measures *what* a model detects, not only how well it scores |

---

## 3. Phase 3 (2026-11-05 → 11-07): the final-phase timeline

| When | Step | Signals | Result | Decision |
|---|---|---|---|---|
| 11-05 | **R0 / R1 / R1b clocks** | None (time-in-drive, epoch index; R1b adds the session label) | Clock **0.366** pooled; R1b 0.70 | The clock is the bar |
| 11-05 | **R7 image statistics** | IR brightness, colour moments, histograms | 0.356 | No global lighting shortcut |
| 11-05 | **R6 tabular statistics** | Window mean / SD / p10 / p90 of face geometry (2 Hz) and 14 telemetry channels (4 Hz) | 0.32–0.38 | ≈ clock |
| 11-05 | **Frozen DINOv2 extraction** | IR face (112, 168 px), L3D / R3D face, Pose (224 px); 2 fps (pose 1 fps) | 712 img/s at 112 px; about 1 h for everything | Frozen self-supervised features, because 290 labels are too few to train an encoder |
| 11-06 | **R4 / R4s**: M5 retrained with inner-validation selection | IR frames 0.5 fps | **0.362** pooled (0.333 fold-mean) | M5 is at clock level; it becomes the H1 comparator |
| 11-06 | **C1–C4 single streams** | IR / L3D / Pose / landmark series; mean-pool vs MIL heads | **IR 112 mean-pool 0.446**; Pose MIL 0.499 (but see LOSO) | IR 112 chosen by inner CV |
| 11-06 | **C2 / C5** | Two stereo views; temporal transformer; 240 s context | No gain beyond noise | Simplest head wins → **V\*** fixed |
| 11-06 | **Session-identity audit** | Same features, target = session | Pose 0.67, L3D 0.73, IR ≈ chance | Colour cameras see session appearance; IR does not |
| 11-06 | **Late fusion** | IR + telemetry (learned) 0.306; IR + pose (learned) 0.405; equal-weight 0.517 (exploratory) | Learned weights overfit | Fusion must be designed, not learned on 13 drivers |
| 11-06 | **C9 → C9d: fallback design** | V\* + 14 telemetry/grip channels + 5 clock features | Residual gate: robust but its gate never moves. **MoE + fast gate: clean 0.428, 60 s blackout 0.349 vs 0.186** | **VIGIL (M8) frozen** |
| 11-06 | **Round 2: events (R2-1/R2-4)** | 27 features from 60 fps FL + PL landmarks | 0.480 | Interpretable features match DINOv2 |
| 11-06 | **Round 2: calibration (R2-7)** | DINOv2 token minus the driver's Alert-drive reference | **0.521** | Largest single gain |
| 11-06 | Failed variants | Same-drive reference (C12) 0.368; absolute + residual tokens 0.405; DINOv3 0.325; 10 fps eye crops 0.409; ordinal loss 0.404 | Negative | Kept as ablations |
| 11-06 | **LOSO (V1)** | — | V\* 0.471; Pose 0.499 → 0.409 | IR is the protocol-stable stream |
| 11-07 | **Final evaluation** | Fusion of calibrated V\* + events (rule fixed in advance), B2, extended metrics, H1–H4, confound analysis, known-driver splits | See §5 | — |
| 11-07 | **Deployment models** | Refit on all 16 drivers → `models/final/` | — | Used only by the demo; in-sample on UL-DD |

---

## 4. The final models and their architectures

There are two "final" models, chosen for different reasons:
- **Best score:** calibrated V\* + events.
- **Frozen system design**, chosen for robustness: **VIGIL (M8)**.

Both share the same frozen vision front end.

### 4.1 Shared front end and V\*

1. IR video at 60 fps, sampled at **2 fps** (120 frames per 60 s window).
2. Face crop from the dataset's landmarks (1.5 × landmark extent, shifted up 8%), resized to
   **112 px**.
3. **DINOv2 ViT-S/14 with registers**, frozen, fp16 on Apple MPS. Each frame becomes a 768-d token
   [CLS ‖ mean patch].
4. Tokens z-scored with statistics fitted on the training rows only.
5. **V\* head:** masked mean over the 120 frames → LayerNorm → dropout 0.2 → Linear(768 → 3).
   That is **3,843 trainable parameters**.
6. Training: AdamW (lr 1e-3, weight decay 0.05), one-cycle schedule, 30 epochs, class-weighted
   cross-entropy, 15% frame dropout plus Gaussian feature noise, 3 seeds.

### 4.2 Best-scoring model: calibrated V\* + events

- **Calibrated V\* (P4):**
  - Compute the driver's mean DINOv2 token over the first 240 s of their Alert drive. No labels
    are used.
  - Subtract it from every token of both drives, then train and test the V\* head on the
    residuals.
  - In deployment, the reference would come from an enrolment drive.
- **Events model:**
  - Inputs are 27 features per 60 s window, from the 60 fps FL and PL landmarks:
    - **Ocular (13):** EAR-based eye closure normalised to the driver's open-eye level, blink
      rate, mean and p90 blink duration, long closures (> 0.5 s), PERCLOS at 70% / 80% /
      absolute thresholds, closing and opening velocity, amplitude–velocity ratio,
      inter-blink-interval variability, EAR mean and SD.
    - **Mouth / head (6):** yawns (MAR > 0.6 for > 1 s), mean MAR, pitch-proxy SD and velocity,
      nods, face-missing rate.
    - **Posture (8):** shoulder-normalised motion energy (nose, shoulders, wrists), stillness,
      head height above the shoulders (mean, min, SD), slouch.
  - Model: standardise → class-balanced logistic regression, with C chosen by inner CV.
  - Why these: before this model, every vision model saw at most 2 fps, so blink dynamics were
    invisible. These are the validated physiological markers of drowsiness, and they are
    human-readable.
- **Fusion:** the mean of the two models' log-probabilities, with no fitted weights. The rule was
  fixed before any fused number was seen.

### 4.3 Robust system: VIGIL (M8)

`z = (1 − g)·z_v + g·z_t`, with `g = σ(w·u + b)`.

- **Vision expert z_v:** the V\* head.
- **Fallback expert z_t:** LayerNorm → Linear 64 → GELU → dropout → Linear.
  - Input: the window mean of 14 telemetry and grip channels (pitch, roll, speed, rpm, gear,
    heading rate with absolute value and SD, Δspeed, left and right grip, grip asymmetry, a
    grip-zero flag, a telemetry-missing flag), multiplied by an availability flag.
  - Plus 5 clock features.
- **Gate input u:** observed-frame fraction, face-missing rate, no-frame flag. The gate sees only
  *vision quality*, never the label signal.
- **Gate settings:** bias −3 at initialisation (g ≈ 0.05, so it starts as the vision model) and a
  10× learning rate for the gate.
- **ModDrop training:**
  - half of the windows get a contiguous 15–60 s camera blackout, and a quarter of those are full
    60 s;
  - telemetry is dropped with probability 0.1;
  - auxiliary cross-entropy losses (weight 0.5) on each expert.
- **Why telemetry, grip and the clock:** they keep working when the camera does not. Phase 1
  showed telemetry is too weak to be an equal fusion partner, so here it is only a fallback.

---

## 5. Final scores in detail

### 5.1 Three-class KSS (y3), subject-independent

| Model | 5-fold pooled [95% CI] | LOSO pooled | Epoch F1 | QWK | Partial ρ (KSS given time) | Δ vs clock (p) |
|---|---|---|---|---|---|---|
| Clock (R1) | 0.366 [0.32, 0.41] | — | 0.369 | 0.19 | +0.04 | — |
| M5 end-to-end CNN (R4s) | 0.362 [0.29, 0.43] | — | 0.363 | 0.27 | +0.35 | −0.004 (0.95) |
| V\* | 0.446 [0.36, 0.52] | 0.471 | 0.440 | 0.13 | +0.17 | +0.079 (0.31) |
| Events LR | 0.480 [0.40, 0.55] | 0.430 | 0.507 | 0.16 | +0.12 | +0.114 (0.07) |
| VIGIL (M8), clean | 0.428 [0.34, 0.50] | — | 0.419 | 0.15 | +0.20 | +0.062 (0.43) |
| Calibrated V\* | 0.521 [0.44, 0.59] | 0.518 | 0.540 | **0.52** | **+0.63** | +0.155 (**0.01**) |
| **Calibrated V\* + events** | **0.536 [0.45, 0.61]** | **0.525** | **0.545** | 0.52 | +0.63 | +0.170 (**< 0.01**) |

- Only the calibrated models beat the clock significantly. They are also the only ones whose
  predictions track KSS beyond time-in-drive.
- **Confusion pattern:**
  - The clock and M5 almost never recognise Alert.
  - V\* spreads its errors over every cell.
  - The calibrated models call only 0–3% of Alert windows Drowsy, but call about half of the
    Drowsy windows Low-vigilant. The remaining error is between adjacent classes.
- **Per-driver LOSO F1** for V\* ranges from 0.03 to 0.99. Drivers vary more than models do.

### 5.2 Robustness to camera failure

| Model | Blackout 0 s | 30 s | 45 s | 60 s (camera off) | Gate mean, 0 → 60 s |
|---|---|---|---|---|---|
| Vision only (same ModDrop training) | 0.430 | 0.433 | 0.424 | **0.186** | — |
| Residual telemetry gate (C9) | 0.435 | 0.435 | 0.423 | 0.339 | 0.12 → 0.12 |
| **VIGIL (M8)** | **0.428** | 0.426 | 0.423 | **0.349** | **0.13 → 0.93** |

### 5.3 Binary B2 (KSS ≤ 5 vs ≥ 6)

| Model | Pooled F1 | AUC |
|---|---|---|
| Clock | 0.570 | 0.56 |
| V\* | 0.569 | 0.58 |
| VIGIL (M8) | 0.577 | 0.60 |
| **Calibrated V\*** | **0.775** (LOSO 0.779) | **0.86** |

B2 coincides with "which drive" for 91% of windows, so read this number alongside §5.4.

### 5.4 What the models detect

| Model | B2 within-drive AUC | B2 drive-AUC | KSS ≥ 7 within-drive AUC | KSS ≥ 7 drive-AUC |
|---|---|---|---|---|
| Clock | **0.81** | 0.50 | **0.90** | 0.50 |
| V\* | 0.50 | 0.66 | 0.64 | 0.79 |
| Events | 0.56 | 0.59 | 0.68 | 0.59 |
| VIGIL (M8) | 0.50 | 0.68 | 0.66 | 0.80 |
| Calibrated V\* | 0.63 | **0.90** | 0.66 | **0.91** |
| Calibrated V\* + events | 0.64 | 0.90 | 0.70 | 0.89 |

- Within a drive, time-on-task beats every vision model.
- The gains over the clock come from separating the two drives.
- **Splits where the test driver is seen in training:**

  | Split | V\* | No video: who + drive + time |
  |---|---|---|
  | Random windows | 0.910 | 0.926 |
  | Held-out 4-minute epochs (60 s purge) | 0.802 | 0.867 |
  | Known driver, new drive | 0.413 | 0.635 |

  These splits measure recording recognition, not drowsiness.

### 5.5 Pre-registered hypotheses

| Hypothesis | Δ pooled F1 | p | Holm p |
|---|---|---|---|
| H1: V\* > M5 end-to-end CNN (R4s) | +0.084 | 0.152 | 0.456 |
| H2: VIGIL does no harm on clean input | −0.002 | 0.802 | 0.802 |
| H3: VIGIL > vision-only, 60 s blackout | +0.163 | 0.076 | 0.306 |
| H4: calibrated > uncalibrated V\* | +0.075 | 0.177 | 0.456 |

None survives Holm correction at n = 16 drivers. The effect sizes and confidence intervals are
reported instead.

---

## 6. The arc in one paragraph

We started with sequence models over extracted signals (M1–M4), moved to learning from raw IR
video (M5–M7), and found that fusion never beat the best single stream. Auditing our own
evaluation then showed that M5's lead came from choosing checkpoints on the test drivers. With a
rebuilt protocol, every Phase-1 model sits at the level of a clock that only knows time-in-drive.
That protocol has canonical video-clock windows, inner subject-wise validation, pooled F1 with
driver-bootstrap CIs, LOSO, a clock control and pre-registered tests. Frozen DINOv2 features with
a 3,843-parameter head (V\*) were the first to clear the clock. Unlabelled driver calibration plus
60 fps blink and posture events gave the best score (0.536 / LOSO 0.525). VIGIL (M8) made the
system survive camera failure (0.349 vs 0.186). The confound analysis shows that, on UL-DD, those
gains over the clock are mostly the model recognising which drive it is watching.
