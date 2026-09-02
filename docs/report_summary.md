# Kavipriya's Report — Summary

Source: `Kavipriya_report (2).html` (root) / `Literature/Kavipriya_Ramasamy_24872_Final_Phase (2).pdf`
— **"Driver Drowsiness Detection Using Camera and CAN Signals"**, Kavipriya Ramasamy, MTech
(Online) Data Sciences and Business Analytics, Indian Institute of Science, project carried out
at **NIQ** (Guide: Srividya Lakshmanan; Faculty mentor: Jaichendar Swaminathan), June 2026.

This is the prior project on the same UL-DD dataset that this Master's project is meant to build
on / improve upon ("similar to the one done in the report" per the project brief).

## 1. Problem framing

- **Task:** 3-class vigilance classification — **Alert (KSS 1–3)**, **Low Vigilant (KSS 4–6)**,
  **Drowsy (KSS 7–9)** — same binning as the UL-DD paper's Low/Med/High.
- **Modalities actually used:** infrared driver-camera frames/video, Facial Action Units (FAU),
  and vehicle telemetry (CAN-style: pitch, roll, speed, RPM, gear).
- **Modalities in UL-DD but NOT used by this report:** biometric wearable signals (HR, EDA, BVP,
  IBI, TEMP, SpO2), grip pressure, RGB/3D depth video, pose video/landmarks. This is the single
  biggest gap relative to the dataset's full potential — the dataset paper's own best result
  (84% acc, early-fusion Bio+Behavioral+Facial, §6.4) already beats every model in this report,
  and it uses signal families this report never touches.
- **Central research question:** "Under a subject-independent evaluation protocol, which
  modality and architecture provide the most reliable driver drowsiness detection performance
  on UL-DD?"
- **5 research questions:** (1) best single modality, (2) does temporal modeling help,
  (3) does multimodal fusion beat the best unimodal model, (4) how much does subject-independent
  evaluation matter, (5) which model is most deployment-suitable.

## 2. Dataset journey before settling on UL-DD

1. **MRL Eye dataset** (Kaggle, ~170k 64×64 open/closed eye crops) — simple binary CNN baseline,
   got Acc 0.95 / F1 0.93, but rejected as final dataset: no full face, no telemetry, no
   graded/subject-level labels, task too simple (binary eye state vs. 3-class vigilance).
2. **CARLA simulator** — planned synthetic video + simulated CAN telemetry collection
   (`carla_collector.py`), abandoned: no real human drowsiness ground truth, domain gap between
   simulated and real driver behavior.
3. **UL-DD selected** because it is the only option with real driver data, full face/video,
   real telemetry/CAN signals, multimodal streams, KSS-based graded labels, and existing
   subject-wise structure suitable for subject-independent CV.

## 3. Methodology

### Labeling & windowing
- KSS given per 4-minute interval → too coarse for enough training samples, so re-windowed into
  **60-second sliding windows at 4 Hz → 240 timesteps per window**, each window inheriting its
  parent interval's 3-class label. Multiple overlapping windows per 4-min interval.
- FAU windows: `(240 timesteps, 30 AU features)`. Telemetry windows: `(240 timesteps, 5 features:
  pitch, roll, speed, rpm, gear)`.

### Subject-independent 5-fold CV (no subject appears in both train and test within a fold)

| Fold | Held-out test subjects |
|---|---|
| 0 | A, B, C, D |
| 1 | E, F, G, H |
| 2 | I, J, K |
| 3 | L, M, N, O |
| 4 | P, Q, R, S |

### Model families evaluated

| Model | Input | Architecture | Notes |
|---|---|---|---|
| **M1** | FAU (240×30) | BiLSTM(128)→BiLSTM(64)→Dense(64)+BN→Dense(3) | facial-dynamics-only baseline |
| **M2** | Telemetry (240×5) | LSTM(64)→LSTM(32)→Dense(32)+BN→Dense(3) | CAN-only baseline |
| **M3** | FAU + Telemetry | BiLSTM + LSTM streams, **bidirectional cross-modal attention**, concat, dense | main early multimodal fusion model |
| M4 | — | — | **designed but excluded** — needs real-time face/window-state sync not feasible offline on UL-DD |
| **M5** | IR frames, 224×224 | **YOLOv8n-cls**, pretrained, fine-tuned 20 epochs, batch 32 | frame-independent image classifier; **best overall model** |
| **M6 Lite** | 16× YOLOv8 embeddings (512-d) + telemetry | Visual BiLSTM(128) + CAN BiLSTM(64), mean-pool, concat, dense | lightweight late fusion |
| **M6 Full** | same inputs as M6 Lite | Visual Transformer + CAN BiLSTM, **bidirectional cross-attention** | richer fusion, more sync-sensitive |
| **M7** | 16 sampled raw IR frames/window | EfficientNet-B0 (frame features) → BiLSTM(256) → attention pooling → deep MLP | direct temporal video model |

### Training / evaluation
- Loss: cross-entropy with class weights (focal loss trialed for M7); Adam/AdamW; YOLOv8n-cls via
  Ultralytics training workflow.
- Checkpoint selection by validation **macro-F1** (not accuracy) — chosen because Alert is
  consistently the hardest class and class balance varies per fold.
- Metrics: accuracy, macro-F1, balanced accuracy, per-class F1, confusion matrices.

## 4. Results

| Model | Description | Accuracy | Macro-F1 | Observation |
|---|---|---|---|---|
| M1 | FAU BiLSTM | 0.391 ± 0.116 | 0.348 | facial dynamics baseline |
| M2 | Telemetry LSTM | 0.387 ± 0.040 | 0.339 | CAN-only baseline |
| M3 | FAU+CAN attention | 0.399 ± 0.098 | 0.378 | best non-visual early-fusion macro-F1 |
| **M5** | **YOLOv8 IR image classifier** | **0.545 ± 0.100** | **0.444** | **best overall** |
| M6 Full | Visual-CAN cross-attention | 0.420 ± 0.038 | 0.377 | richer fusion, still below M5 |
| M6 Lite | Lightweight visual-CAN fusion | 0.405 ± 0.031 | 0.360 | simpler fusion, still below M5 |
| M7 | EfficientNet-B0 temporal IR | 0.434 ± 0.090 | 0.314 | decent accuracy, weak macro-F1 (overfits) |

**M5 confusion matrix (normalized, all folds):**

| True \ Pred | Alert | LowVigilant | Drowsy |
|---|---|---|---|
| Alert | 19.6% | 62.3% | 18.2% |
| LowVigilant | 12.9% | 63.1% | 24.0% |
| Drowsy | 4.8% | 32.6% | 62.6% |

**Alert is the hard class throughout** (F1 ≈ 0.23 on the Fold-0 breakdown shown in the report) —
most Alert frames get misclassified as LowVigilant, because a single infrared frame of a relaxed-
but-alert face looks very similar to a mildly low-vigilant one. Drowsy is comparatively easy to
separate from Alert (only 4.8% confusion). This directly motivates temporal modeling — but M6/M7
(the temporal/fusion attempts) did not actually beat the static frame classifier M5.

## 5. Key findings

- Best model overall: **M5 (YOLOv8n-cls on raw IR frames)** — a *static, frame-level, single-
  modality* visual classifier beat every temporal and multimodal alternative tried.
- More architectural complexity ≠ better generalization here: cross-modal attention (M3, M6 Full)
  and deep temporal video modeling (M7) all underperformed the simple frame classifier.
- Fusion failed specifically because the **auxiliary modalities were weak** (telemetry alone
  maxes out ~0.39 accuracy) — fusing a strong branch with a weak, possibly noisy/misaligned
  branch pulled the strong branch down rather than lifting the result.
- Subject-independent evaluation is called out repeatedly as *the* methodologically important
  choice — without it, results would likely look much better but be an artifact of memorizing
  driver-specific appearance/behavior.
- M4 (a real-time face-tracking + window-state model) was designed but dropped as infeasible for
  offline UL-DD data — a reminder that not every planned architecture fits this dataset's
  offline/pre-recorded nature.

## 6. Stated limitations

- Only 19 subjects total → subject-independent folds are small and high-variance (note the large
  ± std on M1/M3/M5/M7 accuracy).
- Missing video for some subjects (see `dataset_summary.md` §2 — B, I, M) reduces usable windows
  for video-dependent models.
- M5 and other models used different visual preprocessing pipelines, so cross-model comparison
  should lean on the validated aggregate metrics rather than architecture-only claims.
- Report never validates biometric-signal modeling (HR/EDA/BVP/etc.) or pose/grip-pressure at
  all, despite these being part of UL-DD and part of the dataset paper's own strongest result.

## 7. Recommended future work (from the report, prioritized)

| Priority | Direction | Rationale |
|---|---|---|
| 1 | M5 temporal ensembling | average/attention-pool M5 predictions over 5–10 neighboring frames |
| 2 | M5 + FAU late fusion | precompute M5 probabilities/embeddings, fuse with FAU temporal embedding, validate on identical subject splits |
| 3 | M5 + CAN calibrated fusion | learned confidence weighting instead of naive averaging |

## 8. What this means for the new Master's project

Framed against this prior work, clear open opportunities for a genuinely **multimodal / hybrid /
ensemble** model that this report did not attempt:
- Bring in the **biometric wearable streams** (HR, EDA, BVP, IBI, TEMP, SpO2) and **grip
  pressure** — completely unused here, yet part of the dataset's own best-performing
  combination (84% acc in the source paper).
- Investigate why M6 fusion underperformed unimodal M5 — likely a calibration/weighting problem,
  not a fundamental flaw in fusion — and whether a better-weighted or later-stage fusion (as the
  report's own future-work section suggests) recovers the gap.
- The Alert vs. LowVigilant boundary is the dominant source of error across this entire report;
  any new approach should be judged specifically on whether it improves that boundary, not just
  aggregate accuracy.
- Macro-F1, not accuracy, is the report's chosen headline metric because of class imbalance
  across folds — worth keeping for comparability if this project extends the same evaluation
  protocol (same subject-independent folds, same 3-class KSS binning).
