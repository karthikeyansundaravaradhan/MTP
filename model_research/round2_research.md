# Round 2 research: bottlenecks, validation changes, and next ablations

Written 2026-11-06, after M8 was frozen (CP2). This doc covers the window between the current
tasks and the final report, PPT and demo.

Scope (user constraint): inputs are limited to visual cues (IR, stereo, posture video, face and
pose landmarks), telemetry, and time-in-drive. The physiological CSVs (BVP, EDA, HR …) are *not*
model inputs. BVP may be used only as ground truth to validate a video-derived pulse (R2-6).

All numbers come from `scripts/m8/diag_bottlenecks.py` (`results/m8/diag_bottlenecks.json`)
and the run JSONs in `results/m8/`, unless a source is cited.

---

## 1. Where the model loses: measured bottlenecks

| # | Diagnostic | Result | What it means |
|---|---|---|---|
| D1 | KSS variance decomposition (video cohort) | **67 % between the A and D sessions of the same subject**, 16 % between subjects, **18 % within a session**. 8 of 29 sessions have a single 3-class label. | The task is mostly "which drive is this, and how drowsy overall", not tracking change minute by minute. A within-session model has little label variance to learn from. |
| D2 | Does V\* track KSS? (Spearman of E[ŷ] with KSS) | Within session **0.12** (positive in 62 % of sessions). Session-level **0.10**. Overall 0.19. | V\* is weak at *both* levels. It does not even rank sessions well. The signal is simply weak; this is not a smoothing or post-processing problem. |
| D3 | Oracle per-subject offset on E[ŷ] | Macro-F1 0.432 → **0.515** (+0.08) | Subject-specific bias is a real but bounded loss. Personal calibration can recover at most about +0.08. |
| D4 | Windows straddling two KSS epochs | 17 % of windows. Accuracy 0.483 vs 0.475 for clean windows. | Label-boundary noise is **not** a bottleneck. |
| D5 | Learning curve (linear probe on IR-112 window means vs number of training subjects) | 4 subjects → 0.365, 8 → 0.380, 12 → 0.426, **15 (LOSO, pooled) → 0.506** | **Strongly data-limited**: roughly +0.015 F1 per extra subject, and the curve has not flattened. The 5-fold protocol (about 12–13 training subjects) understates what a 15–16-subject model can do. |
| D6 | Can telemetry predict time-in-session? (held-out subjects) | R² = **0.04** | Telemetry is *not* a clock proxy. It simply carries little drowsiness information in this simulator, which matches R6 (≈ clock), the C7 telemetry expert (< clock), and C9 (clock fallback ≥ telemetry). |
| — | Session-identity probe at matched KSS (Day 3) | IR ≈ chance; pose 0.67, L3D 0.73 | Pose and stereo appearance embeddings partly encode the recording session. IR embeddings do not. |
| — | Inner-CV vs test correlation across 14 runs | r = **0.22** | Model selection at a ±0.05 effect size is noise with 12–13 inner-training subjects. |
| — | Temporal resolution actually used | IR / L3D embeddings at **2 fps**, pose at **1 fps**, blinks as **2 Hz bin counts**. FL, FAU and PL are all **60 fps**. | Blink *dynamics* have never been seen by any model: duration, closing velocity, the amplitude-velocity ratio, and micro-sleeps (eye closure > 0.5 s). These are the best-established ocular drowsiness markers (Johns Drowsiness Scale family; survey §3.3). |

**Bottleneck ranking:**
1. **Number of subjects** (D5): the biggest lever, and it can only be partly addressed (LOSO,
   training on all 16 subjects for the deployed model).
2. **Weak, slow, subject-confounded signal** (D1, D2, D3): addressed by richer temporal
   features and by subject calibration.
3. **The protocol cannot resolve small effects** (inner-vs-test correlation 0.22): addressed by
   changing the validation design (§2).
4. **Telemetry is information-poor:** keep it only as a fallback (M8 already does this).

---

## 2. Validation-check changes

These are ordered by value. The ones marked **adopt** are cheap and remove a known weakness.
None of them changes the frozen M8. They change how it, and the round-2 ablations, are judged.

| ID | Change | Why (evidence) | Cost |
|---|---|---|---|
| V1 | **LOSO (16 folds) as a co-primary protocol.** Report it next to the 5-fold protocol, which stays for comparability with Kavipriya. **Adopt.** | D5: training on 15 subjects instead of about 12 is worth about +0.08 for the linear probe. LOSO also gives one score per subject, the natural unit for the paired test, and is standard in the cross-subject literature (e.g. the TTA EEG work). | About 3× the 5-fold cost for heads on cached features (minutes). Not feasible for the YOLO baseline R4; keep R4 on 5-fold and say so. |
| V2 | **Repeated grouped CV for selection** (5 × 5 different subject shuffles), replacing single inner 4-fold OOF. **Adopt for round 2.** | Inner-OOF vs test r = 0.22. Averaging over shuffles shrinks selection variance. Selection is then pre-registered on this, and the test protocol is touched once. | 5× the inner cost (cheap heads only). |
| V3 | **Epoch-level evaluation** (one decision per 4-min KSS epoch, causal mean of its windows) as a secondary metric. | KSS is reported once per 4 min, so the 157 windows per session are pseudo-replicates of 10 labels. Epoch-level metrics match label granularity and give honest effective sample sizes. | Scoring-only. |
| V4 | **Two-level metrics:** (a) *between-session*: per subject, does the model rank the D session above the A session (paired AUC over 13 two-session subjects); (b) *within-session*: Spearman(E[ŷ], KSS) per session. **Adopt.** | D1 and D2: macro-F1 hides which of the two jobs a model does. | Scoring-only. |
| V5 | **"Beats the clock" as an explicit test:** the paired test vs R1, plus a partial correlation of E[ŷ] with KSS controlling for time-in-drive. **Adopt.** | Critique #10. The clock gets 0.355 / B2 0.573. On B2, V\* only ties the clock. | Scoring-only. |
| V6 | **Ordinal / regression metrics:** QWK and KSS-MAE alongside macro-F1. | The 3-class boundaries (≤3 / 4–6 / ≥7) put most windows in an ambiguous middle class. Ordinal metrics credit near misses. | Scoring-only (QWK is already in `metrics`). |
| V7 | **Calibration protocol P4:** use the subject's **Alert session** (a different drive) as an unlabelled reference for per-subject normalisation, *not* the same session (C12 showed that destroys the level signal). | D3 oracle: +0.08 headroom. Deployable: calibrate on a known-fresh drive. The IR session probe is at chance, so the A → D appearance shift is small for IR. | Cheap (an embedding transform). |
| V8 | **Session-shortcut audit for every new stream:** the matched-KSS session probe, before a stream may enter fusion. **Adopt as a gate.** | Pose and L3D failed it. | About 1 min per stream. |

---

## 3. Architecture and feature directions (Mac M4, 16 GB)

All of these keep the frozen-encoder + small-head recipe, which is where the evidence points with
16 subjects. They are listed with cost and the bottleneck each one attacks.

### 3.1 High-temporal-resolution ocular dynamics (bottleneck 2) — strongest candidate

| ID | Idea | Details | Cost |
|---|---|---|---|
| R2-1 | **Blink-event features at 60 fps from the existing FL landmarks** | Per window: blink rate, mean / max blink duration, PERCLOS-70/80, closing / opening velocity, **amplitude-velocity ratio**, count of eye closures > 0.5 s (micro-sleeps), inter-blink-interval CV; plus yawn events (MAR) and head-nod events (pitch velocity). Optionally the **eyelid-angle (ELA)** metric of [Blinking Beyond EAR](https://arxiv.org/abs/2511.19519), described as more viewpoint-stable than EAR. | CPU only, about 30 min to code and 5 min to run. Interpretable. Also a candidate second fallback expert. |
| R2-2 | **IR eye-region crops at 10 fps → DINOv2-S** | Crop both eyes from FL (IR frame-aligned), 112 px. A temporal head over 600 steps per window, or stride. Gives the model eyelid dynamics that 2 fps cannot see. | Extraction about 45 min for all sessions (5× the 2 fps IR run). |
| R2-3 | **Face video clip encoder (MARLIN-S / VideoMAE-v2-S)** | 16-frame clips at about 8–15 fps, face crops. Facial masked-video pretraining captures motion. Already in the plan as A7. | About 1–2 h extraction. Needs a HF / transformers install check (not in timm). |

### 3.2 Appearance-invariant posture cues (bottlenecks 2 and 3; removes the pose shortcut)

| ID | Idea | Details | Cost |
|---|---|---|---|
| R2-4 | **PL keypoint dynamics at 60 fps** instead of pose-video embeddings | Head-drop angle, slouch (shoulder-to-nose distance), movement energy and stillness bouts, nodding frequency. Keypoints carry no lighting or clothing, so they should pass the V8 session audit that pose embeddings failed. Uses a per-joint MLP + temporal head (the plan's LandmarkFormer-lite). | CPU features about 20 min, plus a cheap head. |
| R2-5 | **Stereo depth from L3D/R3D (StereoSGBM)** | Head-to-camera distance and forward-lean time series from disparity on the face box. Turns the stereo pair (whose RGB appearance failed the audit) into a geometric, appearance-free cue. | About 1 h of CPU for 2 fps sampling. Calibration unknown, so use relative disparity. |

### 3.3 Video-derived physiology (bottleneck 2; still "visual")

| ID | Idea | Details | Cost |
|---|---|---|---|
| R2-6 | **rPPG from 60 fps IR face video → pulse rate and HRV** | Skin ROI (forehead / cheeks from FL) mean intensity → band-pass 0.7–3 Hz → spectral HR; RMSSD and LF/HF over the 60 s window. Monochrome NIR rules out RGB methods (CHROM/POS); use single-channel or [NIR-specific](https://www.mdpi.com/1424-8220/26/18/5734) approaches. **Validate against the dataset's own BVP** before use (ground truth only). HR / HRV are among the strongest drowsiness correlates in the literature. | 2–3 h, higher risk (IR rPPG is noisy, and 60 fps is just enough). Treat as exploratory. |

### 3.4 Subject and session robustness (bottleneck 1 and D3)

| ID | Idea | Details | Cost |
|---|---|---|---|
| R2-7 | **Alert-session calibration (V7 as a model):** z-score each subject's embeddings by their Alert-session statistics | Unlike C12 (same-session reference), this keeps between-session level differences while removing subject identity. Upper bound from D3 is about +0.08. | Minutes (a flag on `video_bank`). |
| R2-8 | **Absolute + delta-from-start features** | Concatenate the window embedding with (embedding − running causal mean since drive start). C12 used only the delta, which destroyed the level signal; giving the head both lets it choose. | Minutes. |
| R2-9 | **Subject-adversarial head (gradient reversal, subject ID as the domain)** | Push window embeddings to be subject-invariant. Common in cross-subject EEG fatigue work ([cross-subject DA](https://iopscience.iop.org/article/10.1088/1741-2552/ad546d), [multi-modal DANN](https://www.nature.com/articles/s41598-025-92701-6)). Risk: with 12–15 domains it can also erase drowsiness. | About 1 h. |
| R2-10 | **Test-time adaptation of LayerNorm statistics** on the test subject's own past windows (causal) | Online TTA as in [calibration-free EEG drowsiness](https://arxiv.org/abs/2511.22030). Must be causal and per-subject, never per-session, for the same reason as C12. | About 1 h. |

### 3.5 Time-aware modelling ("including time")

| ID | Idea | Details | Cost |
|---|---|---|---|
| R2-11 | **Clock-residual model:** z = z_clock + z_vision (vision learns only what time-in-drive does not explain) | Directly targets V5. A time-conditioned head (FiLM by time-in-drive) is the soft version. The simulator literature confirms that time-on-task drives both KSS and driving metrics ([time-on-task effects](https://www.researchgate.net/publication/273656144_Leading_Indicators_of_Drowsiness_in_Simulated_Driving)), so time should be modelled explicitly, not left as a hidden confound. | Minutes (another mode in `m8/vigil.py`). |

### 3.6 Encoder swaps (cheap checks, low expected gain)

| ID | Idea | Details | Cost |
|---|---|---|---|
| R2-12 | **DINOv3-S/16** (`vit_small_patch16_dinov3.lvd1689m`, available in timm) on IR-112 | Drop-in replacement for DINOv2-S. Its distilled small models are reported stronger as frozen features. | About 15 min extraction. |
| — | **V-JEPA 2** | Only ViT-L and larger are released. Too heavy to extract at scale on 16 GB, and a 2026 driving-video study found its *self-supervised* objective, not its features, to be the weak point ([Asleep at the Wheel](https://arxiv.org/abs/2608.01336)). Skip. | — |
| — | **LoRA fine-tuning** (A6) | D5 says data, not encoder capacity, is the limit; fine-tuning 22 M parameters on 12 subjects is high-risk. Keep as a last stretch goal only. | — |

---

## 4. Proposed round-2 ablation slate (ranked)

Selection uses V2 (repeated grouped CV). Final numbers are reported on the 5-fold protocol *and*
LOSO (V1). Every new stream must pass the V8 audit before it may enter fusion.

| Rank | Ablation | Hypothesis | Est. time |
|---|---|---|---|
| 1 | **R2-1** blink / yawn / nod events at 60 fps → tabular expert, and fused with V\* | Ocular dynamics carry drowsiness signal that 2 fps embeddings miss | 1 h |
| 2 | **R2-7** Alert-session calibration, and **R2-8** absolute + delta | Recovers part of the +0.08 subject-offset headroom | 0.5 h |
| 3 | **R2-4** PL keypoint dynamics | A posture signal without pose-video's session shortcut | 1 h |
| 4 | **R2-2** IR eye crops at 10 fps | Learned eyelid dynamics > hand-crafted (R2-1) | 1.5 h |
| 5 | **R2-11** clock-residual / time-FiLM | Vision gains measured *beyond* time-on-task | 0.5 h |
| 6 | **V1** LOSO re-evaluation of V\*, M8, best round-2 model | Confirms the D5 learning-curve gain on the real heads | 1 h |
| 7 | **R2-12** DINOv3-S swap | Cheap encoder check | 0.5 h |
| 8 | **R2-6** rPPG HR / HRV (BVP-validated) | Video-derived physiology as an extra visual cue | 3 h, exploratory |
| 9 | R2-9 / R2-10 subject-adversarial / TTA | Subject invariance | 2 h, stretch |
| 10 | R2-5 stereo-disparity head distance | Appearance-free stereo cue | 1 h, stretch |

Ranks 1–7 fit in about 1.5 days of the current 2–3-day buffer, before the report, PPT and Gradio
work. Any winner can enter an "M8+" variant **only through V2 selection**, and the frozen M8
stays the pre-registered headline. That keeps the thesis honest: round 2 is reported as
post-freeze exploration.

## 5. Round-2 results (status 2026-11-06 17:00)

All runs use the video cohort under P0 (5-fold), with 3 seeds where a neural head is used. The
reference is V\* = IR-112 mean-pool, 0.440 fold-mean / 0.446 pooled. Architectures and diagrams
are in [`../docs/m8_models.md`](../docs/m8_models.md) §5.

| Rank | Item | Status | Fold-mean | Pooled | Δ pooled vs V\* (p) | Verdict |
|---|---|---|---|---|---|---|
| 1 | R2-1 events (27 features) → LR | done | 0.440 | 0.480 | +0.034 | Ties V\*. Interpretable. LOSO 0.430 |
| 1 | R2-1 events + V\* (equal weights) | done | 0.445 | 0.461 | +0.016 (0.64) | n.s. |
| 2 | **R2-7** Alert-session calibration | done | **0.500** | **0.521** | +0.075 (0.18) | Best. P4 only: gain is in the A session (+0.092 after 4 min), D −0.050 |
| 2 | R2-8 absolute + delta (concat) | done | 0.383 | 0.405 | −0.041 | Negative |
| 3 | R2-4 posture keypoint events | done | 0.386 | 0.449 | +0.003 | Passes the V8 audit (0.455); pose video fails it (0.67) |
| 4 | R2-2 IR eye crops 10 fps, mean | done | 0.360 | 0.385 | −0.061 | Negative: eyes are ≈ 10 px, blurry when upscaled |
| 4 | R2-2 IR eye crops 10 fps, MIL | done | 0.367 | 0.409 | −0.037 (0.62) | Negative |
| 5 | R2-11 clock-residual | covered by C9 `clock` | 0.413 (clean) | 0.435 | — | No gain |
| 6 | V1 LOSO V\* | done | — | 0.471 | +0.025 vs 5-fold (0.30) | Small data gain; IR is the most protocol-stable stream |
| 6 | V1 LOSO Alert-calibrated V\* | done | — | 0.518 | +0.048 vs LOSO V\* (0.44) | Calibration gain holds under LOSO |
| 6 | V1 LOSO pose MIL | done | — | 0.409 | −0.090 vs its 5-fold | Pose's 5-fold edge does not survive LOSO |
| 7 | R2-12 DINOv3-S/16 | done | 0.329 | 0.325 | −0.121 | Negative (7×7 grid at 112 px) |
| 8 | R2-6 rPPG (BVP-validated) | not started | | | | Optional |
| 9 | R2-9 / R2-10 subject-adversarial / TTA | not started | | | | Stretch |
| 10 | R2-5 stereo disparity | not started | | | | Stretch |

**Validation changes adopted so far:**
- **V1 LOSO:** done for V\*, calibrated V\*, pose MIL (above) and events (0.430).
- **V5 beats-the-clock:**
  - V\* Δ +0.079 (p = 0.31), M8 Δ +0.062, events Δ +0.114 (p = 0.066).
  - Partial ρ given time: 0.17 / 0.20 / 0.12.
- **V8 session audit:** run for events and calibration.
- **V2 / V3 / V4 / V6:** pending. They are scoring-only and will be done in the Day 8 final
  evaluation.

**Interim conclusions:**
1. No round-2 architecture beats V\* significantly under P0 at n = 16.
2. The two real gains are both about the **driver**, not the encoder:
   - Alert-session calibration (P4; 0.52 under both protocols).
   - More training subjects: LOSO gives +0.025 for V\*, smaller than the D5 linear-probe estimate.
   - Pose and events lose 0.05–0.09 under LOSO, so IR-112 is the most robust core.
3. Interpretable 60 fps events match the foundation-model stream. They are the report's
   interpretability result and a candidate demo readout.
4. M8 stays the pre-registered headline. Round 2 is reported as post-freeze exploration.

**Final evaluation update (2026-11-07):** full tables are in
[`../docs/m8_models.md`](../docs/m8_models.md) §0.5.
- **Fusion:** calibrated V\* + events gives y3 0.536 (5-fold) / 0.525 (LOSO). Not significant vs
  calibrated V\* alone.
- **B2:** calibrated V\* scores 0.775 / 0.779 (AUC 0.86), but this is mostly drive detection.
  - B2 equals the session for 91% of windows.
  - Calibrated V\* separates the drives with AUC 0.90, but its within-drive AUC is only 0.63–0.66.
  - The clock's within-drive AUC is 0.81 (B2) and 0.90 (KSS ≥7). Within a drive, time beats every
    vision model.
- **Pre-registered H1–H4:** none is significant after Holm. The smallest Holm-adjusted p is H3,
  0.306.
- **Subject-dependent splits:** random 0.910, epoch-purged 0.802, other-drive 0.413. The first two
  are recording recognition; a no-video who + drive + time LR scores higher. Subject-independent
  evaluation is kept.

## Sources

- [UL-DD dataset paper (arXiv 2507.13403)](https://arxiv.org/abs/2507.13403); [Scientific Data version](https://www.nature.com/articles/s41597-025-06540-1); [Tampere portal (public Zenodo release, Dec 2025)](https://researchportal.tuni.fi/en/datasets/ul-dd-a-multimodal-drowsiness-dataset-using-video-biometric-and-b/)
- [Blinking Beyond EAR: eyelid angle metric (arXiv 2511.19519)](https://arxiv.org/abs/2511.19519)
- [Calibration-free EEG drowsiness detection with online TTA (arXiv 2511.22030)](https://arxiv.org/abs/2511.22030)
- [Cross-scenario / cross-subject domain adaptation for fatigue (J. Neural Eng.)](https://iopscience.iop.org/article/10.1088/1741-2552/ad546d); [Multi-modal DANN fatigue monitoring (Sci. Rep. 2025)](https://www.nature.com/articles/s41598-025-92701-6)
- [Wavelet spectral gating for NIR rPPG (Sensors 2026)](https://www.mdpi.com/1424-8220/26/18/5734); [Contactless pulse rate in a driving simulator (arXiv 2505.01299)](https://arxiv.org/pdf/2505.01299)
- [DINOv3 (arXiv 2508.10104)](https://arxiv.org/abs/2508.10104); [timm DINOv3 ViT-S/16](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)
- [V-JEPA 2 (arXiv 2506.09985)](https://www.alphaxiv.org/abs/2506.09985); [Asleep at the Wheel: JEPA limits on driving data (arXiv 2608.01336)](https://arxiv.org/abs/2608.01336)
- [Leading indicators of drowsiness in simulated driving](https://www.researchgate.net/publication/273656144_Leading_Indicators_of_Drowsiness_in_Simulated_Driving); [Steering wheel reversal rate metric](https://www.researchgate.net/publication/325843375_A_Steering_Wheel_Reversal_Rate_Metric_for_Assessing_Effects_of_Visual_and_Cognitive_Secondary_Task_Load)
