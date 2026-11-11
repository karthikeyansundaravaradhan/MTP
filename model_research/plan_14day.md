# 14-Day Execution Plan: 12 Core Experiments

*Written 2026-11-05. **This file supersedes the schedule and run list in [`ablation_plan.md`](ablation_plan.md).** That file remains the detailed spec for architectures and shapes. This revision applies the fixes from [`plan_critique.md`](plan_critique.md). Day 1 = Thu 2026-11-05, Day 14 = Wed 2026-11-18.*

## 0. What changed from the 18-day plan

| Critique item | Change |
|---|---|
| #1 No 14-day path | The new schedule below has hard cut lines. Everything outside the 12 core experiments is a stretch goal (§4). |
| #2 LandmarkFormer OOM | The landmark stream runs at **2 Hz** (120 steps, matching the video tokens). It uses a per-joint MLP with mean pooling, not spatial attention, and batch ≤ 32. |
| #3 Statistics can't reach significance | Headline tests are a **paired sign-flip permutation test over the 16 test subjects**, using per-subject balanced accuracy. Holm correction is applied across H1–H4. |
| #4 Frame blanking is too easy | Robustness uses **contiguous 15–45 s blackouts across all frontal streams together**, plus held-out corruption types (patch occlusion, brightness saturation). The vision-only comparator is trained with the same ModDrop. |
| #5 Calibration measures the session shortcut | P4 uses the **same-session first 4 min** as the primary reference. The Alert-session reference is secondary. A session probe is run before and after normalisation. |
| #6 Selection on 2 subjects is noisy | Selection uses **inner 4-fold subject-wise CV** (pooled inner macro-F1) with a fixed epoch budget. Decisions are logged in `story_so_far.md`. |
| #7 Near-empty Alert folds | **Pooled OOF macro-F1 with a subject-bootstrap 95% CI is co-primary** with the fold mean. |
| #8 H1 confounds model with post-processing | The comparator is **R4s**: honest M5 + window averaging + the same causal EMA. |
| #9 Fragile overnight runs | R4 retrain runs at **0.5 fps, 10 epochs** under `caffeinate -dims` (plugged in), selected on an inner-validation split. R4a (`last.pt`) is reported as clean only for folds 1, 2 and 4. |
| #10 No clock control | **R1** (time-only) and **C9-clock** (gate fed time features instead of telemetry) are mandatory. Telemetry must beat both. |
| #11 No strong simple baseline | **R6**: LightGBM/logistic regression on hand-crafted window statistics. **R7**: image-statistics shortcut probe. |
| #12–13 Weak diagnostics | Partial correlation controlling for epoch index. The session probe is trained and tested on **different** subjects. |
| #14 Confounded camera comparison | IR and L3D face crops are **both extracted at 168 px**. IR is also extracted at 112 px, to measure the resolution effect separately. |
| #17 Scope too big | Cut to 12 core runs. DINOv3, MARLIN, VideoMAE, LoRA fine-tuning, Mamba, depth proxy and end-to-end VIGIL become stretch goals. |
| #18 Demo can't run M8 on uploads | Two demo paths. **Dataset sessions** use full M8 with precomputed features. **Upload** uses a "live" model: frontal face crop → DINOv2 → MIL → EMA, telemetry optional. |

## Progress (status 2026-11-06 17:00)

The plan is running about 5 days ahead. Days 1–7 finished by Day 2 (2026-11-06), so the Day 8+
schedule now has a buffer, used for round 2
([`round2_research.md`](round2_research.md) §5). Per-variant results and diagrams:
[`../docs/m8_models.md`](../docs/m8_models.md).

| Item | Status | Headline result |
|---|---|---|
| R0, R1, R4a, R4, R4s, R6, R7 | done | Clock 0.355; honest M5 (R4s) 0.333; R6 0.28–0.36; R7 0.306 |
| C1 IR DINOv2 | done | **V\* = IR-112 mean-pool, 0.440 / pooled 0.446** |
| C2 L3D (+R3D TTA) | done | 0.33; fails the session audit (0.73) |
| C3 Pose | done | MIL 0.433 / pooled 0.499; fails the session audit (0.67) |
| C4 Landmarks/AU/geometry | done | 0.31–0.32 |
| C5 Temporal / context / smoothing | done | 240 s tf 0.470, but pooled gain only +0.012; smoothing no gain |
| C6 Multi-view fusion | done | IR + pose equal 0.469 / 0.517 (exploratory only); learned weights worse |
| C7 Telemetry expert | done | 0.362 (≈ clock) |
| C8 Calibrated late fusion | done | IR + tel 0.421 (worse than V\*) |
| C9 VIGIL gate + ModDrop + clock control + MoE | done | See M8 |
| **CP2: M8 frozen** | done (2026-11-06 11:00) | **C9d MoE (tel + clock): clean 0.416, 60 s blackout 0.326 vs 0.190, Δ +0.163 (p = 0.076)** |
| C10 Ordinal | done | Negative, 0.399 |
| C11 B2 binary | done | V\* 0.571 vs clock 0.573 |
| C12 Start-of-drive normalisation | done | Negative (0.351). The Alert-session variant (R2-7) gives 0.500 |
| Round 2 (buffer) | core slate done | Alert calibration 0.500 (P4); events tie V\*; eye crops / DINOv3 / concat negative; LOSO V\* 0.471, calibrated 0.518. Optional rPPG / adversarial / stereo not started |
| Day 8 final evaluation (H1–H4 + Holm, figures, `inference.py`) | partly done (2026-11-07) | H1–H4 none significant after Holm; fusion calibrated + events 0.536 / LOSO 0.525; B2 calibrated 0.775 (mostly drive detection; within-drive AUC 0.63); subject-dependent splits 0.91 / 0.80 / 0.41; final models in `models/final/`. Figures (`results/figures/`) and `m8/inference.py` done 2026-11-08 |
| Day 9–10 Gradio demo, codebase cleanup | demo done (`app/demo.py`); README cleanup open | Replay, upload and results tabs |
| Day 11–13 report, PPT | drafted 2026-11-08 | `report/thesis_report.docx` (from `report/report.md`); 17-slide deck (claude.ai Slides artifact, sources in `report/slides/`); narrative in `docs/thesis_story.md` |

## 1. Reference runs

| ID | What | Day |
|---|---|---|
| R0 | Majority-class and stratified-random baselines | 1 |
| R1 | Clock-only: logistic regression on epoch index and window position (R1b adds the session label, as an upper bound on the shortcut) | 1 |
| R4a | Existing M5 `last.pt`, frame- and window-level (clean for folds 1, 2, 4 only) | 1 |
| R4 | **Honest M5 retrain**: YOLOv8n-cls, 0.5 fps, 10 epochs, `best.pt` selected on inner-val subjects | 1 (overnight) |
| R4s | R4 + window averaging + causal EMA. **The fair comparator for H1.** | 3 |
| R6 | Tabular baseline: hand-crafted EAR/PERCLOS/blink/head-pose + telemetry stats → LightGBM or logistic regression | 3 |
| R7 | Image-statistics shortcut probe: brightness and colour histograms → logistic regression | 3 |

## 2. The 12 core experiments

All runs use frozen, cached features and the same canonical window table, protocol P0 (3-class, subject-independent 5-fold), 3 seeds, and wandb logging. They report fold-mean and pooled macro-F1, a subject-bootstrap CI, and per-class F1.

| # | ID | Experiment | Inputs | Question | Maps to |
|---|---|---|---|---|---|
| 1 | **C1** | DINOv2-S, IR face crop (168 and 112 px) | E2 | Does a frozen foundation ViT + attention-MIL beat honest M5? | A2 |
| 2 | **C2** | DINOv2-S, L3D face crop at 168 px (+ R3D test-time averaging) | E3, E4 | Does the higher-resolution stereo camera beat IR at matched crop and resolution? | A3 |
| 3 | **C3** | DINOv2-S, side-view Pose at 1 fps | E5 | Does posture carry independent signal? | A9 |
| 4 | **C4** | Landmark/AU/geometry stream (2 Hz, per-joint MLP) | E7 | A geometric stream that covers all 19 subjects | A8-lite |
| 5 | **C5** | Temporal head: MIL vs. 2-layer transformer, + 4-min causal context, + causal EMA/HMM | best of C1–C4 | Does time order or longer context fix the Alert/LowVig flicker? | A10 + A12 + A24 |
| 6 | **C6** | Multi-view vision fusion: late fusion vs. 8-latent bottleneck, with leave-one-stream-out analysis | C1–C4 | Does combining cameras help? This is the **vision core V\***. | A13–A16 |
| 7 | **C7** | Enriched telemetry expert (heading rate, grip, variability stats), + time-proxy R² | E8 | Is telemetry better than M2 and the clock (R1)? | A17 |
| 8 | **C8** | Calibrated late fusion (fixed weight / entropy-scaled / stacking) | V\* + C7, OOF | Does the simplest asymmetric fusion already do the job? | A18 |
| 9 | **C9** | **Gated residual telemetry correction + ModDrop (VIGIL gate)**, + C9-clock control, + availability MoE | V\* + C7 + quality u | Does telemetry help mainly when vision fails? | A19 |
| 10 | **C10** | Ordinal objective (CORN or KSS regression → thresholds) | V\* | Does respecting KSS order help? | A21 |
| 11 | **C11** | Binary protocol B2 (KSS ≤ 5 vs ≥ 6), vs. R4 on B2 | final model | The task where higher F1 is realistic | A22 |
| 12 | **C12** | Start-of-drive normalisation (same-session primary, Alert-session secondary) | final model | Does per-driver calibration close the gap? | A23 |

**Final model M8 (VIGIL-Former, two-stage)** = C6 vision core + C9 gate + the C5 temporal/smoothing choice, with C10's objective kept if it wins. It is frozen on Day 7 by the rule: inner-CV selection, and the simpler model wins ties.

**Pre-registered headline tests** (permutation over 16 subjects, Holm-corrected):
- **H1.** M8 vision core vs R4s, under P0.
- **H2.** M8 vs the vision core, under P0 with clean inputs. This is a do-no-harm test, expected to tie.
- **H3.** M8 vs the vision core (same ModDrop) under contiguous blackouts.
- **H4.** Calibrated vs uncalibrated, under P4.

## 3. Day-by-day schedule

| Day | Date | Work | Checkpoint / output |
|---|---|---|---|
| **1** | Thu 11-05 | Power check. Canonical window table (`cache/windows.parquet`). Alignment checks: FL source video, FL row ↔ frame index, PL column order, telemetry t0. R0, R1, R4a. Inner-validation subject assignment. **Overnight:** R4 honest M5 retrain. | Window table; alignment notes; honest-M5 preliminary numbers |
| **2** | Fri 11-06 | Feature extraction: E2 IR face 168 + 112, E3/E4 L3D/R3D face 168, E5 Pose 1 fps, E7 landmarks 2 Hz, E8 telemetry 4 Hz, E9 quality. Training harness: dataset, MIL head, metrics, bootstrap, wandb. | `cache/` populated; harness tested on fold 0 |
| **3** | Sat 11-07 | C1, C2, C3, C4. R4 results → R4s. R6, R7. | Single-stream table |
| **4** | Sun 11-08 | C5 temporal head, context and smoothing. **CP1:** best stream ≥ R4s + 0.02? | Decision logged |
| **5** | Mon 11-09 | C6 multi-view fusion → **V\***. Inner-fold OOF logits for V\*. | V\* chosen |
| **6** | Tue 11-10 | C7, C8, C9 (+ C9-clock). Robustness suite: contiguous blackouts, held-out corruptions, B/I/M coverage. | Gate figures |
| **7** | Wed 11-11 | C10, C11, C12. **CP2: freeze M8.** No more tuning after this. | M8 locked |
| **8** | Thu 11-12 | Final evaluation: tables, confusion matrices, H1–H4 with Holm, clock and session diagnostics, figures. Export weights and `inference.py`. | Results frozen |
| **9** | Fri 11-13 | Gradio demo: dataset-session path (M8) and upload path (live model). | Working demo |
| **10** | Sat 11-14 | Demo polish; record a backup demo video. Codebase cleanup: README, reproduction commands. | Codebase ready |
| **11** | Sun 11-15 | Report: intro, literature review (from `literature_survey.md`), dataset, methods. | Chapters 1–4 drafted |
| **12** | Mon 11-16 | Report: experiments, results, discussion (M5 audit, confounds, limitations), conclusion, abstract. | Full draft |
| **13** | Tue 11-17 | PPT, about 15 slides. | Deck |
| **14** | Wed 11-18 | Buffer: proofread, fix anything broken, final story entry. Commit/push only if the user asks. | Submission-ready |

**Fallbacks.**
- **CP1 fails** (no frozen stream beats R4s): use honest-M5 embeddings (per-fold backbones) with the C5 head, and continue.
- **C6 fusion doesn't beat the best single stream:** V\* = the best single stream, and the fusion result is reported as a negative finding.
- **Nothing beats R4s at CP2:** the thesis rests on four contributions:
  - the M5 audit;
  - the protocol results;
  - the robustness value of the gate (H3);
  - the first use of the unused streams.

## 4. Stretch goals (only with slack, in this order)

1. MARLIN-S face-video clips (A7).
2. LoRA fine-tuning of DINOv2-S (A6).
3. LOSO confirmation on V\* and R4.
4. DINOv3 swap (A5).
5. End-to-end VIGIL with gated cross-attention (A20).
6. Stereo disparity via `cv2.StereoSGBM`, if L3D/R3D are rectified.

## 5. Code layout (new code for this phase)

```
m8/                      # new package, project root
  windows.py             # canonical window table + labels + folds + inner-val
  extract.py             # video → face crop → DINOv2 → cache/emb/...
  tabular.py             # FL/PL/FAU/telemetry/grip → cache/csv, cache/tel
  data.py                # memory-mapped window dataset
  heads.py               # MIL, temporal transformer, bottleneck fusion, gate
  train.py               # CV loop (outer 5-fold × inner CV), wandb
  evaluate.py            # metrics, bootstrap, permutation tests, smoothing
scripts/m8/              # one entry-point script per experiment (C1–C12, R*)
cache/                   # gitignored: windows.parquet, emb/, csv/, tel/, quality/
results/m8/              # per-run JSON (per-window predictions, metrics)
```
