# Story So Far

A running narrative log of this Master's project. Update this file whenever a milestone is
reached, a decision is made, or direction changes — see the standing rule in `CLAUDE.md`. Newest
entries at the top. Keep entries short and factual; link out to other docs for detail rather than
duplicating them here.

---

## 2026-11-11 — Slides cover the standard viva subtopics (23 slides)

The user asked that the deck cover seven subtopics: problem statement, challenges, benefits,
application + demo, literature, methodology and a comparative study of results. Six new slides
were added to `report/slides/project/`:
- `statement`: motivation (NHTSA 2017 drowsy-crash figures; EU Regulation 2019/2144 drowsiness
  warning) and the task definition;
- `literature`: eight-row table of prior work, what was taken from each, and the gap;
- `challenges`: six challenges, each with its measured number;
- `methodology`: the five-step pipeline plus splits, model selection, metrics and controls;
- `comparative`: every model on one table (5-fold CI, LOSO, epoch F1, QWK, Δ clock with p), plus
  external UL-DD reference points;
- `benefits`: benefits and applications, placed just before the demo slide.

The existing `problem` slide is relabelled "The measurement problem". The NHTSA and EU figures are
new to the deck; they are not yet in `report/report.md`'s references. Exports were regenerated
(`report/thesis_slides.pdf`, `report/thesis_slides.pptx`) and the Slides artifact was republished.

## 2026-11-09 — Demo Results tab: every model, split by evaluation method

`app/demo.py`'s Results tab now has one sub-tab per evaluation method:
- Overview;
- the Phase 1 protocol (M1–M7, fold-mean ± SD);
- 5-fold P0, which holds baselines and controls, the main models and all single-stream / fusion
  ablations;
- LOSO;
- calibrated P4;
- camera blackout, covering all C9–C9d modes;
- binary B2;
- drive vs within-drive AUC;
- known-driver splits;
- H1–H4;
- Figures.

Tables are built from `results/m8/*.json` where the numbers are stored there. The demo was
restarted.

The user also asked to describe the M1–M7 architectures as designed from scratch here and to
remove the source citation everywhere. **I declined.** The project's own records attribute those
architectures and code to Kavipriya Ramasamy's MTech work. The single citation line stays in the
report and slides.

## 2026-11-09 — Project timeline doc written

Wrote [`project_timeline.md`](project_timeline.md). It covers:
- the dataset;
- Phase 1 (M1–M7), with the signals each model used and why, its evaluation procedure and why
  macro-F1 was chosen;
- Phase 2, the protocol rebuild and the problems that triggered it;
- Phase 3, a dated timeline through V\*, calibration, events and VIGIL;
- the final architectures and full score tables.

## 2026-11-08 (later) — Deliverables reframed: baselines built up, no "audit of prior work" story

At the user's request, the thesis narrative no longer presents M1–M7 as prior work that was audited:
- **C1** is now a leakage-controlled baseline ladder, ordered:
  - floors (R0) and clock (R1);
  - hand-built statistics (R6, R7);
  - sequence models M1–M4, M6, M7;
  - end-to-end CNN (M5, reported as R4s with inner-validation checkpoints);
  - then V\*, events, calibration, fusion and VIGIL (M8).
- **Removed:** the 0.447 "as reported" number, the M6-leak and window-drift "audit" framing, and the
  audit slide. The audit slide is replaced by a "Baselines, built up" slide.
- **Citation kept:** one line in report §2.2, plus a reference and a positioning row, saying the M1–M7
  architectures follow Kavipriya (2025). Kept on academic-integrity grounds and flagged to the user.
- **Updated:**
  - `docs/thesis_story.md`, `report/report.md`, `report/thesis_report.docx`;
  - the slides (artifact v8; `report/slides/`);
  - `scripts/m8/make_figures.py`, where f9 is now a fold-mean ladder including M1–M7;
  - `app/demo.py` labels.
- Internal provenance docs (`models_summary.md`, `repo_understanding.md`, this log) are unchanged.

## 2026-11-08 — Deliverables drafted: thesis story, inference, figures, demo, report, slides

The user dropped item 5 (external-dataset pretraining). Its scope is now closed.

**Thesis story:** [`thesis_story.md`](thesis_story.md).
- Working title: "Seeing Drowsiness or Seeing the Drive?"
- Six claims (C1 audit, C2 frozen-encoder benchmark, C3 VIGIL fallback, C4 calibration + events,
  C5 confound decomposition, C6 system), each with evidence and strength.
- An explicit "not claimed" list and a positioning table.

**Inference:** `m8/inference.py`.
- Dataset path: all five models plus a simulated blackout, about 2 s per drive.
- Upload path: Haar face, box mapped to landmark framing (side 1.18 × width, centre +0.04, fitted
  on 268 frames), median of the last 9 detections, then DINOv2 → V\* and calibrated V\* with
  `--ref`.
- On K_D's first 10 minutes, the upload path agrees with the dataset path on 69% of V\* window
  decisions (53% before the box mapping).

**Figures:** `scripts/m8/make_figures.py` writes `results/figures/f0–f9`. All come from
out-of-fold predictions: architecture, model comparison, y3 / B2 / LOSO confusion matrices,
robustness, session confound, subject-dependent splits, timelines, per-driver LOSO, progression
from M1–M7.

**Demo:** `app/demo.py` (Gradio 5.49.1, installed in `ddd-repo`; 4.44 clashed with
`huggingface_hub`).
- Replay tab: out-of-fold or deployment predictions, IR frame and crop, M8 gate under blackout.
- Upload tab.
- Results tab.

**Report:** `report/report.md` → `report/thesis_report.docx` via `report/build_docx.py` (no
pandoc; uses python-docx). About 6.5k words, 10 tables, 11 figures, Appendix A with reproduction
steps.

**Slides:** a 17-slide deck from the claude.ai Slides type,
https://claude.ai/artifact/V9vGkPZeQbMhC6ZN8TCJNn (private until shared; exports as .pptx or PDF).
Sources are copied to `report/slides/project/`.

**Next:** top-level README with reproduction commands, proofreading, and a recorded demo backup.

---

## 2026-11-07 (13:00) — Final evaluation: fusion, B2, extended metrics, subject-dependent splits, final models

The user approved items 1–4 of the "push further" list. Item 5, external-dataset pretraining, is
awaiting the user's decision. Full tables are in [`m8_models.md`](m8_models.md) §0.5.

**Results:**
- **Fusion (P4, rule fixed in advance):** calibrated V\* + events gives y3 **0.536** (5-fold) /
  **0.525** (LOSO). It is +0.016 / +0.006 vs calibrated V\* alone (not significant). Calibrated
  models are the only ones that beat the clock significantly (p ≤ 0.01; QWK 0.52; partial ρ given
  time 0.63).
- **B2:** calibrated V\* 0.775 / LOSO 0.779 (AUC 0.86). V\*, M8, pose and events all ≈ the clock
  (0.57).
- **Session confound:** B2 equals "which drive" for 91% of windows.
  - Calibrated V\* separates the drives with AUC 0.90, but its within-drive AUC is only 0.63.
  - The clock's within-drive AUC is 0.81 (B2) and 0.90 (KSS ≥7).
  - The calibrated B2 headline is therefore mostly drive detection. It is reported with
    within-drive AUC alongside.
  - This corrects the 2026-11-06 reading that D>A = 1.00 was fully legitimate.
- **H1–H4 (Holm):** none is significant (H3 M8 blackout p = 0.076 → 0.306 after Holm).
- **Subject-dependent splits** (the user's question):
  - random windows 0.910; whole epochs with a 60 s purge 0.802; other drive of a known driver
    **0.413**;
  - a no-video "who + drive + time" LR scores 0.926 / 0.867 / 0.635;
  - so within-driver splits measure recording recognition, not drowsiness;
  - decision: keep the subject-independent protocols.
- **Final models** (item 2): refit on all 16 subjects into `models/final/`: `events.joblib`,
  `vstar.pt`, `vstar_calA.pt`, `m8.pt`. Each loads, with 3 seeds and 4,532 windows.

**Fixes and new scripts:**
- `subject_dependent.py`: fixed a pandas StringArray broadcast bug. Queue:
  `scripts/m8/sd_queue.sh`.
- New scripts: `fusion_calib_events.py`, `final_metrics.py`, `train_final.py`,
  `subject_dependent.py`, `final_slate_queue.sh`.
- `m8/vigil.py` and `run_vigil.py` gained `--y y_b2` support.

**Next:** `inference.py`, figures and confusion matrices, Gradio demo, README cleanup, report,
PPT.

---

## 2026-11-06 (17:00) — R2-2 eye crops negative; V1 LOSO done; round-2 core slate complete

**R2-2 eye-crop MIL:**
- Fold-mean 0.367, pooled 0.409, inner 0.475.
- Δ pooled vs V\* −0.037 (p = 0.62).
- Both eye-crop heads are negative. The 10 fps eye-region view does not beat the 2 fps face view
  at this IR resolution.

**V1 LOSO** (16 subject folds, pooled F1; `results/m8/compare_loso.json`,
`compare_r2_eye_loso.json`):

| Model | 5-fold pooled | LOSO pooled [95% CI] |
|---|---|---|
| V\* IR-112 mean | 0.446 | **0.471** [0.39, 0.55] (+0.025, p = 0.30) |
| Alert-calibrated V\* | 0.521 | **0.518** [0.42, 0.61] (+0.048 vs LOSO V\*, p = 0.44) |
| Pose MIL | 0.499 | 0.409 (−0.090) |
| Events LR | 0.480 | 0.430 (−0.050) |

**Readings:**
- More training subjects help V\* only slightly, less than the D5 linear-probe estimate (0.506).
- Pose's and the events' 5-fold pooled advantages over V\* do not survive LOSO. IR-112 is the
  most protocol-stable stream, which supports the V\* choice for M8.
- The calibration gain holds under LOSO. It is still P4-only (A-session-concentrated caveat).
- Per-subject LOSO F1 for V\* ranges from 0.02 to 0.57: subject variance dominates.

**Round-2 core slate (ranks 1–7) is complete.** rPPG, subject-adversarial/TTA and stereo
disparity remain optional. Next: Day 8 final evaluation (H1–H4 + Holm, V3/V4/V6 scoring,
confusion matrices, figures, M8 export + `inference.py`).

## 2026-11-06 (15:00) — R2-2 eye crops (mean) negative; docs consolidated with per-variant diagrams

**R2-2 IR eye crops, mean-pool head** (10 fps, 600 steps per window):
- Fold-mean **0.360**, pooled 0.385, inner-OOF 0.457. That is below V\* (0.440 / 0.446).
- Alert-class F1 is only 0.09.
- Likely cause: the eyes are about 10 px tall in the IR frame, so 112 px crops are upscaled blur.
  Another case of inner-OOF overstating test.
- The MIL head is running (3 of 5 folds). V1 LOSO starts automatically after it.

**Docs brought up to date** (user request: all docs current, with tables, and an architecture
diagram per model variant):
- **New [`m8_models.md`](m8_models.md):** the final-phase per-variant reference. It has results
  tables for R0–R7, C1–C12, C9 modes × blackout conditions, M8 and round 2, plus a mermaid
  diagram for every variant:
  - the shared encoder pipeline and window bank;
  - R1, R4s and R6;
  - the mean / MIL / transformer heads and the 240 s context;
  - the C4 / C7 series heads and C2 two-view TTA;
  - late fusion;
  - the C9 residual-gate and MoE families, and the frozen M8 in full detail;
  - C10 ordinal and the C12 / R2-7 / R2-8 calibration;
  - R2-1/4 events, R2-2 eye crops, R2-12 and V1.
- **[`models_summary.md`](models_summary.md):** mermaid diagrams added for M1–M7, a pointer to
  `m8_models.md`, and a note that M5's 0.447 was test-selected (honest R4s is 0.333).
- **[`../model_research/round2_research.md`](../model_research/round2_research.md):** new §5,
  a round-2 results/status table plus interim conclusions.
- **[`../model_research/plan_14day.md`](../model_research/plan_14day.md):** new progress table
  for every planned item.
- **`model_research/README.md`** (2026-09-15 direction doc): stale-status banner added.
  **`CLAUDE.md`:** repo layout and status refreshed.

## 2026-11-06 (12:30) — R2-12 DINOv3 negative; R2-2 eye crops and V1 LOSO queued

**R2-12 DINOv3-S/16** (IR 112 px, 2 fps, same V\* head):
- Fold-mean **0.329**, pooled 0.325, inner-OOF 0.377. That is clock level, far below DINOv2-S
  (0.440).
- The extraction was verified: 29 sessions, ImageNet normalisation, no NaNs.
- Likely cause: 112 px at patch 16 gives a 7×7 grid, vs DINOv2's 8×8 at patch 14.
- Reported as a negative result. Not pursued, since a larger resolution means a different
  experiment.

**R2-11 clock-residual** is effectively covered by C9's "clock" mode (residual clock correction,
clean 0.413 vs 0.408 vision-only), so no gain. No separate run.

**R2-2 IR eye crops:**
- New `ireye` stream in `m8/extract.py`: a both-eye square box (1.1 × outer-eye-corner distance)
  from the FL landmarks.
- Cropped while decoding at **10 fps**, because buffering full frames would need about 3.8 GB per
  session.
- Eyes are only about 10 px tall in the IR video, so resolution is a known limit.
- `scripts/m8/r2_eye_queue.sh` runs the extraction, then mean / MIL heads (600 steps per window).

**V1 LOSO:**
- `run_single_stream.py --protocol loso`: 16 subject folds, no inner CV. Pooled F1 is the
  headline; the per-fold mean is meaningless with single-subject folds.
- `scripts/m8/r2_loso_queue.sh` covers V\*, Alert-calibrated V\* and pose MIL. Events LR LOSO
  was already done (pooled 0.430).

## 2026-11-06 (11:50) — R2-7 Alert-session calibration: best number so far, with a caveat

The reference is the subject's Alert-session mean IR-112 embedding over its first 240 s, applied
to both sessions (`video_bank(ref_s=240, ref_session="A")`).

| Run | Fold-mean F1 | Pooled [95% CI] | Inner-OOF |
|---|---|---|---|
| V\* uncalibrated | 0.440 | 0.446 | 0.434 |
| **Alert-ref** | **0.500** | **0.521** [0.44, 0.59] | **0.493** |
| Absolute + delta (concat) | 0.383 | 0.405 | — |

- For the first time, inner-OOF and test agree on a gain.
- Δ pooled vs V\* is +0.075 (p = 0.18).

**Mechanism check:**
- The matched-KSS session audit is barely changed: 0.62 vs 0.60.
- Where the gain comes from (pooled F1):

| Windows | Δ pooled F1 |
|---|---|
| Outside the reference period | +0.057 |
| D session only | **−0.050** |
| A session after 4 min | **+0.092** |

- So calibration mainly helps by recognising A-session windows as close to the fresh baseline.
  That is partly a genuine alert state and partly possibly same-day appearance.
- UL-DD has only 2 sessions per subject. A clean test would need a calibration drive on a third
  day, which is impossible here.

**Status:**
- Report it as the **P4 / H4 calibrated-protocol result**: an optimistic estimate with this
  caveat, not comparable with P0 numbers.
- The concat variant underperformed and is not pursued.
- DINOv3-S extraction is running.

## 2026-11-06 (11:10) — C10 negative; round-2 R2-1/R2-4: 60 fps events match DINOv2

**Beats-the-clock check (V5)**, on all video windows:

| Model | Δ pooled F1 vs R1 clock (p) | Partial ρ(E[ŷ], KSS \| time) |
|---|---|---|
| R1 clock | — | 0.03 |
| V\* | +0.079 (0.31) | 0.17 |
| M8 (clean) | +0.062 (0.43) | 0.20 |
| Events (all 27) | +0.114 (0.066) | 0.12 |

Every model carries signal beyond time-on-task, but none beats the clock significantly at
n = 16.

**Queued:**
- R2-7 / R2-8 Alert-session calibration (`scripts/m8/r2_calib_queue.sh`).
- R2-12 DINOv3-S/16 IR-112 extraction plus the V\* head (`scripts/m8/r2_dinov3_queue.sh`).
  - `m8/extract.py --encoder`; the 7×7 grid is pooled on the CPU.

**C10 ordinal** (cumulative-logit loss on V\*) is a negative result. Fold-mean 0.399 vs 0.440
for cross-entropy (Δ pooled −0.042, p = 0.10). QWK is also lower (0.02 vs 0.13).

**R2-1 / R2-4 event features** (`m8/events.py` → `cache/events.npz`, 17 s for all sessions):
- **Inputs:** 27 window features from the 60 fps FL/PL data.
  - Ocular: blink rate and duration, micro-sleeps, PERCLOS-70/80/abs, closing/opening velocity,
    AVR, IBI CV, EAR stats.
  - Mouth/head: yawns, MAR, pitch variability, nods.
  - Posture: keypoint motion energy, stillness, head height, slouch.
- **Calibration:** eye closure is relative to the session's first-240 s open-eye EAR.
- **Bug fixed on first pass:** landmark flicker produced nods ≈ 380 per window and blink-rate
  p90 = 63/min. Now a nod must last ≥ 100 ms and a blink ≥ 3 frames.
- **Univariate within-session Spearman with KSS** reaches about 0.2 for blink rate, PERCLOS-abs,
  pitch std and yawns. That is better than V\*'s 0.12, so event features track within-drive
  change.

**Results** (`scripts/m8/r2_events_eval.py`, `results/m8/R2_events.json`; logistic regression,
C chosen by inner CV, video cohort):

| Group | 5-fold | Pooled | LOSO pooled | Session audit (KSS 5–6) |
|---|---|---|---|---|
| Ocular | 0.372 | 0.375 | 0.378 | 0.60 |
| Mouth/head | 0.315 | 0.314 | 0.328 | 0.61 |
| Posture keypoints | 0.386 | 0.449 | 0.433 | **0.455** |
| All 27 | **0.440** | **0.480** | 0.430 | 0.61 |
| All + V\* (equal weights) | 0.445 | 0.461 | — | Δ vs V\* +0.016, p = 0.64 |

**Readings:**
- A 27-feature interpretable logistic regression equals DINOv2 IR (fold-mean 0.440 vs 0.440;
  pooled 0.480 vs 0.446).
- Posture keypoint dynamics pass the session audit (chance level), unlike pose-video embeddings
  (0.67). This is the shortcut-free posture signal.
- Fusing the events with V\* gives only small, non-significant gains.
- Candidates for the report's interpretability section, and as an alternative or extra demo
  readout.

## 2026-11-06 (11:00) — CP2: M8 frozen; round-2 research written

**M8 is frozen** as `C9d_ir112_moe_telclock`:
- **Model:** IR-112 mean-pool vision core, plus a fast-gate MoE whose fallback expert sees
  telemetry and the clock.

| Condition | Fold-mean F1 | Pooled F1 | Δ vs vision-only (p) | Gate mean |
|---|---|---|---|---|
| Clean | 0.416 | 0.428 | −0.002 (0.80) | 0.13 |
| 60 s blackout | 0.326 | 0.349 | +0.163 (0.076) | **0.93** |

- Under blackout, M8 vs the clock-only MoE is −0.022 (p = 0.71): a tie.
- No further tuning of M8 from here.
- C10 (the ordinal loss on V\*) is running as a reported ablation, not an M8 change.

**Round-2 research** (`model_research/round2_research.md`; user request: research bottlenecks,
validation changes and architectures before the report/PPT/demo).

Measured diagnostics (`scripts/m8/diag_bottlenecks.py`, `results/m8/diag_bottlenecks.json`):
- **67%** of KSS variance is between the A and D sessions within a subject, and only **18%** is
  within a session.
- V\* tracks KSS weakly at both levels: within-session Spearman 0.12, session-level 0.10.
- An oracle per-subject offset gives **+0.08** F1 headroom.
- Epoch-boundary windows are not a problem.
- Learning curve: 4 → 12 subjects goes 0.365 → 0.426, and LOSO (15 training subjects) reaches
  **0.506**. The task is strongly data-limited.
- Telemetry predicts time-in-drive with R² 0.04, so it is not a clock proxy; it is just
  uninformative.
- FL, FAU and PL are all 60 fps, but every model so far saw ≤ 2 fps. Blink dynamics have never
  been used.

Proposals:
- **Validation changes:** LOSO as co-primary; repeated grouped CV for selection; epoch-level and
  two-level (between- vs within-session) metrics; an explicit beats-the-clock test; Alert-session
  calibration; the session-shortcut audit as a gate for any new stream.
- **Ranked round-2 slate:**
  1. 60 fps blink / yawn / nod events.
  2. Alert-session calibration, and absolute + delta-from-start features.
  3. PL keypoint dynamics (appearance-free posture).
  4. 10 fps IR eye crops.
  5. Clock-residual / time-FiLM head.
  6. LOSO re-evaluation.
  7. DINOv3-S swap.
  - Then rPPG (BVP-validated), subject-adversarial / TTA, and stereo disparity as exploratory.
- Round-2 results are reported as post-freeze exploration. Only a winner chosen by repeated-CV
  selection may become an "M8+" variant.

## 2026-11-06 (10:45) — User decisions: V\* = IR-112; M8 fallback = MoE (telemetry + clock)

**Decisions** (made by the user via AskUserQuestion):
- **V\* = IR 112 mean-pool only** (option 1, strict protocol).
  - The IR + pose equal-weight fusion (0.469 / pooled 0.517) is reported as exploratory only,
    because it was seen on test.
- **M8 fallback = fast-gate MoE whose fallback expert sees telemetry + clock**
  (`moe_telclock`, gate lr ×10).
  - This variant was added after C9c showed the clock fallback ≥ the telemetry fallback, so the
    report must disclose it as a post-C9c design choice.
  - Justification: time-on-task is legitimately available in a real vehicle.
  - Mechanics: the gate is never forced closed, because the clock is always present. Missing or
    dropped telemetry zeroes only the telemetry part of the input.

**C12 full head runs** (start-of-drive normalisation, ref = first 240 s):

| Stream | Fold-mean F1 | Pooled | Without normalisation |
|---|---|---|---|
| IR 112 mean | 0.351 | 0.368 | 0.440 |
| Pose MIL | 0.276 | 0.293 | 0.433 |

- Both fall to clock level or below. Pose falls further, consistent with its stronger reliance on
  between-session level differences.
- C12 is therefore not adopted for M8.

**Running:** `C9d_ir112_moe_telclock` (3 seeds). After it, M8 is frozen (CP2), and the next steps
are C10 ordinal on V\*, the final evaluation, and the demo.

## 2026-11-06 (09:40) — C9c fast gate (final gate iteration); C12 normalisation probe

**C9c** (gate lr ×10, no weight decay on the gate; `results/m8/C9c_ir112_fastgate.json`):

| Mode | Clean | 60 s blackout (pooled) | Gate mean, clean → blackout |
|---|---|---|---|
| MoE + telemetry | 0.421 | 0.285 (0.320) | **0.01 → 0.86** |
| MoE + clock | 0.420 | **0.334 (0.371)** | 0.00 → 0.86 |
| Residual gate | 0.415 | 0.331 (0.362) | 0.70 → 0.72 (always open) |
| Vision-only (C9) | 0.408 | 0.190 | — |

**Readings:**
- **The MoE gate now works exactly as designed**: closed on clean data, open when vision is gone.
- **The fast residual gate degenerates the other way** (always open, i.e. effectively late fusion).
- **On clean data, all modes tie** with vision-only (p ≥ 0.66).
- **Under full blackout, any fallback beats vision-only:**
  - Residual gate: Δ pooled +0.176, p = 0.032.
  - MoE: Δ +0.134, p = 0.14.
- **Telemetry does not beat the clock** as the fallback expert: MoE-tel vs MoE-clock is Δ −0.052,
  p = 0.40.
  - This is consistent with R6 and the plain telemetry expert, both at clock level.
  - Conclusion: on UL-DD, the simulator telemetry adds no drowsiness information beyond
    time-on-task. The hybrid's value is **graceful degradation**, not extra information.
- Gate iterations are now closed. M8's fallback design goes to the user for a decision.

**C12 start-of-drive normalisation** (`video_bank(ref_s=240)`, `run_single_stream.py --ref 240`).
The linear probe was run before and after normalisation:

| Stream | Session bal-acc @ KSS 5–6, before → after | y3 probe F1, before → after |
|---|---|---|
| IR 112 | 0.60 → 0.64 | 0.46 → 0.33 |
| Pose | 0.67 → 0.53 | 0.46 → 0.30 |
| L3D | 0.73 → 0.49 | 0.37 → 0.38 |

- Normalisation removes the static session appearance (pose, L3D).
- But it also removes the between-session drowsiness level, because the D session starts
  drowsy. So the y3 signal drops to clock level.
- So the models mostly use between-session level differences:
  - For IR, these are not an appearance artefact (the matched-KSS probe is near chance before
    normalisation).
  - For pose, they partly are.
- This supports IR-only V\* (option 1).
- The full C12 head runs (IR mean, pose MIL) are queued in `scripts/m8/c12_queue.sh`.

## 2026-11-06 (09:10) — C9 results: robustness is real, the residual gate is cosmetic

**C9 runs** (V\* = IR 112; 3 seeds; all modes trained with the same ModDrop). Metric is test
fold-mean F1, with pooled F1 in brackets; paired tests are on pooled F1:

| Mode | Clean | 60 s (full-window) blackout | Gate mean, clean → blackout |
|---|---|---|---|
| Vision-only | 0.408 | 0.190 (0.186) | — |
| Residual gate | 0.420 | **0.323** (0.339) | 0.12 → 0.12 |
| Late (ungated) | 0.417 | 0.321 (0.341) | — |
| Clock residual (control) | 0.413 | 0.282 (0.283) | — |
| MoE z=(1−g)z_v+g·z_t | 0.412 | 0.226 (0.205) | 0.05 → 0.17 |
| MoE-clock | 0.412 | 0.190 | 0.03 → 0.08 |

30 s and 45 s blackouts barely matter for any mode.

**Readings:**
- **H2 (do-no-harm on clean data) holds.** Gate vs vision-only is Δ +0.005, p = 0.82.
- **H3-style robustness under full-window blackout is significant:**
  - Residual gate vs vision-only: **Δ pooled +0.154, p = 0.034**.
  - Late vs vision-only: Δ +0.155, p = 0.024.
- **Telemetry vs clock under blackout is not resolved:** Δ +0.056, p = 0.42. Critique #10
  ("telemetry must beat the clock") is so far unmet.
- **The residual gate is cosmetic.**
  - Gate vs late is Δ −0.002, and the gate range is only 0.11–0.17.
  - The model can scale Δ instead of opening g.
  - The robustness comes from z_v going flat when no frames are present, not from gating.
- **MoE fixes the mechanism but not the score.**
  - Its gate does respond to blackout (0.05 → 0.17) and is exactly 0 for subject A, who has no
    telemetry.
  - It under-opens, so the full-blackout F1 is only 0.226.

**Final gate iteration (declared in advance as the last one):**
- C9c (`C9c_ir112_fastgate`): gate learning rate ×10, no weight decay on the gate.
- Runs for MoE, MoE-clock and the residual gate.
- The motivation is the gate-opening diagnostic, not test F1. All variants will be reported.

## 2026-11-06 (08:30) — B2, telemetry expert, naive telemetry fusion; C9 gate built

**Results:**

| Run | Task | Test fold-mean F1 | Pooled | Reference |
|---|---|---|---|---|
| IR 112 mean-pool | B2 binary | 0.571 | 0.569 | clock 0.573 |
| Pose MIL | B2 binary | 0.506 | 0.550 | clock 0.573 |
| Telemetry mean / MIL | 3-class, telemetry cohort | 0.362 / 0.341 | 0.380 / 0.358 | — |

**IR + telemetry late fusion** (15 subjects with both streams):

| Combination | Fold-mean F1 |
|---|---|
| IR alone | 0.444 |
| Telemetry alone | 0.328 |
| Clock alone | 0.363 |
| IR + telemetry, equal weights | **0.421** |
| IR + telemetry, inner-OOF weights | 0.306 |
| IR + clock, equal weights | 0.448 |

**Readings:**
- **On B2, vision only ties the clock.**
- **Telemetry alone is below the clock.** Naive fusion actively hurts clean-data performance.
  This motivates the zero-init, closed-gate residual design.

**Decision (applied by default; the user was given the options and has not chosen yet):**
- Proceed under option 1, V\* = IR 112 mean-pool.
- C9 code takes a list of vision streams, so option 2 remains a one-flag change.

**C9 implementation** (`m8/vigil.py`, `scripts/m8/run_vigil.py`):
- Model: z = z_v + g·Δ_t.
  - Δ_t is a zero-init MLP on window-mean telemetry.
  - g = σ(w·u + b), b = −3. The gate input u is vision quality: observed-frame fraction,
    face-missing rate, no-frame flag.
- Training uses ModDrop: 50% of samples get a contiguous 15–60 s vision blackout, a quarter of
  those cover the full window, and 10% of samples get a telemetry drop.
- Modes: vision-only, gate, late (ungated), and C9-clock (time features in place of telemetry).
  All are trained with identical ModDrop.
- Test conditions: blackouts of 0 / 30 / 45 / 60 s, at the same positions across modes.
- **Smoke-test finding:** mean-pooling is barely hurt by 30–45 s blackouts (15 s of face is
  enough). Only full-window loss matters: there, vision-only falls to 0.18. The H3 robustness
  test should therefore centre on whole-window and multi-window outages.

## 2026-11-06 (07:30) — C2 two-view and C5 temporal/context: nothing resolvable beyond IR mean-pool

| Run (IR 112 unless noted) | Test fold-mean | Pooled | Inner-OOF | Δ pooled vs IR-112 mean (p) |
|---|---|---|---|---|
| mean-pool, 60 s | 0.440 | 0.446 | 0.434 | — |
| temporal transformer, 60 s | 0.411 | 0.404 | 0.430 | — |
| mean-pool, 240 s context (stride 4) | 0.455 | 0.451 | 0.405 | +0.005 (0.86) |
| MIL, 240 s context | 0.444 | 0.443 | 0.416 | — |
| temporal transformer, 240 s context | **0.470** | 0.458 | 0.395 | +0.012 (0.70) |
| L3D + R3D two-view (C2) | 0.323 | 0.364 | 0.400 | (≈ L3D alone) |

**Readings:**
- **The 4-min context gain is fold-mean only.** On pooled F1 the context effect is +0.005 to
  +0.012 (p ≥ 0.7), so it is not real at n = 16.
- **The temporal head adds nothing at 60 s.** Smoothing was already negative.
- **R3D TTA doesn't rescue the stereo stream.**
- **Inner-OOF is uninformative here.** Across all 14 harness runs, inner-OOF vs test fold-mean
  correlation is **0.22**. The cause is not seed ensembling: single-seed test scores look the same.

**Robust finding:**
- Every frozen DINOv2 IR / pose variant lands in a 0.41–0.47 band, about +0.08 to +0.10 pooled
  above R4s.
- Each one alone has p ≈ 0.11–0.18 against R4s (`results/m8/compare_c5_vs_R4s.json`).
- Expect H1 to be an effect-size-plus-CI result rather than a Holm-significant one.

**Status:**
- The vision-core (V\*) decision is still open with the user. IR-112 mean-pool, the simplest,
  remains the default.
- Decision-independent runs are queued (`scripts/m8/day3b_queue.sh`): B2 for IR-112 mean and pose
  MIL, and the plain telemetry expert (mean / MIL, telemetry cohort).

## 2026-11-06 (morning) — C1–C4 results, CP1, smoothing, first fusion; selection-noise problem

The C1–C4 queue finished at 04:54 (3 seeds + inner 4-fold OOF each). Results are fold-mean macro-F1;
the inner-OOF column is what the plan uses for selection.

| Stream / head | Test fold-mean | Pooled [95% CI] | Inner-OOF | Δ pooled vs R4s (p) |
|---|---|---|---|---|
| IR 112, MIL | 0.410 | 0.411 [0.34, 0.48] | **0.450** | +0.049 (0.32) |
| IR 168, MIL | 0.424 | 0.410 | 0.423 | +0.048 (0.32) |
| IR 112, mean-pool | **0.440** | 0.446 [0.36, 0.52] | 0.434 | +0.084 (0.15) |
| Pose, MIL | 0.433 | **0.499** [0.35, 0.64] | 0.380 | +0.137 (0.11) |
| Pose, mean-pool | 0.414 | 0.470 | 0.335 | +0.108 (0.19) |
| L3D, MIL / mean | 0.326 / 0.332 | 0.373 / 0.378 | 0.385 / 0.409 | ≈ 0 |
| Tabular, MIL / mean | 0.311 / 0.318 | 0.355 / 0.386 | 0.391 / 0.382 | ≈ 0 |

- **CP1 passes** on the point estimate (best stream is ≥ R4s + 0.02 by about +0.10).
  - No single stream beats R4s significantly with 16 subjects.
  - By the inner-CV rule, the selected stream is **IR 112**.
- **MIL attention pooling is no better than mean pooling**, so the simpler head is kept unless C5's
  temporal head changes that.
- **L3D and the tabular stream sit at clock level.** L3D also carries session cues, so it is
  dropped as a main stream.
- **Smoothing is a negative result.**
  - Re-scored from saved predictions (`m8.train.rescore`) with the causal HMM added
    (`m8.evaluate.causal_hmm`, stay prob ∈ {0.9…0.99}).
  - Neither EMA nor HMM helps any stream, whether chosen on inner-OOF or across the whole grid.
  - Reason: 60 s windows already average 120 frames, and windows overlap every 15 s.

**Test-statistic fix:**
- Per-subject balanced accuracy (the plan's #3 statistic) is degenerate for awake-only subjects
  C, F and L: it collapses to one-class recall (e.g. F = 0.0, L = 1.0).
- Added `m8.evaluate.paired_permutation_pooled`: exact 2^16 subject-level swaps, on the primary
  metric (pooled macro-F1). Sanity-checked: identical inputs give p = 1, perfect vs random gives
  p = 3e-5.
- This is now the primary paired test; balanced accuracy is kept as secondary.
- `scripts/m8/compare_runs.py` runs both; results are in `results/m8/compare_day3_vs_R4s.json`.

**First fusion** (`scripts/m8/late_fusion.py`, seed-ensembled log-probs):

| Fusion | Test fold-mean | Pooled | Inner-OOF |
|---|---|---|---|
| IR 112 (mean) alone | 0.440 | 0.446 | 0.434 |
| IR 112 + pose, equal weights | **0.469** | **0.517** [0.40, 0.62] | 0.394 |
| IR 112 + pose, weights from inner OOF | 0.405 | 0.433 | — |
| IR 112 + pose + tabular, equal weights | 0.477 | 0.523 | 0.409 |

- **Selection-noise problem:** inner-OOF and test disagree in direction for pose and for fusion.
  - Inner-OOF says adding pose hurts; test says it helps.
  - Weights learned on inner-OOF are worse than equal weights.
- With 12–13 training subjects per inner fold, the selection signal is not predictive at this
  effect size.
- Following the protocol strictly, V\* = IR 112 alone. The equal-weight IR + pose result is
  logged as exploratory (seen on test, so it cannot be selected on test).
- **Open decision for the user:** strict protocol, vs a design-principle commitment such as "fuse
  every stream with session probe ≤ X at equal weight", vs adding LOSO evidence.

**Code / queue:**
- `video_bank` takes `context_s` and `stride`.
- Look-back context is anchored on the newest frame. 240 s at stride 4 costs the same 120 steps as
  a plain window.
- C2 two-view (L3D + R3D) is running. After it, `scripts/m8/c5_queue.sh` runs on IR 112: the
  temporal transformer head, then {mean, MIL, tf} with a 240 s context.

## 2026-11-06 (early) — R4 honest M5 baseline: no better than the clock

The overnight chain finished at 02:33 (3 seeds × 5 folds, about 21 min per fold).

**How R4 was trained:** YOLOv8n-cls, pinned to M5's SGD(0.01, 0.9) for 10 epochs. The
checkpoint was chosen on inner-fold-0 subjects only, never on test.

**Results** (window level, `results/m8/R4_seed*_probs.json`):
- Per seed, fold-mean F1 is **0.347 / 0.299 / 0.303**.
- Frame-level accuracy is 0.40–0.46.
- Causal EMA with β picked per fold on inner-val odd-second frames changes almost nothing.

**R4s** (3-seed ensemble + window averaging + EMA, `results/m8/R4s_ens3_probs.json`):
- Fold-mean **0.333 ± 0.096**, pooled 0.362 [0.29, 0.43].
- Fold 2 (IJK) is the hardest at 0.225; fold 3 (LMNO) is the easiest at 0.51.
- `scripts/m8/score_frame_run.py` now takes several runs and averages them as a seed ensemble.

**Takeaway:**
- Honest M5 lands *below* the clock-only baseline R1 (0.355).
- Only test-set checkpoint selection (0.449) or the provisional DINOv2 smoke result (0.456)
  clears it.
- R4s is the H1 comparator. C1–C4 started at 02:34 (`cache/logs/day2_queue.log`).

**R6 tabular baseline** (`scripts/m8/r6_tabular.py`, `results/m8/R6_tabular.json`):
- Method: class-balanced logistic regression on per-window mean/std/p10/p90 summaries, with C
  chosen per fold by inner CV.
- Results:

  | Feature set | Video cohort | All windows |
  |---|---|---|
  | face/pose | 0.332 | 0.361 |
  | telemetry | 0.314 | 0.362 |
  | combined | 0.276 | 0.354 |

- Hand-crafted features sit at clock level, and stacking them does not help.
- Telemetry on its own is weak, which supports using it as a correction rather than an equal
  partner.

**Session-identity probe** (`scripts/m8/probe_session_identity.py`):
- A linear probe on window-mean DINOv2 tokens predicts the session (A vs D) of unseen subjects at
  balanced accuracy 0.59–0.73.
- That number is partly real label signal, because session D has higher KSS. The control repeats
  the probe on KSS 5–6 windows only, where both sessions are present:

  | Stream | Session bal-acc, KSS 5–6 only | Meaning |
  |---|---|---|
  | IR 168 | 0.49 | chance |
  | IR 112 | 0.60 | mostly clean |
  | L3D | 0.73 | session cues |
  | R3D | 0.70 | session cues |
  | Pose | 0.67 | session cues |

- So the RGB stereo and pose cameras carry session-level appearance cues (lighting, clothing,
  scene) beyond drowsiness. IR does not.
- Consequence: any C6 gain from L3D or pose must be checked against this shortcut, and C12
  (same-session start-of-drive normalisation) becomes a required control, not just an ablation.
- The same frozen embeddings give a linear y3 probe of 0.44–0.46 (IR, pose), already above R4s.

## 2026-11-05 (evening) — Day 2 work pulled forward while R4 trains

**Video extraction is done** (21:12):
- `cache/emb/{ir_168, ir_112, l3d_168, r3d_168}` at 2 fps, and `pose_224` at 1 fps.
- Each file holds DINOv2-S/14-reg tokens: [CLS, mean-patch, 2×2 grid] × 384.
- L3D/R3D Haar face detection misses fewer than 7% of frames per session.

**Tabular extraction** (`python -m m8.tabular`, about 75 s, all 35 sessions):
- `cache/tab` at 2 Hz holds 11 geometry features (EAR, MAR, PERCLOS, blinks, head-pose proxies,
  inter-ocular distance, face-missing), plus 30 AUs, 136 normalised landmarks and 33 upper-body
  pose values.
- `cache/tel` at 4 Hz holds pitch, roll, speed, rpm, gear, heading rate (with |·| and std),
  Δspeed, and grip L/R/asymmetry/zero-mask.
- Raw FAU contains some `inf` values; these are mapped to 0.
- Face is missing for 26% of O_D, 15% of E_D and 11% of A_D (A_D's video is short). These are
  natural vision-failure cases for the robustness tests.

**Training harness:**
- `m8/data.py`: per-stream frame banks that live on the device and are gathered per window.
- `m8/heads.py`: MIL, mean-pool and temporal-transformer heads.
- `m8/train.py`: outer 5-fold CV, 3 seeds, inner 4-fold out-of-fold logits, EMA β per fold.
- `scripts/m8/run_single_stream.py` is the entry point.
- Moving the gather onto the device made training about 3× faster, to roughly 86 s per model
  while the GPU is shared.

**First signal** (smoke test: `ir_112` + MIL, 1 seed, no inner CV, no smoothing): fold-mean
**0.456 ± 0.093**, pooled 0.458 [0.37, 0.54]. That already matches test-selected M5 (0.449) and is
well above `last.pt` M5 (0.363). It is provisional until the 3-seed run, the honest R4 baseline
and the session-identity probe are in.

## 2026-11-05 — Day 1 of the 14-day plan: canonical windows, alignment, clock baselines

Plan: `model_research/plan_14day.md`, with 12 core experiments C1–C12 and reference runs R0–R7.
It supersedes the schedule in `ablation_plan.md` and applies the critic's fixes. New code lives
in the `m8/` package (`config`, `windows`, `evaluate`) and in `scripts/m8/`. Caches go in
`cache/`, which is gitignored.

**Canonical window table** (`cache/windows.csv`, built by `python -m m8.windows`):
- 35 sessions × 157 windows (60 s long, 15 s stride) on the true 60 fps video clock, rather than
  on FAU row index as in Kavipriya's pipeline. FAU has 2.4% missing rows, so her windows drift
  against the video.
- Labels use the KSS epoch at the window centre.
- 4,532 windows have full video. A_D (2,144 s) and N_D (2,360 s) are shorter than 2,400 s.
- Each outer fold's training subjects are split into 4 subject-wise inner folds, balanced for
  Alert epochs (`cache/inner_folds.json`). All model selection happens on these.

**Alignment checks:**
- FL and FAU have a 1-indexed `Frame` column. FL frame *n* overlays exactly on IR video frame
  *n*−1, so FL was computed on the IR video.
- FL does not match L3D, so L3D needs its own face detection.
- PL is 0-indexed and its values are interleaved (x, y, z) per joint, despite the
  x-block/y-block/z-block header.
- Telemetry has 144,000 rows (60 Hz) and grip has 7,200 rows (3 Hz), both covering 2,400 s. Row
  *i* is assumed to align with video time *i*/rate.

**Clock baselines (R0/R1, P0 video windows, fold-mean macro-F1):**

| Baseline | 3-class | Binary B2 |
|---|---|---|
| Stratified random | 0.320 | 0.498 |
| R1: time-in-session only | **0.355** | 0.573 |
| R1b: time + session label | **0.668** | 0.896 |

- Time alone matches M1 and M7.
- Time plus "which session" reaches 0.67, so any vision model that can recognise the session
  (clothing, lighting, day) can score high without seeing drowsiness at all.
- This makes the session-identity probe and R7 image-statistics probe central to the thesis.
- Results: `results/m8/R0_R1_baselines.json`.

**R7: image-statistics probe.** Inputs are brightness, colour moments and histograms, averaged
over each window. Result: 0.306 F1 for drowsiness, and only 58% accuracy for session A vs D on
held-out subjects. Global lighting and colour do not transfer across subjects as a shortcut.

**R4a: existing M5 checkpoints re-scored on the canonical windows:**

| Checkpoint | Window macro-F1 (fold mean) | Pooled F1 [95% CI] |
|---|---|---|
| `best.pt` (selected on test) | 0.449 | 0.481 [0.39, 0.57] |
| `last.pt` | **0.363** | 0.399 [0.32, 0.48] |

- Honest-ish M5 is at clock-baseline level (R1 = 0.355).
- Causal EMA smoothing changes it by less than 0.005.
- `last.pt` is still not fully clean for folds 0 and 3, where test accuracy triggered early
  stopping. R4 gives the clean number.

**Code review** (workflow `day1-code-review`; 3 review lenses, each finding checked by a
refutation agent): 12 issues confirmed, 4 refuted, all fixed before launch.
- EMA β is now picked per outer fold, from that fold's own inner-val subjects. Previously it was
  pooled across folds, which let test subjects influence β, and it averaged several models'
  predictions within one window.
- β now uses only the odd-second frames, which were not used to pick `best.pt`.
- The paired sign-flip permutation test is now exact (2^16 sign vectors) instead of Monte Carlo.
- **R4 is pinned to SGD (lr 0.01, momentum 0.9), the optimizer M5 actually got.** Ultralytics'
  `optimizer=auto` would have silently switched R4 to AdamW, because R4 has fewer iterations.
- Smaller fixes: atomic cache writes, smoke runs written to a separate directory, deduplicated FL
  frame rows, positional `groupby` indices, and a safer resume order.

**Overnight chain launched** (`scripts/m8/day1_overnight.sh`, under `caffeinate`, plugged in):
1. DINOv2-S/14-reg extraction: IR face at 168 and 112 px, L3D and R3D face at 168 px (2 fps),
   and Pose at 224 px (1 fps), written to `cache/emb/`.
2. R4 honest M5 retrain: 3 seeds, 0.5 fps, 10 epochs, `best.pt` selected on inner-fold-0 subjects.
3. Window-level scoring of each R4 seed.

## 2026-11-05 — Final-phase scope set; extensive literature survey + ablation plan started

Scope decisions from the user for the final phase:
- **Deadline:** 14–21 days. Deliverables are the project report, PPT, codebase, and a working
  **Gradio** demo of the final chosen model.
- **Evaluation protocol:** open to redesign (e.g. binary, ordinal KSS), but it must stay
  subject-independent. Any new-protocol results get reported alongside M5 re-evaluated under the
  same protocol.
- **Pretrained weights** (e.g. frozen DINOv2) are allowed; training uses UL-DD only.
- **Modelling intent:**
  - ViT-style or modern transformer models that fit the Apple M4 (16 GB, MPS), optionally with a
    frozen generic vision encoder in front.
  - Ablations across IR face, stereo depth, and side-pose video, plus the extracted features.
  - Telemetry acts as a gated fallback for when visual cues fail, not as an equal partner.
    Vision-only models are also acceptable.

Launched a multi-agent research workflow: vision encoders, drowsiness literature, and
fallback-fusion research tracks, plus a local feature inventory and an MPS throughput benchmark.
It writes `model_research/literature_survey.md` and `model_research/ablation_plan.md`, and a
feasibility critic then reviews both.

**Outputs (workflow complete):**
- `model_research/literature_survey.md`: 44-paper master table and a "gaps this thesis fills"
  section.
- `model_research/ablation_plan.md`: 25 ablations, the VIGIL-Former flagship (to be published as
  M8), and an 18-day schedule.
- `model_research/plan_critique.md`: the critic's review. Verdict: the strategy is sound, but the
  plan is not executable as written.
  - Blockers: there is no 14-day path, and the A8 LandmarkFormer needs about 24 GB at batch 64.
  - Statistical design: the 5-fold Holm tests can never reach significance. Use a subject-level
    permutation test or bootstrap instead.
  - Missing controls: random frame blanking doesn't actually degrade vision; use contiguous
    blackouts instead. Also missing are a clock-only control for telemetry, a smoothed M5
    comparator, and a tabular baseline.
  - Scope: cut to about 12 core runs.
  - Plan revision is still pending.

**M5's 0.447 is test-selected** (verified in code). `scripts/m5/train_m5.py:89` links each fold's
test subjects in as the Ultralytics `val/` split. Ultralytics then uses that split to pick
`best.pt` and to early-stop. This mirrors Kavipriya's original pipeline, which is why the numbers
reproduced so closely. The per-fold best epochs (1 / 6 / 5 / 2 / 15) give mean accuracy 0.541;
the last epochs give about 0.441. A retrain selected on an inner-validation split ("R4" in the
plan) becomes the honest comparator. The 0.447 is still reported for continuity with the thesis.

Findings from the local grounding pass:
- **M6 embeddings leak across folds** (verified). `models/embeddings_uldd/` was extracted once,
  with `M5_fold0.pt` (trained on subjects E–S), and reused for every M6 fold. So folds 1–4 used
  visual features that had been trained on their own test subjects. M6's numbers (0.29–0.30) are
  optimistic, and these caches must not be reused. New caches should come from frozen generic
  encoders or per-fold backbones.
- **The "3D" videos are not depth maps.** L3D and R3D are ordinary colour frames from the left and
  right stereo cameras. The project treats them as extra frontal views unless disparity is computed.
- **Alert is near-empty in some folds** once B, I, M are dropped. Video-subject Alert epochs per
  fold are 4 / 21 / 3 / 9 / 34. The most balanced binary scheme is KSS ≤5 vs ≥6.
- **Effective sample size is about 290 epochs** (16 video subjects × 10 four-minute epochs, plus
  the Drowsy sessions), not about 5k windows.
- **Within-session confound:** KSS rises within every session. Session identity and time-on-task
  are therefore confounds to discuss in the report.
- **Measured MPS throughput** (in Low Power Mode, so lower bounds): DINOv2 ViT-S/14 fp16 runs at
  89–185 img/s at 224 px and 712 img/s at 112 px. Embedding IR + L3D + Pose at 2 fps takes about
  1 h in total. Decoding the video is never the bottleneck. Full report:
  `model_research/grounding_report.md`.

## 2026-09-19 — Wrote `docs/models_summary.md` (detailed per-model reference)

One dedicated section per model (M1–M7, M6 split into Lite/Full), each covering: modality,
architecture (with layer-by-layer diagram), exact training protocol, dataset/windowing specifics,
code provenance (unmodified Kavipriya code vs. new-this-project vs. locally-patched bugs),
full per-fold results table, thesis comparison, and a "Notable" callout with the model's key
finding. Pulled fresh exact numbers (including M7's per-class F1 breakdown, not previously written
up in this much detail) directly from the `results/reports*/*.json` files rather than
re-transcribing from memory. Ends with the cross-model summary table. Complements
`docs/repo_understanding.md` (which is about the *codebase*) and `docs/report_summary.md` (which
is about the *thesis*) — this new doc is the model-results reference specifically.

## 2026-09-15 — New `model_research/` track: video+ECU fusion targeting 0.70–0.80 macro-F1

On the `model_research` git branch (already created and checked out by the user). Wrote
`model_research/README.md`: a research plan for a transformer/fusion architecture combining video
+ telemetry ("ECU") specifically, aimed at beating M5 (0.447 macro-F1) with a stated target of
0.70–0.80. Builds directly on `docs/literature_review.md` and the M6/M7 findings rather than
starting over — key content: an explicit calibration section stating the 0.70–0.80 target is
1.6–1.8× anything achieved on this dataset so far and likely unreachable via architecture alone
given the small-N (19 subject) subject-independent protocol; a video-encoder and telemetry-encoder
options table; a ranked fusion-strategy list (gated/residual correction first, since it directly
targets the "fusion underperforms M5" pattern seen in M3/M6 by construction); a data-efficiency
section (cross-dataset pretraining, self-supervised pretraining, synthetic/manifold augmentation,
ordinal-aware loss) since architecture alone hasn't closed the gap in three attempts (M3, M6 lite,
M6 full) so far; a 6-phase experiment plan; and explicit honesty checkpoints for when to conclude
the target isn't reachable under the strict protocol rather than tuning indefinitely. Not yet
committed to git — pending the user's go-ahead.

## 2026-09-15 — Reorganized scripts/ (one folder per model + shared helpers/archive)

Now that M1–M7 are all reproduced/built, tidied up the flat `scripts/` directory (13 files with
no structure) into:

```
scripts/
├── m1/train_m1.py            ├── m5/train_m5.py
├── m2/train_m2.py            ├── m6/train_m6.sh   (wraps her src.models.m6_train CLI directly)
├── m3/train_m3.py            └── m7/{extract_m7.py, train_m7.py}
├── m4/train_m4.py
├── common/                   preprocess_m1_m2_m3.py, extract_m4_m5_features.py (shared prep)
├── helpers/                  verify_zips.py, wandb_backfill_m1_m4.py, wandb_backfill_m6_m7.py
└── archive/                  reproduce_m123.py, chain_m4_m5.sh, chain_m6_m7(_resume).sh
```

The old combined `regen_and_train.py` (preprocessing + M1 + M2 + M3 all in one file) was split
into a shared `common/preprocess_m1_m2_m3.py` plus one dedicated `train_m1.py`/`train_m2.py`/
`train_m3.py` per model, matching M4/M5/M7's existing one-script-per-model shape. M6 still has no
dedicated Python script of its own (training always ran through Kavipriya's own
`src.models.m6_train` CLI directly) — added `m6/train_m6.sh` as a thin, documented wrapper rather
than inventing a script that never existed. Superseded/one-off scripts (the early broken-
checkpoint evaluation attempt, and the two chain scripts that were fully superseded once their
jobs finished) moved to `archive/` with header notes explaining why, rather than deleted — all
still syntax-valid, none were touched functionally. Older entries below still reference the old
flat paths (e.g. `scripts/m4_train.py`) — that's correct for what existed at the time; not
retroactively rewritten.

## 2026-09-08 — M6 alignment fix completed: both variants got *worse*, not better

Three real bugs in her already-written code had to be fixed before M6 could even run (all local,
uncommitted patches to `driver-drowsiness-detection-system/src/models/m6_train.py` and
`m6_validate.py` — not pushed anywhere, not a redesign, just unblocking her own intended pipeline):
1. Validation wrongly expected video embeddings for B/I/M (never had video) and wrongly excluded
   subject A (does have video) — a `NO_TELEMETRY` check was used where `NO_VIDEO` was needed, in
   two separate functions (`validate_embeddings_cache`, `build_session_window_index`). Added a
   proper `NO_VIDEO` constant and fixed both.
2. A hardcoded tolerance gate blocked her own `_allocate_session_counts` proportional-reallocation
   logic from ever running when the video-vs-FAU-CSV window-count gap exceeded 2% — real gaps
   observed ranged 12% (fold 0) to 33% (fold 2), varying by which subjects a fold happens to test
   on. Removed the arbitrary cutoff entirely; the reallocation logic doesn't need one, and no
   single fixed threshold fit the dataset's actual variance.
3. Device selection had no MPS check at all (`"cuda" if available else "cpu"`), silently training
   on CPU — same class of bug already found and fixed in M5. Added the MPS branch.

**Results, after all three fixes, full 60fps re-extraction (using our regenerated M5 fold-0
backbone), both variants, 5-fold CV:**

| Variant | Fixed-alignment result | Kavipriya's original (broken, 1fps) result |
|---|---|---|
| M6 Lite | acc 0.332 ± 0.032, F1 0.303 ± 0.022 | acc 0.405, F1 0.360 |
| M6 Full | acc 0.318 ± 0.058, F1 0.292 ± 0.044 | acc 0.420, F1 0.377 |

**Both variants got worse after the alignment fix, not better.** This is a genuinely useful
negative result, not a failure to hide: the leading hypothesis is that the proportional window
reallocation needed for the 12–33%-per-fold video/FAU-CSV duration mismatch (item 2 above)
introduces its own label-to-embedding alignment noise, since it can't guarantee the *specific*
windows it adds/drops are the temporally-correct ones — trading a coarse-but-consistent 1fps
sampling error for a finer-grained but less-certain window-correspondence error. Worth a deeper
look before drawing final conclusions, but the headline finding — "the known bug wasn't actually
the dominant source of M6's underperformance vs. M5" — stands regardless and is itself a useful
data point for where this project's own novel work (`docs/literature_review.md`) should focus:
not on visual-embedding alignment precision, but likely on the fusion mechanism itself (Direction
1: gated/residual fusion, so a weak or noisy auxiliary signal literally cannot drag the visual
baseline down, regardless of alignment quality).

**M7 (built from scratch this session) result**: accuracy 0.452 ± 0.052, macro-F1 0.355 ± 0.063
across 5 folds (per-fold: 0.474/0.281, 0.444/0.358, 0.534/0.425, 0.376/0.287, 0.433/0.423 —
fold 3 is again the weakest, consistent with the pattern already seen in M1/M2/M3). Notably close
to, and slightly above, the thesis's own informally-stated M7 number (0.434 acc / 0.314 F1) —
which `docs/repo_understanding.md` §7 item 8 had flagged as unverifiable, since it came from a
document describing a *different* ("hybrid ensemble") M7 than the EfficientNet-B0+BiLSTM+attention
one actually documented as the thesis's real design. Landing in the same range on a faithful build
of the real architecture is a reasonable cross-check on both numbers, without over-claiming either.
Fold 0's per-class breakdown shows Alert F1=0.0 for that fold — the "Alert is the hardest class"
pattern holds even for this brand-new model.

**Session-final model comparison (everything reproduced or newly built so far):**

| Model | Accuracy | Macro-F1 | Status |
|---|---|---|---|
| M1 | 0.390 ± 0.114 | 0.357 ± 0.108 | reproduced (near-exact match to thesis) |
| M2 | 0.324 ± 0.069 | 0.259 ± 0.074 | reproduced (somewhat below thesis) |
| M3 | 0.352 ± 0.120 | 0.322 ± 0.125 | reproduced (somewhat below thesis) |
| M4 | 0.353 ± 0.056 | 0.275 ± 0.058 | trained for the first time ever (no target) |
| M5 | 0.541 ± 0.080 | 0.447 ± 0.071 | reproduced (near-exact match to thesis) — still the best model |
| M6 Lite | 0.332 ± 0.032 | 0.303 ± 0.022 | alignment-fixed, but *worse* than the original broken version |
| M6 Full | 0.318 ± 0.058 | 0.292 ± 0.044 | alignment-fixed, but *worse* than the original broken version |
| M7 | 0.452 ± 0.052 | 0.355 ± 0.063 | built from scratch, first-ever result |

**M5 remains the strongest model by a clear margin**, exactly as Kavipriya's thesis found — none
of M6's fix or M7's from-scratch build managed to beat pure frame-level visual classification.
Combined with the literature review's Direction-1 hypothesis (gate/residual-correct a frozen
visual baseline rather than fusing as equals), this is a strong signal for where the project's
genuinely new architectural work should go next, once resumed.

M6 had no wandb integration at all in `m6_train.py`, so all 10 folds (5 lite + 5 full) were
backfilled into wandb from their saved per-epoch history via new `scripts/wandb_backfill_m6_m7.py`
(same `_backfill`/`_summary` run-naming convention as the existing M1-4 backfill script), plus the
missing `M7_summary` aggregate run (M7's individual folds were already logged live during
training).

All local, uncommitted code changes made to unblock this session's work
(`driver-drowsiness-detection-system/src/models/m6_train.py`, `m6_validate.py`; new
`scripts/m7_extract_features.py`, `scripts/m7_train.py`, `scripts/chain_m6_m7*.sh`) are documented
above and in `docs/repo_understanding.md` — nothing pushed to Kavipriya's remote.

## 2026-09-07 — Pivoted from new-architecture work to M6/M7: both now running

- User asked to pause the STFTransNet-derived new-architecture work (from
  `docs/literature_review.md`) and instead pursue the two known, well-scoped gaps identified in
  `docs/repo_understanding.md`: finish Kavipriya's own coded-but-never-run M6 alignment fix, and
  build M7 from scratch (no code for it exists anywhere in her repo).
- **M6**: timed her `m6_extractor_from_video.py`'s per-frame seek approach first (~0.03s/frame) —
  full 60fps extraction over all 29 available-video sessions is ~4-5 hours, feasible with
  `num_workers=8`. Launched via her existing CLI (unmodified source, just `--ckpt`/`--video-root`/
  `--fps 60` overrides pointed at our regenerated `M5_fold0.pt` and local dataset path). Also
  discovered `datasets/processed/ul_dd/` already contains the correct M6 metadata (byte-identical
  to `ul_dd1`) — the `ul_dd1`→`ul_dd` copy flagged as needed in `repo_understanding.md` §7.3 had
  already happened at some point; `m6_train.py` needs no further data-path fixing.
- **M7**: confirmed (again) that no `src/models/m7_*.py` exists anywhere in her repo — wrote it
  from scratch as `scripts/m7_extract_features.py` (EfficientNet-B0 frozen feature extraction, 16
  frames/window sampled from the already-existing `datasets/processed/uldd_m7/fold_*.npz` window
  metadata) and `scripts/m7_train.py` (BiLSTM(256, bidir) + attention pooling + MLP classifier,
  PyTorch, same wandb/checkpoint/results conventions as the other `_regen` reproductions).
- All four stages (M6 extraction → M6 lite → M6 full → M7 extraction → M7 training) chained into
  `scripts/chain_m6_m7.sh`, running detached under `caffeinate` — not complete as of this entry.

## 2026-09-04 — M5 training completed: full M1–M5 reproduction now done

- All 5 folds finished cleanly (~1 day wall-clock, running detached across two Mac sleep/wake
  cycles thanks to `caffeinate`): fold accuracies 0.597, 0.482, 0.461, 0.670, 0.493 → **mean
  0.541 ± 0.080 acc, 0.447 ± 0.071 macro-F1**.
- **Near-exact match to the thesis's own M5 result** (0.545 ± 0.100 acc / 0.444 macro-F1) — macro-
  F1 landed marginally higher. Combined with M1's earlier near-exact match, this is strong
  end-to-end validation that the regenerated dataset, fixed paths, and training pipeline are
  faithful to Kavipriya's original work. Live-logged to wandb per epoch throughout (the MPS-fixed
  run, not the earlier CPU-only false start).
- **All of M1–M5 are now reproduced.** Final numbers vs. thesis, all in `results/reports_regen/`
  and on wandb (project `drowsiness_project`):

  | Model | Reproduced | Thesis |
  |---|---|---|
  | M1 | 0.390±0.114 acc / 0.357±0.108 F1 | 0.391±0.116 / 0.348 |
  | M2 | 0.324±0.069 / 0.259±0.074 | 0.387±0.040 / 0.339 |
  | M3 | 0.352±0.120 / 0.322±0.125 | 0.399±0.098 / 0.378 |
  | M4 | 0.353±0.056 / 0.275±0.058 | *(never trained by Kavipriya — no target)* |
  | M5 | 0.541±0.080 / 0.447±0.071 | 0.545±0.100 / 0.444 |

- Next decision point (with the user): dig into fold 3's weak M1/M2/M3 performance, or move
  straight to the new work outlined in `docs/literature_review.md` (STFTransNet-derived
  video+telemetry fusion, targeting the M6-regression and Alert/LowVigilant-boundary weaknesses).

## 2026-09-02 — M5 relaunched (with a real perf fix); literature-review research kicked off in parallel

- **M5 relaunch #1 was inefficient**: caught early (4 iterations in) that Ultralytics was training
  on pure CPU with 0 dataloader workers, despite MPS being available on this Mac — killed it and
  added `device='mps', workers=8` to `scripts/m5_train.py`'s `model.train()` (and `device='mps'`
  to the eval `model.predict()` call too). **Relaunch #2 confirmed ~10x speedup** (0.3 it/s → 3.2
  it/s, GPU_mem went from 0G to 1.09G). Running detached under `caffeinate` as usual, live-logging
  to wandb per epoch.
- User asked to start a parallel research track: find published SOTA work specifically on
  **video + vehicle/CAN-telemetry fusion** for drowsiness (explicitly excluding biometric
  wearables — narrower than the "bring in biometrics" idea floated earlier), pick one to
  implement, then design a custom hybrid/multimodal/ensemble model that improves on it. Launched
  a background research agent to search the literature, check for papers citing the UL-DD dataset
  itself, survey the broader SOTA landscape, recommend one implementation-target paper, and
  propose 2-3 extension directions grounded in what this project has already observed (M6 fusion
  underperforming unimodal M5, Alert-vs-LowVigilant being the dominant error). Writing findings to
  `docs/literature_review.md` — not complete as of this entry, check that file directly.

## 2026-09-02 — Pushed to GitHub (github.com/karthikeyansundaravaradhan/MTP)

- Set up Git LFS for the outer repo (`git lfs install` + tracking `*.pdf`, `*.mp4`, `*.pt`,
  `*.keras`, `*.zip`, `*.npz`, etc. — forward-looking, only `Literature/ULDD Dataset.pdf` (6MB)
  actually needed it so far).
- Updated `.gitignore` to also exclude: `driver-drowsiness-detection-system/` (Kavipriya's
  separate nested repo — never absorbed into this one, standing rule) and Kavipriya's personal
  report (`Kavipriya_report (2).html`, `Literature/Kavipriya_Ramasamy_24872_Final_Phase (2).pdf`)
  — excluded by default since it carries her name/roll number/guide names and this repo is
  public; **pending her consent or a decision to make the repo private** before including those.
- Scanned every staged file for secrets before committing (clean).
- **Push initially failed**: this Mac's cached git credentials (`osxkeychain`) belonged to a
  different GitHub account (`idifro`), not `karthikeyansundaravaradhan` — 403 on push. Installed
  `gh` CLI and ran `gh auth login --web`; that in turn hit a second, unrelated issue —
  `~/.config` was owned by `root` (leftover from some earlier `sudo`-run install), so `gh` couldn't
  write its own config. User fixed ownership (`sudo chown karthikeyansundaravaradhan:staff
  ~/.config`) themselves — not something to run without asking, since it's a sudo action outside
  this project. Second login attempt succeeded (`gh auth setup-git`), push went through cleanly.
- Repo is now live and public at **github.com/karthikeyansundaravaradhan/MTP**, `main` branch,
  root commit `4a380b1` (docs, scripts, non-sensitive literature — dataset and Kavipriya's repo/
  report excluded per above).

## 2026-09-02 — Weights & Biases experiment tracking wired up

- User asked to start logging training runs to wandb, and pasted credentials directly into chat
  and into `CLAUDE.md` (twice, two different keys). **Both raw keys are treated as burned** —
  authenticated via `wandb login <key>` instead, which stores credentials in this machine's
  `~/.netrc` (outside the repo), then removed both keys from `CLAUDE.md` entirely, replacing that
  section with a short pointer to this setup instead. Nothing wandb-related was ever committed to
  git. If auth ever needs resetting, generate a fresh key at wandb.ai/authorize.
- Entity `m25ai2065-iitj`, project `drowsiness_project`.
- **Backfilled** the already-completed M1–M4 runs from their saved `results_regen/*.json` files
  (which include full per-epoch history — accuracy/loss/val_accuracy/val_loss/learning_rate —
  saved at training time) rather than retraining just for logging's sake:
  `scripts/wandb_backfill.py` replays that history into wandb as one run per (model, fold), plus
  one aggregate summary run per model. All 24 runs (5 folds × 4 models + 4 summaries) synced
  successfully — browsable at https://wandb.ai/m25ai2065-iitj/drowsiness_project.
- **Wired up live logging for M5** (`scripts/m5_train.py`) via an Ultralytics `on_fit_epoch_end`
  callback, so the still-pending M5 retrain will log per-epoch metrics live rather than needing a
  backfill. M1–M4 don't have live hooks (they go through her shared `src/models/train.py::
  cross_validate`, which wasn't forked just for this) — the established pattern for any future
  model using that shared path is: train first, backfill after via the same script's pattern.

## 2026-09-01 — M4 training completed; M5 blocked by a disk-space/iCloud issue

- **M4 finished successfully**: accuracy 0.3533 ± 0.0558, macro-F1 0.2746 ± 0.0584 across all 5
  folds — the first-ever UL-DD result for this model (see `docs/repo_understanding.md` §10 for
  context on why this is a reasonable outcome vs. M1's ~0.39/~0.35). Checkpoints and results.json
  saved to `models/checkpoints_regen/` and `results/reports_regen/`.
- **M5 crashed early** (fold 0, during frame-linking) with a filesystem `TimeoutError`. Root cause
  traced to the machine's disk being **99% full**, and — more seriously — clear evidence that
  this Mac's iCloud Desktop/Documents sync is auto-evicting and re-fetching this project's large
  dataset files under disk pressure: a single test read of one video frame took **11 seconds**
  (should be milliseconds for a local file) and visibly re-materialized ~66MB of "vanished" data
  the moment it was touched; `du` showed the raw `Video_Data`/`Extracted_Features` folders at
  near-zero size immediately after the crash despite `ls`/`stat` still reporting full original
  file sizes; and there's a stuck iCloud Trash item (`Extracted_Features.zip`) failing to sync
  with a "quota exceeded" CloudKit error.
- **This was flagged to the user rather than resolved unilaterally** — it's an account/storage-
  level issue (iCloud quota, whether Desktop sync should be paused for this project, whether to
  free space by deleting the already-redundant `.zip` archives) that only the user can decide, and
  repeatedly touching the dataset while this is happening risks further instability or iCloud
  bandwidth/quota usage. Resolution and next steps pending the user's input.
- **Resolved**: user approved deleting the redundant zips once verified intact. Wrote a
  file-count + exact-size verification (`zipfile` central-directory listing vs. on-disk
  `stat()` — deliberately not a full checksum read, since that would force iCloud to re-fetch
  everything and worsen the exact problem being fixed) for all 4 zips in
  `DriverDrowsinessDataset/`. `CSV_Files.zip`, `Extracted_Features.zip`, `Video_Data.zip` verified
  byte-exact (every file present, every size matching) and were deleted (~10.9 GB freed, disk
  usage dropped from 85% to 29%, 40 GB free). `Codes.zip` had one 3-byte mismatch in
  `Biometric_Plots.ipynb` — left untouched (it's tiny, ~69 MB, not part of the space problem, and
  the mismatch is almost certainly trivial but wasn't worth the risk given it's costless to keep).
  Confirmed `~/Desktop` is genuinely iCloud-synced on this Mac (macOS 26.5.2) — the
  `~/Library/Mobile Documents/com~apple~CloudDocs/Desktop` path is a symlink back to the real
  `~/Desktop`. Disabling that sync is a System Settings action only the user can perform (no safe
  scriptable/terminal path for it) — instructions given, not automated.
- M5 is safe to relaunch now that disk space is healthy again; pending user's go-ahead on timing.

## 2026-08-31 — M4 (new) and M5 (reproduction) pipelines built and launched

- Started exploring M4 and M5, per user request. **M4 was never trained on UL-DD by Kavipriya at
  all** — only an architecture built for the live webcam demo, explicitly excluded from her
  thesis evaluation. Training it is genuinely new work for this project, not reproduction.
- Hit a real environment blocker: MediaPipe's modern Tasks API (`FaceLandmarker`, what her
  `src/face_detection.py` itself uses) hard-crashes on this Mac (Apple Silicon) with a Metal-GPU
  service-unavailable error, regardless of delegate settings — a genuine incompatibility, not a
  one-off. Fixed by using MediaPipe's older `mp.solutions.face_mesh` legacy API instead, which
  required a **separate, dedicated conda env `mp-extract`** (`mediapipe==0.10.21`, older
  numpy/protobuf) — that older mediapipe conflicts with `ddd-repo`'s TensorFlow if installed in
  the same env (confirmed: broke TF's import chain), so extraction and training now run in
  cleanly separated environments. Full detail in `docs/repo_understanding.md` §10.
- Built one combined video-decode pipeline (`m4_m5_extract.py`) that produces both M4's windowed
  10-feature sequences (EAR/MAR/PERCLOS/blink-rate/head-pose/brow/jaw, reusing Kavipriya's exact
  formulas from `face_detection.py`, adapted from live wall-clock timing to offline video-frame
  timing) and M5's 224×224 classification frames, in a single per-video pass — avoids decoding
  each ~40-minute IR video twice. Smoke-tested on all 29 available sessions before committing to
  the full run.
- Full extraction (~4-5 hours estimated) launched detached under `caffeinate`, chained
  automatically into M4 training (TensorFlow, `ddd-repo` env) then M5's full 5-fold YOLOv8n-cls
  retraining (`ddd-repo` env, using Apple Silicon MPS) — one autonomous background job, no manual
  restart needed between stages. Also rebuilding the M5 classification dataset from scratch since
  the existing one on disk was a partial dump (missing a class, a test split, and 4 of 5 folds).
- Next: check back once the chain completes; compare M5 against the thesis's 0.5451 acc/0.4439
  macro-F1 (existing checkpoints already reproduce this without retraining — this run validates
  the full pipeline, not just the weights), and M4 against no external target (first-ever result,
  M1's ~0.39 acc used only as a rough sanity anchor, see `docs/repo_understanding.md` §10).

## 2026-08-31 — M1/M2/M3 reproduction complete

- The retraining-from-scratch run (see previous entry) initially died silently around
  M1/fold0/epoch 48 — the machine went to idle sleep and killed the unprotected background
  process. Relaunched wrapped in `caffeinate -ims` + fully detached (`nohup ... & disown`) so it
  survives both system sleep and Claude Code session restarts; that run completed cleanly,
  all 5 folds × 3 models, no errors, in ~6 hours on CPU.
- Results: **M1 acc 0.390±0.114 / macro-F1 0.357±0.108** (thesis: 0.391±0.116 / 0.348 — almost
  exact match); **M2 acc 0.324±0.069 / macro-F1 0.259±0.074** (thesis: 0.387±0.040 / 0.339 —
  somewhat lower); **M3 acc 0.352±0.120 / macro-F1 0.322±0.125** (thesis: 0.399±0.098 / 0.378 —
  somewhat lower). Full breakdown and per-fold table in `docs/repo_understanding.md` §9.
- M1's near-exact match is strong validation that the regenerated dataset, fold split,
  architecture, and training loop are all faithful to Kavipriya's original. M2/M3's shortfall vs.
  thesis is concentrated almost entirely in one fold (fold 3 = subjects L/M/N/O), which is the
  worst fold for all three models — flagged as worth a dedicated look before assuming it's just
  sampling variance, since it's consistent across model families.
- Next: decide whether to dig into fold 3 first, or move on to verifying M5's existing checkpoints
  and then the M6 alignment-fix rebuild (see `docs/repo_understanding.md` §7.1, §8).

## 2026-08-31 — Environment set up, checkpoints recovered, checkpoint/data mismatch found, retraining from scratch

- Created a dedicated conda env **`ddd-repo`** (Python 3.11) for
  `driver-drowsiness-detection-system/`, installed its `requirements.txt`
  (TensorFlow 2.21/Keras 3, torch 2.13 with **MPS/Apple-GPU support**, opencv, ultralytics,
  mediapipe — PyTorch came in transitively via ultralytics/mediapipe despite not being listed).
  All imports verified working.
- Discovered `models/checkpoints/*.keras` and `*.pt` were **Git LFS pointer stubs**, not real
  weights (`git-lfs` wasn't installed, so the clone only pulled 132-byte text files). Installed
  `git-lfs` via Homebrew, ran `git lfs install --local` (repo-scoped, not global) +
  `git lfs pull` — all real binaries now present and correctly sized.
- Confirmed M1/M2/M3 checkpoints load fine under TF 2.21/Keras 3 (M3 needs
  `custom_objects=...` + `compile=False`, which `src/inference/predict.py` already does).
- **Found a real problem**: evaluated the existing M1/M2/M3 checkpoints against
  `datasets/processed/ul_dd1/` (the fold data `docs/repo_understanding.md` recommended reusing
  in place of the empty `ul_dd/`) and got numbers far below the thesis's reported ones — and,
  more tellingly, the models collapse to predicting almost entirely one class even on their own
  **training** split of that data. Since `src/data/preprocess.py`'s windowing/fold-assembly is
  fully deterministic (no randomness), this means `ul_dd1` is *not* the same data generation the
  checkpoints were actually trained on — most likely an artifact of the repo's messy iterative
  history (see `docs/repo_understanding.md` §1). **Correction to `docs/repo_understanding.md`**:
  reusing `ul_dd1` is fine for M6's metadata *schema* needs, but is not a safe substitute for
  numeric reproduction of M1/M2/M3.
- Decided the correct path is to **regenerate fold data fresh from the raw dataset using her
  unmodified `ULDDProcessor` code, then retrain M1/M2/M3 from scratch** with her unmodified
  `src/models/train.py` — reproducing her *methodology* exactly rather than trusting
  possibly-stale binary checkpoints. Confirmed and fixed the path-nesting bug along the way:
  `ULDDProcessor` hardcodes a triple-nested `Extracted_Features/Extracted_Features/Extracted_Features/`
  path; this project's actual dataset copy is double-nested — overrode `fau_root`/`tele_root` by
  one level (no source edits needed) rather than restructuring the dataset folder.
  Preprocessing ran cleanly against the real data once that was fixed.
- Kicked off a background run that regenerates `datasets/processed/ul_dd_regen/` and retrains
  M1, M2, M3 (5-fold CV each, her exact hyperparameters/architecture) into
  `models/checkpoints_regen/` + `results/reports_regen/`. Will compare the resulting numbers
  against the thesis's reported M1/M2/M3 results once it finishes.

## 2026-08-31 — Repo deep-dive complete: real results vs. fabricated documentation

- The deep-dive research pass on `driver-drowsiness-detection-system/` finished →
  `docs/repo_understanding.md`. Key finding: the repo's root-level docs contain **at least three
  mutually contradictory narratives** about the project's models/results, produced by one large
  "final" commit that dumped ~34 markdown files at once. Only one narrative is corroborated by
  actual code and `results/reports/*.json` — it matches `docs/report_summary.md`. Two files
  (`PROJECT_COMPLETION_REPORT.md`, `MULTI_MODAL_FUSION_SUMMARY.md`) describe a fabricated model
  taxonomy and a 74.2%-accuracy result with zero backing code/data anywhere in the repo — flagged
  as never-to-cite.
- Confirmed reproducible-without-retraining today: M1, M2 (all 5 folds), M5 (all 5 folds via
  existing checkpoints), M3 fold 0 only, M6 Lite/Full (existing but known-misaligned embeddings).
- Confirmed **M7 was designed in full but never actually coded** — only its preprocessed fold
  data exists, no model/training code was ever committed. M8 is pure vaporware (docs only, zero
  code). Both will need to be built from scratch for this project if pursued.
- Found a concrete, ready-made improvement target: Kavipriya fully coded a fix
  (`m6_extractor_from_video.py`) for a known temporal-misalignment bug that caps her M6 fusion
  models at ~41% accuracy, but never ran it. Running it is a strong candidate first step.
- Correct branch to build from is `feature/final` (not `main`, which is stale/pre-M5).
- Repo is Windows-authored throughout (hardcoded `C:\Users\raka1005\...` paths) and is missing
  PyTorch from `requirements.txt` despite M6 depending on it — both will need fixing before
  anything runs on this Mac.
- Next: decide (with the user) whether to reproduce her exact validated results first, or go
  straight for the M6 alignment fix / M7 rebuild / biometric-signal extension, then start doing so.

## 2026-08-31 — Cloned Kavipriya's actual codebase; started deep repo analysis

- User finished cloning Kavipriya's real thesis codebase into
  `driver-drowsiness-detection-system/` (`github.com/kavipriyariisc/driver-drowsiness-detection-system`,
  currently on branch `feature/final`) — this supersedes the inferred, report-only understanding
  in `docs/report_summary.md` wherever the two disagree.
- Launched a deep-dive research pass over that repo (it's large and messy: 25+ overlapping
  root-level status/summary `.md` files, `temp_check/`/`temp_src_check/` duplicate source trees,
  many one-off debug/test scripts, an `M8` model family mentioned nowhere in her written report)
  to produce `docs/repo_understanding.md`: canonical vs. superseded docs, a full `src/` module
  map, notebook map, end-to-end reproduction steps, data-expectation checks against the actual
  UL-DD folder layout, and known issues/red flags.
- Plan once that lands: reproduce her validated results (M1, M2, M3, M5, M6 Lite/Full, and M7 if
  its code is findable) on this machine, then design improvements on top — most obviously,
  bringing in the biometric/grip-pressure signals her work never used (see
  `docs/report_summary.md` §8) and revisiting why her fusion models (M6) underperformed her
  single-modality visual model (M5).
- Standing rule established: this file must be kept current as the project progresses (see
  `CLAUDE.md`).

## 2026-08-30 — Dataset + prior-report understanding, environment and git setup

- Read and summarized the UL-DD dataset paper and the raw dataset folder structure →
  `docs/dataset_summary.md` (modalities, sample rates, file formats, per-subject data gaps,
  the dataset authors' own SVM/RF baseline results).
- Read and summarized Kavipriya's written thesis report (at that point, only the PDF/HTML
  write-up was available, not her code) → `docs/report_summary.md` (her M1–M7 models, results,
  and the modalities/approaches she never tried).
- Created `CLAUDE.md` as the project onboarding doc.
- Asked the user clarifying questions on project scope, compute, environment setup, and version
  control. Answers: scope decision deferred until after her codebase is set up here; compute is
  local-first with Colab/Azure VM as fallback; set up the Python environment now; initialize git
  now.
- Set up a working environment: conda env `uldd` (Python 3.11.11) via the machine's existing
  Anaconda install, with all of `DriverDrowsinessDataset/Codes/Codes/requirements.txt` installed
  (`dlib` specifically via `conda-forge`, since pip tried to compile it from source and the
  system C++ toolchain is blocked by an unaccepted Xcode license — see `CLAUDE.md` Environment
  section for the full workaround).
- Initialized git at the project root with a `.gitignore` excluding the (tens-of-GB) dataset
  folder; nothing staged/committed yet.
- Mid-setup, noticed the user was independently cloning Kavipriya's real codebase into
  `driver-drowsiness-detection-system/` — left it alone per the user's instruction and picked
  this up once the clone finished (see entry above).
