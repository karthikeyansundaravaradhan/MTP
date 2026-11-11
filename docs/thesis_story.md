# Thesis Story and Novelty Claims

The narrative spine for the report, slides and viva. Every number here comes from
subject-independent cross-validation on the 16 UL-DD drivers with video (4,532 windows of 60 s).
Full tables: [`m8_models.md`](m8_models.md) §0.5. Chronology: [`story_so_far.md`](story_so_far.md).

---

## 1. Working title

**Seeing Drowsiness or Seeing the Drive? Frozen Foundation Features, Driver Calibration and
Graceful Fallback for Subject-Independent Drowsiness Detection on UL-DD**

## 2. Thesis statement (one paragraph)

On a naturalistic, KSS-labelled driving dataset, this thesis builds drowsiness detection from
the ground up under strict subject-independent evaluation, one baseline at a time. The first
rungs are floors and a model that knows only how long the person has been driving (0.366). Above
them come hand-built face and telemetry statistics, an image-statistics probe, six sequence
models over action units, telemetry, geometry and video (M1–M4, M6, M7) and an end-to-end
fine-tuned CNN on IR frames (M5). None of them clears the clock (0.26–0.38). The thesis then shows
three things:

- A frozen self-supervised vision encoder (DINOv2) on low-resolution IR face crops, with a
  3,843-parameter head, is the first rung that recovers real signal: 0.446 (5-fold), 0.471 (LOSO).
- Two additions build on that core:
  - an unlabelled 4-minute alert-drive calibration plus interpretable 60 fps eye/head/posture
    events lifts it to 0.536 / 0.525, the only models that significantly beat the clock;
  - a vision-quality-gated mixture of experts (VIGIL, M8) keeps working when the camera fails:
    0.349 vs 0.186 under a full 60 s blackout.
- A set of diagnostics shows what all of these models actually learn. On UL-DD, most of the
  learnable signal separates the *Drowsy drive from the Alert drive*. Within a single drive,
  time-on-task orders drowsiness better than any vision model. Splits that let the test driver
  into training reach 0.80–0.91, but a model with no video at all scores higher still. This
  reframes what "accuracy" means on this dataset and on others like it.

## 3. The story arc (how the report and slides flow)

| Act | Question | Answer, with the number |
|---|---|---|
| 1. The bar | What does a model with no vision get? | Majority 0.215, stratified random 0.327. Time-in-drive alone (R1): **0.366**. Time + "which drive": 0.70. Every model must clear the clock, and must not just recognise the drive. |
| 2. Baselines, built up | Do conventional pipelines clear it? | Hand-built face-geometry and telemetry window statistics (R6) 0.32–0.38; image statistics (R7) 0.356; sequence models M1–M4, M6, M7 (FAU BiLSTM, telemetry LSTM, attention fusion, geometry BiLSTM, embedding fusion, CNN-BiLSTM) 0.26–0.36 fold-mean; an end-to-end fine-tuned CNN on IR frames (M5 as R4s, YOLOv8n-cls, checkpoints chosen on inner-validation drivers) 0.362. All sit at clock level: with 290 labelled epochs, learning features from scratch does not work. |
| 3. Better features | Do frozen foundation features help? | DINOv2 ViT-S on 112 px IR face, mean-pool head (V\*): **0.446 / LOSO 0.471**. Bigger heads, temporal transformers, longer context, other cameras, DINOv3 and eye crops all fail to beat it. Simplest wins. |
| 4. Robustness | What happens when the camera fails? | VIGIL (MoE: vision expert + telemetry/clock expert, gate driven by vision quality, trained with contiguous ModDrop blackouts). Clean: no harm (0.428 vs 0.430). 60 s blackout: **0.349 vs 0.186**. Gate opens 0.13 → 0.93. |
| 5. Personalisation | Can 4 unlabelled minutes of the driver help? | Subtracting the driver's Alert-drive reference token: **0.521 / LOSO 0.518**. Adding 27 interpretable 60 fps events: **0.536 / 0.525**. QWK 0.13 → 0.52. Beats the clock, p ≤ 0.01. |
| 6. The twist | What are the models actually detecting? | Drive-AUC vs within-drive AUC. Calibrated V\* separates the drives at 0.90, but orders windows within a drive at only 0.63–0.70. The clock: 0.81–0.90 within-drive. Binary KSS ≥6 equals the drive for 91% of windows, so calibrated binary F1 0.775 is mostly drive detection. Known-driver splits: random 0.91, epoch 0.80, new drive **0.41**; a no-video "who + drive + time" model 0.93 / 0.87. |
| 7. Takeaway | What should the field and a deployer believe? | Subject-independent three-class KSS on UL-DD tops out around 0.45–0.54 without labels. Gains over the clock are drive-level. Robustness is the most deployable contribution. Report drive vs within-drive signal, a clock baseline and LOSO next to any headline. |

## 4. Novelty claims (C1–C6) and the evidence behind each

Each claim is phrased so that it survives an examiner's challenge. "Strength" says how hard the
evidence is.

### C1. A leakage-controlled baseline ladder for UL-DD video, built from the ground up

- **Claim:** under one fixed subject-independent protocol, a ladder of baselines of increasing
  complexity (floors → no-sensor clock → hand-built statistics → sequence models M1–M7 →
  end-to-end CNN) shows that every
  conventional pipeline sits at the level of a model that only knows time-in-drive. This sets the
  honest bar for everything that follows.
- **Evidence:**
  - R0 majority 0.215 / stratified 0.327; R1 clock 0.366; R1b clock + session 0.699 (shortcut
    upper bound).
  - R6 tabular window statistics: face geometry 0.379, telemetry 0.354, both 0.316.
  - R7 image-statistics probe 0.356, which rules out a global lighting shortcut.
  - Sequence models, fold-mean: M1 FAU BiLSTM 0.357, M2 telemetry LSTM 0.259, M3 FAU +
    telemetry attention 0.322, M4 geometry BiLSTM 0.275, M6 embedding + telemetry fusion
    0.303 / 0.292, M7 CNN-BiLSTM on IR frames 0.355. Fusion never beats the better single stream.
  - M5 end-to-end YOLOv8n-cls on IR frames, three seeds, checkpoints on inner-validation
    drivers (R4s): 0.362 pooled (vs clock −0.004, p 0.95). Selecting `best.pt` on the test
    drivers would give 0.449 fold-mean, which is why all selection stays inside training drivers.
  - Canonical windows anchored on the true 60 fps video clock; windowing by FAU row index would
    drift by 2.4% because FAU misses rows.
- **Strength:** strong (direct measurements under one protocol, all model selection inside the
  training drivers).
- **Novel because:** no published subject-independent UL-DD video benchmark exists, and none
  reports a time-in-drive control. The dataset paper's 83–88% uses stratified, likely leaky
  folds; an independent re-implementation drops to 30–34% per driver (ildahun).

### C2. The first subject-independent raw-video benchmark on UL-DD, with frozen foundation encoders

- **Claim:** frozen DINOv2 features on 112 px IR face crops plus a 3,843-parameter mean-pool head
  are the first rung of the ladder that clears the clock under subject-independent evaluation. A
  systematic ablation shows the simplest configuration is best.
- **Evidence:**
  - V\* 0.446 (5-fold) / 0.471 (LOSO).
  - 20+ ablations in `m8_models.md` §0.2–0.4, none of which beats V\*: L3D/R3D/Pose views, MIL and
    temporal-transformer heads, 240 s context, ordinal loss, DINOv3, 10 fps eye crops.
  - Session-identity audit: L3D/R3D/Pose carry session cues (balanced accuracy 0.67–0.73 at
    matched KSS 5–6); IR does not (≈ chance). Pose's 5-fold advantage disappears under LOSO
    (0.499 → 0.409).
- **Strength:** strong for "first" (no published subject-independent UL-DD video result found)
  and for "simplest wins". V\* vs the end-to-end CNN is not significant after Holm (p 0.15 → 0.46).
- **Novel because:** no peer-reviewed DINOv2 application to drowsiness exists, and none on UL-DD
  video.

### C3. VIGIL (M8): a vision-primary, quality-gated mixture of experts that degrades gracefully

- **Claim:** gating a telemetry + clock expert by *vision quality* (observed-frame fraction,
  face-missing rate, no-frame flag), trained with contiguous camera blackouts (ModDrop) and a
  fast gate, preserves performance under camera failure without hurting clean accuracy.
- **Evidence:**
  - Clean: 0.428 vs vision-only 0.430 (H2, Δ −0.002, p 0.80).
  - 60 s blackout: 0.349 vs 0.186 (H3, Δ +0.163, p 0.076; Holm 0.31).
  - Gate mean 0.13 → 0.93.
  - The residual-gate variant (C9) is "cosmetic": its gate never moves. That negative result
    motivates the MoE design.
- **Strength:** medium. The effect is large and mechanistically clear (gate behaviour) but not
  significant at n = 16 after Holm.
- **Novel because:** the literature explains why symmetric fusion fails on small data, and our own
  learned late-fusion attempts show it on UL-DD (IR + telemetry 0.306). No drowsiness paper
  evaluates an asymmetric, quality-gated fallback with clean and degraded regimes reported
  separately.

### C4. Unlabelled driver calibration of foundation tokens, plus interpretable 60 fps events

- **Claim:** subtracting the mean DINOv2 token of the driver's first 4 minutes of an alert drive
  (no labels) is the largest single improvement. Fusing it with 27 interpretable event features
  gives the best model. These are the only models that significantly beat the clock and track
  KSS beyond time.
- **Evidence:**
  - Calibrated V\* 0.521 / LOSO 0.518. Fusion 0.536 / 0.525. QWK 0.52 vs 0.13 for V\*.
  - Partial ρ with KSS given time: 0.63 vs 0.17.
  - vs clock: +0.155 / +0.170, p ≤ 0.01.
  - Events alone (blink duration, PERCLOS, long closures, yawns, nods, posture) score 0.480,
    matching the foundation model, and are human-readable.
- **Strength:** medium. The protocol (P4) assumes an alert reference drive is available, and C5
  shows much of the gain is drive-level. H4 (calibrated vs plain V\*) is +0.075, p 0.18.
- **Novel because:** ildahun showed alert-session normalisation for *tabular* features. This is
  its first application to *video foundation embeddings*, and the first quantification of what it
  adds within vs between drives.

### C5. A confound-decomposition protocol: what does a drowsiness model actually detect?

- **Claim:** splitting AUC into a *drive* part (Alert vs Drowsy recording) and a *within-drive*
  part, alongside no-video shortcut baselines, shows that on UL-DD the gains of every vision
  model over the clock are drive-level. Known-driver evaluation splits measure recording
  recognition.
- **Evidence:**
  - Within-drive AUC: clock 0.81 (B2) / 0.90 (KSS ≥7); best vision model 0.64–0.72.
  - Drive-AUC: calibrated models 0.89–0.91.
  - B2 equals "which drive" for 91% of windows.
  - Known-driver splits:

    | Split | V\* | No-video "who + drive + time" |
    |---|---|---|
    | Random windows | 0.910 | 0.926 |
    | Held-out KSS epochs (purged) | 0.802 | 0.867 |
    | New drive of a known driver | 0.413 | — |

- **Strength:** strong. These are direct measurements, reproducible from saved predictions.
- **Novel because:** papers on UL-DD and NTHU/RLDD report single headline accuracies. Leaky-split
  inflation is known qualitatively (ildahun: 75% random vs 30% per-driver), but no work
  decomposes drive vs within-drive signal or shows that a no-video identity + drive + time model
  beats the video model under the leaky splits.

### C6. A reproducible, edge-budget system with pre-registered statistics

- **Claim:** the whole pipeline runs on a 16 GB Apple M4. Hypotheses H1–H4 were fixed before
  results and tested with exact subject-level permutation tests and Holm correction. Final models,
  `m8/inference.py` and a Gradio demo are released.
- **Evidence:**
  - DINOv2 runs at 712 img/s at 112 px; a head trains in about 1.5 min.
  - `models/final/`, `app/demo.py`.
  - H1–H4 with Holm: none is significant. This is reported as a power statement (n = 16), not
    hidden.
- **Strength:** strong (engineering), modest (scientific).

## 5. What we deliberately do not claim

- **Not "state of the art accuracy".** The dataset paper's 88% and NTHU-style 95–99% come from
  leaky or acted protocols. Our numbers are comparable only to subject-independent three-class
  KSS work: ildahun 0.362 uncalibrated, UTA-RLDD 52–65% accuracy, human judges under 60%.
- **Not statistically significant improvements after multiple-testing correction.** With 16
  subjects, the smallest Holm-adjusted p is 0.31.
- **Not "the calibrated binary model detects drowsiness at 0.78".** It detects the drive. The
  within-drive AUC of 0.63 is the honest number.
- **Not "VIGIL improves accuracy".** It improves *robustness*. Clean accuracy is unchanged.
- **Not "within-driver splits would make the model better".** They make it look better while
  measuring something else.
- **The demo's deployment models are in-sample on UL-DD.** Only CV numbers are results.

## 6. Positioning against the literature

| Work | Protocol | Inputs | Result | Relation to this thesis |
|---|---|---|---|---|
| UL-DD paper (Bodaghi et al. 2025) | Stratified 5-fold (likely leaky) | Biometric + behavioural + facial features | 83.8–88.0% acc | Not comparable; no video model |
| ildahun (2025/26) | LOSO | Tabular face + bio + vehicle | 0.362 F1 uncalibrated; 0.626 with labelled personalisation | Closest; our unlabelled video calibration reaches 0.518–0.525 LOSO |
| Kavipriya (2025), MTech report on UL-DD | Subject-independent 5-fold | FAU, telemetry, IR frames | — | Source of the M1–M7 architectures used in our sequence-model rung |
| UTA-RLDD (Ghoddoosian 2019; Wolter 2025) | Subject-wise folds | Blink features | 52.5–65.2% acc, 3-class; 80.4% binary | Same task family; same 3-class vs binary gap |
| Massoz 2018 | LOSO, PVT labels | NIR face video | 94% binary at 60 s | Objective labels; supports 60 s windows |
| Literature's realistic targets (survey §10.6) | — | — | 3-class calibration-free 0.50–0.55; with calibration ~0.60–0.65 | We reach 0.45–0.47 calibration-free, 0.52–0.54 with unlabelled calibration |

## 7. Chapter plan (report)

1. Introduction: problem, the honesty problem in the field, contributions C1–C6.
2. Background and related work: vision drowsiness, protocol leakage, fusion and missing
   modalities, foundation encoders, personalisation.
3. Dataset and protocol: UL-DD, label budget (290 epochs), canonical windows, 5-fold + LOSO,
   metrics, statistics.
4. Baselines, built up (C1): floors, clock, hand-built statistics, sequence models M1–M7,
   end-to-end CNN.
5. Methods: frozen-encoder pipeline and V\* (C2); VIGIL (C3); calibration and events (C4);
   confound diagnostics (C5).
6. Results: single streams and ablations, robustness, calibration and fusion, binary, extended
   metrics, H1–H4.
7. What the models detect (C5): drive vs within-drive, subject-dependent splits.
8. System and demo (C6).
9. Discussion and limitations: n = 16, label resolution, simulator, P4 assumption.
10. Conclusion and future work.

## 8. One-line versions (for the abstract, slide 1 and the viva opener)

- **Short:** "Built up from a clock baseline, conventional pipelines stall at clock level. Frozen
  foundation features, driver calibration and a quality-gated fallback clear it robustly. But on
  UL-DD, most of what any model learns is which drive it is watching."
- **Elevator:** "We built drowsiness detection on UL-DD up from honest baselines, made it robust
  to camera failure, and measured how much of it is really drowsiness."
