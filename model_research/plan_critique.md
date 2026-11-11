## Adversarial review of `model_research/ablation_plan.md` and `literature_survey.md`

What I checked locally (I modified no files):
- **The M5 test-selection finding holds.** `scripts/m5/train_m5.py` is at the project root, not in the nested repo. `datasets/yolo_cls_regen/fold_0/val/` holds C_A frames, and the best epochs match.
- **Epoch timing.** From the `results.csv` files: 6.3–7.2 min per epoch, fold 0's first epoch took 39 min, and there are two stalls of 1.7 h and 27.6 h that look like the machine sleeping.
- **MPS ops.** Conv3d in fp32 and fp16, bicubic+antialias interpolation and cumsum backward all work under `torch 2.13` (tested with no CPU fallback).
- **A8 spatial stage, measured.** One transformer layer with d = 32 over 84 tokens × 600 frames runs at **37 windows/s and uses 6.1 GB at batch 16**.
- **Power.** The machine is still on battery right now.

### Blockers

1. **There is no plan for a 14-day deadline.** (§5, Days 13–18)
   - The user said 14–21 days. The plan assumes 18, starts the demo on Day 13, writes the results chapters on Days 15–16 and the PPT on Day 17.
   - If the deadline is Day 14, nothing gets submitted.
   - **Fix:** add a 14-day path with hard cut lines:
     - Days 1–3: R0/R1/R4, extraction, A2/A3/A8/A9.
     - Days 4–6: A10/A12/A24, A16, A17/A18/A19.
     - Day 7: freeze M8.
     - Days 8–9: demo.
     - Days 9–12: report.
     - Day 13: PPT.
     - Day 14: buffer.
   - Everything else is a stretch goal, used only if the deadline turns out to be 21 days.

2. **A8 LandmarkFormer does not fit memory or the time estimate as specified.** (A8, also feeds A15/A16/A19)
   - Measured: batch 16 already takes 6.1 GB of MPS memory, so the default batch of 64 would need about 24 GB and run out of memory or swap.
   - At 37 windows/s, one epoch over 4.4k windows takes about 2 min. Fifteen runs with early stopping come to about 10 h or more, against the planned 45 min.
   - **Fix:** run the spatial stage at 2 Hz instead of 10 Hz, matching the 120 video tokens (about 5× cheaper). Alternatively, replace spatial attention with a per-joint MLP and mean pooling, or cache the spatial stage once. Use batch ≤ 16 and re-benchmark on Day 2.

### Major

3. **The pre-registered statistics cannot produce a significant result.** (§1.3, H1–H5)
   - With 5 folds, a paired Wilcoxon test can never go below p = 0.0625 two-sided, so nothing can pass Holm correction.
   - "Wins ≥ 4 of 5 folds" is not a p-value (a 4/5 sign test gives p ≈ 0.19), so Holm cannot be applied to it.
   - **Fix:** define each H-test as a paired sign-flip permutation test or paired bootstrap over the **16 test subjects**. Use pooled macro-F1, or per-subject balanced accuracy or Brier score, since per-subject macro-F1 is undefined when a subject lacks a class. Apply Holm to those p-values and keep the "beats" threshold of Δ ≥ 0.03.

4. **H3 and A19 test a corruption that does not hurt vision, and that the gate was trained on.** (P5b, A19 success "+0.05 at 50% blanking", §7)
   - Adjacent frames have cosine similarity 0.9996, and MIL pools over the frames that remain. Blanking 50% of 2 fps frames at random removes almost no information, so V* will barely drop, the gate will not open, and H3 will come out null.
   - ModDrop trains the gate on exactly this corruption, while the vision-only comparator never sees it. The comparison is confounded.
   - **Fix:**
     - Use **contiguous** blackouts (15–45 s), correlated across all frontal streams (IR+L3D together, as a face turn or occlusion would cause), plus held-out corruption types (patch occlusion, brightness saturation) that are never used in training.
     - Give the vision-only baseline the same ModDrop augmentation.
     - Count the natural-failure windows (P5a, face-missing > 20%) before relying on them. At 2.4% missing frames overall there may be almost none.

5. **P4 / A23 calibration probably measures the session shortcut, and its expected effect is inflated.**
   - Normalising frozen DINOv2 embeddings against the **Alert session's** epoch 0 means the Drowsy session differs from its reference by clothing, lighting and day. That difference is itself a "not the Alert session" signal, and session is nearly the label.
   - ildahun's own control fits this: a same-drive reference gives 0.351, against 0.626 with the Alert-session reference.
   - The plan describes P4 as unlabelled, but cites ildahun's 0.362 → 0.626, which includes **labelled** head fine-tuning (survey §2.2 step b). The unlabelled-only gain on UL-DD has not been isolated by anyone.
   - **Fix:**
     - Make a **same-session start reference** (first 4 min of each session) the primary P4 variant, and keep the Alert-session reference as secondary.
     - Run the subject-independent session-identity probe before and after calibration.
     - Re-derive the H4 expectation without the +0.26 figure.

6. **Model selection on one inner-validation pair of 2 subjects is very noisy.** (§1.1)
   - Early stopping, hyperparameters, thresholds, temperatures, gate penalties, β and the choice of V* all ride on about 20 epochs from 2 people. In practice people will look at the outer test.
   - **Fix:** select on the inner 4-fold subject-wise CV that is already planned for OOF logits, using pooled inner macro-F1. Use a fixed epoch budget instead of patience, and keep a timestamped decision log in `story_so_far.md`.

7. **The primary metric is dominated by near-empty Alert folds.** (§1.1)
   - Fold 0 has 4 Alert epochs. Fold 2 has 3, and its test set is only J and K.
   - One epoch flips a fold's Alert F1 by about 0.3, so the fold-mean ± std mostly measures noise.
   - **Fix:** make **pooled out-of-fold macro-F1 with a subject-bootstrap CI** co-primary. Keep the fold mean for comparability with M1–M7, and use the binary cut B2 as the stability check.

8. **H1 confounds model with post-processing.** (H1, R4)
   - New models get window aggregation, MIL, epoch context and EMA smoothing; R4 is a per-frame CNN.
   - **Fix:** add **R4 + window averaging + the same causal EMA/HMM** as the comparator, so any gain is attributed to the encoder and head.

9. **R4b and R5-bin overnight runs are fragile, and R4a is not fully clean.** (R4, Day 1 overnight)
   - The real cost is 6.3–7.2 min per epoch plus a 39-min first epoch, about 12 h in total. The recorded 1.7 h and 27.6 h stalls point to sleep.
   - The run will overlap Day 2's GPU extraction, and two GPU jobs at once on 16 GB will slow both.
   - R4a `last.pt` is still test-contaminated for folds 0 and 3, where patience on the **test** accuracy decided the stopping epoch.
   - R4b uses a single seed while the comparators use 3.
   - **Fix:** run under `caffeinate -dims`, plugged in, with Low Power Mode off. Cap at 10 epochs and subsample to 0.5 fps (the frames are redundant), which gives about 3 h per seed, so 2–3 seeds fit. Report R4a only for folds 1, 2 and 4 as clean.

10. **The time/clock control for telemetry is missing.** (G4)
    - Telemetry may encode route position, and therefore time. The A17 R² diagnostic does not test whether the *gated hybrid's* gain comes from the clock.
    - **Fix:**
      - Add **A19-clock**: the same gate with e_t replaced by time-in-session features. Telemetry earns its place only if it beats this.
      - Add **R1b**: session label + epoch index → 3-class, as an upper bound on the shortcut.

11. **No strong simple baseline.** (§3.1)
    - R0 and R1 are trivial baselines.
    - **Fix:** add **R6**: logistic regression or LightGBM on hand-crafted window statistics (EAR/PERCLOS/blink/head pose from E7, plus E8 telemetry stats), ildahun-style, on the canonical windows. Also add **R7**: an image-statistics shortcut probe (brightness/colour histograms → classifier), to catch lighting or session cues that DINOv2 may pick up.

12. **The within-session Spearman diagnostic does not remove the time shortcut.** (§1.2 diagnostics)
    - KSS rises monotonically within sessions, so a clock alone scores well on this metric.
    - **Fix:** use partial correlation controlling for epoch index, and report accuracy on the off-trend epochs (D_A and J_A reaching 7, G_D and O_D LowVig).

13. **The session-identity probe is under-specified.**
    - If it is trained and tested on the same subjects it reaches about 1.0 from clothing alone, which tells us nothing.
    - **Fix:** train it on outer-train subjects and test it on held-out subjects.

14. **The G1 ablations are confounded.** (A1/A2/A3)
    - Camera, crop and resolution change together: IR full at 224, IR crop at 112, L3D crop at 168. The A3 vs A2 difference cannot be attributed to the camera.
    - **Fix:** add A2 at 168 px (or A3 at 112 px). That is about 7 min of extraction.

15. **A4 uses in-fold overfit embeddings.**
    - The H-std head is trained on embeddings from a backbone that memorised those same training subjects. This is the stacking bias the plan already guards against for the gates.
    - **Fix:** use an early-epoch backbone, or note the bias explicitly when interpreting A4's "why" diagnostic.

16. **The OOF and LOSO costs are underestimated.** (A19 "1 h incl. OOF"; Day 11 LOSO "1–2 h")
    - OOF for V* (A16) means 4 inner folds × 5 outer folds × 3 seeds, which is 60 trainings or about 4 h.
    - LOSO for the gated model needs inner OOF inside each of 16 folds, about 5 h or more with one seed.
    - The degraded z_v used for ModDrop must also come from the **inner-fold** models, not the outer ones.
    - **Fix:** budget these runs, use 1 seed for OOF, and run LOSO only on V* and R4.

17. **Implementation time is not budgeted, and the scope is too large.**
    - The plan has 13 extraction streams and 25 ablations with sub-variants. The time column counts compute only, not coding the Perceiver, LandmarkFormer, MARLIN loader, LoRA, CORN, HMM, MoE and missing-aware prompts, or debugging them.
    - **Fix:** cut A5, A7b, A11, A16d, A20, E4, E11, E12 and E13 first, and A25 down to (a) and (b). Treat about 12 runs as the core.

18. **The demo's live upload path cannot run the chosen model.** (§5.2)
    - M8 needs FL (dlib), PL (MediaPipe, not installed), L3D, Pose and telemetry. An uploaded mp4 provides one frontal view.
    - Live crops would come from YuNet while training crops come from FL bounding boxes, a mismatch.
    - **Fix:** define a separate "live" model (frontal stream trained on YuNet crops + MIL + EMA, telemetry optional). Use the full M8 only on precomputed dataset sessions. Decide whether the live path uses a 5-fold ensemble or a model retrained on all subjects.

### Minor

- **A12:** the first 15 windows of each session have padded histories, so the padding length tells the model where it is in the first 4 min. Randomise history length in training and report early windows separately. The same applies to how the EMA/HMM is initialised.
- **A6:** it is compared at 16 frames and 1 seed against frozen models at 120 frames and 3 seeds. Compare it with a frozen head on the same 16 frames.
- **Extraction margins** (§2.2): the "2× margin" is not applied consistently. E1's raw time is already 12.5–26 min, and E6 MARLIN has no margin and is not measured. Realistic core extraction is about 2.8–5 h, still inside the 6 h cap. Benchmark MARLIN on Day 1; Conv3d in fp16 works on MPS.
- **E6:** MARLIN on IR upsamples 80–100 px faces to 224. L3D crops are a better source.
- **E7 storage:** at 10 Hz for 35 sessions the parquet is about 0.9 GB in fp16, not 0.4 GB. This does not matter for disk.
- **YuNet** is not built in: the ONNX model has to be downloaded. `marlin-pytorch`'s video utilities expect ffmpeg, which is not installed, so pass tensors in directly.
- **"Stereo depth" ablation:** before falling back to Depth Anything, check whether L3D/R3D are rectified, i.e. whether matching keypoints sit on the same rows. If they are, `cv2.StereoSGBM` at 1 fps on CPU gives real relative disparity in under 1 h, which matches the user's request better than monocular depth.
- **Biometrics are excluded without a stated reason.** ildahun's skin temperature alone scores 0.415, which is above telemetry, and it is likely a clock proxy. Add one sentence explaining the exclusion.
- **Low Power Mode and battery:** the machine is still on battery, so fix this before Day 1.

### Citations

I spot-checked nine; the ones that need changes:
- **Adaptive cross-modal fusion (S2666521226001481):** 76.35% is a **single LOSO fold (S006)**, not the LOSO mean, and 91.51% comes from a shared-backbone ablation. Reword survey §3.2 and table row 12.
- **Lakhani 2022:** the 44% is confirmed, but it is a non-peer-reviewed preprint trained for 10 epochs on a K80. It is weak support for the "freeze" policy (§0, §3.2), so lean on LP-FT and M5's own curves instead.
- **SI multimodal 2026 (S174680942601061X):** κ = 0.274 and the +0.566 F1 gap are confirmed. The 0.527 and 0.388 F1 values were not, so mark them [unverified]; they appear in §10.2 and §10.6.
- **Over-flagged as unverified but correct:** Mamba arXiv:2312.00752, Perceiver IO (ICLR 2022), PoseFormer (ICCV 2021). Remove the tags.

Verified as stated:
- Wolter 52.5% three-class / 80.4% binary. It cites UL-DD but does not use it.
- Massoz 70/85/89/94% at 5/15/30/60 s.
- Kong, Sensors 24(2):455: 0.948 → 0.972.
- ildahun: 0.362 / 0.626 / 0.651 / 0.295 / 0.415 / 0.351, and 75.2% vs 29.5–33.8%.
- Light-VTD (PMC13554065) and Sensors 26:6193 both exist.

### Verdict

The core strategy is feasible and well grounded:
- Frozen DINOv2 encoders with cached 2 fps embeddings and tiny heads keep GPU time under about 5 h and disk under 8 GB, with no MPS op blockers found.
- The M5 test-selection audit is correct and is a real contribution.

The plan is not ready to execute as written. It has no 14-day path, and A8 does not fit memory as specified. Several of the headline claims cannot be tested or are confounded:
- the Holm-on-folds statistics;
- H3's random frame blanking;
- the P4 Alert-session reference;
- the H1 comparison against an R4 without smoothing;
- 2-subject inner validation.

It also lacks the clock control and the simple tabular baseline needed to show that the transformer and the telemetry gate earn their place. Fix the blockers and items 3–13 before Day 1, cut the scope to about 12 core runs, and the plan becomes a defensible thesis inside the deadline.

Sources:
- [Lakhani, arXiv:2207.12148](https://arxiv.org/pdf/2207.12148)
- [Wolter et al., arXiv:2511.19519](https://arxiv.org/html/2511.19519)
- [ildahun/Driver-Drowsiness-Detection](https://github.com/ildahun/Driver-Drowsiness-Detection)
- [Massoz et al. 2018, PMC6165048](https://pmc.ncbi.nlm.nih.gov/articles/PMC6165048/)
- [Kong et al. 2024, PMC11154312](https://pmc.ncbi.nlm.nih.gov/articles/PMC11154312/)
- [Adaptive cross-modal fusion, S2666521226001481](https://www.sciencedirect.com/science/article/pii/S2666521226001481)
- [SI multimodal 2026, S174680942601061X](https://www.sciencedirect.com/science/article/abs/pii/S174680942601061X)
- [Sensors 26(19):6193](https://www.mdpi.com/1424-8220/26/19/6193)
- [Light-VTD, PMC13554065](https://pmc.ncbi.nlm.nih.gov/articles/PMC13554065/)