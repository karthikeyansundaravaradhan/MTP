# Model Research — Video + ECU Fusion for Driver Drowsiness Detection

> **Status note (2026-11-06):** this is the original 2026-09-15 direction doc, kept for history.
> - Its premise that M5 = 0.447 is superseded: that number was test-selected, and the honest
>   retrain R4s is 0.333.
> - Current plan: [`plan_14day.md`](plan_14day.md). Round 2: [`round2_research.md`](round2_research.md).
> - Final-phase models and results: [`../docs/m8_models.md`](../docs/m8_models.md).
> - Current best under P0: V\* 0.440. Frozen final model: M8.

**Goal:** design a transformer-based or fusion architecture that combines **video** (IR facial
video) and **ECU/vehicle-telemetry** data from UL-DD to beat the current best model (**M5**,
YOLOv8n-cls on IR frames, **macro-F1 0.447**) with a **target of 0.70–0.80 macro-F1**, under the
same subject-independent, 3-class (Alert/LowVigilant/Drowsy) evaluation protocol used throughout
this project.

This doc is the living research plan for that effort. Read `docs/story_so_far.md`,
`docs/repo_understanding.md`, and `docs/literature_review.md` first — this doc builds directly on
all three rather than re-deriving what they already established.

## 0. Read this first: calibrating the 0.70–0.80 target

Stated plainly, because it shapes every decision below: **every model built on UL-DD so far in
this project, and every model in Kavipriya's thesis, tops out well under 0.50 macro-F1** (M5:
0.447; M1: 0.357; M3: 0.322; M7: 0.355; M6 lite/full after the alignment fix: 0.303/0.292). A jump
to 0.70–0.80 would be roughly **1.6–1.8×** the best result achieved on this dataset by anyone,
under this protocol, to date. `docs/literature_review.md` §3.2 already flagged the reason this is
hard, not just unattempted: essentially every published 90%+ drowsiness result in the literature
comes from an **easier task regime** — binary or near-binary labels (alert/drowsy, open/closed
eye), **acted or simulated** drowsiness rather than gradual self-reported KSS, and/or
**subject-dependent** evaluation (the same driver's data can appear in both train and test).
UL-DD's actual regime — 3-class, self-reported/ordinal KSS labels, strict subject-independent
5-fold CV, only 19 subjects — is intrinsically harder, and none of the 90%+ papers surveyed are
doing the same task.

This doesn't mean 0.70–0.80 is impossible — it means **hitting it will very likely require more
than a better fusion architecture alone**. Sections 3–4 below lay out which levers (architecture,
data efficiency, evaluation protocol, additional data) are realistically available, and section 6
sets honest checkpoints for reassessing the target as evidence comes in, rather than chasing a
number that may not be achievable under the strict protocol without additional data.

## 1. What "video + ECU" means for this dataset

UL-DD's telemetry is simulator-generated, not a real vehicle CAN bus, but it's the closest
analog available and is what Kavipriya's thesis (and this project) already treat as the "CAN"
signal:

| Signal | Source | Rate | Columns |
|---|---|---|---|
| Video | `*_IR_*.mp4` (IR camera, face-focused, 240×220) | 60 fps | RGB frames |
| "ECU"/telemetry | `*_Telemetry_*.csv` (American Truck Simulator output) | 60 Hz | pitch, roll, speed, rpm, gear (+ raw timestamps) |

Scope for this research track is explicitly **video + telemetry only** — no biometric wearables
(HR/EDA/BVP/etc.), no grip pressure, no pose video. That's a deliberate narrowing (matches the
brief that produced `docs/literature_review.md`), not an oversight; bringing biometrics back in is
a separate, larger research track noted in `docs/report_summary.md` §8 if this one doesn't pan out.

Known data constraints to design around (all documented in `docs/dataset_summary.md` and
`docs/repo_understanding.md`):
- **19 subjects total**, subject-independent 5-fold CV (per `ULDDProcessor.FOLDS`) — small-N,
  high-variance regime. Every model result so far has large per-fold std (e.g. M5: ±0.080 acc).
- **B, I, M have no video at all.** Any video-dependent architecture drops to 16 usable subjects
  unless a missing-modality strategy is used (see §3.4).
- **A has video but no telemetry.** Symmetric problem for fusion models.
- **C, F, L have Alert-only sessions** (no Drowsy session).
- Video and telemetry streams have a **12–33% duration mismatch per fold** (`docs/story_so_far.md`,
  2026-09-08 entry) — real, not a bug, and a likely contributor to why M6's fusion attempt got
  *worse* after fixing its alignment bug rather than better.
- KSS labels are **ordinal**, self-reported every 4 minutes, with documented inter-rater
  reliability that degrades at fine granularity (κ=0.619 on the 9-level scale vs. κ=0.88 on the
  3-class binning — `docs/dataset_summary.md` §5). No model built so far has used an ordinal-aware
  loss; every one has treated the 3 classes as nominal.

## 2. Where this project already stands (don't re-derive these)

| Model | Modalities | Macro-F1 | Notes |
|---|---|---|---|
| M1 | FAU (facial landmarks) | 0.357 | facial-only baseline |
| M2 | Telemetry only | 0.259 | weakest model |
| M3 | FAU + Telemetry, cross-modal attention | 0.322 | fusion, still below M1 |
| M4 | 10 real-time geometric features | 0.275 | new this project, no prior target |
| **M5** | **Raw IR frames, YOLOv8n-cls** | **0.447** | **best model — the one to beat** |
| M6 Lite | Visual embeddings + Telemetry, BiLSTM | 0.303 | fusion, *worse* than pre-fix version |
| M6 Full | Visual embeddings + Telemetry, cross-attention | 0.292 | fusion, *worse* than pre-fix version |
| M7 | 16-frame EfficientNet-B0 + BiLSTM + attention | 0.355 | new this project |

**The dominant pattern across every fusion attempt (M3, M6 Lite, M6 Full) is that fusion
underperforms the best single modality (M5).** This is the single most important empirical fact
this research track needs to explain and fix — not a detail, the core problem. `docs/story_so_far.md`'s
leading hypothesis (2026-09-08 entry): the video/telemetry duration mismatch forces a proportional
window-reallocation that introduces its own label-to-embedding alignment noise, on top of
whatever mismatch already exists between concatenation/attention fusion and a genuinely weaker
auxiliary signal (telemetry alone: 0.259 macro-F1, barely better than chance).

**The second dominant pattern**: Alert vs. LowVigilant is the hardest class boundary in every
model that reports per-class F1 (M5's own confusion matrix: 62.3% of Alert samples predicted as
LowVigilant, vs. only 4.8% Alert↔Drowsy confusion — `docs/report_summary.md` §4).

Any new architecture proposed here should be evaluated against **both** of these patterns
specifically, not just aggregate macro-F1 — a model that improves aggregate F1 by improving
Drowsy detection while leaving Alert/LowVigilant just as confused is not actually solving the
project's real bottleneck.

## 3. Architecture research agenda

### 3.1 Video encoders to evaluate

| Encoder | Type | Why consider it | Risk |
|---|---|---|---|
| YOLOv8n-cls backbone (existing M5) | CNN, frame-level | Already trained, warm-startable, cheap | Frame-level only, no temporal modeling (M5's own stated limitation) |
| EfficientNet-B0 (existing M7 pipeline) | CNN, frame-level | Already have a 16-frame extraction pipeline built | Same frame-level limitation as above |
| **ViT / DeiT (small/tiny)** | Transformer, frame-level | Stronger spatial representations than CNNs at similar param count when pretrained; literature review found ViT/Swin hitting 99%+ on MRL Eye (binary, easier task) | Needs more data or heavier pretraining to avoid overfitting on 16 subjects; likely needs ImageNet-pretrained weights, not from-scratch |
| **VideoMAE / TimeSformer (frozen, feature-extraction only)** | Video transformer, clip-level | Purpose-built for temporal video understanding; using as a frozen feature extractor (not fine-tuning) sidesteps the small-N problem | Heavy; needs to be evaluated for feasibility on this Mac (MPS) or via Colab; may be overkill for 240×220 IR footage very unlike its Kinetics-400 pretraining domain |
| **Two-stream (frame + optical flow)** | CNN/Transformer hybrid | Explicitly captures motion (blink dynamics, head movement) that frame-level models miss — this is exactly the temporal gap the thesis's own M5 analysis flagged (`docs/report_summary.md` §4.4) | Optical flow computation cost over IR video; flow quality on low-res/IR footage untested |

### 3.2 Telemetry ("ECU") encoders to evaluate

| Encoder | Why consider it | Risk |
|---|---|---|
| LSTM/BiLSTM (existing M2/M3/M6) | Baseline, already implemented | Already tried multiple times, ceiling appears low (0.259–0.322) |
| **1D-CNN / FCN-LSTM** (per `docs/literature_review.md` §1.1, IIETA paper) | Purpose-built for CAN-bus time series, reported 99% on driver-behavior classification (different task, but strongest CAN-only architecture found in the literature review) | That result is for behavior/anomaly classes, not drowsiness — transfer to drowsiness unverified |
| **Temporal Convolutional Network (TCN, dilated)** | Matches STFTransNet's temporal stage (§3.3 of the literature review); cheaper than attention over long sequences, wide receptive field | Needs tuning of dilation schedule for UL-DD's 240-timestep windows |
| **Small Transformer encoder** over the 5-channel telemetry sequence | Consistent architecture family with the video side if both are transformer-based; self-attention could surface which timesteps (e.g. sudden steering correction) matter most | 5-dim input is very low-dimensional for a transformer — likely needs a learned projection to a wider embedding first, and is easy to overfit |

### 3.3 Fusion strategies to evaluate (ranked by what the evidence so far suggests)

1. **Gated / residual correction fusion** (from `docs/literature_review.md` Direction 1) —
   telemetry stream produces a *correction* to a frozen, pretrained visual baseline's logits/
   embedding, rather than being concatenated or attended to as an equal partner. This is the
   highest-priority experiment: it directly targets the "fusion underperforms M5" pattern by
   construction (the correction is gated, so it can learn to output ~0 when telemetry isn't
   helpful, structurally preventing regression below the frozen visual baseline).
2. **Cross-attention with telemetry as query** (STFTransNet-style, weak-queries-strong) — already
   detailed in `docs/literature_review.md` §4. Second priority: more expressive than gating, but
   without an explicit "can't hurt the baseline" guarantee.
3. **Multimodal Bottleneck Transformer (MBT)-style fusion** — a small number of learned "bottleneck"
   tokens mediate information exchange between the two streams (video and telemetry each attend to
   the bottleneck tokens, not directly to each other). Worth evaluating as a middle ground between
   full cross-attention and naive concatenation; caps how much a noisy telemetry stream can
   directly corrupt visual features, similar in spirit to gating but at the token level.
4. **Late fusion / learned ensemble** (from `docs/literature_review.md` Direction 3, mixture-of-
   experts) — train visual and telemetry specialists fully separately, combine only at the
   decision level via a learned router. Lowest architectural risk (each specialist trains exactly
   as well as its unimodal counterpart), but historically late fusion captures less cross-modal
   interaction than earlier fusion — worth it primarily as a strong, safe baseline to beat with 1–3.
5. **Naive early fusion (concatenation)** — already effectively what M3 and M6 Lite do (via
   concatenated pooled representations). **Not worth re-trying as-is** — already tested twice,
   underperformed twice. Only revisit if 1–4 all fail to beat naive concatenation, as a sanity
   check that the fusion *mechanism* was ever the bottleneck at all (vs. e.g. data quality).

### 3.4 Missing-modality handling (B, I, M have no video; A has no telemetry)

Whatever fusion architecture is chosen needs an explicit policy for these subjects, not silent
exclusion (which is what M5/M6/M7 currently do, discarding real labeled data):
- **Modality dropout during training** — randomly zero out one modality's input during training
  (a standard robustness technique) so the fused model learns to function on telemetry-only or
  video-only inputs, then route B/I/M through telemetry-only and A through video-only at inference.
- **Gating network naturally handles this** if Direction 3.3.1/3.3.3 is used — a learned gate/
  router can be trained to assign near-zero weight to a missing modality without special-casing.

## 4. Data efficiency — the lever architecture alone can't pull

Given 19 subjects is a hard ceiling on this project (no more data collection is possible), closing
the gap toward 0.70–0.80 likely needs at least one of the following, independent of which fusion
architecture is chosen:

1. **Cross-dataset pretraining.** Pretrain the video encoder (or a shared visual backbone) on a
   larger drowsiness/facial dataset — NTHU-DDD, YawDD, MRL Eye, RLDD (all in `docs/dataset_summary.md`
   §8's comparison table) — before fine-tuning on UL-DD's 16 video-available subjects. This is
   standard practice for small-N fine-tuning and is likely the single highest-leverage data-side
   intervention available.
2. **Self-supervised pretraining on UL-DD's own unlabeled structure.** Frame-order prediction,
   masked-frame reconstruction, or contrastive learning across the many more finely-windowed
   sub-clips available before any labels are needed — makes use of the full 1,400 minutes of
   recording rather than only the labeled 4-minute epochs.
3. **Synthetic augmentation.** `docs/literature_review.md` §3.3 item 4 (GAN-based augmentation for
   class-imbalanced/insufficient data) and item 6 (the "Blinking Beyond EAR" paper's rigged-avatar
   synthetic data approach) are both directly applicable and already scouted.
4. **Manifold-level augmentation for subject-independence** (`docs/literature_review.md` §3.3
   item 5) — augmentation performed in embedding space specifically to improve generalization to
   unseen subjects, which is UL-DD's exact evaluation regime.
5. **Ordinal-aware training** (§0 above) — not more data, but a better use of the label signal
   already present; a soft/ordinal loss that treats Alert↔Drowsy as a bigger mistake than
   Alert↔LowVigilant may recover some of the "wasted" signal in the current nominal-softmax setup.

## 5. Proposed phased experiment plan

| Phase | What | Exit criteria before moving on |
|---|---|---|
| **0** (done) | Reproduce M1–M7, establish baselines, literature review | Complete — see `docs/story_so_far.md` |
| **1** | Implement gated/residual fusion (§3.3.1) on top of the existing M5 backbone + a TCN telemetry encoder (§3.2) | Does it beat M5's 0.447 macro-F1? If not, is it at least closer than M6's 0.30 attempts? |
| **2** | If Phase 1 shows promise: add cross-attention (§3.3.2) and/or MBT-style fusion (§3.3.3); ablate against Phase 1's gated baseline | Which fusion mechanism wins, and by how much? |
| **3** | Add ordinal-aware loss + auxiliary continuous-KSS regression head (§4.5) to the Phase 1/2 winner | Does the Alert/LowVigilant confusion rate specifically improve (not just aggregate F1)? |
| **4** | Add cross-dataset pretraining for the video encoder (§4.1) — highest-effort, highest-expected-payoff data intervention | Re-run Phase 1–3's winning architecture with the pretrained backbone; compare |
| **5** | Missing-modality handling (§3.4) — recover B/I/M and A's currently-discarded windows | Does using more of the 19 subjects' data change the fold-to-fold variance or mean? |
| **6** | Final architecture, full ablation table, honest writeup | See §6 |

## 6. Honesty checkpoints (don't skip these)

- **After Phase 1**: if gated fusion doesn't at least match M5 (0.447), that's important evidence
  the bottleneck isn't fusion *mechanism* at all — revisit whether the telemetry signal itself
  carries enough drowsiness-relevant information to be worth fusing (M2's 0.259 ceiling is a real
  warning sign already).
- **After Phase 3**: if macro-F1 is still meaningfully below 0.55–0.60, treat 0.70–0.80 as
  **out of reach without Phase 4's data intervention** — say so explicitly in whatever writeup
  follows, rather than continuing to tune architecture indefinitely against a data ceiling.
- **At every phase**: report macro-F1 **and** the Alert-class per-class F1 **and** the fold-to-fold
  std, not just the headline mean accuracy/F1 — this project's own history (`docs/story_so_far.md`)
  shows repeatedly that aggregate numbers alone hide which class or which fold is actually driving
  a result.
- If 0.70–0.80 ultimately proves unreachable under the strict subject-independent protocol, the
  right response is to **document that finding rigorously** (it would itself be a legitimate,
  useful thesis contribution — "here's why, and here's the ceiling we found and why") rather than
  relax the evaluation protocol without saying so, which would make results incomparable to
  everything already established in this project.

## 7. Open questions for the next research session

- Is a Colab/cloud GPU session worth setting up for Phase 1 experimentation, or is this Mac's MPS
  sufficient? (M5/M6/M7 all trained locally in reasonable time — informative prior.)
- Should Phase 4's cross-dataset pretraining target NTHU-DDD specifically (most literature
  precedent, per `docs/literature_review.md` §3.1) or a broader multi-dataset pretraining mix?
- Worth a dedicated literature pass specifically on **modality-dropout / missing-modality training
  techniques** (§3.4) and **ordinal deep learning for affective/graded labels** (§4.5) — neither
  was covered in-depth by `docs/literature_review.md`'s original video+CAN-focused search.
