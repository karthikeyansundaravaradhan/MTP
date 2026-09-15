# Literature Review — Video + Vehicle-Telemetry Fusion for Driver Drowsiness Detection

Purpose of this doc: find genuinely published state-of-the-art work in the specific niche of
**camera/video-derived features fused with vehicle CAN-bus / driving-telemetry / vehicle-dynamics
signals** for drowsiness/fatigue detection, check whether anyone has published on UL-DD itself,
survey the broader SOTA bar, recommend one paper as an implementation target, and propose hybrid/
ensemble extensions grounded in that literature. This is scoped deliberately to **video +
telemetry only** — the biometric wearable channels (HR/EDA/BVP/IBI/TEMP/SpO2) are explicitly out
of scope for this document (see `docs/dataset_summary.md` for those). Read `docs/dataset_summary.md`
and `docs/report_summary.md` first — this review does not re-derive facts already established
there (UL-DD's own baselines, Kavipriya's M1–M7 results) except where verifying or expanding them.

Search conducted 2026-09-02 via web search across arXiv, IEEE Xplore, ScienceDirect, Nature
(Scientific Reports / npj), PMC, Springer, and Semantic Scholar's citation graph. Search strategy:
targeted queries for the video+CAN niche first (precision-first, per brief), then a citation-graph
check for arXiv:2507.13403 (UL-DD), then broader SOTA queries to calibrate the accuracy/F1 bar and
harvest transferable architectural techniques.

---

## 1. The video + CAN/telemetry fusion niche: what actually exists

**Finding, stated plainly: no published paper was found that fuses camera/video-derived driver
features with vehicle CAN-bus/telemetry signals specifically for drowsiness or vigilance
classification with a KSS-style graded/ordinal label.** This is consistent with the task brief's
expectation that this is a narrow niche. What exists instead falls into four adjacent buckets,
each missing one piece of the exact combination:

### 1.1 Vehicle-signal-only drowsiness/fatigue detection (no camera)

| Paper | Signals | Architecture | Data / Result |
|---|---|---|---|
| **"An Efficient Deep Learning Model Based on Driver Behaviour Detection Within CAN-BUS Signals"**, *Revue d'Intelligence Artificielle* (IIETA), n.d. — [link](https://iieta.org/journals/ria/paper/10.18280/ria.380106) | Acceleration, RPM, speed, accelerator pedal, throttle position, GPS, IMU — **CAN-bus only, no video** | **FCN-LSTM**: parallel CNN branch (univariate time series, multi-scale) + LSTM branch (reshaped multivariate sequence), outputs concatenated → dense classifier | 4 real drivers, urban driving, 80/20 split, 10s sliding windows w/ 90% overlap → **99.01% acc / 99.00% F1** for driver-behavior classification (not drowsiness specifically — behavior/anomaly classes) |
| Online/real-time steering-wheel-angle fatigue detection line of work (e.g. classic "Real Time Detection of Drowsiness Related Lane Departures Using Steering Wheel Angle"; "Online Detection of Driver Fatigue Using Steering Wheel Angles for Real Driving Conditions", PMC5375781) | Steering wheel angle (SWA), steering wheel movement (SWM), standard deviation of lane position (SDLP) | Statistical/entropy features (approximate entropy of SWA) + classical classifiers, some LSTM variants | Vehicle-dynamics-only; treated in newer surveys as a "reliable, real-time, non-invasive" signal family, but consistently weaker alone than facial/physiological signals — directly consistent with UL-DD's own telemetry-only baseline (40–45% acc, dataset_summary.md §6) and Kavipriya's M2 (38.7% acc) |

**Relevance to this project:** the FCN-LSTM CAN-only design is a good template for the
**telemetry branch** of a hybrid model (a small dual-path CNN+LSTM encoder is a stronger telemetry
backbone than Kavipriya's plain 2-layer LSTM in M2), but neither of these papers touches video.

### 1.2 Camera-only drowsiness/fatigue detection (no vehicle signals)

Large literature; see §3 below for the SOTA bar. Not itemized here since it's outside the video+
telemetry niche by definition, but it establishes that vision-only methods are frequently the
strongest single modality — the same conclusion Kavipriya's M5 already reached on UL-DD.

### 1.3 Camera + *other* physiological/tactile signals (fusion, but not vehicle telemetry)

These are the closest **fusion-architecture** analogs even though the second modality differs
from UL-DD's telemetry:

| Paper | Modalities fused | Architecture | Result |
|---|---|---|---|
| **"A Multi-Source Fusion Approach for Driver Fatigue Detection Using Physiological Signals and Facial Image"**, *IEEE Transactions on Intelligent Transportation Systems*, 2024 — [DOI](https://doi.org/10.1109/TITS.2024.3420409) | Physiological signals + facial video, 21 participants, simulated driving | End-to-end neural fusion (no hand-crafted features) | 93.15% acc in a 3s window (94.04% specificity, 91.71% sensitivity) |
| **"Dynamic cross-domain transfer learning for driver fatigue monitoring: multi-modal sensor fusion with adaptive real-time personalizations"**, *Scientific Reports*, 2025 — [PMC12056195](https://pmc.ncbi.nlm.nih.gov/articles/PMC12056195/) | EEG + ECG + video | **Attention-Based Sensor Fusion Transformer** (cross-modal attention weights each modality's contribution) + a **gating mechanism** that suppresses unreliable/missing modalities in real time + **online personalized fine-tuning (OPFT)** for per-driver adaptation; domain-adaptation component reduces cross-domain accuracy gap to <5% | Cross-domain accuracy improved from ~10% to as high as 15% gain; robust to missing/degraded sensor channels |
| **"A multimodal framework for fatigue driving detection via feature fusion of vision and tactile information"**, *npj Flexible Electronics*, 2026 — [link](https://www.nature.com/articles/s41528-026-00543-7) | Vision (CNN image features) + tactile/flexible-sensor signals (grip-pressure-like) | CNN image branch + fully-connected tactile-signal branch, projected into a shared embedding space, concatenated | Not independently confirmed in detail (paywalled); listed because "tactile" here is functionally the closest published analog to UL-DD's own grip-pressure channel, which is *not* in this project's video+telemetry scope but is architecturally the same fusion pattern (image CNN + scalar-sensor MLP → shared embedding → concat) |
| **STFTransNet**, Kim & Choi, *Sensors* 25(18):5819, 2025 — [DOI](https://doi.org/10.3390/s25185819) | **Two visual streams**: full driver frame + cropped face region (not video+CAN, but the fusion mechanics are the most reproducible found and are structurally what a video+telemetry model needs) | See full spec in §4 — this is the recommended implementation target | 95.86% acc / 0.957 F1 on NTHU-DDD; full ablations available |

**Why these matter despite not being video+CAN:** the "Dynamic cross-domain..." paper's
**gating mechanism for missing/unreliable modalities** is directly transferable to UL-DD's known
data gap (subjects B, I, M have no raw video — see `dataset_summary.md` §2) — a hybrid model could
use the same style of gate to fall back gracefully to telemetry-only inference for those subjects
instead of dropping them. Its **OPFT personalization** is also relevant given UL-DD's small
19-subject pool: a lightweight per-subject calibration step, even under subject-independent
evaluation (calibrate only on the *held-in* training subjects' early data), could reduce variance.

### 1.4 Broader driver-behavior surveys touching vision+vehicle fusion

**"Using Visual and Vehicular Sensors for Driver Behavior Analysis: A Survey"**, Bikram Adhikari,
arXiv:2308.13406, 2023 — [link](https://arxiv.org/abs/2308.13406). General survey (distraction,
risk, behavior — not drowsiness-specific) concluding that "integrating vision and vehicular
information can significantly enhance the accuracy and effectiveness of driver behavior analysis."
Confirms the *general* premise that vision+vehicle-signal fusion is a recognized, actively pursued
direction in driver-monitoring research broadly, even though the specific drowsiness/KSS
sub-literature hasn't converged on a canonical video+CAN paper yet. Also worth flagging a newer,
not-yet-fully-verifiable ScienceDirect entry, **"Multi-modal driving fatigue detection with
causal-aware fusion and Kolmogorov-Arnold network"** (2026, forthcoming/very recent —
[link](https://www.sciencedirect.com/science/article/abs/pii/S156625352600134X)) — title suggests
causal fusion across driving-behavior + physiological modalities using KAN layers; too new to have
full details extracted in this pass, but worth a follow-up check as it becomes indexed, since
"causal-aware fusion" directly targets the failure mode (a weak/misaligned modality dragging down a
strong one) that this project has already hit with Kavipriya's M6.

**Bottom line for §1:** per the brief's own fallback instruction, since no exact video+CAN-for-
drowsiness paper exists, the recommendation in §4 broadens to the closest architectural analog
(STFTransNet's two-stream cross-attention + TCN design) rather than a topically exact but
non-existent paper, and separately borrows the missing-modality gating idea from the EEG+ECG+video
fusion paper in §1.3.

---

## 2. UL-DD dataset citation check

UL-DD (Bodaghi, Hosseini, Gottumukkala, Bhupatiraju, Ahmad, Gabbouj — arXiv:2507.13403, posted
July 2025; also published as **"A multimodal drowsiness dataset using video, biometric, and
behavioral data"**, *Scientific Data*, 2025 — [DOI](https://www.nature.com/articles/s41597-025-06540-1))
is very new. Semantic Scholar's citation graph for arXiv:2507.13403 (queried directly) returns
essentially nothing yet:

- **"Blinking Beyond EAR: A Stable Eyelid Angle Metric for Driver Drowsiness Detection and Data
  Augmentation"**, Wolter, Perez, Shan, *2026 IEEE Intelligent Vehicles Symposium (IV)* —
  [arXiv:2511.19519](https://arxiv.org/pdf/2511.19519). This paper does cite UL-DD, but only as a
  reference to a *related public dataset* in its background/related-work discussion — it does
  **not** train or evaluate its Eyelid-Angle (ELA) metric on UL-DD itself. Its actual
  experiments run on other "public driver monitoring datasets" (not further identified in the
  fetched excerpt) plus its own Blender-rendered synthetic avatar data. Its core contribution
  (a 3D-landmark-derived, viewpoint-robust replacement for Eye Aspect Ratio, plus synthetic-data
  augmentation via rigged-avatar animation) is nonetheless a genuinely useful technique this
  project could borrow for the FL-based facial feature engineering, given UL-DD already ships
  Dlib 68-point 2D landmarks.
- One additional candidate surfaced (**"Comparison of Histogram Equalization and Multi-Scale
  Retinex Methods for Near-Infrared Image Enhancement in Drowsiness Detection"**, Subowo et al.,
  *IJACSA*, 2026) could **not be confirmed** to actually use UL-DD — the citation-graph tool's own
  summary of it referenced the "DROZY" (University of Liège) dataset instead, which is a
  different, older, unrelated multimodal drowsiness dataset with a similarly-styled acronym. Flag
  this as a likely tool/summary confusion rather than a real UL-DD citation; do not treat it as
  confirmed.

**Conclusion:** as of this search (Sept 2026), **no published paper trains or evaluates a model on
UL-DD** beyond the dataset authors' own baselines (`dataset_summary.md` §6). This project would be
among the first, if not the first, to build a custom model on this dataset — which is itself a
notable framing point for the thesis (novelty is inherent to the dataset choice, not just the
method). It also means there is no existing "UL-DD leaderboard" to beat other than the dataset
paper's own 84% (SVM, Bio+Behavioral+Facial, 3-class) and Kavipriya's 0.545 acc / 0.444 macro-F1
(M5, vision-only) — both already documented in `dataset_summary.md` and `report_summary.md`.

---

## 3. Broader SOTA landscape for driver drowsiness/fatigue detection

### 3.1 Verifying/expanding the benchmarks already cited in `report_summary.md`

- **Lyu, Yuan, Chen — "Long-term Multi-granularity Deep Framework for Driver Drowsiness
  Detection"**, arXiv:1801.02325 (2018) — **confirmed**: Multi-granularity CNN (parallel
  extractors on facial patches at different granularities, robust to head-pose variation) +
  deep LSTM for long-term temporal dependency. **90.05% accuracy, ~37 fps on NTHU-DDD**,
  state-of-the-art at time of publication; also introduced a new dataset, FI-DDD, with more
  precise temporal drowsy-interval annotation. Full author list: Jie Lyu, Zejian Yuan, Dapeng
  Chen; arXiv preprint only (no confirmed peer-reviewed venue found beyond arXiv in this search).
- **"Shen et al., 94.46%"** (as cited in `report_summary.md`) — **could not be independently
  re-confirmed** in this search pass; no paper by a "Shen" reporting exactly 94.46% on a
  drowsiness benchmark surfaced. Closest confirmed near-matches found instead: a DeepLabv3+ /
  Gaussian-weighted eye-state fatigue method at **94.4%** ("Driver Drowsiness Detection Using
  Deep Learning", ResearchGate) and a ViT/Swin-Transformer blink-classification approach at
  **94.44%** ("Real-time driver drowsiness detection using transformer architectures: a novel
  deep learning approach", *Scientific Reports*, 2025 — [PMC12092738](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092738/),
  MRL Eye dataset, open/closed eye binary classification). Recommend treating the 94.46% figure
  in `report_summary.md` as needing its original citation re-checked directly against
  Kavipriya's report references rather than assuming it is one of these two — they are plausible
  substitutes but not verified identical.

### 3.2 New SOTA data points found

| Approach | Task / Dataset | Result | Source |
|---|---|---|---|
| ViT / Swin Transformer, transfer-learned | MRL Eye (binary open/closed) | **99.15% / 99.03%** acc | *Scientific Reports* 2025, PMC12092738 |
| STFTransNet (two-stream cross-attention + TCN) | NTHU-DDD (4-class: normal/drowsy/yawning/inattention) | **95.86% acc / 0.957 F1** | Kim & Choi, *Sensors* 2025 |
| Multi-Attention Fusion (MAF) — IResNet50 + local-attention modules | RLDD (binary alert/drowsy) | **99.11% acc / 0.991 F1** | arXiv:2312.17052 |
| MAF (self-built, day/night, occluded/low-light) | Custom, 32 drivers | 96.88% acc / 0.960 F1 | arXiv:2312.17052 |
| FCN-LSTM (CAN-bus only) | Driver-behavior classes (not drowsiness per se), 4 drivers | 99.01% acc / 99.00% F1 | IIETA 2024 |
| Multi-source physio+facial fusion | Simulated driving, 21 participants | 93.15% acc (3s window) | IEEE TITS 2024 |

**Important caveat for calibrating expectations on UL-DD:** essentially every one of these
high-accuracy numbers (95–99%) comes from **binary or near-binary tasks** (alert/drowsy,
open/closed eye), **acted/simulated drowsiness** (not gradual, self-reported KSS), and/or
**subject-dependent or non-held-out evaluation**. None of them do 3-class ordinal classification
under strict subject-independent evaluation with only ~19 subjects, which is UL-DD's actual
regime. Kavipriya's own results (0.545 acc / 0.444 macro-F1 for M5, the best model) are much lower
than these headline numbers precisely *because* of that harder, more realistic protocol — this is
consistent with the general pattern in the literature (see also the dataset paper's own
Kruskal-Wallis findings that graded, self-reported drowsiness is intrinsically noisier than
binary/acted labels). **Any target this project sets should be benchmarked against Kavipriya's
M5 (0.444 macro-F1) and the dataset paper's 84% acc/whatever macro-F1 that corresponds to under
the *same* protocol — not against the 95–99% numbers above, which are not comparable tasks.**

### 3.3 Transferable architectural techniques identified

1. **Cross-modal attention fusion** (query/key/value across modality streams) — used in
   STFTransNet, the EEG+ECG+video Sensor Fusion Transformer, and already attempted (with mixed
   results) in Kavipriya's M3/M6. The literature consistently frames cross-attention as
   outperforming naive concatenation (STFTransNet's own ablation: cross-attention beats
   concatenation baseline by only ~0.24% in their setting — a reminder that attention fusion is
   not a silver bullet by itself; it needs the right auxiliary signal quality and calibration,
   which is precisely where Kavipriya's M6 broke down).
2. **Modality-gating for missing/unreliable channels** — the Sensor Fusion Transformer's gate
   that down-weights degraded sensors in real time is a direct, practical fit for UL-DD's B/I/M
   video gap and any per-window signal dropouts.
3. **Multi-task learning with a shared backbone** — "Multi-Task Learning for Fatigue Detection and
   Face Recognition of Drivers via Tree-Style Space-Channel Attention Fusion Network"
   (arXiv:2405.07845, Qu, Gao et al., 2024) uses a shared trunk with task-specific branches
   appended deeper in the network, trained via alternating updates + gradient accumulation to
   cope with the fact that no single dataset has labels for all tasks — a pattern directly
   reusable if this project wants to jointly predict e.g. KSS class *and* an auxiliary task
   (subject ID, session type, or a continuous KSS regression head) without needing extra labels.
4. **GAN-based augmentation for class-imbalanced/insufficient data** — "A Generative Adaptive
   Convolutional Neural Network with Attention Mechanism for Driver Fatigue Detection with
   Class-Imbalanced and Insufficient Data" (He, Zhang, Sun, Lin — *Computer Methods and Programs
   in Biomedicine* or similar venue per ScienceDirect/PubMed, 2024 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/38382711/))
   proposes ARMFCN-LSTM plus a WGAN-GP-based generative variant (GARMFCN-LSTM) specifically to
   handle the class-imbalanced, small-sample regime — directly relevant to UL-DD's small N and
   its known class-imbalance-by-fold issue (see `report_summary.md` §4, why macro-F1 was chosen
   over accuracy).
5. **Manifold-level augmentation for calibration-free / subject-independent classification** —
   "Calibration-Free Driver Drowsiness Classification based on Manifold-Level Augmentation"
   (arXiv:2212.13887) targets exactly the small-N, no-per-subject-calibration setting that UL-DD's
   subject-independent protocol requires; full text wasn't extractable in this pass but the
   framing (data augmentation performed in a learned manifold/embedding space, not raw signal
   space, specifically to improve generalization to unseen subjects) is worth a follow-up read
   before finalizing an augmentation strategy.
6. **3D-landmark-based, viewpoint-robust eye-openness metric (Eyelid Angle)** — from the UL-DD-
   citing "Blinking Beyond EAR" paper (§2 above); since UL-DD ships Dlib 68-point 2D landmarks
   (FL.csv) rather than 3D landmarks, this technique would need adaptation (MediaPipe re-extraction
   for 3D, or a 2D approximation), but the underlying idea — a geometrically stable drowsiness-
   correlated feature more robust than EAR — is directly usable as an engineered feature to
   augment or replace FAU inputs to the facial branch.

---

## 4. Recommended implementation target

### Recommendation: **STFTransNet** (Kim & Choi, *Sensors* 25(18):5819, 2025)

**Why this one, against the three stated criteria:**

- **(a) Relevance to video+telemetry fusion:** Not a video+CAN paper (it fuses two *visual*
  streams), but it is the single most directly *transplantable* fusion architecture found for
  this project's actual problem shape — two parallel input streams of differing character (one
  visually rich/high-dimensional, one lower-dimensional/behavioral) that need to be fused via
  cross-attention and then modeled temporally. The adaptation required is well-defined: replace
  its second visual stream (the cropped-face RCN branch) with a **telemetry encoder** (pitch,
  roll, speed, rpm, gear — UL-DD's actual telemetry columns) producing an embedding of the same
  shape, and keep the rest of the pipeline unchanged. This is explicitly stated as a broadened
  "closest analog" recommendation per the task brief's own fallback instruction (§1.4), since no
  literature paper does drowsiness video+CAN fusion directly.
- **(b) Reproducible method:** Full architectural spec is available (see below) — pipeline stages,
  exact cross-attention formulation, TCN dilation structure, training hyperparameters, and a
  component-by-component ablation table, which is unusually complete for a paper found in this
  niche.
- **(c) Feasible for UL-DD's small-data / subject-independent constraints:** The base
  architecture (~31.5M params, ResNet18+CBAM backbones, 3-layer TCN) is modest by modern
  video-transformer standards and trainable from a pretrained ResNet18 backbone rather than
  requiring large-scale pretraining — realistic on a local machine or Colab, per this project's
  compute constraints (`CLAUDE.md` "Known open questions"). Caution: its published results (95%+)
  come from *much larger, class-balanced, non-subject-independent* datasets (NTHU-DDD 36 subjects
  binary/4-class, StateFarm, YawDD) — direct transplant to UL-DD's 19-subject, 3-class,
  subject-independent, small-N regime should be expected to land far closer to Kavipriya's
  0.444 macro-F1 ceiling than to STFTransNet's own 95%+ numbers. Treat the architecture, not the
  accuracy, as the transferable asset.

### Architecture, adapted for this project (spec detailed enough to implement)

Original STFTransNet pipeline (Sensors 2025):
1. **Stream inputs**: original driver-cabin frame `I_orig` + cropped face region `I_face`
   (MediaPipe Face Mesh, 468 3D landmarks, used to locate/crop the face).
2. **RCN backbone** (ResNet18 + CBAM — channel attention via avg/max-pool, spatial attention via
   per-location weighting) applied to each stream independently → `F'_orig`, `F'_face`.
3. **Cross-attention fusion**: `F'_face` as query, `F'_orig` as key/value:
   `A = softmax(Q·Kᵀ)`, `F_O = A·V`.
4. **Temporal stage**: `F_O` reshaped to a sequence, passed through a **3-layer dilated TCN**
   (`y(t) = Σ f(i)·x(t − r·i)`) → `F_TO`.
5. **Classification head**: `F_TO` recombined with a residual spatial term from `F'_orig`,
   `Prediction = softmax(Linear(F_combined))`.
6. **Training**: AdamW, lr 1e-4, weight decay 1e-3, batch 64, 50 epochs, GeLU activations,
   10-frame input window.

**Adaptation for UL-DD (video + telemetry, not two visual streams):**

- Stream 1 (unchanged in spirit): IR video frames (UL-DD's `*_IR_*.mp4`, 60 fps) through a
  CNN/RCN backbone — can reuse Kavipriya's already-fine-tuned M5 YOLOv8n-cls backbone (or its
  penultimate-layer embeddings) as a warm-started `F'_orig`, avoiding training a visual backbone
  from scratch on only 19 subjects.
- Stream 2 (replaces the cropped-face branch): a small **CNN+LSTM telemetry encoder** — borrow the
  IIETA FCN-LSTM design from §1.1 — over UL-DD's telemetry window (pitch, roll, speed, rpm, gear
  at 60 Hz, matching `dataset_summary.md`'s CSV_Files/Telemetry schema and Kavipriya's M2/M3
  windowing convention of 240-timestep windows) to produce `F'_telemetry` in the same embedding
  dimension as `F'_orig`.
- Cross-attention: query = `F'_telemetry` (the weaker, lower-info modality queries the stronger
  visual one — this direction, weak-queries-strong, is the opposite of naive concatenation and
  is exactly the mechanism the report's own future-work section (§7, priority 3: "learned
  confidence weighting instead of naive averaging") was reaching for without naming it).
- Temporal stage: keep the dilated TCN (matches UL-DD's 240-timestep sliding windows well —
  dilated convolutions cover long receptive fields cheaply, cheaper than the BiLSTM/Transformer
  stacks Kavipriya's M6 used, which is worth trying given M6's underperformance).
- **Add the modality-gate from §1.3** (Sensor Fusion Transformer) between streams 1 and 2, so that
  for subjects B/I/M (no video at all) the model can fall back to telemetry-only inference
  instead of being excluded from those folds entirely — recovering usable training/eval data that
  Kavipriya's video-dependent models (M5–M7) implicitly discarded.

---

## 5. Proposed hybrid/ensemble extension directions

Grounded in §1–4 above, and targeted specifically at this project's two already-observed
weaknesses: **(a)** fusion underperforming the strong visual unimodal baseline (M6 problem), and
**(b)** the Alert-vs-LowVigilant boundary being the dominant error source.

### Direction 1 — Gated, asymmetric cross-attention fusion with confidence-weighted late blending

Directly implements the STFTransNet-derived architecture in §4, but explicitly targets weakness
**(a)**. The core hypothesis for *why* M6 underperformed M5 (per `report_summary.md` §4/§8: "fusion
failed because the auxiliary modality was weak... pulled the strong branch down") is a
**calibration problem, not a fundamentality problem** — this is exactly what the Sensor Fusion
Transformer's gating mechanism (§1.3) and STFTransNet's asymmetric query/key-value roles (§4) are
built to solve: let the telemetry stream *modulate* the visual stream's confidence rather than
being concatenated as an equal partner. Concretely: train the visual-only branch (M5-equivalent)
to convergence first, freeze/warm-start from it, then add a small gated telemetry-conditioning
head that only *adjusts* the visual logits (residual correction) rather than replacing them —
structurally guaranteeing the fusion model can never do worse than the visual baseline on its own
training objective, addressing M6's regression directly. This mirrors the report's own priority-3
future-work item ("M5 + CAN calibrated fusion... learned confidence weighting instead of naive
averaging") but grounds it in a named, ablated architecture (STFTransNet) rather than starting
from scratch.

### Direction 2 — Auxiliary ordinal/contrastive objective targeting the Alert/LowVigilant boundary

Directly targets weakness **(b)**. Two complementary ideas from the literature converge here:
(i) KSS is fundamentally **ordinal**, not nominal (`dataset_summary.md` §3 notes KSS is
continuous/ordinal, with `.5` interpolated values even in the label file) — yet every model
Kavipriya built (M1–M7) used a plain 3-way softmax cross-entropy, discarding the ordinal structure
entirely. An ordinal loss (e.g. CORN/CORAL-style ordinal regression heads, or a soft-label
cross-entropy that penalizes Alert↔Drowsy confusions more than Alert↔LowVigilant) would let the
model learn that the Alert/LowVigilant boundary is inherently a harder, closer decision than
Alert/Drowsy — consistent with M5's own confusion matrix (4.8% Alert↔Drowsy vs. 62.3%
Alert↔LowVigilant confusion, `report_summary.md` §4). (ii) Add a **multi-task auxiliary head**
(pattern from arXiv:2405.07845, §3.3) that jointly predicts a continuous KSS regression target
(not just the 3-class bin) alongside the classification head, sharing the same backbone — this
gives the model a finer-grained training signal near the class boundaries where the discretization
itself is the noisiest (recall the dataset paper's inter-rater agreement, §5 of
`dataset_summary.md`: κ=0.619 on 9-level KSS vs. κ=0.88 on the 3-class binning — some of the
Alert/LowVigilant "errors" may be genuine label-boundary ambiguity that a regression-aware loss
handles more gracefully than a hard 3-way classifier does).

### Direction 3 — Ensemble of modality-specialist experts with missing-modality-aware routing

A structurally different (not just architecturally tweaked) direction: instead of one fused model,
train **separate specialist models** — a visual specialist (M5-style), a telemetry specialist
(FCN-LSTM-style per §1.1), and (if budget allows revisiting biometrics later) a biometric
specialist — then combine via a **learned gating/routing network** (small MLP over lightweight
meta-features, e.g. per-window signal-quality/SNR proxies) that decides per-window how much to
trust each expert's prediction, rather than fusing raw features. This is a mixture-of-experts
framing of the same missing-modality-gate idea in §1.3/§4, but pushed to the ensemble level rather
than the feature level — directly solving both weaknesses at once: **(a)** because a weak expert
that predicts randomly still can't drag down a confident strong expert (the router learns to
down-weight it, rather than the strong branch's own features being contaminated during joint
training), and **(b)** because the router can be trained/calibrated specifically on the
Alert/LowVigilant boundary cases (e.g. weighted more heavily in the router's loss), directly
optimizing for the project's stated weakest point rather than aggregate accuracy. This also
naturally solves the B/I/M no-video-data problem: those subjects simply route to 100% telemetry
weight, with no special-casing needed elsewhere in the pipeline — the "generative adaptive CNN"
and "manifold-level augmentation" techniques from §3.3 (items 4–5) are natural additions to
strengthen the individual specialists given the small per-fold sample sizes, before the ensemble
stage combines them.

---

## 6. Summary table — this project's position relative to the literature

| | Best on UL-DD so far | Video+CAN literature ceiling | General SOTA ceiling (different task/data regime) |
|---|---|---|---|
| Accuracy | 0.545 (Kavipriya M5) / 84% (dataset paper, uses biometrics too) | none published (niche empty) | 95–99% (binary/near-binary, acted, subject-dependent) |
| Macro-F1 | 0.444 (Kavipriya M5) | n/a | rarely reported at this granularity in the 95%+ papers above |
| Task | 3-class KSS, subject-independent | n/a | mostly binary alert/drowsy or open/closed-eye |

This project's realistic target band, given the above, is **improving macro-F1 meaningfully beyond
0.444 and specifically improving Alert-vs-LowVigilant per-class F1** (currently ≈0.23 per
`report_summary.md`'s Fold-0 breakdown) under the *same* subject-independent, 3-class protocol —
not chasing the 90%+ numbers from unrelated task regimes.

---

## Sources (full list)

- Bodaghi et al., "UL-DD: A Multimodal Drowsiness Dataset Using Video, Biometric Signals, and
  Behavioral Data," arXiv:2507.13403 (2025); also *Scientific Data* (2025),
  https://www.nature.com/articles/s41597-025-06540-1
- Kavipriya Ramasamy, "Driver Drowsiness Detection Using Camera and CAN Signals," MTech thesis,
  IISc/NIQ, 2026 (`Kavipriya_report (2).html`, local)
- "An Efficient Deep Learning Model Based on Driver Behaviour Detection Within CAN-BUS Signals,"
  *Revue d'Intelligence Artificielle* (IIETA), https://iieta.org/journals/ria/paper/10.18280/ria.380106
- "A Multi-Source Fusion Approach for Driver Fatigue Detection Using Physiological Signals and
  Facial Image," *IEEE T-ITS*, 2024, https://doi.org/10.1109/TITS.2024.3420409
- "Dynamic cross-domain transfer learning for driver fatigue monitoring: multi-modal sensor fusion
  with adaptive real-time personalizations," *Scientific Reports*, 2025,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12056195/
- "A multimodal framework for fatigue driving detection via feature fusion of vision and tactile
  information," *npj Flexible Electronics*, 2026, https://www.nature.com/articles/s41528-026-00543-7
- Kim, M. & Choi, G., "STFTransNet: A Transformer Based Spatial Temporal Fusion Network for
  Enhanced Multimodal Driver Inattention State Recognition System," *Sensors* 25(18):5819, 2025,
  https://doi.org/10.3390/s25185819
- Adhikari, B., "Using Visual and Vehicular Sensors for Driver Behavior Analysis: A Survey,"
  arXiv:2308.13406 (2023)
- "Multi-modal driving fatigue detection with causal-aware fusion and Kolmogorov-Arnold network,"
  ScienceDirect, 2026 (forthcoming/very recent — flagged for follow-up),
  https://www.sciencedirect.com/science/article/abs/pii/S156625352600134X
- Lyu, J., Yuan, Z., Chen, D., "Long-term Multi-granularity Deep Framework for Driver Drowsiness
  Detection," arXiv:1801.02325 (2018)
- "Real-time driver drowsiness detection using transformer architectures: a novel deep learning
  approach," *Scientific Reports*, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12092738/
- "Multi-Attention Fusion Drowsy Driving Detection Model," arXiv:2312.17052 (2023)
- Wolter, M., Perez, J.S.B., Shan, M., "Blinking Beyond EAR: A Stable Eyelid Angle Metric for
  Driver Drowsiness Detection and Data Augmentation," IEEE IV 2026, arXiv:2511.19519
- Qu, S., Gao, Z., et al., "Multi-Task Learning for Fatigue Detection and Face Recognition of
  Drivers via Tree-Style Space-Channel Attention Fusion Network," arXiv:2405.07845 (2024)
- He, L., Zhang, L., Sun, Q., Lin, X., "A generative adaptive convolutional neural network with
  attention mechanism for driver fatigue detection with class-imbalanced and insufficient data,"
  2024, https://pubmed.ncbi.nlm.nih.gov/38382711/
- "Calibration-Free Driver Drowsiness Classification based on Manifold-Level Augmentation,"
  arXiv:2212.13887 (2022)
- "Enhancing Fatigue Detection through Heterogeneous Multi-Source Data Integration and
  Cross-Domain Modality Imputation," arXiv:2507.16859 (2025)
- "A Survey on Drowsiness Detection – Modern Applications and Methods," arXiv:2408.12990 (2024)
