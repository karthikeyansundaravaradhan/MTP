# Literature Survey: Vision-Primary, Telemetry-Assisted Driver Drowsiness Detection on UL-DD

*Prepared 2026-11-05 for the thesis report. It brings together four research passes (vision encoders, the drowsiness literature, fallback fusion, and local dataset grounding with benchmarks) into one survey. Each claim keeps its source link. Anything a research pass could not verify is marked **[unverified]** where it appears, or was left out. The companion execution plan is [`ablation_plan.md`](ablation_plan.md).*

---

## Contents

1. [Scope and framing](#1-scope-and-framing)
2. [The UL-DD dataset and every known use of it](#2-the-ul-dd-dataset-and-every-known-use-of-it)
3. [Vision-based drowsiness detection](#3-vision-based-drowsiness-detection)
4. [Vehicle-signal methods](#4-vehicle-signal-methods)
5. [Multimodal fusion under subject-independent evaluation](#5-multimodal-fusion-under-subject-independent-evaluation)
6. [Foundation vision and video encoders](#6-foundation-vision-and-video-encoders)
7. [Temporal heads](#7-temporal-heads)
8. [Fallback, gated and uncertainty-aware fusion](#8-fallback-gated-and-uncertainty-aware-fusion)
9. [Small data and subject-independent generalisation](#9-small-data-and-subject-independent-generalisation)
10. [Labels, protocol and realistic targets](#10-labels-protocol-and-realistic-targets)
11. [Compute considerations on Apple Silicon](#11-compute-considerations-on-apple-silicon)
12. [Master comparison table](#12-master-comparison-table)
13. [Gaps this thesis fills](#13-gaps-this-thesis-fills)
14. [References](#14-references)

---

## 1. Scope and framing

This thesis builds on Kavipriya Ramasamy's prior MTech work on UL-DD (models M1–M7, reproduced in this repository; see `docs/models_summary.md`). That work found that **a single-stream vision model, M5 (YOLOv8n-cls fine-tuned on 1 fps IR frames), was the best result**: subject-independent macro-F1 0.447 ± 0.071, accuracy 0.541. Every fusion attempt scored lower:

- M3, FAU + telemetry cross-attention: 0.322
- M6-Lite, YOLO embedding + telemetry concatenation: 0.303
- M6-Full, transformer cross-attention: 0.292
- M7, temporal EfficientNet-B0 + BiLSTM: 0.355

The dominant error is confusing Alert with LowVigilant. In M5's pooled per-class F1, Alert scores 0.258, LowVig 0.565 and Drowsy 0.620, and 59.6% of Alert frames are predicted as LowVig.

The survey is organised around five questions that set the design space for this thesis:

1. What does UL-DD actually contain, and what have others reported on it under honest protocols?
2. Which published vision results transfer to a naturalistic, KSS-labelled, subject-independent setting, and which are inflated by leaky splits?
3. Which modern frozen image, video and face encoders, with which temporal heads, fit an Apple M4 with 16 GB of memory and only about 290 independent labels?
4. How can a weak secondary modality (vehicle telemetry and grip pressure) be fused so that it helps mainly when vision fails, rather than diluting a strong vision model?
5. Which protocol and label choices (binary vs. 3-class, ordinal, window length, per-subject normalisation, smoothing) matter more than architecture?

---

## 2. The UL-DD dataset and every known use of it

### 2.1 The dataset paper

**Bodaghi, Hosseini, Gottumukkala, Bhupatiraju, Ahmad & Gabbouj, "UL-DD: A Multimodal Drowsiness Dataset Using Video, Biometric Signals, and Behavioral Data."**

- Preprint: arXiv:2507.13403 (July 2025), https://arxiv.org/abs/2507.13403.
- Journal version: *Scientific Data* (2026), "A multimodal drowsiness dataset using video, biometric, and behavioral data", doi:10.1038/s41597-025-06540-1, https://www.nature.com/articles/s41597-025-06540-1, full text at PMC13039290, https://pmc.ncbi.nlm.nih.gov/articles/PMC13039290/.
- Data: Zenodo record 17978727, released 18 Dec 2025.
- Affiliation: **University of Louisiana at Lafayette**, not Limerick, with co-affiliations at Tietoevry and Tampere University.

**Recording setup.** There are 19 subjects (A–S), each recorded in an Alert session and a Drowsy session of about 40 minutes in a driving simulator.

- Video: frontal "IR" 240×220, ZED 2 stereo left/right (L3D/R3D) 440×370, and side-view "Pose" 620×470, all at 60 fps.
- Pre-extracted features: 68 Dlib facial landmarks (FL), 30 facial action units (FAU), and 33 MediaPipe pose joints (PL).
- Biometrics: HR, EDA, BVP, TEMP, IBI, SpO2.
- Vehicle signals: telemetry at 60 Hz (heading, pitch, roll, speed, rpm, gear) and left/right grip pressure (LGP/RGP) at 3 Hz.
- Labels: one KSS value per 4-minute epoch.

**Technical validation, as reported.**

- **Labels**: Alert KSS < 4, Medium 4–6, High > 6. This project uses the same bins.
- **Pre-processing**: all modalities are resampled to 4 Hz, gaps are filled by linear interpolation plus forward fill, then MinMax scaling and PCA keeping 90% of variance. **No windowing is described.**
- **Protocol**: 5-fold CV "stratified to maintain the distribution of drowsiness levels across subjects". The paper never states that subjects are held out.
- **Results** (3-class, accuracy):

| Model | Inputs | Accuracy |
|---|---|---|
| SVM | biometric + facial | 73.06% (F1 72.43) |
| SVM | biometric + behavioural | 79.87% (F1 80.04) |
| SVM | all three | 83.75% (F1 83.95) |
| Random forest | all three | 75.14% |
| Early-fusion Transformer | all three | 86.39% (F1 86.61) |
| I2M2 product of experts | all three | 88.03% (F1 88.24) |

- **Unimodal results** appear only in a figure. They include telemetry at about 40% (SVM) and 45% (RF), grip at about 54% (RF), and "Pose" at 65–68%.
- **No video model is benchmarked.** "Facial" means the extracted FAU and landmark features.

**Other facts in the paper that matter for modelling.**

- **KSS inter-rater reliability**: unweighted κ = 0.619 on the 9-level scale, quadratic-weighted κ = 0.967, and κ = 0.88 on the 3-level scale. Experts annotated 30% of the data.
- **Grip pressure has the lowest SNR**: 14.68 dB on average, with 20% of recordings below 10 dB.
- **Biometric separation is weak**: BVP does not differ significantly across levels, and HR does not separate Medium from High (p = 0.2).
- **Known data gaps**: no video for B, I and M; no Drowsy session for C, F and L; N's drowsy session runs 39:20. Telemetry is also missing for subject A.

### 2.2 Independent re-analysis: `ildahun/Driver-Drowsiness-Detection`

**ildahun, *Multimodal driver fatigue detection on UL-DD*** (Greek MSc thesis code, c. 2025/26). https://github.com/ildahun/Driver-Drowsiness-Detection

This is the most directly comparable subject-independent benchmark on UL-DD. It uses tabular features only.

**Features.**
- **Windows**: 30 s non-overlapping, 8 per KSS epoch, each inheriting the epoch label.
- **Biometrics**: mean, std, min, max and slope.
- **Telemetry**: mean, std and std of differences. Heading is converted to a rate.
- **Grip**: mean, std and left/right asymmetry, with zero readings masked.
- **Face**: EAR, PERCLOS (EAR < 0.20), blink count and AU means.
- **Quality gates**: a modality branch is set to NaN when fewer than 50% of its samples are valid.

**Findings.**

1. **The dataset paper's protocol leaks.** Re-implementing the paper's random forest on all modalities gives **75.2% on a random split but 29.5–33.8% on a per-driver split.**
2. **Calibration-free versus personalised LOSO.** A unified LOSO model reaches macro-F1 **0.362 without calibration**. With per-driver personalisation it reaches **0.626**, and 18 of 19 drivers improve (p < 0.001). Adding all sensors, realistic faults and time features gives 0.651.
3. **How personalisation works.**
   - (a) An unlabelled step subtracts the per-subject feature mean of the first *k* minutes (2, 4 or 8) of that subject's **Alert session**.
   - (b) A labelled step fine-tunes an MLP head with the encoder frozen, using the first *k* minutes of each drive plus replay of other subjects' data. Test windows always come after the calibration period.
   - Using the same drive as the normalisation reference, instead of the Alert session, gives only 0.351.
4. **Ablations** (5 seeds, Holm-corrected):
   - Removing personalisation costs 17–26 points.
   - With 1-minute context and the face branch missing, the score falls to 0.389.
   - Removing the time feature costs 1–2 points.
5. **Attribution**:
   - SHAP gives face 52%, biometrics 33% and vehicle 15%.
   - Within vehicle signals, grip carries 66%.
   - Time-in-drive alone carries 50% of attributions.
   - A time-only model scores 0.295, and skin temperature alone scores 0.415.

Calibration-free, subject-independent performance on tabular features is therefore about 0.36, the same range as Kavipriya's M1–M4 (0.26–0.36). **M5's 0.447 on raw IR frames sits above it.** The time-only and temperature-only results show that the *session clock* is a real confound.

### 2.3 Other UL-DD-related items

- **Bodaghi, *Adaptive Attention-Based LSTM Framework for Multimodal Driver Drowsiness Prediction*** (MSc thesis, UL Lafayette, 2024). Cited by the dataset paper; the full text was not found.
- **Bodaghi, Hosseini & Gottumukkala, *A Multimodal Intermediate Fusion Network with Manifold Learning for Stress Detection*** (IEEE ICMI 2024). Same group, but about stress, not drowsiness.
- **Wolter et al., *Blinking Beyond EAR*** (arXiv:2511.19519, 2025), https://arxiv.org/abs/2511.19519. Cites UL-DD but runs its experiments on DMD and UTA-RLDD (see §3.2).
- The Google Scholar "cited by" list could not be queried. No peer-reviewed paper reporting subject-independent *video* results on UL-DD was found, so **this thesis is very likely the first**.

### 2.4 What this project verified locally (2026-11-05)

These observations change several modelling assumptions.

1. **L3D and R3D are not depth maps.** They are the left and right RGB views of the ZED 2 stereo camera: the same frontal upper-body scene with a small horizontal offset. The release contains no disparity or depth stream. Getting "depth" would need either stereo matching, with calibration that was not released, or monocular depth estimation.
   - `docs/dataset_summary.md` line 87, which describes them as "3D depth info", needs correcting.
2. **The "IR" video is a low-resolution frontal view, not a tight face crop.** It is 240×220, low in colour saturation (mean HSV S ≈ 19 against ≈ 113 for L3D) but still 3-channel. The head fills only about 80–100 px.
   - L3D shows the same frontal view in colour at roughly twice the face resolution.
   - R3D shows occasional hand occlusion and glare on glasses.
3. **The "Pose" video is a side view** showing the full body, the hands on the wheel and the face in profile.
4. **The label budget is tiny.**
   - `Labels.csv` has 35 sessions × 10 epochs = 350 KSS labels.
   - Only **290 epochs** belong to the 16 subjects with video.
   - The ~4.4–5.4k windows (60 s long, 15 s stride, about 16 per epoch) are heavily overlapping views of these labels.
   - KSS histogram over all epochs: 1:21, 2:30, 3:41, 4:40, 4.5:1, 5:36, 6:49, 6.5:1, 7:54, 7.5:3, 8:56, 9:18.
5. **Time-on-task confound.** Mean KSS rises steadily within each session: from 2.53 to 4.55 across Alert-session epochs, and from 6.19 to 8.28 across Drowsy-session epochs. 51.6% of Alert-session epochs are not 3-class "Alert".
6. **Fold-level class scarcity.** With video subjects only, Alert epochs per fold are 4, 21, 3, 9 and 34 (folds 0–4). This explains the Alert F1 of 0.000 in some M5 and M7 folds.
7. **The provided FAU csv is derived from FL**, so it is geometric rather than appearance-based. This partly explains why M1 and M4 are weak.
8. **60 fps is very redundant.** Adjacent-frame embedding cosine similarity is 0.9996.
9. **FL/FAU are missing for about 2.4% of frames** in the session checked (E_A), where no face was detected. PL is complete. PL's header lists all x columns, then all y, then all z, but the values appear interleaved as (x, y, z) triplets, so the layout must be checked before use.
10. **The existing 60 fps `embeddings_uldd` cache leaks the test subjects for folds 1–4.** It was produced by the fold-0 M5 backbone, which was trained on E–S.

---

## 3. Vision-based drowsiness detection

### 3.1 Why most headline numbers are not comparable

Most reported vision accuracies of 95–99.6% come from random frame- or clip-level splits. That lets the same subject appear in both training and test data. Many also use NTHU-DDD, whose drowsiness is *acted* and labelled in binary.

- **Hassan et al. 2025**, *Real-time driver drowsiness detection using transformer architectures*, *Sci. Rep.* 15:17493, https://pmc.ncbi.nlm.nih.gov/articles/PMC12092738. ViT-B/16 reaches 99.52% and Swin-T 98.76% on NTHU-DDD, using an image-level 80/20 split with no cross-subject test.
- **Light-VTD** (2026), *Sci. Rep.* 16:28136, https://pmc.ncbi.nlm.nih.gov/articles/PMC13554065/. A MobileOne-style stem with Swin stages, a design that suits a Mac. It reports 95–98.7% within datasets on stratified random 80/5/15 splits, and 90.3–94.5% across datasets, but those datasets are mostly eye-state or acted.
- **STFTransNet** (Kim & Choi, *Sensors* 25(18):5819, 2025), https://pmc.ncbi.nlm.nih.gov/articles/PMC12473511/. A two-stream ResNet18 with CBAM, then cross-attention, then a TCN. It reports 95.86% on NTHU-DDD with shared subjects. Its own ablation shows **cross-attention adds nothing** over a simple ensemble (93.76 vs. 93.94).
- **Krishna et al. 2022**, *Vision Transformers and YoloV5 based Driver Drowsiness Detection Framework*, arXiv:2209.01401, https://arxiv.org/abs/2209.01401. A binary ViT on UTA-RLDD reaches 97.4% validation accuracy, and 95.5% on a custom 39-person set.
- **Zaman et al. 2025**, arXiv:2511.12438, https://arxiv.org/abs/2511.12438. 99.6% on NTHU-DDD.
- **Fu et al. 2024 survey**, arXiv:2408.12990, https://arxiv.org/html/2408.12990v1. Collects NTHU-DDD accuracies of 85.6–97.1% and RLDD results up to 97.5%, mostly without stated subject independence. The authors recommend balanced accuracy.

### 3.2 Subject-independent vision results

**NTHU-DDD** (acted drowsiness, binary labels; official held-out evaluation drivers).

- **Lyu et al. 2018**, multi-granularity CNN + LSTM, arXiv:1801.02325, https://arxiv.org/abs/1801.02325: 90.05%.
- **Shen et al. 2020**, two-stream multi-feature network with eye, mouth and head patches plus optical flow, arXiv:2010.06235, https://arxiv.org/abs/2010.06235: 94.46%.
- **Tüfekci et al. 2022** (ECCV-W), arXiv:2209.05269, https://arxiv.org/abs/2209.05269. Treats drowsiness as an anomaly using an LSTM autoencoder trained on normal driving, with a 12/6/4 driver split. AUC 0.874, about 81.6% accuracy. **CLAHE contrast enhancement added more than 5 AUC points.**
- **Lakhani 2022**, *Spatiotemporal Attention with Vision Transformers*, arXiv:2207.12148, https://arxiv.org/abs/2207.12148. A Video Swin model reached **only 44% test accuracy**, which the author attributes to overfitting on a small dataset. This is a direct warning against end-to-end video transformers on small data.
- **Adaptive cross-modal fusion with temporal modelling** (ScienceDirect 2026, abstract only), https://www.sciencedirect.com/science/article/pii/S2666521226001481. Bidirectional gated cross-attention followed by a Transformer. Its 91.51% headline falls to **76.35% under leakage-free LOSO** and to **55.93% F1 zero-shot across datasets**. A plain BiLSTM-attention concatenation baseline stays competitive.
- **A Subject-Independent Temporal Framework for Behavioural Eye-Based Driver Drowsiness Detection**, *Sensors* 26(19):6193 (2026), https://www.mdpi.com/1424-8220/26/19/6193. MediaPipe EAR features with leave-one-driver-out over four NTHU drivers. Mean accuracy is about 0.556 against an always-drowsy baseline of about 0.530, and drowsy-class F1 is about 0.617. These figures come from search snippets because the page returned HTTP 403.

**UTA-RLDD** (naturalistic, KSS-based, three classes: alert, low-vigilant and drowsy; 60 subjects; official subject-wise folds). This is the closest public analogue to UL-DD.

- **Ghoddoosian et al. 2019**, *A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection* (CVPRW), arXiv:1904.07312, https://arxiv.org/abs/1904.07312. Blink features fed to an HM-LSTM reach **65.2% on three classes** (as cited by later papers). **Human judges scored under 60%** on the same task.
- **Wolter et al. 2025**, *Blinking Beyond EAR*, arXiv:2511.19519. Blink-phase features from the eyelid angle (ELA) give **52.5% on three classes and 80.4% binary**. The jump of about 28 points from three classes to two is the clearest published evidence that the **Alert vs. Low-Vigilant boundary is the hard part**, which is exactly the confusion M5 shows.
- **Magán et al. 2022**, *Appl. Sci.* 12(3):1145. A CNN+RNN on 60 s image sequences; about 60% test accuracy according to secondary sources **[unverified exact figure]**.
- **GitHub projects (anecdotal only).**
  - `Arnav-Dugad/Driver-Detection-System` reports 53.3% three-class video accuracy on the official folds.
  - `RuchirBhatia4/DriveMind` reports 44–56% for held-out people using per-minute blink-feature logistic regression. It also notes that **2 of 12 drowsy drivers barely closed their eyes**, which caps how far eye-only cues can go.

**NIR video, LOSO, objective labels.**

- **Massoz et al. 2018**, *Multi-Timescale Drowsiness Characterization Based on a Video of a Driver's Face*, *Sensors* 18(9):2801, https://pmc.ncbi.nlm.nih.gov/articles/PMC6165048/. Kinect v2 NIR video, 29 subjects, LOSO, binary labels derived from PVT reaction times.
  - Accuracy rises with window length: **70.7% at 5 s, 85.5% at 15 s, 89.8% at 30 s, 94.2% at 60 s.**
  - Sharing multi-timescale context lifted 5 s accuracy from 61.9% to 70.7%.
  - This is the most IR-relevant paper found, and the strongest evidence that longer temporal aggregation matters.
- **Vural et al. 2007**, *Drowsy Driver Detection Through Facial Movement Analysis* (ICCV-HCI workshop, LNCS 4796), https://inc.ucsd.edu/~marni/pubs/Vural_ICCV07.pdf. Thirty automatic AU detectors predicted crash and sleep episodes at 96% within subjects and above 90% across subjects (98% across subjects in the later ICAT 2008 version). The episodes were binary, extreme and event-locked rather than KSS-labelled.

**Takeaway.** Naturalistic, three-class, KSS-labelled, subject-independent tasks give about **50–65% accuracy**. Acted or binary NTHU-style tasks give 75–94%. UL-DD belongs to the first group, so M5's 0.447 macro-F1 and 0.541 accuracy are in line with the literature.

### 3.3 Hand-crafted ocular, mouth and head cues

- **PERCLOS** is the most validated single measure. In the FHWA/NHTSA lab validation it was the most reliable predictor of PVT lapses (https://rosap.ntl.bts.gov/view/dot/113). One camera study reports a PERCLOS–KSS correlation of ρ = 0.74 (Friedrichs & Yang, IEEE IV 2010) **[unverified context, snippet only]**.
- **Blink duration vs. PERCLOS.** Reviews disagree on which predicts better (ResearchGate review, "PERCLOS-based technologies…", 2023).
- **Eyelid angle (ELA)** from 3D landmarks (Wolter et al. 2025) is more viewpoint-stable than 2D EAR. It yields blink-phase durations (closing, closed, reopening), amplitude and PEROPENING. UL-DD's 68-point Dlib landmarks are 2D; MediaPipe could supply 3D landmarks.
- **Personalised EAR/MAR thresholds** add about 2–3% (arXiv:2604.22479, 2026, https://arxiv.org/abs/2604.22479).
- **Body posture.** Validated side-view work barely exists. A graph-based study of train drivers combines facial and skeletal features (arXiv:2505.08800, https://arxiv.org/abs/2505.08800); its results were not read. UL-DD's Pose video and PL joints are therefore a new direction with an uncertain payoff. The dataset paper's own pose baseline (65–68%, under a non-subject-independent protocol) is its strongest non-biometric unimodal result.

### 3.4 Remote photoplethysmography (rPPG) from IR video

- **MERL TR2020-161**, *Near-Infrared Imaging Photoplethysmography During Driving*, https://www.merl.com/publications/docs/TR2020-161.pdf. A narrow-band NIR camera recovers heart rate reliably under changing light and motion.
- **Kong et al. 2024**, RPPMT-CNN-BiLSTM, *Sensors* 24(2):455, https://pmc.ncbi.nlm.nih.gov/articles/PMC11154312/. Combines IR rPPG HRV (SDNN, LF/HF), PERCLOS and yawn rate.
  - Adding rPPG raises accuracy from 0.948 to 0.972.
  - rPPG-derived HRV scores 0.895, against 0.894 for ECG-derived HRV.
  - Labels are binary KSS, and the paper does not say whether the protocol is subject-independent.
- **Tools and datasets**: rPPG-Toolbox (arXiv:2210.00716, NeurIPS 2023 D&B, https://arxiv.org/abs/2210.00716); PhysDrive (arXiv:2507.19172, https://arxiv.org/abs/2507.19172; RGB + NIR + mmWave, 48 drivers); MS-rPPG (arXiv:2606.21115, https://arxiv.org/abs/2606.21115); VDMoE, a mixture-of-experts for driver state plus rPPG (arXiv:2410.21086, https://arxiv.org/abs/2410.21086).
- **Limits for UL-DD.** The face region is small, although 60 fps suits pulse estimation. Contact BVP/HR is available for validation. However, the dataset paper found that HR does not separate Medium from High, which caps the likely benefit.

### 3.5 Depth and multi-view in-cabin data

Only older or patent work on depth for drowsiness was found: face depth maps for mouth state, and low-cost 3D sensors for head pose.

- **DMD** (Ortega et al., ECCV-W 2020), arXiv:2008.12085, https://arxiv.org/pdf/2008.12085. RGB, depth and IR from three cameras covering the face, body and hands of 37 drivers. It is the closest public analogue to UL-DD's multi-view rig, though drowsiness is only a minor part of it.
- For UL-DD, "depth" would first have to be estimated, either by monocular depth (§6.4) or by stereo matching. Because no published evidence exists for it, it is best treated as an exploratory ablation.

---

## 4. Vehicle-signal methods

- **Standard indicators** are the standard deviation of lane position (SDLP), steering-wheel angle variability, steering reversal rate and speed variability (survey: arXiv:2110.00472, https://arxiv.org/abs/2110.00472). Reversal rate is very sensitive to the reversal threshold (patent US8519853). **UL-DD has neither steering angle nor lane position.**
- **Steering entropy.** Li et al. 2017, *Online detection of driver fatigue using steering-wheel angles*, *Sensors* 17(3):495, https://pmc.ncbi.nlm.nih.gov/articles/PMC5375781/. Approximate entropy of the steering angle, combined with DTW against an online-learned "awake" reference, reaches **78.01%** on real roads (6 drivers, 1-minute samples, binary expert labels). A lab study they cite reached 86.1%.
- **Heading and yaw as a steering proxy.**
  - A study titled "Lane heading difference…" found the standard deviation of yaw angle to be the best indicator, scoring 0.72 on unseen data (ResearchGate 275363392; snippet only).
  - Patent US9205844 detects drowsiness from a frequency component of the yaw rate.
  - Patent US4594583 detects an abrupt steering correction after a period with no steering.
  - **For UL-DD**: the heading channel, dropped by every model so far, is the closest steering proxy. Unwrapping and differentiating it gives heading rate, from which its std, sample entropy, reversal rate and "quiet-then-jerk" events can be computed.
- **Grip force.**
  - Grip force falls as subjective drowsiness rises (Chung-Ang FSR study; *Detecting Driver Fatigue by Steering Wheel Grip Force*, Int. J. Contents 2016).
  - A 2021 on-road study of 36 drivers found that grip-force variation correlates with fatigue (ScienceDirect S0169814121000019; venue unverified).
  - In UL-DD, grip is the noisiest channel, yet it carried 66% of vehicle-signal importance in the ildahun SHAP analysis.
- **Vehicle signals vs. camera.** *The detection of drowsiness using a driver monitoring system*, Traffic Injury Prevention 2019, doi:10.1080/15389588.2019.1622005 (abstract only). Vehicle-only random forests did worse than camera features, and the authors recommend combining the two. This matches M2's score of 0.259.

**Implication.** Telemetry in UL-DD is expected to be a weak, partly time-confounded signal. Its value for the thesis is likely robustness when vision degrades or is missing, not higher F1 on clean data.

---

## 5. Multimodal fusion under subject-independent evaluation

### 5.1 Reported fusion gains are mostly leaky

- **Dual-sensing** (arXiv:2505.17392, https://arxiv.org/abs/2505.17392): vision 92%, physiology 94% and fusion 98%, all on a session-level split.
- **UL-DD's I2M2** result of 88% uses stratified folds.

Neither shows a gain across subjects.

### 5.2 Evidence under LOSO

- **Subject-independent multimodal drowsiness detection using behavioural and physiological features** (ScienceDirect 2026, abstract and snippets), https://www.sciencedirect.com/science/article/abs/pii/S174680942601061X. HRV plus OpenPose EAR, 12 participants.
  - LOSO F1 is **0.527 for binary KSS and 0.388 for 3-class KSS**, against 0.836 with observer-rated labels.
  - The gap between LOSO and 10-fold CV reaches **+0.566 F1** on 3-class KSS.
  - Agreement between KSS and observer ratings is only κ = 0.274.
- **Cao et al. 2025**, *Sci. Rep.* 15:12240, https://pmc.ncbi.nlm.nih.gov/articles/PMC11985911/. DROZY NIR plus polysomnography, ResNet18 + LSTM, leave-one-subject-out over 10 runs: 98.41%. The model depends on EEG, and window-level redundancy is suspected but not verified, so it does not transfer.
- **An explainable multimodal study** (ScienceDirect S0169260723005916; snippet): 80.1% leave-one-participant-out accuracy with a random forest.
- **ildahun**: fusion drops to 0.389 when the face branch is missing. Its reliability-weighted voting was not confirmed in the 5-seed ablation.

### 5.3 Why naive fusion underperforms the best single modality

- **Wang, Tran & Feiszli, *What Makes Training Multi-modal Classification Networks Hard?*** (CVPR 2020), arXiv:1905.12681, https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf. Multimodal networks overfit faster than unimodal ones, and the best single modality often beats them. **Gradient-Blending**, which weights auxiliary unimodal losses by their overfitting-to-generalisation ratio, gains 3.2% over the best unimodal model.
- **OGM-GE** (Peng et al., CVPR 2022 oral), arXiv:2203.15332, https://github.com/GeWu-Lab/OGM-GE_CVPR2022; journal extension arXiv:2410.11582. The dominant modality starves the others, and on-the-fly gradient modulation with Gaussian noise counteracts this.
- **Greedy learner hypothesis** (Wu et al., ICML 2022), arXiv:2202.05306, https://proceedings.mlr.press/v162/wu22d.html. Multimodal networks rely on one modality and underfit the rest.
- **Modality competition** (Huang et al., ICML 2022), arXiv:2203.12221, https://proceedings.mlr.press/v162/huang22e.html. In jointly trained late fusion, only a subset of modalities gets learned.
- **Uni-modal feature learning** (Du et al., ICML 2023), arXiv:2305.01233, https://mlanthology.org/icml/2023/du2023icml-unimodal/. Joint training under-learns unimodal features. When paired cross-modal features are weak, a plain uni-modal ensemble is the better choice.
- **PMR** (Fan et al., CVPR 2023), arXiv:2211.07089, https://arxiv.org/abs/2211.07089: prototype-based rebalancing. **MMPareto** (Wei & Hu, ICML 2024), arXiv:2405.17730, https://arxiv.org/abs/2405.17730: Pareto-style combination of the multimodal and unimodal gradients.
- **Ma et al., *Are Multimodal Transformers Robust to Missing Modality?*** (CVPR 2022), arXiv:2204.05454, https://arxiv.org/abs/2204.05454. The fusion strategy decides robustness, and a multimodal model can fall below a unimodal one when a modality is missing.
- **Survey**: Zhang, Wei et al., *Multimodal Fusion on Low-quality Data: A Comprehensive Survey*, arXiv:2404.18947, https://arxiv.org/abs/2404.18947.

### 5.4 Reading M3 and M6 through this lens

M3 and M6 fused telemetry *symmetrically* and trained everything jointly. M6 also fused M5's *embeddings* rather than its calibrated *decision*, so the fusion head had to relearn the vision decision boundary from about 12 training subjects. The literature above predicts this outcome. The failure mode is also specific: OGM-GE-style rebalancing *amplifies* the weak modality, which is the wrong remedy when telemetry carries little subject-independent signal. **For telemetry, the design requirement is "do no harm"**: the fused model should fall back to the vision model exactly whenever telemetry adds nothing. Rebalancing methods remain useful when fusing *several vision streams* (IR, stereo views, pose), where IR is likely to dominate and the posture stream to be under-learned.

---

## 6. Foundation vision and video encoders

### 6.1 Frozen image encoders

| Encoder | Small variant | Licence (as found) | Notes |
|---|---|---|---|
| **DINOv2** (Oquab et al., TMLR 2024), arXiv:2304.07193, https://arxiv.org/abs/2304.07193; repo https://github.com/facebookresearch/dinov2; timm https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m | ViT-S/14, 21M params, dim 384 (also `_reg` register variants) | Apache-2.0 | Strongest track record for frozen features without fine-tuning. Ungated. Measured on this M4: 185 img/s at 224 px fp16 (88.6 when throttled) and 712 img/s at 112 px. |
| **DINOv3** (2025), arXiv:2508.10104, https://arxiv.org/pdf/2508.10104; repo https://github.com/facebookresearch/dinov3; HF https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m; licence https://ai.meta.com/resources/models-and-libraries/dinov3-license/ | ViT-S/16, 21M, dim 384, 4 registers; S+ 29M; ConvNeXt-T 29M | Custom DINOv3 licence (attribution required); HF weights gated | The natural "newer encoder" ablation at the same cost. Needs transformers ≥ 4.56 or timm ≥ 1.0.20. |
| **SigLIP 2** (2025), arXiv:2502.14786, https://arxiv.org/abs/2502.14786 | Smallest is B/16 (86M) | Apache-2.0 **[unverified for weights]** | Language-supervised contrast to the self-supervised DINO models; about 3× the cost of ViT-S. |
| **MobileCLIP** arXiv:2311.17049, https://arxiv.org/abs/2311.17049; **MobileCLIP2** arXiv:2508.20691, https://arxiv.org/abs/2508.20691 | S0–S2, B | Apple licence **[unverified]** | CLIP-style distillation; latency-oriented. |
| **EVA-02** (timm cards: https://huggingface.co/timm/eva02_small_patch14_224.mim_in22k, https://huggingface.co/timm/eva02_tiny_patch14_224.mim_in22k) | Ti 5.5M; S 21.6M, dim 384 | MIT (timm card) | MIM with an EVA-CLIP teacher; same cost as ViT-S. |
| **MAE** (He et al., CVPR 2022), arXiv:2111.06377, https://arxiv.org/abs/2111.06377 | ViT-B | — | Features are "less linearly separable"; linear-probe and fine-tune accuracy are "largely uncorrelated". Needs a non-linear head. |
| **FastViT** (ICCV 2023), arXiv:2303.14189, https://arxiv.org/abs/2303.14189; **MobileViT** (ICLR 2022), arXiv:2110.02178, https://arxiv.org/abs/2110.02178; **EfficientViT**, arXiv:2205.14756, https://arxiv.org/abs/2205.14756 | 2–30M | Various | Latency-optimised supervised backbones. Good for a CPU demo, not where accuracy gains are expected. |

Domain prior art:

- **CT-Net** (Gao et al., *Sensors* 24(24):7948, 2024), https://pmc.ncbi.nlm.nih.gov/articles/PMC11679248/. A fine-tuned CLIP-ResNet50 with a Transformer temporal encoder reaches AUC 0.892, against 0.836 for an ImageNet encoder with the same Transformer and 0.858 for CLIP+LSTM. It used 25k fleet videos on a random split. This suggests that foundation features plus a transformer head beat ImageNet+LSTM, but the number is not comparable.
- The **TelUP drowsiness** repository uses frozen DINOv2-L embeddings with a subject-disjoint split (https://github.com/agungibr/telup-drowsiness); its numbers were not verified.
- **No peer-reviewed application of DINOv2 to drowsiness was found.**

### 6.2 Video transformers and video foundation models

| Model | Smallest usable variant | Licence | Assessment |
|---|---|---|---|
| **VideoMAE** (Tong et al., NeurIPS 2022), arXiv:2203.12602, https://arxiv.org/abs/2203.12602; HF https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics | ViT-S, 16×224² clips (1568 tokens) | CC BY-NC 4.0 | Data-efficient on 3–4k videos; authors stress that pre-training/target domain shift matters. ~0.1 s per clip estimated. |
| **VideoMAE V2** (CVPR 2023), arXiv:2303.16727, https://github.com/OpenGVLab/VideoMAEv2 | Distilled ViT-S (K400 83.7) | Repo MIT | Same cost as VideoMAE-S, stronger features. |
| **InternVideo2** (ECCV 2024), arXiv:2403.15377, https://arxiv.org/pdf/2403.15377; zoo https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/single_modality/MODEL_ZOO.md | dist-S/14 (K400 85.4, SSv2 71.6) | Not stated | Cheap, but needs a custom loader. |
| **V-JEPA 2** (2025), arXiv:2506.09985, https://arxiv.org/abs/2506.09985; repo https://github.com/facebookresearch/vjepa2; HF https://huggingface.co/facebook/vjepa2-vitl-fpc64-256; V-JEPA 2.1, arXiv:2603.14482, https://www.alphaxiv.org/abs/2603.14482v1 | ViT-L/16 (300M); 2.1 adds ViT-B/16 (80M, not on HF) | MIT | Feasible only frozen, at 16 frames, overnight (~1.1–1.8 s per clip estimated). |
| **TimeSformer** (2021), arXiv:2102.05095, https://arxiv.org/abs/2102.05095 | ViT-B, divided space-time attention | — | Mainly a design template now. |
| **ViViT** (ICCV 2021), arXiv:2103.15691, https://arxiv.org/abs/2103.15691 | Factorised encoder | — | "Model 2" (frozen spatial encoder + separate temporal encoder) is the design pattern this thesis adopts. |
| **Video Swin** (CVPR 2022), arXiv:2106.13230, https://arxiv.org/abs/2106.13230; torchvision https://docs.pytorch.org/vision/main/models/video_swin_transformer.html | swin3d_t | BSD | Supervised on K400, action-biased. Lakhani's 44% (§3.2) is the cautionary tale. |
| **MViTv2** (CVPR 2022), arXiv:2112.01526, https://arxiv.org/abs/2112.01526; **X3D** (CVPR 2020), arXiv:2004.04730, https://arxiv.org/abs/2004.04730, hub https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/; **UniFormerV2** (ICCV 2023), arXiv:2211.09552, https://arxiv.org/abs/2211.09552 | — | — | Supervised or heavier; low priority. X3D-XS is a cheap non-transformer video baseline. |

Kinetics-style supervised video models are biased toward actions and objects. Drowsiness cues are subtle facial dynamics: blink duration, slow eyelid closure, yawns and head nods. Self-supervised masked video models are better aligned with those cues and cheap enough to run frozen on every clip.

### 6.3 Face-specific pretrained encoders

- **MARLIN** (Cai et al., CVPR 2023), arXiv:2211.06627, https://arxiv.org/abs/2211.06627; code https://github.com/ControlNet/MARLIN.
  - A masked autoencoder trained on face *videos* (YouTube Faces), with masking guided by the eye, nose and mouth regions.
  - Variants: ViT-S (384-d), B (768-d) and L (1024-d). Checkpoints include HF `ControlNet/marlin_vit_base_ytf`. Licence CC BY-NC 4.0, which is acceptable for academic use.
  - **The best-aligned frozen encoder for facial dynamics.** As an MAE, it needs an attentive (non-linear) probe rather than a linear one.
- **FaRL** (CVPR 2022), arXiv:2112.03109, https://arxiv.org/abs/2112.03109; https://github.com/FacePerceiver/FaRL. CLIP + MIM pre-training on LAION-Face (~20M pairs), ViT-B/16. Strong low-data transfer. Licence **[unverified]**.
- **FSFM** arXiv:2412.12032, https://arxiv.org/html/2412.12032, and **FS-VFM** arXiv:2510.10663, https://arxiv.org/html/2510.10663v1. Face pre-training for security tasks. Likely to encode identity and texture more than state, so low priority.
- **EmoNet** (Toisoul et al., *Nat. Mach. Intell.* 2021), https://www.nature.com/articles/s42256-020-00280-0; https://github.com/face-analysis/emonet. Continuous valence and arousal. A per-frame *arousal* scalar is a plausible drowsiness feature. Licence CC BY-NC-ND, so it can only be used as a frozen extractor.
- **LibreFace** (WACV 2024), arXiv:2308.10713, https://arxiv.org/pdf/2308.10713. Appearance-based AU detection and intensity estimation (DISFA PCC 0.63) that runs on CPU. A drop-in replacement for UL-DD's geometric FAU csv.
- **ME-GraphAU / OpenGraphAU** (IJCAI 2022), https://www.ijcai.org/proceedings/2022/0173.pdf; https://github.com/lingjivoo/OpenGraphAU. 41 AUs. Apache-2.0.
- **MediaPipe Face Landmarker**, https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker. 478 3D points plus 52 blendshapes (eyeBlinkL/R, jawOpen and others), fast on CPU. It also provides the 3D landmarks that ELA needs.

### 6.4 Depth proxies and skeleton encoders

- **Depth Anything V2** (NeurIPS 2024), arXiv:2406.09414, https://arxiv.org/abs/2406.09414v2; Small model HF https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf. Licence Apache-2.0 (Base/Large/Giant are CC BY-NC). Its encoder is a DINOv2-S, so encoder features would duplicate the face stream. The novel signal is the output depth map: head-to-camera distance, forward lean and slump. Because relative depth is scale-ambiguous, each map should be normalised per frame.
- **FoundationStereo** (CVPR 2025), arXiv:2501.09898, https://arxiv.org/abs/2501.09898, and **Fast-FoundationStereo**, arXiv:2512.11130, https://arxiv.org/pdf/2512.11130. Accurate zero-shot stereo, but heavy on MPS, and ZED calibration is not released. Not justified within the deadline.
- **PoseFormer** (2021, venue ICCV 2021 **[unverified]**), arXiv:2103.10455, https://arxiv.org/abs/2103.10455. A spatial transformer over joint tokens followed by a temporal transformer over frame tokens. This is the template for the landmark/skeleton stream.
- **MotionBERT / DSTformer** (ICCV 2023), arXiv:2210.06551, https://arxiv.org/abs/2210.06551. A pretrained motion encoder expecting 17-joint input (the MediaPipe-to-H36M joint mapping is **[unverified]**). Its pre-training domain is full-body motion, not seated micro-motion.
- **ST-GCN** (AAAI 2018), arXiv:1801.07455, https://arxiv.org/abs/1801.07455, and **CTR-GCN** (ICCV 2021), arXiv:2107.12213, https://arxiv.org/pdf/2107.12213, https://github.com/Uason-Chen/CTR-GCN. Small, mature graph-convolution baselines.

The PL and FL csvs cover **all 19 subjects**, including B, I and M, whose video was withheld. The landmark/skeleton stream is therefore the only vision-family stream with full cohort coverage.

---

## 7. Temporal heads

At 1–4 fps, a 60 s window holds 60–240 frame tokens. Attention cost is therefore trivial (N² ≤ 10⁶), and **overfitting, not efficiency, is the constraint**.

| Head | Data-regime notes |
|---|---|
| **Attention-MIL pooling** (Ilse et al., ICML 2018), arXiv:1802.04712, https://arxiv.org/abs/1802.04712 | Learned weighted average of frame embeddings (~50k params). Ignores order but is interpretable: its weights point at the frames that drive the prediction, which is useful for the demo. **The strongest baseline for ~290 labels.** |
| **Small temporal transformer** (ViViT factorised-encoder style) | 1–2 layers, d = 64–128, 2–4 heads, CLS or attention pooling, ~100–250k params; frame-drop and time-shift augmentation. |
| **TCN** (Bai et al.), arXiv:1803.01271, https://arxiv.org/abs/1803.01271 | Dilated 1-D convolutions; fast on MPS; receptive field easily set to 60 s. |
| **S4** (ICLR 2022), arXiv:2111.00396, https://arxiv.org/abs/2111.00396; **Mamba** (arXiv:2312.00752 **[unverified id]**) | Their advantage appears at 10k+ steps, which this data never reaches. Official `mamba-ssm` needs CUDA (https://github.com/state-spaces/mamba/issues/324); an unofficial macOS port claims MPS support (https://github.com/purohit10saurabh/mamba-ssm-macos). At L ≈ 120 a plain-PyTorch selective scan is cheap, so an SSM row is possible without custom kernels. |
| **Perceiver IO** (venue ICLR 2022 **[unverified]**), arXiv:2107.14795, https://ar5iv.labs.arxiv.org/html/2107.14795 | 8–16 latents cross-attend to all tokens from all streams. Cost is linear in input length, and a missing stream is handled by masking its tokens. |
| **Attention bottleneck fusion (MBT)** (Nagrani et al., NeurIPS 2021), arXiv:2107.00135, https://arxiv.org/abs/2107.00135 | Streams exchange information only through a few bottleneck tokens. Beats vanilla cross-attention by more than 2 mAP at under half the compute. A principled way to stop a weak stream from dominating. |
| **Token Merging** (ICLR 2023), arXiv:2210.09461, https://arxiv.org/abs/2210.09461 | A training-free speed-up inside the encoder; irrelevant for the heads. |

Two further points from the literature:

- **Hierarchical aggregation.** Massoz et al. (window length) and the 4-minute KSS resolution both favour aggregating 60 s window predictions over the full epoch. Treating each epoch as a bag of windows is a multiple-instance learning framing borrowed from video anomaly detection (Lv et al., CVPR 2023). No drowsiness-specific MIL paper was found.
- **Smoothing.** Temporal smoothing of predicted drowsiness scores also has precedent: MSTN, https://www.researchgate.net/publication/315162000, and Ghoddoosian's HM-LSTM.

---

## 8. Fallback, gated and uncertainty-aware fusion

This section covers the user's design intent: telemetry should be *unequal*, and should speak mainly when visual cues fail.

### 8.1 Gated residual logit correction

The vision model's calibrated logits form the prediction backbone. Telemetry adds a small gated correction, `z = z_v + g(u)·Δ_t`, where:

- the reliability vector u holds vision entropy and margin, a face-missing rate, landmark jitter, IR brightness, and telemetry availability;
- the gate's bias is initialised so that g ≈ 0.05.

This follows the **zero-init gating of Flamingo** (Alayrac et al., arXiv:2204.14198, https://ar5iv.labs.arxiv.org/html/2204.14198), where each new cross-attention block is scaled by tanh(α) with α initialised at 0, so the model starts as the pretrained one.

The critical pitfall is **cross-fitting**. If the vision model was trained on the same subjects that train the gate, its training logits are over-confident, and the gate learns that telemetry never helps. The gate must therefore be trained on *out-of-fold* vision logits from inner subject-wise splits.

### 8.2 Feature-level gates and conditioning

- **GMU** (Arevalo et al.), arXiv:1702.01992, https://arxiv.org/abs/1702.01992. Computes `h = z⊙h_v + (1−z)⊙h_t`. It can be made asymmetric by initialising z toward vision. It reintroduces joint representation learning, which is what hurt M3 and M6, so it should be used only on frozen embeddings.
- **FiLM** (Perez et al., AAAI 2018), arXiv:1709.07871, https://arxiv.org/abs/1709.07871. Telemetry predicts a zero-initialised affine transform of the vision features, acting as driving context. Conceptually attractive, but it lets the weak modality perturb every vision feature.

### 8.3 Uncertainty sources for the gate

- **Entropy weighting.** In the spirit of MLA (Zhang et al., CVPR 2024), arXiv:2311.10707, https://arxiv.org/abs/2311.10707, higher-entropy modalities get smaller fusion weights. PAMRL (Applied Sciences 2025, https://doi.org/10.3390/app15105229) points out that entropy cannot separate "confident and right" from "confident and wrong", which is a real risk for Alert/LowVig confusions.
- **Evidential deep learning** (Sensoy et al., NeurIPS 2018), arXiv:1806.01768, https://arxiv.org/abs/1806.01768. Dirichlet evidence gives a vacuity score that separates "no evidence" from "conflicting evidence". It is sensitive to the KL annealing schedule.
- **MC-dropout** (Gal & Ghahramani), arXiv:1506.02142, https://arxiv.org/abs/1506.02142, and **deep ensembles** (Lakshminarayanan et al., NeurIPS 2017), arXiv:1612.01474, https://arxiv.org/abs/1612.01474. Mutual information over passes or over fold models gives an epistemic uncertainty estimate. It is cheap when only the head is stochastic.
- **Temperature scaling** (Guo et al., ICML 2017), arXiv:1706.04599, https://arxiv.org/abs/1706.04599. A prerequisite for any fusion that consumes probabilities.

### 8.4 Quality-aware dynamic fusion

- **QMF** (Zhang et al., ICML 2023), arXiv:2306.02050, https://arxiv.org/abs/2306.02050. Proves that dynamic fusion beats static fusion when fusion weights are negatively correlated with each modality's generalisation error.
- **Predictive Dynamic Fusion** (Cao et al., ICML 2024), arXiv:2406.04802, https://arxiv.org/abs/2406.04802. Builds fusion weights from mono- and holo-confidence.
- **Multimodal Dynamics** (Han et al., CVPR 2022), https://openaccess.thecvf.com/content/CVPR2022/html/Han_Multimodal_Dynamics_Dynamical_Fusion_for_Trustworthy_Multimodal_Classification_CVPR_2022_paper.html. Uses each modality's true-class probability as its confidence.
- **DynMM** (Xue & Marculescu, CVPR-W 2023), arXiv:2204.00102, https://arxiv.org/abs/2204.00102. Routes each input through different forward paths.

A light QMF-style recipe trains tiny per-modality correctness predictors on out-of-fold correctness labels and caps telemetry's weight.

### 8.5 Missing modalities and mixture of experts

- **ModDrop** (Neverova et al., TPAMI 2016), arXiv:1501.00102, https://arxiv.org/abs/1501.00102. Drops whole modalities during training. It is the only way a gate ever sees "vision absent, telemetry must carry the prediction" examples, because natural vision failures are too rare (about 2.4% of frames).
- **FuseMoE** (Han et al., NeurIPS 2024), arXiv:2402.03226, https://github.com/aaronhan223/FuseMoE, and **Flex-MoE** (Yun et al., NeurIPS 2024), arXiv:2410.08245, https://github.com/UNITES-Lab/flex-moe. Availability-aware routing. UL-DD has only three availability patterns: vision + telemetry, vision only (subject A), and telemetry only (B, I, M). A three-expert soft router is the light version.
- **Missing-aware prompts** (Lee et al., CVPR 2023), arXiv:2303.03369, https://arxiv.org/abs/2303.03369. Learnable prompt tokens per missing pattern, under 1% extra parameters.
- **ShaSpec** (Wang et al., CVPR 2023), arXiv:2307.14126, https://github.com/billhhh/ShaSpec; **MMANet** (Wei et al., CVPR 2023), arXiv:2304.08028, https://arxiv.org/abs/2304.08028; **SMIL** (Ma et al., AAAI 2021), arXiv:2103.05677, https://arxiv.org/abs/2103.05677. These are heavier than this problem needs. MMANet's margin-aware distillation becomes relevant if the deployed model is vision-only.

### 8.6 Late fusion, stacking and evidential combination

- **I2M2** (Madaan et al., NeurIPS 2024), arXiv:2405.17613, https://arxiv.org/abs/2405.17613. A product of unimodal experts and a joint expert; the UL-DD authors used it.
- **MultiBench** (Liang et al., NeurIPS 2021 D&B), https://github.com/pliang279/MultiBench. Simple late fusion is a strong baseline there; this is secondhand evidence, via https://pmc.ncbi.nlm.nih.gov/articles/PMC11106632/.
- **Weighted log-linear pooling** `log p ∝ w_v log p_v + w_t log p_t` with a small, inner-fold-tuned w_t is the simplest form of "not equal weight". **OOF stacking** with multiplicative terms (e.g. `u_v·log p_t`) learns the confidence gate in closed form.
- **TMC** (Han et al., ICLR 2021), arXiv:2102.02051, https://github.com/Han-Zongbo/TMC, combines Dirichlet opinions with the reduced Dempster rule. Applying **Shafer discounting** to telemetry first means confident vision dominates.
  - Follow-up work warns that the Dempster rule is sensitive to conflicting views, and that independently estimated per-view uncertainties may not be comparable (arXiv:2604.09288, https://arxiv.org/pdf/2604.09288; title only seen).

### 8.7 Privileged information and distillation

- **Learning Using Privileged Information** (Vapnik & Vashist, *Neural Networks* 22(5–6), 2009, doi:10.1016/j.neunet.2009.06.042) and **generalised distillation** (Lopez-Paz et al., ICLR 2016), arXiv:1511.03643, https://arxiv.org/abs/1511.03643. Telemetry or biometrics are available only at training time.
- **Cross-modal distillation** (Gupta, Hoffman & Malik, CVPR 2016), arXiv:1507.00448, https://arxiv.org/abs/1507.00448.
- The **Modality Focusing Hypothesis** (Xue et al., ICLR 2023 oral), https://github.com/zihuixue/MFH: cross-modal distillation helps only to the extent that the teacher relies on decisive features the student's modality also contains.
- **C2KD** (Huo et al., CVPR 2024), https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html, filters misaligned soft labels.

A teacher that is weaker than the student under subject-independent CV will not help. Telemetry, at 0.259, is weaker than vision.

---

## 9. Small data and subject-independent generalisation

- **Freeze first, adapt last.** Kumar et al., *Fine-Tuning can Distort Pretrained Features* (LP-FT, ICLR 2022), arXiv:2202.10054, https://arxiv.org/abs/2202.10054. Full fine-tuning averaged 2% better in-distribution but **7% worse out-of-distribution** than linear probing, while LP-FT gave about 10% better OOD than full fine-tuning. A new subject is an OOD problem.
- **PEFT options**: LoRA (arXiv:2106.09685, https://github.com/microsoft/LoRA), rank 4–8 on the q/v projections of the last blocks; AdaptFormer (NeurIPS 2022), arXiv:2205.13535, https://arxiv.org/abs/2205.13535, with under 2% extra parameters.
- **Subject normalisation.** Per-session z-scoring would erase the label, because session ≈ label. Two options are viable:
  - Per-subject statistics, a transductive approach that must be reported as such.
  - An **early-session personal reference**, which is realistic for deployment. ildahun's 0.362 → 0.626 is the strongest evidence for it. Gao & Lv (*Frontiers in Neuroscience* 2026), https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1997832/full, use a session-opening reference for cross-subject EEG fatigue, and removing it sharply degrades performance.
  - Test-time BatchNorm adaptation is a related idea (arXiv:2511.22030, https://arxiv.org/abs/2511.22030; AdaBN **[unverified attribution]**).
  - A camera study (*Technologies* 14(9):544, 2026; snippet only) found that 5–50 target samples did not close a cross-*dataset* gap.
- **Subject-adversarial training.** DANN (Ganin et al., JMLR 2016), arXiv:1505.07818, https://ar5iv.labs.arxiv.org/html/1505.07818. UL-DD-specific risk: C, F and L are awake-only, so subject identity correlates with the label. Use a class-conditional adversary or exclude those three subjects. Reports from EEG work describe DANN as unstable under large inter-subject differences.
- **Augmentation across subjects.**
  - mixup (Zhang et al.), arXiv:1710.09412, https://arxiv.org/pdf/1710.09412, restricted to cross-subject pairs.
  - MixStyle (Zhou et al., ICLR 2021), arXiv:2104.02008, https://arxiv.org/abs/2104.02008, mixing subject "style" statistics.
  - Kim et al., *Calibration-Free Driver Drowsiness Classification based on Manifold-Level Augmentation*, arXiv:2212.13887, https://arxiv.org/abs/2212.13887 (EEG).
- **Self-supervision on unlabelled frames.**
  - TS2Vec (AAAI 2022), arXiv:2106.10466, https://arxiv.org/abs/2106.10466, or SupCon (Khosla et al.), arXiv:2004.11362, https://arxiv.org/abs/2004.11362, applied on top of frozen features.
  - VideoMAE pre-training on UL-DD would be a multi-day job on an M4.
  - Time-neighbour positives tend to learn subject identity first, so positives should be cross-subject.
- **Anomaly framing.** Tüfekci et al. 2022 train on alert driving only. An alert-only autoencoder could also use the awake-only subjects C, F and L.
- **IR-specific augmentation.** CLAHE gave more than 5 AUC points in Tüfekci et al.

---

## 10. Labels, protocol and realistic targets

### 10.1 Label noise and granularity

- **Noise is a main barrier.** Perkins et al. 2021 (IEEE T-IV), arXiv:2109.08355, https://arxiv.org/abs/2109.08355, name subjective and incoherent labelling, together with subject diversity, as the main barriers.
- **Self-report and observer ratings disagree.** Their agreement is κ = 0.274 (§5.2).
- **UL-DD's agreement is good but coarse.** Three-level agreement is κ = 0.88, but there is only one value per 4 minutes, and it is inherited by about 16 overlapping windows.

### 10.2 Binary vs. three-class

The three-class to two-class gap is large and consistent:

| Source | Three-class | Binary |
|---|---|---|
| ELA (Wolter et al.) | 52.5% | 80.4% |
| 2026 LOSO study | 0.388 F1 | 0.527 F1 |

- **Common binary cuts** are KSS ≥ 7 or 8 for drowsy, sometimes with ≤ 6 for alert (Dreissig et al., arXiv:2009.13276, https://arxiv.org/abs/2009.13276).
- **Locally**, KSS ≤ 5 vs. ≥ 6 is the most balanced cut in every fold (140/150 epochs overall), and it avoids the near-empty Alert class of folds 0 and 2.

### 10.3 Ordinal structure

- **CORAL** (Cao, Mirjalili & Raschka, *Pattern Recognition Letters* 2020), arXiv:1901.07884, https://arxiv.org/abs/1901.07884, and **CORN**, arXiv:2111.08851, https://arxiv.org/abs/2111.08851. Both give rank-consistent ordinal outputs.
- Reporting MAE on the 9-level KSS and quadratic-weighted κ alongside macro-F1 penalises Alert↔Drowsy errors more than adjacent errors.
- Ordinal heads mostly reduce *non-adjacent* errors. They help calibration and smoothing more than they fix the adjacent Alert/LowVig confusion directly.

### 10.4 Window length and smoothing

- **Longer windows help.** Massoz et al. show a 24-point gain from 5 s to 60 s, and Magán uses 60 s sequences.
- **Causal smoothing** (an EMA, or a three-state ordinal HMM with no direct Alert↔Drowsy transitions between consecutive windows) is cheap and well motivated.
- A linear-chain CRF (Lafferty et al., ICML 2001 **[unverified]**) is a learned alternative.
- **Caution.** Bidirectional whole-session smoothing turns into *session classification*, because session strongly predicts the label. Causal filtering must be the primary number.

### 10.5 Confounds and imbalance

- **Time and temperature confounds.** Time-only (0.295) and temperature-only (0.415) baselines on UL-DD (ildahun) show the session clock is informative. Every report should include a time-in-session baseline and check that vision models are not keying on session-level appearance such as lighting or clothing.
- **Telemetry may encode route position.** It may partly encode position on the simulator route, and hence time. A diagnostic is to predict time-in-session from telemetry.
- **Imbalance.** Use macro-F1 and balanced accuracy (Fu et al. survey) with class-weighted cross-entropy.

### 10.6 Is 0.70–0.80 macro-F1 realistic?

Subject-independent, KSS-based, three-class evidence:

| Result | Source |
|---|---|
| 0.362 macro-F1, tabular, UL-DD LOSO, no calibration | ildahun |
| 0.388 F1, KSS-3, LOSO | ScienceDirect 2026 |
| 52.5–65.2% accuracy, UTA-RLDD three-class | Wolter 2025; Ghoddoosian 2019 |
| < 60%, human judges on UTA-RLDD three-class | Ghoddoosian 2019 |
| 0.447 macro-F1, UL-DD IR frames | Kavipriya M5 |

The 0.70–0.80 range has been reached only with one of the following:
- leaky splits;
- acted binary data;
- objective PVT or EEG labels;
- labelled per-driver calibration (0.63–0.65).

**Defensible thesis targets:**

| Protocol | Target macro-F1 |
|---|---|
| Three-class, calibration-free | 0.50–0.55 |
| Three-class, short start-of-drive calibration | about 0.60–0.65 |
| Binary | about 0.65–0.80 |

---

## 11. Compute considerations on Apple Silicon

- **No published PyTorch-MPS ViT benchmark was found** for M-series chips. The one hard figure: a base M4's MPS FP32 GEMM peaks at about 2.9 TFLOPS against 4.26 theoretical, with about 100 GB/s of GPU memory bandwidth (*Apple vs. Oranges*, arXiv:2502.05317, https://arxiv.org/html/2502.05317v1).
- Apple-Silicon training generally trails CUDA, more so in FP16 (*Profiling Apple Silicon for ML Training*, arXiv:2501.14925, https://arxiv.org/abs/2501.14925).
- `torch.compile` on MPS is immature (*State of PyTorch Hardware Acceleration 2025*, https://tunguz.github.io/PyTorch_Hardware_2025/).
- MPS autocast support: fp16 autocast landed in PR #99272, https://github.com/pytorch/pytorch/pull/99272. bf16 and version behaviour are inconsistent (issues #139386, https://github.com/pytorch/pytorch/issues/139386, and #139190, https://github.com/pytorch/pytorch/issues/139190). `torch.compile` ignores nested autocast on MPS (issue #201630, https://github.com/pytorch/pytorch/issues/201630).
- **Local measurements** (2026-11-05, Low Power Mode on battery, so lower bounds). Frozen inference at fp16, batch 64, MPS memory 1.1–1.5 GB in every case:

  | Model | Resolution | Throughput |
  |---|---|---|
  | DINOv2 ViT-S/14 | 224 px | 88.6–185 img/s |
  | DINOv2 ViT-S/14 | 112 px | 712 img/s |
  | ViT-B/14 | 224 px | 27 img/s |
  | YOLOv8n-cls | 224 px | 1,986 img/s |

  A 4-layer temporal transformer trains at 532 windows/s.

  OpenCV sequential decode with `grab()`-skip covers all 29 sessions in about 1.5 min for IR and 4.6 min for Pose. Seeking is 5–10× slower.

  **Frozen encoders with cached embeddings and small heads are therefore the realistic main path.** Full fine-tuning of ViT-S at 224 px takes about 17 min per epoch per fold and uses 5.5 GB.

---

## 12. Master comparison table

Abbreviations: SI = subject-independent; acc = accuracy; F1m = macro-F1; 3c/2c = three-class/binary.

| # | Paper / work (year, venue, id) | Modality | Dataset | Protocol | Reported result | Key idea | Relevance to this thesis |
|---|---|---|---|---|---|---|---|
| 1 | UL-DD (2025/26, Sci Data; arXiv:2507.13403) | Bio + behavioural + facial features | UL-DD | Stratified 5-fold, SI unclear | 83.75% (SVM) to 88.03% (I2M2) acc, 3c | Early fusion, PCA, I2M2 | Official baseline; likely leaky; not comparable |
| 2 | ildahun (GitHub thesis, ~2025/26) | Face + bio + vehicle tabular | UL-DD | LOSO | F1m 0.362 → 0.626 with calibration; RF 75.2% random vs. 29.5–33.8% per-driver | Alert-session normalisation, head fine-tuning, quality gates | Most comparable SI benchmark |
| 3 | Kavipriya M5 (MTech thesis; reproduced here) | IR frames, YOLOv8n-cls | UL-DD | SI 5-fold, frame-level | F1m 0.447 ± 0.071, acc 0.541 | Fine-tuned tiny CNN per frame | Baseline to beat |
| 4 | Ghoddoosian et al. (2019, CVPRW; arXiv:1904.07312) | RGB blink features | UTA-RLDD | Subject-wise 5-fold | 65.2% 3c; humans < 60% | HM-LSTM on blinks | Closest naturalistic 3c KSS analogue |
| 5 | Wolter et al. (2025; arXiv:2511.19519) | 3D-landmark ELA | UTA-RLDD, DMD | RLDD folds | 52.5% 3c, 80.4% 2c | Eyelid angle, blink phases | Shows 3c vs. 2c gap; feature upgrade |
| 6 | Massoz et al. (2018, Sensors 18:2801) | NIR face video | Own (29) | LOSO, PVT labels | 70.7% (5 s) → 94.2% (60 s), 2c | Multi-timescale CNN | IR + window-length evidence |
| 7 | Vural et al. (2007, ICCV-HCI) | AUs | Simulator | Across subjects | > 90% (binary episodes) | AU classifiers | Supports appearance-AU branch |
| 8 | Shen et al. (2020; arXiv:2010.06235) | RGB/IR + flow | NTHU-DDD | Held-out subjects | 94.46% | Patch two-stream | Acted, binary upper bound |
| 9 | Lyu et al. (2018; arXiv:1801.02325) | Video | NTHU-DDD | Eval set | 90.05% | Multi-granularity CNN + LSTM | Patch design |
| 10 | Tüfekci et al. (2022, ECCV-W; arXiv:2209.05269) | IR/RGB video | NTHU-DDD | Driver-disjoint | AUC 0.874 | LSTM-AE anomaly; CLAHE +5 AUC | Alert-only pre-training |
| 11 | Lakhani (2022; arXiv:2207.12148) | Video Swin | NTHU-DDD | Clip split | 44% test | End-to-end video transformer | Warning: overfits |
| 12 | Adaptive cross-modal fusion (2026, ScienceDirect S2666521226001481) | Visual cues | NTHU-DDD, YawDD | LOSO + zero-shot | 91.51% → 76.35% LOSO; 55.93% F1 cross-dataset | Gated cross-attention + Transformer | Concatenation baseline competitive |
| 13 | SI eye-based framework (2026, Sensors 26:6193; snippet-level) | MediaPipe EAR | NTHU-DDD (4) | LODO | acc ~0.556 vs. 0.530 trivial | LSTM/BiLSTM vs. RF | Cross-subject collapse |
| 14 | Hassan et al. (2025, Sci Rep 15:17493) | ViT-B, Swin-T | MRL, NTHU, CEW | Random 80/20 | 99.52% | Fine-tuned ViT | Leaky ViT example |
| 15 | STFTransNet (2025, Sensors 25:5819) | Frame + face streams | NTHU, YawDD | Random, shared subjects | 95.86%; cross-attention no gain | CNN + cross-attention + TCN | Ablation caution |
| 16 | Light-VTD (2026, Sci Rep 16:28136) | Swin hybrid | MRL, NTHU, RLDD | Random + cross-dataset | 95–98.7% / 90.3–94.5% | MobileOne stem + Swin | Mac-friendly ViT |
| 17 | Krishna et al. (2022; arXiv:2209.01401) | ViT + YOLOv5 | UTA-RLDD, custom | Validation split | 97.4% / 95.5% (2c) | ViT classification | Leaky ViT example |
| 18 | CT-Net (2024, Sensors 24:7948) | CLIP-RN50 + Transformer | Fleet videos (25k) | Random 80/10/10 | AUC 0.892 vs. 0.836 ImageNet | Foundation encoder + temporal transformer | Direction supports frozen-FM design |
| 19 | SI multimodal behavioural + physiological (2026, ScienceDirect S174680942601061X) | HRV + EAR | Own (12) | LOSO vs. 10-fold | KSS-3 F1 0.388; KSS-2 0.527; observer 0.836 | Observer labels vs. KSS | Realistic ceiling |
| 20 | Cao et al. (2025, Sci Rep 15:12240) | NIR + PSG | DROZY | LOSO ×10 | 98.41% | ResNet18 + LSTM | EEG-dependent; not transferable |
| 21 | Kong et al. RPPMT (2024, Sensors 24:455) | IR rPPG + PERCLOS + yawn | Own (52) | Unclear | 98.2%; rPPG 0.948 → 0.972 | IR rPPG HRV | Optional rPPG cue |
| 22 | MERL TR2020-161 (2020) | NIR video | In-car | — | Reliable HR | Narrow-band NIR iPPG | rPPG feasibility |
| 23 | Li et al. (2017, Sensors 17:495) | Steering angle | Real road (6) | Pooled | 78.01% | ApEn + DTW vs. awake reference | Template for heading-rate entropy |
| 24 | Wang, Tran & Feiszli (2020, CVPR; arXiv:1905.12681) | General | Kinetics etc. | — | G-Blend +3.2% over unimodal | Auxiliary unimodal losses | Explains M3/M6 failures |
| 25 | OGM-GE (2022, CVPR; arXiv:2203.15332) | General | — | — | — | Gradient modulation | Multi-vision-stream balancing |
| 26 | Ma et al. (2022, CVPR; arXiv:2204.05454) | General | — | — | Multimodal < unimodal when modality missing | Fusion choice decides robustness | Gated fallback design |
| 27 | ModDrop (2016, TPAMI; arXiv:1501.00102) | General | Gesture | — | — | Modality dropout | Essential for the gate |
| 28 | I2M2 (2024, NeurIPS; arXiv:2405.17613) | General | — | — | — | Unimodal + joint experts | Used by UL-DD authors |
| 29 | QMF (2023, ICML; arXiv:2306.02050) | General | — | — | — | Quality-aware dynamic weights | Confidence-gated telemetry |
| 30 | Flamingo (2022; arXiv:2204.14198) | Vision-language | — | — | — | tanh(α) zero-init gated cross-attention | Gate initialisation |
| 31 | TMC (2021, ICLR; arXiv:2102.02051) | General | — | — | — | Dirichlet + Dempster–Shafer | Discounted evidential fusion |
| 32 | MBT (2021, NeurIPS; arXiv:2107.00135) | Audio-visual | — | — | > 2 mAP over cross-attention, < ½ compute | Bottleneck tokens | Multi-view fusion |
| 33 | Perceiver IO (arXiv:2107.14795) | General | — | — | — | Latent cross-attention | Missing-stream-tolerant fusion |
| 34 | DINOv2 (2024, TMLR; arXiv:2304.07193) | Image SSL | LVD-142M | — | — | Frozen ViT features | Primary frame encoder |
| 35 | DINOv3 (2025; arXiv:2508.10104) | Image SSL | LVD-1689M | — | — | Newer frozen ViT | Encoder ablation |
| 36 | MARLIN (2023, CVPR; arXiv:2211.06627) | Face video MAE | YouTube Faces | — | — | Facial-region masking | Face-dynamics encoder |
| 37 | VideoMAE / V2 (arXiv:2203.12602; 2303.16727) | Video MAE | K400, SSv2 | — | Works on 3–4k videos | Tube masking | General-video clip encoder |
| 38 | V-JEPA 2 (2025; arXiv:2506.09985) | Video SSL | Large web video | — | — | Latent prediction | Large-model upper-bound ablation |
| 39 | LP-FT (2022, ICLR; arXiv:2202.10054) | General | — | — | FT −7% OOD vs. LP; LP-FT +10% OOD vs. FT | Probe then fine-tune | Fine-tuning policy |
| 40 | Attention-MIL (2018, ICML; arXiv:1802.04712) | General | — | — | — | Attention pooling | Default temporal head |
| 41 | CORAL / CORN (arXiv:1901.07884; 2111.08851) | General | — | — | — | Rank-consistent ordinal heads | Ordinal KSS protocol |
| 42 | Depth Anything V2 (2024, NeurIPS; arXiv:2406.09414) | Monocular depth | — | — | — | DINOv2-S + DPT | Depth-proxy ablation |
| 43 | PoseFormer / CTR-GCN (arXiv:2103.10455; 2107.12213) | Skeleton | H36M / NTU | — | — | Spatial→temporal transformer / GCN | Landmark-skeleton stream |
| 44 | DMD (2020, ECCV-W; arXiv:2008.12085) | RGB + depth + IR, 3 cameras | DMD (37) | — | — | Multi-view cabin dataset | Closest multi-view analogue |

---

## 13. Gaps this thesis fills

1. **First subject-independent *video* benchmark on UL-DD.** The dataset paper benchmarks only extracted features, under a stratified (likely leaky) protocol. The only independent subject-independent study (ildahun) is tabular. No published work uses UL-DD's raw IR, stereo or side-view video under a subject-independent protocol.
2. **First use of the never-touched streams.** The L3D/R3D stereo RGB views, the side-view Pose video, the PL skeleton, grip pressure and the heading channel have not been used by any UL-DD model, published or in Kavipriya's work. This thesis also corrects the dataset documentation: L3D/R3D are RGB, not depth.
3. **First frozen-foundation-encoder study for drowsiness under a subject-independent protocol.** No peer-reviewed paper was found applying DINOv2/DINOv3, MARLIN or V-JEPA 2 to drowsiness. The related works found (CT-Net, TelUP) use random splits or have unverified numbers.
4. **An explicit "vision-primary, telemetry-as-fallback" fusion study.** The literature explains *why* symmetric fusion fails on small data, and M3 and M6 show it does on UL-DD. No drowsiness paper evaluates an asymmetric, quality-gated fusion with clean, degraded-vision and missing-video (B, I, M) regimes reported separately.
5. **Protocol honesty on a naturalistic KSS dataset.** The thesis quantifies, on one dataset with one fold split, how the score changes across 3-class, binary, ordinal, epoch-level and calibrated protocols. It reports the time-on-task confound baseline and the per-fold class scarcity. The field mostly reports one inflated number; this gives a calibrated picture for UL-DD.
6. **Edge-budget feasibility.** Every model is trained and run on a 16 GB Apple M4 with MPS, with measured throughput, which bears on in-vehicle deployment. A working Gradio demo shows the gate value and frame attention for interpretability.

---

## 14. References

URLs are as found during the research passes. Venue details marked **[unverified]** were not confirmed.

**UL-DD and directly related**
- Bodaghi et al., UL-DD, arXiv:2507.13403 — https://arxiv.org/abs/2507.13403 ; Sci Data doi:10.1038/s41597-025-06540-1 — https://www.nature.com/articles/s41597-025-06540-1 ; PMC — https://pmc.ncbi.nlm.nih.gov/articles/PMC13039290/
- ildahun, Driver-Drowsiness-Detection — https://github.com/ildahun/Driver-Drowsiness-Detection
- Kavipriya Ramasamy, MTech report (local: `Kavipriya_report (2).html`) and code (`driver-drowsiness-detection-system/`)

**Vision-based drowsiness**
- Ghoddoosian et al. 2019, arXiv:1904.07312 — https://arxiv.org/abs/1904.07312 ; CVPRW PDF — https://openaccess.thecvf.com/content_CVPRW_2019/papers/AMFG/Ghoddoosian_A_Realistic_Dataset_and_Baseline_Temporal_Model_for_Early_Drowsiness_CVPRW_2019_paper.pdf
- Wolter et al. 2025, arXiv:2511.19519 — https://arxiv.org/abs/2511.19519
- Massoz et al. 2018 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6165048/
- Vural et al. 2007 — https://inc.ucsd.edu/~marni/pubs/Vural_ICCV07.pdf
- Shen et al. 2020, arXiv:2010.06235 — https://arxiv.org/abs/2010.06235
- Lyu et al. 2018, arXiv:1801.02325 — https://arxiv.org/abs/1801.02325
- Tüfekci et al. 2022, arXiv:2209.05269 — https://arxiv.org/abs/2209.05269
- Lakhani 2022, arXiv:2207.12148 — https://arxiv.org/abs/2207.12148
- Adaptive cross-modal fusion 2026 — https://www.sciencedirect.com/science/article/pii/S2666521226001481
- SI eye-based framework, Sensors 26:6193 — https://www.mdpi.com/1424-8220/26/19/6193
- Hassan et al. 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12092738
- STFTransNet 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12473511/
- Light-VTD 2026 — https://pmc.ncbi.nlm.nih.gov/articles/PMC13554065/
- Krishna et al. 2022, arXiv:2209.01401 — https://arxiv.org/abs/2209.01401
- Zaman et al. 2025, arXiv:2511.12438 — https://arxiv.org/abs/2511.12438
- Fu et al. 2024 survey, arXiv:2408.12990 — https://arxiv.org/html/2408.12990v1
- FHWA/NHTSA PERCLOS validation — https://rosap.ntl.bts.gov/view/dot/113
- Personalised EAR/MAR thresholds, arXiv:2604.22479 — https://arxiv.org/abs/2604.22479
- Train-driver facial + skeletal graph study, arXiv:2505.08800 — https://arxiv.org/abs/2505.08800
- CT-Net, Sensors 24:7948 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11679248/
- TelUP drowsiness — https://github.com/agungibr/telup-drowsiness
- DMD, arXiv:2008.12085 — https://arxiv.org/pdf/2008.12085
- Arnav-Dugad/Driver-Detection-System and RuchirBhatia4/DriveMind (GitHub; anecdotal)

**rPPG**
- MERL TR2020-161 — https://www.merl.com/publications/docs/TR2020-161.pdf
- Kong et al. 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11154312/
- rPPG-Toolbox, arXiv:2210.00716 — https://arxiv.org/abs/2210.00716
- PhysDrive, arXiv:2507.19172 — https://arxiv.org/abs/2507.19172
- MS-rPPG, arXiv:2606.21115 — https://arxiv.org/abs/2606.21115
- VDMoE, arXiv:2410.21086 — https://arxiv.org/abs/2410.21086

**Vehicle signals**
- Survey, arXiv:2110.00472 — https://arxiv.org/abs/2110.00472
- Li et al. 2017 — https://pmc.ncbi.nlm.nih.gov/articles/PMC5375781/
- Traffic Injury Prevention 2019, doi:10.1080/15389588.2019.1622005
- Patents US8519853, US9205844, US4594583; ResearchGate 275363392 (yaw-angle std); grip-force studies (Int. J. Contents 2016; ScienceDirect S0169814121000019)

**Multimodal learning and fusion**
- Wang, Tran & Feiszli 2020, arXiv:1905.12681 — https://ar5iv.labs.arxiv.org/html/1905.12681
- OGM-GE, arXiv:2203.15332 — https://arxiv.org/pdf/2203.15332 ; extension arXiv:2410.11582
- Wu et al. 2022, arXiv:2202.05306 — https://proceedings.mlr.press/v162/wu22d.html
- Huang et al. 2022, arXiv:2203.12221 — https://proceedings.mlr.press/v162/huang22e.html
- Du et al. 2023, arXiv:2305.01233 — https://mlanthology.org/icml/2023/du2023icml-unimodal/
- PMR, arXiv:2211.07089 — https://arxiv.org/abs/2211.07089
- MMPareto, arXiv:2405.17730 — https://arxiv.org/abs/2405.17730
- Low-quality fusion survey, arXiv:2404.18947 — https://arxiv.org/abs/2404.18947
- Ma et al. 2022, arXiv:2204.05454 — https://arxiv.org/abs/2204.05454
- ModDrop, arXiv:1501.00102 — https://arxiv.org/abs/1501.00102
- I2M2, arXiv:2405.17613 — https://arxiv.org/abs/2405.17613
- MultiBench — https://github.com/pliang279/MultiBench ; secondary — https://pmc.ncbi.nlm.nih.gov/articles/PMC11106632/
- Dual-sensing, arXiv:2505.17392 — https://arxiv.org/abs/2505.17392
- SI multimodal 2026 — https://www.sciencedirect.com/science/article/abs/pii/S174680942601061X
- Cao et al. 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11985911/
- GMU, arXiv:1702.01992 — https://arxiv.org/abs/1702.01992
- FiLM, arXiv:1709.07871 — https://arxiv.org/abs/1709.07871
- Flamingo, arXiv:2204.14198 — https://ar5iv.labs.arxiv.org/html/2204.14198
- TMC, arXiv:2102.02051 — https://github.com/Han-Zongbo/TMC
- QMF, arXiv:2306.02050 — https://arxiv.org/abs/2306.02050
- Predictive Dynamic Fusion, arXiv:2406.04802 — https://arxiv.org/abs/2406.04802
- Multimodal Dynamics, CVPR 2022 — https://openaccess.thecvf.com/content/CVPR2022/html/Han_Multimodal_Dynamics_Dynamical_Fusion_for_Trustworthy_Multimodal_Classification_CVPR_2022_paper.html
- DynMM, arXiv:2204.00102 — https://arxiv.org/abs/2204.00102
- MLA, arXiv:2311.10707 — https://arxiv.org/abs/2311.10707
- PAMRL — https://doi.org/10.3390/app15105229
- EDL, arXiv:1806.01768 — https://arxiv.org/abs/1806.01768
- MC-dropout, arXiv:1506.02142 — https://arxiv.org/abs/1506.02142
- Deep ensembles, arXiv:1612.01474 — https://arxiv.org/abs/1612.01474
- Temperature scaling, arXiv:1706.04599 — https://arxiv.org/abs/1706.04599
- SMIL, arXiv:2103.05677 — https://arxiv.org/abs/2103.05677
- ShaSpec, arXiv:2307.14126 — https://github.com/billhhh/ShaSpec
- MMANet, arXiv:2304.08028 — https://arxiv.org/abs/2304.08028
- Missing-aware prompts, arXiv:2303.03369 — https://arxiv.org/abs/2303.03369
- FuseMoE, arXiv:2402.03226 — https://github.com/aaronhan223/FuseMoE
- Flex-MoE, arXiv:2410.08245 — https://github.com/UNITES-Lab/flex-moe
- Dempster–Shafer critique, arXiv:2604.09288 — https://arxiv.org/pdf/2604.09288
- MBT, arXiv:2107.00135 — https://arxiv.org/abs/2107.00135
- Perceiver IO, arXiv:2107.14795 — https://ar5iv.labs.arxiv.org/html/2107.14795

**Distillation and privileged information**
- Vapnik & Vashist 2009, doi:10.1016/j.neunet.2009.06.042
- Lopez-Paz et al., arXiv:1511.03643 — https://arxiv.org/abs/1511.03643
- Gupta et al., arXiv:1507.00448 — https://arxiv.org/abs/1507.00448
- Modality Focusing Hypothesis — https://github.com/zihuixue/MFH
- C2KD — https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html

**Encoders (image, video, face, depth, skeleton)**
- DINOv2 — https://arxiv.org/abs/2304.07193 ; https://github.com/facebookresearch/dinov2 ; https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
- DINOv3 — https://arxiv.org/pdf/2508.10104 ; https://github.com/facebookresearch/dinov3 ; https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m ; https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- SigLIP 2 — https://arxiv.org/abs/2502.14786
- MobileCLIP — https://arxiv.org/abs/2311.17049 ; MobileCLIP2 — https://arxiv.org/abs/2508.20691
- EVA-02 — https://huggingface.co/timm/eva02_small_patch14_224.mim_in22k ; https://huggingface.co/timm/eva02_tiny_patch14_224.mim_in22k
- MAE — https://arxiv.org/abs/2111.06377
- FastViT — https://arxiv.org/abs/2303.14189 ; MobileViT — https://arxiv.org/abs/2110.02178 ; EfficientViT — https://arxiv.org/abs/2205.14756
- VideoMAE — https://arxiv.org/abs/2203.12602 ; https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics
- VideoMAE V2 — https://github.com/OpenGVLab/VideoMAEv2 (arXiv:2303.16727)
- InternVideo2 — https://arxiv.org/pdf/2403.15377 ; https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/single_modality/MODEL_ZOO.md
- V-JEPA 2 — https://arxiv.org/abs/2506.09985 ; https://github.com/facebookresearch/vjepa2 ; https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 ; V-JEPA 2.1 — https://www.alphaxiv.org/abs/2603.14482v1
- TimeSformer — https://arxiv.org/abs/2102.05095 ; ViViT — https://arxiv.org/abs/2103.15691
- Video Swin — https://arxiv.org/abs/2106.13230 ; https://docs.pytorch.org/vision/main/models/video_swin_transformer.html
- MViTv2 — https://arxiv.org/abs/2112.01526 ; X3D — https://arxiv.org/abs/2004.04730 ; https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/ ; UniFormerV2 — https://arxiv.org/abs/2211.09552
- MARLIN — https://arxiv.org/abs/2211.06627 ; https://github.com/ControlNet/MARLIN
- FaRL — https://arxiv.org/abs/2112.03109 ; https://github.com/FacePerceiver/FaRL
- FSFM — https://arxiv.org/html/2412.12032 ; FS-VFM — https://arxiv.org/html/2510.10663v1
- EmoNet — https://www.nature.com/articles/s42256-020-00280-0 ; https://github.com/face-analysis/emonet
- LibreFace — https://arxiv.org/pdf/2308.10713
- OpenGraphAU — https://www.ijcai.org/proceedings/2022/0173.pdf ; https://github.com/lingjivoo/OpenGraphAU
- MediaPipe Face Landmarker — https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- Depth Anything V2 — https://arxiv.org/abs/2406.09414v2 ; https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- FoundationStereo — https://arxiv.org/abs/2501.09898 ; Fast-FoundationStereo — https://arxiv.org/pdf/2512.11130
- PoseFormer — https://arxiv.org/abs/2103.10455 ; MotionBERT — https://arxiv.org/abs/2210.06551
- ST-GCN — https://arxiv.org/abs/1801.07455 ; CTR-GCN — https://arxiv.org/pdf/2107.12213 ; https://github.com/Uason-Chen/CTR-GCN

**Temporal heads, efficiency**
- Attention-MIL — https://arxiv.org/abs/1802.04712 ; TCN — https://arxiv.org/abs/1803.01271 ; S4 — https://arxiv.org/abs/2111.00396
- Mamba (arXiv:2312.00752 **[unverified id]**); MPS status — https://github.com/state-spaces/mamba/issues/324 ; https://github.com/purohit10saurabh/mamba-ssm-macos
- Token Merging — https://arxiv.org/abs/2210.09461
- MSTN — https://www.researchgate.net/publication/315162000

**Small data, adaptation, labels**
- LP-FT — https://arxiv.org/abs/2202.10054 ; LoRA — https://github.com/microsoft/LoRA (arXiv:2106.09685) ; AdaptFormer — https://arxiv.org/abs/2205.13535
- DANN — https://ar5iv.labs.arxiv.org/html/1505.07818
- mixup — https://arxiv.org/pdf/1710.09412 ; MixStyle — https://arxiv.org/abs/2104.02008
- Kim et al. manifold augmentation — https://arxiv.org/abs/2212.13887
- Gao & Lv 2026 — https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1997832/full
- Test-time BN adaptation, arXiv:2511.22030 — https://arxiv.org/abs/2511.22030
- TS2Vec — https://arxiv.org/abs/2106.10466 ; SupCon — https://arxiv.org/abs/2004.11362
- CORAL — https://arxiv.org/abs/1901.07884 ; CORN — https://arxiv.org/abs/2111.08851
- Perkins et al. 2021 — https://arxiv.org/abs/2109.08355 ; Dreissig et al. — https://arxiv.org/abs/2009.13276

**Compute**
- Apple vs. Oranges (M4 MPS) — https://arxiv.org/html/2502.05317v1
- Profiling Apple Silicon for ML Training — https://arxiv.org/abs/2501.14925
- State of PyTorch HW 2025 — https://tunguz.github.io/PyTorch_Hardware_2025/
- MPS autocast — https://github.com/pytorch/pytorch/pull/99272 ; https://github.com/pytorch/pytorch/issues/139386 ; https://github.com/pytorch/pytorch/issues/139190 ; https://github.com/pytorch/pytorch/issues/201630
