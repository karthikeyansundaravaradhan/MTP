# UL-DD Dataset Summary

Source paper: `Literature/ULDD Dataset.pdf.md` (also `ULDD Dataset.pdf.html` at repo root) —
**"UL-DD: A Multimodal Drowsiness Dataset Using Video, Biometric Signals, and Behavioral Data"**,
Bodaghi, Hosseini, Gottumukkala, Bhupatiraju, Ahmad, Gabbouj — University of Louisiana at Lafayette
(arXiv:2507.13403, 2025). Metadata sheet: `DriverDrowsinessDataset/Info.xlsx`.

## 1. What the dataset is

UL-DD is a lab-collected, multimodal **driver drowsiness** dataset: 19 subjects (15 M, 4 F,
ages 25–40) drove an American Truck Simulator rig (three monitors, wheel, pedals) in two
40-minute sessions each — once **Alert** and once **Drowsy** — while facial video, biometric
wearables, and driving telemetry/grip-pressure sensors were recorded simultaneously. Drowsiness
was self-rated every 4 minutes on the 9-point **Karolinska Sleepiness Scale (KSS)** (Cronbach's
α = 0.99). Total recorded length ≈ 1,400 minutes. This is the dataset Kavipriya's report
(see `report_summary.md`) is built on, and the one this project will extend with a
multimodal/hybrid/ensemble model.

Participants: 20 attended, 3 did second-session no-shows, 1 withdrew consent after session 1 →
**16 subjects with both sessions (A/D) + 3 subjects with only the Alert session = 19 total**.
In this local copy those single-session subjects are **C, F, L** (no `*_Drowsy` row in
`Labels.csv`, no `D` sub-folder on disk for those three).

## 2. Local folder layout (`DriverDrowsinessDataset/`)

```
DriverDrowsinessDataset/
├── Info.xlsx                  # 2 sheets: modality/sample-rate catalogue + per-subject availability matrix
├── Labels.csv                 # 35 rows: "<Subject>_<Session>,kss_1,...,kss_10" (4-min KSS scores)
├── CSV_Files/CSV_Files/{A-S}/{A,D}/        # raw per-modality time series (556 MB)
├── Extracted_Features/Extracted_Features/{A-S}/{A,D}/   # precomputed FL/PL/FAU csvs (13 GB)
├── Video_Data/Video_Data/{A-S}/{A,D}/      # raw trimmed videos, MP4 (7.2 GB present, missing for B/I/M)
├── Codes/Codes/                # official preprocessing + baseline notebooks (see §6)
├── ULDD Dataset/               # empty leftover folder — no data, ignore
└── *.zip                       # original zipped downloads of the four folders above (~11 GB, redundant with the unzipped copies)
```

Every subject has a **19-way A–S ID**, and within each subject two session folders: **A = Alert**,
**D = Drowsy**. File naming convention throughout: `<Subject>_<Signal>_<Session>.<ext>`, e.g.
`A_HR_D.csv` = subject A, heart rate, drowsy session; `A_FL_A.csv` = subject A, facial landmarks,
alert session.

**Known local gaps** (also flagged in the paper's Limitations, §9):
- **B, I, M** have `0 B` `Video_Data` folders — no raw video released for these subjects (consent),
  but their `FL`/`FAU`/`PL` *extracted* features are still present for all 19 subjects.
- **C, F, L** have no Drowsy-session data anywhere (single-session subjects).
- `Info.xlsx` sheet "Publicity" encodes per-subject/session/modality availability as `a` = available,
  `r` = removed/not released — decode column headers via sheet row 1 (Participants, Session, IR
  Video, 3D depth Video, Pose Video, ACC, BVP, EDA, HR, IBI, Left GP, Right GP, SPO2, PR, Motion,
  TEMP, Telemetry, PL, FL, FAU, Length). Worth parsing this sheet programmatically before assuming
  a subject/session/modality combo exists.

## 3. Modalities and file formats

### Raw signals — `CSV_Files/<Subject>/<Session>/`

| File | Columns | Rate | Notes |
|---|---|---|---|
| `*_HR_*.csv` | 1 col: heart rate (bpm) | 1 Hz | Empatica E4 |
| `*_ACC_*.csv` | x,y,z accelerometer | 32 Hz | Empatica E4, wrist |
| `*_EDA_*.csv` | 1 col: electrodermal activity | 4 Hz | Empatica E4 |
| `*_TEMP_*.csv` | 1 col: skin temperature (°C) | 4 Hz | Empatica E4 |
| `*_BVP_*.csv` | 1 col: blood volume pulse | 64 Hz | Empatica E4 |
| `*_IBI_*.csv` | timestamp, inter-beat interval | event-based (~0.5 Hz avg) | Empatica E4; not a fixed-Hz signal |
| `*_O2M_*.csv` | SpO2, pulse rate, motion flag | 0.5 Hz | Checkme O2 Max pulse oximeter |
| `*_LGP_*.csv` / `*_RGP_*.csv` | 1 col each: left/right grip pressure | 3 Hz | steering-wheel pressure sensors |
| `*_Labels_*.csv` | 10 values | one row, per 4-min interval | duplicate of that subject/session's row in top-level `Labels.csv` |
| `*_Telemetry_*.csv` (Drowsy-session dir only in the sample checked, per-file) | timestamp, raw rendering/simulation/paused-sim timestamps, heading, pitch, roll, speed (m/s), rpm, gear | 60 Hz | American Truck Simulator output; 18/19 subjects only |

### Extracted features — `Extracted_Features/<Subject>/<Session>/`

| File | Columns | Rate | Source |
|---|---|---|---|
| `*_FL_*.csv` | `Frame_Number` + 68×(X,Y) = 136 cols | 60 fps | Dlib 68-point 2D facial landmarks, from IR video |
| `*_FAU_*.csv` | `Frame` + 30 facial action unit scores | 60 fps | derived from FL.csv (see AU table in paper §3.3) |
| `*_PL_*.csv` | `frame` + 33×(x,y,z) = 99 cols | 60 fps | MediaPipe Pose, from Pose video |

These three are provided for **all 19 subjects/sessions available**, even where the raw video
itself was withheld for consent reasons — useful fallback for facial/pose modeling without
needing the video files.

### Video — `Video_Data/<Subject>/<Session>/`

| File | Content | Notes |
|---|---|---|
| `*_IR_*.mp4` | Infrared face video, 60 fps, cropped ~240×220 | Source for FL/FAU extraction |
| `*_L3D_*.mp4` / `*_R3D_*.mp4` | Left/right views of ZED 2 depth camera, 60 fps, ~440×370 | 3D depth info; only 16/19 subjects |
| `*_Pose_*.mp4` | Side-view RGB posture webcam, 60 fps, ~620×470 | Source for PL extraction; only 16/19 subjects |

### Labels — `Labels.csv` (top level)

35 rows (19 subjects × 2 sessions − 3 missing Drowsy sessions = 35), one row per
`<Subject>_<Session>`, 10 comma-separated KSS scores (one per 4-minute sub-interval of the
40-minute drive). Scale is 1–9 (see KSS table below). A handful of rows contain `.5` values
(e.g. `7.5`) — averaged/interpolated scores, not integers, so treat KSS as ordinal/continuous
rather than assuming clean integers.

| KSS | Meaning |
|---|---|
| 1 | Extremely alert |
| 2 | Very alert |
| 3 | Alert |
| 4 | Rather alert |
| 5 | Neither alert nor sleepy |
| 6 | Some signs of sleepiness |
| 7 | Sleepy, no effort to stay awake |
| 8 | Sleepy, some effort to stay awake |
| 9 | Very sleepy, fighting sleep |

Common downstream binning used by both the source paper and Kavipriya's report:
**Low/Alert = KSS < 4 (1–3)**, **Medium/LowVigilant = 4–6**, **High/Drowsy = KSS > 6 (7–9)**.

## 4. Synchronization

All modalities were aligned to manually-recorded session start/end times; the E4 and O2 Max
wristbands carry epoch timestamps; grip-pressure was aligned by file-internal start time;
telemetry was aligned by observing simulator start in the video. The paper's own multimodal
validation (§6.4) resampled **everything to a common 4 Hz** before fusion — this is the
pragmatic reference approach if you need one shared clock across modalities (IBI, being
event-based, needs explicit resampling/interpolation to fit any fixed-Hz grid).

## 5. Technical validation already done by the dataset authors (paper §6)

- **Normality:** Shapiro-Wilk showed biometric signals are *not* normally distributed →
  paper uses **Kruskal-Wallis** (non-parametric) for Low/Med/High group comparisons, and a
  **linear mixed-effects model** (subject as random effect) for effect sizes. Nearly everything
  is significant across drowsiness bins except **BVP** (never reaches significance) and
  **HR** at the Medium-vs-High boundary (p=0.2, PR is recommended as the more reliable
  heart-rate-family signal over the E4's HR channel).
- **Direction of effects:** EDA ↓ with drowsiness, TEMP ↑, SpO2 slight ↑ at lower drowsiness, PR ↑ with alertness.
- **Signal-to-noise (Table 6):** HR cleanest (45.7 dB avg), grip pressure noisiest (14.7 dB avg,
  20% of samples <10 dB) — sensor-contact-dependent.
- **Video quality:** 0% frame-drop rate across all recordings.
- **Label reliability:** expert re-labeled 30% of video against KSS self-report — unweighted
  Cohen's κ = 0.619 (substantial), quadratic-weighted κ = 0.967 (almost perfect) on the 9-level
  scale; κ = 0.88 on the 3-level binning. Labels are trustworthy but treat 9-level exact-value
  agreement with more caution than the 3-class version.

## 6. Baseline results reported in the dataset paper (§6.4, Fig. 8)

3-class (Low/Med/High) classification, 4 Hz resample, MinMax-scaled, linear interpolation +
forward-fill for missing values, SVM(RBF, C=0.1, balanced) and RF(max_depth=5, n_estimators=100,
min_samples_leaf=10), 5-fold stratified CV, **early fusion** for multimodal:

| Modality / Combination | SVM | RF |
|---|---|---|
| Biometric | 63% | 71% |
| Telemetry | 40% | 45% |
| Grip | 48% | 54% |
| Pose | 68% | 65% |
| FAU | 52% | 59% |
| Bio + Facial | 73% | 71% |
| Bio + Behavioral | 80% | 74% |
| **Bio + Behavioral + Facial** | **84%** | 75% |

Takeaway: even a naive early-fusion SVM over all three modality families clearly beats every
unimodal baseline — this is the dataset authors' own evidence for why a multimodal approach is
worth pursuing, and it is a materially higher number than anything in Kavipriya's later report
(see `report_summary.md`), largely because that report never brought biometric signals into its
models at all.

## 7. Code already provided (`DriverDrowsinessDataset/Codes/Codes/`)

Sequential notebooks (Python 3.11.11, see `requirements.txt`):
1. `1.Combine_Data_and_Change_Frequency.ipynb` — merge subjects, resample (example: → 4 Hz)
2. `2_1/2_2/2_3.Concatenate_*_and_Labeling.ipynb` — biometric / behavioral / facial concatenation + labeling
3. `3_1/3_2/3_3.Classification_*.ipynb` — unimodal classification per modality family
4. `4.Multimodal_Analysis.ipynb` — early-fusion multimodal classification (produces Fig. 8 numbers)
5. `Feature_Extraction/` — `Main.ipynb` + per-feature notebooks (`2D_Facial_Landmarks_Generator`,
   `3D_Facial_Landmarks_Generator` [paper notes 3D landmarks aren't actually in the released
   data], `Facial_Action_Unit_Generator`, `Pose_Landmarks_Generator`) + Dlib's
   `shape_predictor_68_face_landmarks.dat` model file
6. `Biometric_Plots.ipynb`, `Kruskal_Wallis.ipynb`, `Mixed_Linear_Model.ipynb` — the statistical
   validation notebooks behind paper §6.1
7. `requirements.txt` / `setup_env.sh` / `setup_env.bat` — env setup

**Environment note:** this machine's `python3` (`/usr/bin/python3`) is the Xcode Command Line
Tools stub and refuses to run until `sudo xcodebuild -license` is accepted; none of the
notebooks above can run until a real Python (e.g. via Homebrew, pyenv, or conda) is installed.
Flagged here rather than resolved automatically since accepting an Xcode license needs `sudo`.

## 8. How UL-DD compares to other public drowsiness datasets (paper Table 1)

UL-DD is the only one in the comparison table offering video + grip pressure + telemetry +
7 biometric channels + facial landmarks/AUs + pose landmarks simultaneously, with **9-level**
graded labels (vs. binary/3/5-class elsewhere) and continuous 40-minute sessions (vs. 10–30 min
elsewhere) — its whole value proposition is capturing *gradual* transitions into drowsiness
rather than acted/binary drowsy-vs-not snapshots. Comparable datasets: RLDD, NTHU-DDD, DROZY,
YawDD, DMD, 3MDAD, MRL Eye, NITYMED, SEED-VIG (full citations in the paper's References).
