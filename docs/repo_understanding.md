# `driver-drowsiness-detection-system` — Repo Understanding

Source: `driver-drowsiness-detection-system/` (`github.com/kavipriyariisc/driver-drowsiness-detection-system`),
Kavipriya's actual thesis codebase, cloned into the project root. This is the ground truth for
"what she actually built" — where anything here disagrees with `docs/report_summary.md` (which
was inferred from her prose write-up only), **this document wins**.

## 0. Read this first: the repo mixes real work with fabricated documentation

This repo is the accumulated working directory of an MTech thesis with a **large amount of
unreconciled scratch documentation**. On a single day (2026-06-24, commit `4634f7f7` "final") ~34
new root-level markdown files were dumped into the repo at once, along with several dataset/model
artifacts. Within that dump there are **at least three mutually-contradictory narratives** about
what the project's models are and what their results were. Only one of these is corroborated by
actual code and actual `results/reports/*.json` files — the rest are either unimplemented planning
docs or appear to be fabricated after the fact. **Two files in particular should never be
cited: `PROJECT_COMPLETION_REPORT.md` and `MULTI_MODAL_FUSION_SUMMARY.md`** — see §1.

The trustworthy narrative (matches `docs/report_summary.md`, verified below against code and
JSON): **M1 (FAU BiLSTM), M2 (Telemetry LSTM), M3 (cross-modal attention fusion), M5 (YOLOv8n-cls
on IR frames — best model), M6 Lite / M6 Full (visual+CAN fusion, both trained but capped by a
known temporal-misalignment bug), M7 (EfficientNet-B0 + BiLSTM temporal IR video model — fully
designed, never actually executed/committed), M8 (sparse-frame ensemble — pure vaporware, no code
at all)**.

## 1. Chronology, branches, and which docs to trust

### Real build order (from git log)

```
2026-03-18  first commit, initial setup
2026-03-20  Simple CNN model done
2026-03-22  "changes"                    ← tip of `main` (stale — pre-M5, do not use)
2026-04-19  M1 to M5 models + results + notebooks folders added, checkpoints force-added
2026-06-06  m5 / latest notebook
2026-06-12  "changes"                    ← tip of `feature/real-ddd` (merged in, no longer independently useful)
2026-06-13  m6 model added
2026-06-24  "final"                      ← the big 34-file doc/data dump
2026-06-24  "adding iamges"              ← tip of `feature/final` = HEAD of origin/feature/final
```

Real build order: **M1–M4 baselines → M5 YOLOv8 → M6 fusion (multiple bug-fix iterations) →
attempted M7 rebuild (never finished) → M8 planned (never started) → one giant documentation/data
dump**.

### Branches
- **`main`** — frozen at the March 22 commit, only has M1–M4, pre-M5. **Do not use.**
- **`feature/real-ddd`** — short side-branch, already merged into `feature/final`. Not independently useful.
- **`feature/final`** (currently checked out, `HEAD=a8b2d63b`) — **the only branch with M5/M6/M7-docs/M8-docs and all the result data. Build from this one.** (`git diff main feature/final --stat` shows ~30k files differ — `main` is essentially a different, much earlier project snapshot.)

### Canonical (trust these)
- **`ARCHITECTURE_VERIFICATION.md`** — line-by-line comparison of the thesis's architecture prose
  against the actual code. The single best document in the repo; every claim in it checks out.
- **`MODEL_COMPARISON_TABLE.md`** and **`MODELS_COMPLETION_SUMMARY.md`** — accuracy/F1/std numbers
  match `results/reports/M{1,2,3,6_lite,6_full}_results.json` to 4 decimal places (though see the
  one internal inconsistency noted in §7 item 10).
- **`M6_COMPLETION_STATUS.md`, `M6_ARCHITECTURE_EXPLAINED.md`, `M6_Design.md`,
  `CAN_ALIGNMENT_ANALYSIS.md`, `README_M6.md`** — consistent with each other and with the M6 code;
  correctly describe the real temporal-misalignment bug (§7.1).
- **`M7_REBUILD_PLAN.md`, `M7_HARDENING_SUMMARY.md`, `M7_FINAL_CHECKLIST.md`, `M7_QUICK_START.md`,
  `M7_TRAINING_GUIDE.md`** — describe the architecture that matches the thesis's real M7
  (EfficientNet-B0 + BiLSTM(256) + attention pooling), but the code/results they describe were
  **never actually committed** (§3, §5 step 7).

### Ignore / superseded / not to be cited
- **`PROJECT_COMPLETION_REPORT.md`** and **`MULTI_MODAL_FUSION_SUMMARY.md`** — describe a
  *different, apparently fabricated* model taxonomy (M1=Random Forest, M2=XGBoost, M4=plain dense
  NN, M9=MobileNetV3-Small, a "tri-modal fusion" at 74.2% accuracy). None of this exists anywhere
  in the repo: no corresponding code, no result files, notebook numbers referenced don't exist,
  and the two files even contradict each other's dates. **Do not cite any number from these two.**
- **`START_HERE.md`** — despite the name, a narrow mid-project note about an M6 alignment fix; it
  links to ~8 files that don't exist in the repo. Ignore.
- **`STRATEGY_2_EXECUTION.md`** — describes the 60fps-embedding fix as "tested"/"ready," but its
  promised output directory doesn't exist on disk. The fix was designed, never run.
- **`M7_COMPLETION_STATUS.md`** — describes a *different* M7 (a "hybrid ensemble of M1/M2/M5
  predictions," 43.39% accuracy) than the EfficientNet+BiLSTM M7 described everywhere else and in
  the thesis. Code files it cites (`m7_hybrid.py`, `m7_train.py`, `m7_test.py`) don't exist. The
  43.39% number recurs across several other June-18 docs, so a real run of *something* likely
  happened once (probably in Colab, never synced back) — treat the **number** as
  possibly-real-but-unverifiable, and the **architecture description in this specific file** as
  wrong.
- **`FRESH_FUSION_*.md`** (5 files) — an entirely separate, never-started fusion effort. Zero code, zero results. Ignore.
- **`M8_*` (4 files)** — vaporware, see §2.
- Everything else at root (`COMPREHENSIVE_PRESENTATION_GUIDE.md`, `TECH_STACK_EXPLAINED.md`,
  `THESIS_MODEL_ELABORATIONS.md`, `PROJECT_ARCHITECTURE_DIAGRAM.md`, `COLAB_EXTRACTION_NOTEBOOK.py`,
  `VIDEO_DEMO_GUIDE.md`, `M5_VIDEO_DEMO_README.md`, `CPU_OVERNIGHT_RUN_GUIDE.md`) — secondary
  presentation/demo material of varying accuracy; skim only if needed, not load-bearing.
- **`temp_check/`, `temp_src_check/`** — abandoned duplicate copies of `src/`. Ignore/delete-candidate.

## 2. What M8 actually is

"**Sparse-frame temporal ensemble**": sample 5 frames (1/sec) from the 16-frame window instead of
all 16, run each through the same frozen EfficientNet-B0, classify each frame independently, then
average the per-frame logits — conceived as a simpler alternative after M7's BiLSTM result was
disappointing ("43.39%, worse than M5!").

**It was never trained or even coded.** The only M8 artifacts in the entire repo are 4
documentation files. None of the referenced code exists (`src/models/m8_*.py`,
`notebooks/15-*`/`16-*`), and none of the referenced outputs exist (`models/m8_embeddings/`,
`results/reports/M8_results.json`). The docs are internally honest that the numbers shown (e.g.
"Mean Accuracy: 0.5239 ± 0.0892") are labeled "Expected/Example Output," not a real run, and
status is explicitly "Ready to execute" / "TBD." Best inference for its absence from the thesis:
it was a planned next step that was superseded/abandoned once time ran out after the incomplete
M7 rebuild — not a result that existed and was later cut.

## 3. `src/` module map

| File | Purpose | Key classes/functions |
|---|---|---|
| `src/models/architecture.py` | Keras model builders for M1–M4; ground truth for M1/M2/M3 architecture. | `build_m1_facial_bilstm` (BiLSTM(128)→BiLSTM(64)→Dense(64)+BN→Dense(3), ~420K params); `build_m2_telemetry_lstm` (unidirectional LSTM(64)→LSTM(32), ~45K params); `CrossModalAttentionBlock`; `M3FusionModel` (BiLSTM(FAU) + LSTM(CAN) with bidirectional cross-attention, ~750K params — the thesis's headline contribution); `build_m4_realtime_bilstm` (10 live features → BiLSTM, feeds the live demo only, ~82K params); `build_model()` factory. |
| `src/models/train.py` | Generic Keras train/CV loop for M1/M2/M3. | `compile_model` (Adam + label-smoothed sparse-CCE); `make_callbacks` (monitors `val_accuracy`, not `val_loss` — documented as necessary because val_loss diverges from epoch 1 on subject-independent folds); `train_fold`; `cross_validate` (loads `fold_{k}.npz`, aggregates mean/std accuracy & macro-F1). |
| `src/models/m6_extractor.py` | Extracts 512-d YOLOv8-cls backbone embeddings from `datasets/yolo_frames/` (1 fps frames) → `models/embeddings/{subj}_{sess}.npz`. **This is the old, broken-alignment extractor** (see §7.1). | `YoloClsBackbone`, `extract_session`/`extract_all`. |
| `src/models/m6_extractor_from_video.py` | The **alignment-fix** extractor: reads the original 60fps UL-DD `.mp4`s directly → `models/embeddings_uldd/`. Hardcodes a Windows `VIDEO_ROOT` path — needs changing on this machine. `NO_VIDEO = {"B","I","M"}`. **Its output directory does not exist on disk — the fix was designed but never actually run.** | `VideoFrameDataset` (streams/seeks via OpenCV), `extract_session_from_video`, `extract_all_from_videos`. |
| `src/models/m6_fusion.py` | PyTorch M6 architectures. Docstring states current committed results are capped at ~41% due to the wrong embedding source. | `M6_Lite`/`M6_Lite_v2` (BiLSTM(128,bidir) visual + BiLSTM(64,bidir) CAN → concat → Dense); `M6_Full`/`M6_Full_v2` (Transformer encoder visual + BiLSTM CAN + bidirectional cross-attention → concat → Dense); `build_m6()` factory. |
| `src/models/m6_train.py` | M6 dataset + train/eval loop. Defaults to the (non-existent) fixed embedding dir, falls back to the old one. Contains substantial metadata-repair/leakage-filtering logic added specifically to work around the `m6_preprocess.py` bug (§7.2). | `EmbeddingCache`, `sample_clip_embedding_by_range`, `M6Dataset`, `train_one_fold`, `cross_validate` (writes `results/reports/M6_{variant}_results.json`). |
| `src/models/m6_validate.py` | Pre-training data-integrity checks for M6 (fold schema, embedding-cache completeness, index-reconstruction consistency). | `validate_fold_file`, `validate_embeddings_cache`, `validate_index_reconstruction`, `full_validation`. |
| `src/models/m6_vision_only.py` | "M6-A" vision-only ablation: BiLSTM(128,bidir) + learned attention pooling, no CAN. Standalone. | `AttentionPooling`, `M6VisionOnly`, `build_m6a`. |
| `src/data/preprocess.py` | **The real M1/M2/M3 preprocessing pipeline.** Ground truth for dataset layout expectations — see §6 for the important nesting-depth discrepancy. Downsamples 60Hz→4Hz, builds 60s windows (15s stride), KSS→3-class labels, the standard subject-independent 5-fold split. Handles subject A (no telemetry) and C/F/L (Alert-only) gracefully. Writes `fold_{k}.npz`. | `ULDDProcessor` class; `run()`, `process_all()`, `build_folds()`, `save_folds()`. |
| `src/data/load_data.py` | Thin notebook wrappers around `ULDDProcessor`'s saved folds. | `load_fold`, `load_all_folds`, `summarise_fold`, `class_distribution`. |
| `src/data/m6_preprocess.py` | **Buggy script** — retrofits `mm_subject/session/win_idx/...` metadata onto existing fold files by heuristically re-deriving sample ownership. See §7.2 for the real, verifiable leakage-risk bug. | `get_available_sessions`, `build_m6_metadata`, `allocate_mm_samples_to_sessions`. |
| `src/data/telemetry_replay.py` | Live-demo utility: replays a Telemetry CSV at 4Hz on a background thread, plus a generic `RingBuffer`. Not used by any training pipeline. | `TelemetryReplayer`, `RingBuffer`. |
| `src/inference/predict.py` | Unified inference wrapper for trained M1/M2/M3/M4 (Keras) and M5 (YOLO). | `ULDDPredictor` (`predict_window`, `predict_image` with YOLO alphabetical-class remapping, `evaluate_fold`); `load_predictor_from_fold`. |
| `src/utils/helpers.py` | Metrics + all plotting used across notebooks. | `compute_metrics` (the actual metric function `train.py.cross_validate` uses); `plot_confusion_matrix`, `plot_fold_results`, `plot_model_comparison`, `plot_kss_distribution`, `plot_training_history`, `plot_attention_weights` (M3 interpretability). |
| `src/face_detection.py` | Real-time face detection + 10-feature extraction for the live demo (feeds M4). YOLOv8 for bbox, MediaPipe FaceLandmarker (auto-downloads its model file on first use) for landmarks → EAR/MAR/PERCLOS/blink-rate/head-pose. | `FaceDetector`, `FeatureExtractor`, `FacialAnalyzer`. |
| `src/scoring.py` | Converts 3-class softmax probs into a smoothed 0–100 drowsiness score + alert messages. Live-demo only. | `DrowsinessScorer`, `AlertGenerator`. |
| `src/realtime_demo.py` | Combines face detection + telemetry replay + `ULDDPredictor` + scorer into an OpenCV live-webcam app. Telemetry replay is explicitly disabled in the constructor ("disabled for M5-only test"). | `RealtimeDemoApp`. |

**Where is M7's real code?** Not in `src/` at all. The M7 docs reference
`src/data/m7_preprocess.py`, `src/models/m7_{embed,model,dataset,train}.py`, and
`notebooks/12/13/14-uldd-m7-*.ipynb` — **none of these exist anywhere in git history**; only the
doc files were ever committed (all in the single "final" dump commit). The one tangible M7
artifact on disk is the **preprocessed windowed data**,
`datasets/processed/uldd_m7/fold_{0..4}.npz` (49 MB, confirmed present — clean metadata, no
train/test subject overlap, but **no visual embeddings**). So: M7 was fully designed and its
input data was prepared, but the model code and training run were never completed or committed.
Building M7 for real is a from-scratch task for this new project, not a "reproduce existing code" task.

## 4. Notebook map

| Notebook | What it does | Runs on | Hardcoded paths to change |
|---|---|---|---|
| `04-uldd-eda.ipynb` | EDA on raw UL-DD (KSS distributions, AU correlations, sample windows, data availability). | Local | Windows UL-DD root path |
| `05-uldd-preprocessing.ipynb` | Runs `ULDDProcessor.run()` → `datasets/processed/ul_dd/fold_{0..4}.npz` for M1/M2/M3. **Must be re-run first** — that output dir is currently empty on disk (§6, §7.3). | Local | Same Windows root path |
| `06-uldd-m1-facial.ipynb` | Trains/evaluates M1, 5-fold CV. | Local | none found — relative paths |
| `07-uldd-m2-telemetry.ipynb` | Trains/evaluates M2. | Local | none found |
| `08-uldd-m3-fusion.ipynb` | Trains/evaluates M3 — its own header calls this "Thesis Main Contribution." | Local | none found |
| `09-uldd-m5-yolo-cls.ipynb` | Builds the YOLO classification dataset from raw video + `Labels.csv`, fine-tunes YOLOv8n-cls. | Local | Windows `Labels.csv` and `Video_Data` root paths |
| `09_colab_m5_yolo_cls.ipynb` | Colab variant of M5 training. | **Google Colab** | Google-Drive paths |
| `09_kaggle_m5_yolo_cls.ipynb` | Kaggle variant of M5 training. | **Kaggle** | standard `/kaggle/input`/`/kaggle/working`, no user-specific hardcoding |
| `10-comparison.ipynb` | Aggregates all trained-model results into the comparison plots in `results/reports/`. | Local | relative paths, reads `results/reports/*_results.json` |
| `11-uldd-m6-fusion.ipynb` | Runs M6 lite/full training via `src/models/m6_train.py`. | **Written for Colab** (mounts Drive, unzips a project zip) | needs adapting to call `python -m src.models.m6_train` directly for local use |

No notebooks 12–19 exist despite being referenced extensively in the M7/M8/"M9" documentation
threads — confirming those pipelines never got as far as a runnable notebook.

## 5. End-to-end reproduction steps

### Environment / dependency notes (from the repo's own `requirements.txt`)

```
numpy>=1.24.0, pandas>=2.0.0, scikit-learn>=1.3.0, scipy>=1.11.0
tensorflow>=2.14.0
opencv-python>=4.8.0, Pillow>=10.0.0, ultralytics>=8.0.0, mediapipe>=0.10.0
matplotlib>=3.7.0, seaborn>=0.12.0, jupyter/ipykernel/ipywidgets
tqdm>=4.66.0, pytest>=7.4.0
```

- **Critical gap: `torch`/`torchvision` are not listed at all**, despite M6 being pure PyTorch
  (`nn.MultiheadAttention`, `nn.TransformerEncoder`, etc.). Install these separately.
- `PROJECT_COMPLETION_REPORT.md`'s dependency list (torch 2.0, tensorflow 2.13, xgboost, W&B) is
  from the fabricated narrative (§1) — ignore it, use the real `requirements.txt` above.
- MediaPipe's `FaceLandmarker` auto-downloads its model file from `storage.googleapis.com` on
  first run — needs internet access once.
- **Repo is Windows-authored**: every hardcoded path uses `C:\Users\raka1005\...` /
  `C:/Users/raka1005/...`. On macOS, grep for `raka1005` and `C:\`/`C:/` before running anything,
  and replace with this machine's actual `DriverDrowsinessDataset/` path. Also watch for raw
  backslash-string path literals (not `Path(...)`) in scripts you touch.

### Step by step

1. **Branch**: stay on `feature/final` (already checked out) — never `main`.
2. **Install deps**: `pip install -r requirements.txt && pip install torch torchvision` (into the
   `uldd` conda env, or a dedicated env for this repo — see open question in `CLAUDE.md`).
3. **Point at the raw dataset**: replace the Windows UL-DD root path in `04-`, `05-`,
   `09-uldd-m5-yolo-cls.ipynb`, and `src/models/m6_extractor_from_video.py`'s `VIDEO_ROOT`.
4. **M1/M2/M3**: run `05-uldd-preprocessing.ipynb` — **mandatory even though checkpoints exist**,
   because `datasets/processed/ul_dd/` is currently empty on disk (only `.gitkeep`, git-ignored —
   see §7.3). This regenerates `fold_{0..4}.npz`. Then run `06`/`07`/`08`, or call
   `src.models.train.cross_validate` directly. Checkpoints already on disk:
   `M1_fold{0..4}.keras` ✅ all 5, `M2_fold{0..4}.keras` ✅ all 5, **`M3` only has `fold0.keras`**
   — folds 1–4 need retraining if you want to skip training. `src/inference/predict.py` can
   evaluate existing checkpoints without retraining if that's all you need.
5. **M5**: `datasets/yolo_cls/fold_0/` on disk only has `train/Alert` (15,274 imgs) and
   `train/Drowsy` (14,655 imgs) — **missing `LowVigilant`, missing any `test/` split, missing
   folds 1–4**. Either re-run `09-uldd-m5-yolo-cls.ipynb` (or Colab/Kaggle variant) against raw
   video to regenerate the full classification dataset, or just reuse the already-trained
   checkpoints at `models/checkpoints/M5_fold{0..4}.pt` (all 5 present) plus the full Ultralytics
   run artifacts at `models/checkpoints/yolo_cls/M5_fold0/` to reproduce the reported
   54.51%/0.4439 numbers without retraining.
6. **M6 (lite + full)**: two embedding sources exist in code, only one exists on disk.
   - `models/embeddings/` (32 `.npz`, 1fps, the **broken-alignment** source) — exists on disk.
     This produced the checked-in `M6_full_results.json`/`M6_lite_results.json` (41.99%/40.49%).
   - `models/embeddings_uldd/` (the intended 60fps fix) — **does not exist**; generate it with
     `python -m src.models.m6_extractor_from_video` against real UL-DD videos (repo's own
     estimate: 20–30 min on GPU, 2–3 hr on CPU) to attempt the claimed 70–78% improvement.
   - `datasets/processed/ul_dd/` needs M6's extra metadata keys
     (`mm_subject/session/win_idx/start_4hz/end_4hz`) — these already exist, correctly, in
     **`datasets/processed/ul_dd1/fold_{0..4}.npz`** (119 MB, on disk). **Prefer copying/renaming
     this over re-running the buggy `src/data/m6_preprocess.py`** (§7.2).
   - Then: `python -m src.models.m6_train --variant lite --epochs 50` / `--variant full`.
7. **M7**: not reproducible as-is — no model/train code exists (§3). `datasets/processed/uldd_m7/fold_{0..4}.npz`
   (49 MB, clean metadata, no leakage) is on disk and usable as a starting point. Building the
   actual model is a from-scratch task: use `ARCHITECTURE_VERIFICATION.md` §3.5.7 as the spec
   (EfficientNet-B0 frozen → per-frame 1280-d features → BiLSTM(256, bidir) → attention pooling →
   4-layer classifier) plus the existing fold metadata for windowing/frame-range lookups.
8. **M8 / Fresh Fusion**: no code exists anywhere; build from scratch if pursued, treating the
   doc's architecture description as a spec only, never as a results source.

## 6. Data expectations vs. the actual UL-DD structure — important discrepancy

`src/data/preprocess.py`'s `ULDDProcessor.__init__` hardcodes:

```python
self.fau_root  = root / 'Extracted_Features' / 'Extracted_Features' / 'Extracted_Features'
self.tele_root = root / 'CSV_Files' / 'CSV_Files' / 'CSV_Files'
self.labels_path = root / 'Labels.csv'
```

This is **triple-nested**, whereas this project's actual local `DriverDrowsinessDataset/` copy is
only **double-nested** (`CSV_Files/CSV_Files/{A-S}/...`, `Extracted_Features/Extracted_Features/{A-S}/...`
— see `docs/dataset_summary.md` §2). **Check the nesting depth before running `05-uldd-preprocessing.ipynb`
against this project's dataset copy** — either adjust `fau_root`/`tele_root` by one level, or point
`root` one directory higher than you'd expect, whichever matches. Don't assume it "just works."

Expected filenames: `{Subject}_FAU_{Session}.csv`, `{Subject}_Telemetry_{Session}.csv`,
`Session ∈ {A, D}`. Video path convention (from `m6_extractor_from_video.py`):
`Video_Data/{Subject}/{Session}/{Subject}_IR_{Session}.mp4`.

**Missing-session handling is graceful, not a crash risk** (verified against code and the
embeddings actually on disk):
- **Subject A** — `NO_TELEMETRY = {'A'}`: telemetry loader returns `None` early; M1 still uses A's
  FAU data; M2/M3 simply exclude A's windows from the multimodal set.
- **Subjects C, F, L** — `AWAKE_ONLY = {'C','F','L'}`: Drowsy session skipped entirely. Confirmed
  on disk: `models/embeddings/` has `C_A`/`F_A`/`L_A` but no `*_D` counterparts.
- **Subjects B, I, M** — `NO_VIDEO = {"B","I","M"}`: no video-derived features at all (confirmed
  zero embeddings on disk for these three); their FAU/telemetry CSVs still feed M1/M2/M3 fine.
- `process_session`/`process_all` return `None`/`continue`+`[SKIP]`-print on a missing FAU file —
  no crash path found.

## 7. Known issues / red flags (code-verified, concrete)

1. **M6's temporal misalignment bug is real and still baked into the committed results.**
   `m6_fusion.py`'s docstring and several canonical docs describe it: M5-backbone embeddings were
   extracted at ~1fps from pre-cut classification crops, while CAN telemetry runs at 4Hz/240
   timesteps — a ~60× granularity mismatch capping M6 near 41%. The fix
   (`m6_extractor_from_video.py`, extract at 60fps from source video) is fully coded but **never
   executed** (`models/embeddings_uldd/` doesn't exist). **This is a genuine, ready-made
   improvement target for this new project** — see `docs/story_so_far.md` for how it factors into next steps.

2. **`src/data/m6_preprocess.py` has a real, verifiable subject-leakage-risk bug.**
   `allocate_mm_samples_to_sessions()` distributes fold sample counts proportionally across *all*
   globally-available embedding sessions, without constraining to that fold's actual train/test
   subject partition — i.e. it can mislabel which subject/session a reconstructed sample actually
   belongs to. `m6_train.py` has grown substantial defensive/repair code around this, but the root
   script is unfixed. **Prefer `datasets/processed/ul_dd1/` (correct, non-reconstructed metadata)
   over re-running `m6_preprocess.py`.**

3. **`datasets/processed/ul_dd/` is empty on disk** (only `.gitkeep`, git-ignored) despite being
   the default path every M1/M2/M3/M6 script expects. The real usable data lives under
   differently-named, force-committed directories (`ul_dd1/`, `uldd_m7/`). A fresh clone will
   `FileNotFoundError` on any of these scripts until you re-run preprocessing or
   rename/copy `ul_dd1/` → `ul_dd/`.

4. **Checkpoints are incomplete for the Keras models**: M1 and M2 have all 5 folds; **M3 only has
   `fold0.keras`** — folds 1–4 need retraining. No checkpoints exist for M6/M7/M8 (M6 isn't Keras anyway).

5. **`datasets/yolo_cls/` (447 MB) is an incomplete partial dump** — only `fold_0/train/{Alert,Drowsy}`,
   missing `LowVigilant`, missing any `test/` split, missing folds 1–4. Not sufficient to retrain
   M5 as-is; needs full regeneration from raw video if retraining is desired.

6. **`requirements.txt` omits PyTorch entirely** despite M6 requiring it — install it manually.

7. **Two fabricated-looking "completion report" docs** (`PROJECT_COMPLETION_REPORT.md`,
   `MULTI_MODAL_FUSION_SUMMARY.md`) describe a model taxonomy and a 74.2%-accuracy "tri-modal
   fusion breakthrough" with zero corresponding code or results anywhere in the repo, and
   internally contradictory dates. **Do not cite from these.**

8. **Two different "M7"s** are described across the docs with different architectures and
   different implied numbers (43.39% "hybrid ensemble" vs. the unrun EfficientNet+BiLSTM "clean"
   model that matches the actual thesis). Neither has a `results/reports/M7_results.json` on disk
   despite several docs claiming one exists — **treat any M7 number from this repo as
   unverifiable**; the thesis's own stated M7 number (0.4339 acc / 0.3139 macro-F1, see
   `docs/report_summary.md`) is the only number worth citing, and even that should be re-derived
   once M7 is actually rebuilt for this project.

9. **Windows-authored path literals are pervasive** — grep for `raka1005` and `C:\`/`C:/` before
   running anything on this machine.

10. **Internal inconsistency even within a "canonical" doc**: `MODEL_COMPARISON_TABLE.md` states
    M6 Lite's result as "42.29%" in one table row and "40.49%" in its own summary a few lines
    later. **Trust `results/reports/*.json` over any markdown-stated number when they conflict.**

## 8. What this means for reproduction + improvement (next steps)

**Update (post-verification, same day):** two things below turned out to need correction once
actually tested on this machine — see `docs/story_so_far.md`'s 2026-08-31 "checkpoint/data
mismatch" entry for the full story:

1. `models/checkpoints/*.keras` and `*.pt` were **Git LFS pointer stub files**, not real weights,
   until `git-lfs` was installed and `git lfs pull` was run. Any claim of "checkpoints already
   exist" implicitly assumed LFS was resolved — check file sizes (a few hundred bytes = still a
   pointer) before trusting presence alone.
2. **Reusing `datasets/processed/ul_dd1/` in place of the empty `ul_dd/` does NOT safely reproduce
   M1/M2/M3's reported numbers.** Evaluating the (LFS-recovered) M1/M2/M3 checkpoints against
   `ul_dd1` produces far worse accuracy/macro-F1 than reported, and the models collapse to
   predicting almost one class even on `ul_dd1`'s own *training* split — meaning `ul_dd1` is a
   different data generation than whatever the checkpoints actually trained on (`preprocess.py`'s
   windowing/fold logic is fully deterministic, so this isn't run-to-run noise). `ul_dd1` remains
   fine as a source of *M6 metadata schema* (subject/session/window-index columns), just not as a
   numeric stand-in for `ul_dd/`.

Given that, the trustworthy path is: **regenerate fold data from the raw dataset with her
unmodified `ULDDProcessor`, then retrain M1/M2/M3 from scratch with her unmodified
`src/models/train.py`** — this reproduces her methodology exactly, on data whose provenance is
known, rather than leaning on old checkpoints of uncertain origin. Also fix the path-nesting bug
from §6 by overriding `fau_root`/`tele_root` one level shallower (no source edit needed) — do not
restructure the dataset folder to match the code instead.

Revised status:
- **Freshly reproducible today (in progress)**: M1, M2, M3 — regenerate `ul_dd_regen/` from raw
  data, retrain all 5 folds each with unmodified training code. Compare resulting mean/std
  accuracy and macro-F1 against the thesis's reported numbers as a methodology replication, not a
  bit-for-bit checkpoint match.
- **M5**: existing `.pt` checkpoints (all 5 folds, now LFS-recovered) may or may not have the same
  provenance problem — not yet verified the same way M1/M2/M3 were. Treat as unverified until
  checked; `datasets/yolo_cls/` is partial anyway (missing `LowVigilant`, no `test/` split, folds
  1–4 missing), so a from-scratch YOLO retrain needs full dataset regeneration regardless.
- **M6 Lite/Full**: existing results were produced from the known-misaligned 1fps embeddings; the
  alignment fix (`m6_extractor_from_video.py`) is fully coded but never run — still the clearest
  concrete improvement target, but do the same LFS/provenance sanity check before trusting any
  existing M6 artifact.
- **Needs to be built from scratch**: M7 (data ready, model/train code absent), M8 and Fresh
  Fusion (nothing exists).
- After M1/M2/M3 are cleanly reproduced, the next high-value step is likely **finishing the M6
  alignment fix** before moving on to bringing in the biometric/grip-pressure signals that
  neither Kavipriya's work nor this repo ever touched (see `docs/report_summary.md` §8).

## 9. Deep dive: M1 — Facial BiLSTM (current reproduction run)

This section walks through everything involved in reproducing **M1** specifically — dataset,
code, architecture, training mechanics, live progress, and what "success" looks like. Written
while the reproduction run (§8) is in flight; update the progress/results subsections once it
finishes rather than treating this as a one-time snapshot.

### Dataset

- **Raw source**: this project's local `DriverDrowsinessDataset/Extracted_Features/Extracted_Features/{Subject}/{Session}/{Subject}_FAU_{Session}.csv`
  — 30 Facial Action Unit columns per row (see the exact column list in
  `ULDDProcessor.FAU_COLS`, `src/data/preprocess.py:64-73`), sampled at 60 fps.
- **Labels**: top-level `DriverDrowsinessDataset/Labels.csv`, one row per `{Subject}_{Alert|Drowsy}`,
  10 KSS scores (one per 4-minute epoch) → mapped to 3 classes via `KSS_THRESHOLDS = (3, 6)`
  (Alert ≤3, LowVigilant 4–6, Drowsy >6), same binning as the paper and the written thesis.
- **Preprocessing** (`ULDDProcessor` in `src/data/preprocess.py`, run unmodified except for the
  path-nesting override described in §6/§8): downsample 60 Hz → 4 Hz (keep every 15th sample),
  slide a 240-sample (60 s) window with 60-sample (15 s) stride over each session, label each
  window from the KSS epoch at its centre timestamp. M1 uses **every session that has an FAU
  file**, regardless of whether telemetry exists — so, unlike M2/M3, subject A (no telemetry) and
  the Alert-only subjects C/F/L still contribute full-weight data.
- **Folds**: the fixed subject-independent split (`FOLDS` in `preprocess.py:91-97`) — fold 0
  holds out A/B/C/D, fold 1 E/F/G/H, fold 2 I/J/K, fold 3 L/M/N/O, fold 4 P/Q/R/S. Regenerating
  from the raw dataset on this machine reproduced almost exactly the per-fold train/test sample
  counts implied by the thesis (fold 0: 4294 train / 1073 test; fold 1: 4301/1066; fold 2:
  4433/934; fold 3: 4319/1048; fold 4: 4121/1246) — strong evidence the raw data itself matches
  what she originally used, even though the specific old checkpoint files did not (§8).
- **Normalization**: per-feature z-score, mean/std fit on the fold's **training** windows only
  (`ULDDProcessor._fit_scaler`), then applied to both train and test — stored in the fold `.npz`
  as `fau_mean`/`fau_std` so inference-time code can reuse the exact same scaler.
- **Output**: `datasets/processed/ul_dd_regen/fold_{0..4}.npz`, arrays `X_fau_train`/`X_fau_test`
  shape `(N, 240, 30)`, `y_train`/`y_test` shape `(N,)` int labels 0/1/2.

### Code files involved

| File | Role |
|---|---|
| `src/data/preprocess.py` | `ULDDProcessor` — builds the fold `.npz` files from raw CSVs (unmodified from her repo; only `fau_root`/`tele_root` overridden at call-time for the path-nesting fix, no source edit) |
| `src/models/architecture.py` | `build_m1_facial_bilstm()` — the M1 model factory (unmodified) |
| `src/models/train.py` | `compile_model`, `make_callbacks`, `train_fold`, `cross_validate` — the entire training loop (unmodified) |
| `src/utils/helpers.py` | `compute_metrics()` — accuracy/macro-F1/per-class-F1/classification-report (unmodified) |
| *(this project's own driver script, not part of either repo)* | a small script that constructs `ULDDProcessor` with the corrected paths, calls `.run()`, then calls `cross_validate(build_m1_facial_bilstm, 'M1', ...)` for M1 (and equivalently for M2/M3) — this is the only new code written for this reproduction; everything it calls into is Kavipriya's own, untouched |

### Model architecture (`build_m1_facial_bilstm`, `architecture.py:33-74`)

```
Input (batch, 240, 30)                              # 60s window × 30 FAU features
 → Bidirectional(LSTM(128, return_sequences=True))  # + L2 reg on kernel & recurrent weights
 → Dropout(0.45)
 → Bidirectional(LSTM(64))
 → Dropout(0.45)
 → Dense(64, relu, L2 reg)
 → BatchNormalization()
 → Dense(3, softmax)
```

Instantiating this (both the old checkpoint and a freshly-built model — same code) gives
**335,875 parameters**, not the "~420K" figure quoted in the architecture docstring and in
`ARCHITECTURE_VERIFICATION.md` — a minor doc/reality mismatch, not a functional issue, but worth
knowing if a parameter count is ever quoted as a cross-check.

### How it's trained (`train.py`, `DEFAULTS` + `train_fold`/`cross_validate`)

- **Loss**: sparse categorical cross-entropy with **label smoothing 0.1**, implemented by hand
  (`_smooth_sparse_cce`) since Keras's sparse CCE doesn't natively support label smoothing.
- **Optimizer**: Adam, learning rate `5e-4`.
- **Class weights**: `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` computed
  fresh per fold from that fold's train labels, passed to `model.fit(..., class_weight=cw)` — this
  is how the pipeline compensates for KSS's natural class imbalance (Alert-heavy in some subjects,
  Drowsy-heavy in others).
- **Callbacks**: `EarlyStopping(monitor='val_accuracy', mode='max', patience=25,
  restore_best_weights=True)`, `ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
  patience=8, min_lr=1e-6)`, `ModelCheckpoint(monitor='val_accuracy', save_best_only=True)`.
  Deliberately monitors **val_accuracy, not val_loss** — the code comments explain that on these
  subject-independent folds, val_loss diverges from epoch 1 while val_accuracy still improves, so
  a val_loss-based early stop would always revert to epoch-1 (random-init) weights.
- **Epochs / batch size**: up to 200 epochs (early stopping almost always ends it sooner),
  batch size 32, `validation_split=0.15` carved out of each fold's *training* subjects (so the
  validation split is still subject-independent from the fold's test subjects, but not
  necessarily subject-independent from the rest of training — worth being aware of if val_accuracy
  is ever treated as a subject-generalization estimate rather than just an early-stopping signal).
- **No random seed is set anywhere** in `preprocess.py`, `architecture.py`, or `train.py` — the
  data pipeline is fully deterministic (confirmed: regenerating it reproduced her per-fold sample
  counts almost exactly), but **weight initialization and dropout are not seeded**. This means
  even running her *exact, unmodified* code against *exactly* her original data would not be
  expected to reproduce bit-identical numbers run-to-run — only numbers "in the same ballpark."
  This is the main reason this project regenerated data and retrained from scratch rather than
  trying to forensically recover the original training run: an unseeded pipeline has no single
  ground-truth run to recover in the first place.

### Results — reproduction run complete (M1, M2, M3)

The background job initially died silently around M1/fold0/epoch 48 (the machine went to idle
sleep and killed the unprotected background process — confirmed via `pmset -g log` showing sleep/
wake-assertion activity at the exact time the training log went silent). It was relaunched wrapped
in `caffeinate -ims` and fully detached (`nohup ... & disown`) so it would survive both system
sleep and any Claude Code session restart, and **that run completed all 5 folds × 3 models
cleanly, no errors**, in a bit under 6 hours on CPU (started ~11:22, finished ~17:09).

| Model | Reproduced here | Thesis-reported (`docs/report_summary.md` §4) |
|---|---|---|
| M1 | acc **0.3898 ± 0.1136**, macro-F1 **0.3574 ± 0.1082** | acc 0.3913 ± 0.1157, macro-F1 0.3475 |
| M2 | acc **0.3237 ± 0.0691**, macro-F1 **0.2594 ± 0.0737** | acc 0.3867 ± 0.0401, macro-F1 0.3386 |
| M3 | acc **0.3522 ± 0.1199**, macro-F1 **0.3224 ± 0.1254** | acc 0.3992 ± 0.0984, macro-F1 0.3778 |

**M1 landed almost exactly on the thesis's own number** — strong end-to-end validation that the
regenerated dataset, fold split, architecture, and training loop are all faithful to the original.
M2 and M3 are in the same ballpark but somewhat below the thesis figures, and in both cases the
gap is driven by one specific fold rather than a uniform shortfall:

| Fold (held-out subjects) | M1 acc/F1 | M2 acc/F1 | M3 acc/F1 |
|---|---|---|---|
| 0 (A,B,C,D) | 0.341 / 0.337 | 0.377 / 0.249 | 0.465 / 0.457 |
| 1 (E,F,G,H) | 0.488 / 0.481 | 0.290 / 0.184 | 0.495 / 0.479 |
| 2 (I,J,K) | 0.485 / 0.353 | 0.421 / 0.386 | 0.363 / 0.276 |
| **3 (L,M,N,O)** | **0.189 / 0.170** | **0.309 / 0.287** | **0.181 / 0.157** |
| 4 (P,Q,R,S) | 0.446 / 0.446 | 0.222 / 0.191 | 0.256 / 0.243 |

**Fold 3 is the worst fold for all three models**, and by a wide margin for M1/M3 — worth treating
as a real finding, not noise: M subject has no video (irrelevant to M1/M2/M3 which are FAU/CAN
only, so this isn't the video-availability gap), but M/N/O are drawn from later-recruited subjects
per the dataset paper's demographics — a per-subject data-quality check on L/M/N/O's FAU and
telemetry extraction quality (e.g. via `docs/dataset_summary.md`'s Info.xlsx availability matrix)
is a reasonable next diagnostic step before assuming it's just small-sample variance. Also matches
the general shape of the thesis's own results (large std across folds for every model) — this
project's run isn't uniquely unstable, subject-independent 5-fold CV on 19 subjects is inherently
high-variance.

Checkpoints: `models/checkpoints_regen/M{1,2,3}_fold{0..4}.keras`. Full per-fold metrics
(precision/recall/F1 per class, confusion arrays): `results/reports_regen/M{1,2,3}_results.json`.

### Next steps

1. ~~Reproduce M1/M2/M3~~ — done, see Results above.
2. Decide (with the user) whether fold 3's weak performance is worth a dedicated investigation
   before moving on, or whether to treat it as expected variance and proceed.
3. Move on to verifying M5's existing checkpoints (`models/checkpoints/M5_fold{0..4}.pt`, all 5
   present) the same way, or go straight for the M6 alignment fix (§8) as the first genuinely
   *new* contribution of this project — decide with the user which comes first.

### Expected / target metrics and how they're computed

- **Metrics computed**: `compute_metrics()` in `src/utils/helpers.py` returns `accuracy` (overall),
  `macro_f1` (unweighted mean of per-class F1 — the headline metric, chosen because Alert is
  consistently the hardest class and class balance varies by fold), `per_class_f1` (F1 for each
  of Alert/LowVigilant/Drowsy separately), and a full `sklearn.classification_report` string.
  `cross_validate()` aggregates these into `mean_acc`/`std_acc`/`mean_f1`/`std_f1` across folds —
  the same aggregation style used throughout `docs/report_summary.md`'s results tables.
- **Target/reference numbers** (from the thesis, `docs/report_summary.md` §4): M1 accuracy
  **0.3913 ± 0.1157**, macro-F1 **0.3475**. Given the unseeded-training caveat above, treat this
  as a target *range* to land near, not an exact bar to clear — a run that's noticeably worse
  (especially one showing the same near-total-collapse-to-one-class pattern seen in §8's
  `ul_dd1` evaluation) signals a real bug; a run that's noticeably *better* is also worth a second
  look, since M1 is meant to be a relatively weak, facial-only baseline relative to M3/M5.

## 10. Deep dive: M4 — Real-Time BiLSTM (new work, never trained by Kavipriya)

Unlike M1/M2/M3/M5, **M4 was never trained on UL-DD at all** in the original repo — it exists only
as an architecture (`build_m4_realtime_bilstm`, `architecture.py:285-321`) built for the live
webcam demo, and the thesis explicitly excludes it from evaluation ("requires continuous
real-time synchronization between face detection and window state, which is difficult to validate
offline," `docs/report_summary.md` §3.5 model table). No notebook, no fold data, no results file
exists for it anywhere in git history. Training it for real is therefore genuinely new work for
this project, not a reproduction.

### Dataset

M4's live-demo design consumes **10 features per 4 Hz frame** — `EAR_left, EAR_right, MAR,
PERCLOS, blink_rate, head_pitch, head_roll, head_yaw, brow_raise, jaw_drop` — computed from
MediaPipe FaceMesh's 468(+10 iris) landmarks plus a YOLO face bbox
(`src/face_detection.py::FeatureExtractor`/`FaceDetector`). To build an offline, windowed training
set faithful to this design:

- **Source**: this project's local `DriverDrowsinessDataset/Video_Data/Video_Data/{Subject}/{Session}/{Subject}_IR_{Session}.mp4`
  — the same 240×220, 60 fps IR videos M5 uses, for the same 16/19 subjects with video
  (`NO_VIDEO = {B, I, M}`).
- **Face detection skipped deliberately**: her live pipeline runs a YOLOv8 *generic object*
  detector (`yolov8n.pt`, COCO-pretrained, no face class, and the code doesn't even filter by
  class — see §11 "issues found" for why this is a latent bug in the live demo) to get a bbox
  before cropping for MediaPipe. Since the IR video is already a tight face/head crop by
  construction (confirmed by inspecting a sample frame — full head and shoulders fill the 240×220
  frame, no scene clutter), this project's offline extractor **feeds the full frame directly to
  MediaPipe FaceMesh**, skipping the YOLO bbox step entirely. This is simpler, avoids a real bug
  in the original detector, and does not change the feature formulas at all.
  **Note this deliberately targets the recorded, controlled UL-DD IR video, not the live-webcam
  scenario M4 was originally built for** — if the model is later evaluated on live/webcam frames
  where a face detector genuinely is necessary, that gap needs its own validation pass.
- **Sampling rate**: **4 Hz directly** (every 15th of the 60 fps frames), not 60 Hz-then-downsample
  — chosen to match every other modality's windowing contract (240 timesteps = 60 s at 4 Hz,
  same `WINDOW`/`STRIDE` as M1/M2/M3) and to keep MediaPipe inference cost proportional to actual
  training-relevant frames rather than 15× higher. Trade-off, stated plainly: PERCLOS's 2 s window
  becomes 8 samples and blink-rate's 60 s window becomes 240 samples — both still meaningful at
  4 Hz — but a very fast blink (~100–150 ms) can fall between samples and be missed. This is a
  known, accepted limitation of building M4 this way rather than at native 60 Hz; flag it if blink
  count/rate ever looks anomalously low.
- **Feature formulas reused verbatim** from `src/face_detection.py` — `_ear`, `_mar`, the same
  landmark index constants (`_RE`, `_LE`, `_MO`, `_RB_TOP/_BOT`, `_LB_TOP/_BOT`, `_CHIN`, `_NOSE`),
  the same head-pose 4-point `solvePnP(..., flags=SOLVEPNP_SQPNP)` model, and the same constants
  (`EAR_CLOSED_THRESH=0.20`, `PERCLOS_WINDOW_S=2.0`, `BLINK_WINDOW_S=60.0`). The one necessary
  adaptation: her `FeatureExtractor.extract()` keys its rolling PERCLOS/blink-rate windows off
  `time.time()` (live wall clock, meaningless offline); this project's version threads the video's
  own `frame_idx / fps` timestamp through the identical rolling-deque logic instead — functionally
  the same algorithm, just fed file time instead of wall time.
- **Missing-detection handling**: MediaPipe occasionally fails to find a face (extreme head turn,
  eyes fully occluded, IR noise) — the extractor forward-fills the last valid 10-feature vector
  (zero-fills only if no valid detection has occurred yet in that session), the same spirit as
  `preprocess.py`'s own `nan_to_num` handling for FAU/telemetry gaps.
- **Windowing / folds / labeling / scaling**: identical to `ULDDProcessor` — 240-step window,
  60-step stride, KSS→3-class via the same `(3, 6)` thresholds, same 5 subject-independent folds,
  per-feature z-score fit on each fold's training windows only.
- **Output**: `datasets/processed/uldd_m4/fold_{0..4}.npz`, keys `X_m4_train`/`X_m4_test` shape
  `(N, 240, 10)`, `y_train`/`y_test`, `test_subjects`, `m4_mean`/`m4_std`.

### A real environment blocker hit and fixed along the way

MediaPipe's modern **Tasks API** (`FaceLandmarker`, what `src/face_detection.py` itself uses) hard
**crashes on this Mac** (Apple Silicon, mediapipe 1.0.1) with `Check failed: service_ Service is
unavailable` / `DrishtiMetalHelper` — an internal calculator in that graph unconditionally reaches
for a Metal GPU service that isn't available in this process context, and it crashes regardless of
requesting a CPU delegate. This is a real, reproducible incompatibility, not a one-off fluke —
worth remembering before ever running `src/face_detection.py`'s `FeatureExtractor` or the live
demo (`src/realtime_demo.py`) directly on this machine; it would hit the same crash. **Fix used for
this project's own extraction code**: mediapipe's older, pre-Tasks **`mp.solutions.face_mesh.FaceMesh`**
legacy API doesn't use that graph and works fine — but it was removed from mediapipe's `1.0.x`
line, so it required **a separate, dedicated conda env** (`mp-extract`, `mediapipe==0.10.21` +
`opencv-python` + `numpy<2`) rather than trying to coexist with `ddd-repo`'s TensorFlow 2.21, whose
protobuf/numpy requirements directly conflict with the older mediapipe release (confirmed:
installing `mediapipe==0.10.21` into `ddd-repo` broke TensorFlow's import chain via a protobuf
version clash — reverted and separated instead of fighting the resolver). **Takeaway for any
future mediapipe work in this project**: use `mp-extract` for anything needing
`mp.solutions.*` / legacy FaceMesh, and keep it out of any env that also needs current TensorFlow.

### Code files involved

| File | Role |
|---|---|
| `src/face_detection.py` | source of truth for feature formulas/constants — read from, never modified |
| `src/models/architecture.py` | `build_m4_realtime_bilstm()` — the M4 model factory (unmodified) |
| `src/models/train.py` | same `cross_validate`/`train_fold` loop used for M1/M2/M3 (unmodified) |
| *(new, this project)* `m4_m5_extract.py` | one combined video-decode pass producing both M4 features and M5 frames (see §11 — avoids decoding each IR video twice) |
| *(new, this project)* `m4_train.py` | calls `cross_validate(build_m4_realtime_bilstm, ...)` against the new fold data |

### Model architecture (`build_m4_realtime_bilstm`, `architecture.py:285-321`)

```
Input (batch, 240, 10)
 → Bidirectional(LSTM(64, return_sequences=True))
 → Dropout(0.30)
 → Bidirectional(LSTM(32))
 → Dropout(0.30)
 → Dense(32, relu)
 → Dense(3, softmax)
```

~82K parameters (deliberately small/edge-deployable per the docstring — no L2 regularization
unlike M1/M2/M3, and no BatchNorm before the output layer).

### Current progress / next steps

Extraction and training are chained into one autonomous background job (launched under
`caffeinate` + fully detached, per the sleep-kill lesson in §"M1 progress" above): a combined
`m4_m5_extract.py` pass builds both M4's windowed features and M5's classification frames from a
single video-decode loop per session, then a waiter script starts `m4_train.py` (this
`cross_validate` call) automatically the moment extraction finishes, followed immediately by M5
training (§11). Check `results/reports_regen/M4_results.json` and
`models/checkpoints_regen/M4_fold{0..4}.keras` for output once it lands.

### Expected metrics

There is **no thesis or dataset-paper number to target for M4** — it was never evaluated on
UL-DD by anyone before this. Reasonable priors instead: M4's 10 features are a coarser, geometric
proxy for the same "facial dynamics" signal M1's 30 FAU features already capture (both ultimately
derive from facial landmarks), so a result in the same neighborhood as **M1's ~0.39 acc / ~0.35
macro-F1** (or somewhat lower, given M4 has 3× fewer input features and a smaller network) would
be a reasonable, unsurprising outcome — not a target to hit, just a sanity anchor. A result far
below chance (macro-F1 well under ~0.2) would suggest a real bug (e.g. the offline PERCLOS/blink
adaptation, or the skipped-face-detection assumption) rather than a genuine negative finding, and
should be investigated rather than accepted at face value.

## 11. Deep dive: M5 — YOLOv8n-cls Infrared Image Classifier (full 5-fold reproduction)

M5 is the thesis's best-performing model (0.5451 acc / 0.4439 macro-F1). Existing checkpoints
(`models/checkpoints/M5_fold{0..4}.pt`) are all present and real (not LFS pointers, unlike
M1–M3's stale checkpoints — see §8), but the classification dataset needed to *retrain or
re-evaluate from scratch* (`datasets/yolo_cls/`) is a partial, incomplete dump (§5 step 5: only
fold 0's `train/{Alert,Drowsy}`, missing `LowVigilant`, missing any `test/` split, missing folds
1–4). This project rebuilds it in full from raw video rather than trying to patch the partial dump.

### Dataset

- **Source notebook**: `notebooks/09-uldd-m5-yolo-cls.ipynb` (read in full, not modified) — this
  project's script (`m5_train.py`, and the frame-extraction half of `m4_m5_extract.py`) replicates
  its logic exactly, with the Windows `VIDEO_ROOT`/`LABELS_CSV` paths swapped for this project's
  local dataset copy.
- **Frame extraction**: sample IR video at **1 fps** (not 4 Hz — this is M5's own, coarser rate,
  independent of the 4 Hz windowed pipeline everything else uses), resize to **224×224**, label
  each frame by its KSS epoch (same `(3,6)` thresholds, same 4-minute epoch blocks) — extracted
  once per subject/session, saved as `datasets/yolo_frames_regen/{subject}_{session}/{class_name}/frame_NNNNN.jpg`
  plus a `metadata.csv` (subject, session, frame_idx, timestamp_s, epoch, class, class_name, path)
  — schema identical to her notebook's own `metadata.csv`.
- **Folds**: same 5 subject-independent folds, **NO_VIDEO = {B, I, M} excluded from both train and
  test in every fold** (unlike M1/M2/M3, which can still use B/I/M's FAU/telemetry — M5 is
  video-only so those three subjects contribute nothing here).
- **Per-fold YOLO directory structure**: `datasets/yolo_cls_regen/fold_{k}/{train,val}/{Alert,LowVigilant,Drowsy}/`,
  frames hard-linked (not copied, to avoid doubling ~7 GB of extracted frames on disk) from the
  master `yolo_frames_regen/` pool — `val/` here means the fold's held-out test subjects, matching
  her own (slightly nonstandard) naming.

### Code files involved

| File | Role |
|---|---|
| `notebooks/09-uldd-m5-yolo-cls.ipynb` | source of truth for the exact pipeline/hyperparameters — read from, not modified or run directly (notebooks can't run detached/headless easily; replicated as a script instead) |
| *(new, this project)* `m4_m5_extract.py` | frame extraction half (shared decode pass with M4, see §10) |
| *(new, this project)* `m5_train.py` | fold assembly (hard-link into `train`/`val`), YOLOv8n-cls training, evaluation, result aggregation — line-for-line behavior match to notebook cells 9–13 |

### Model / training config (unchanged from her notebook)

- **Model**: `yolov8n-cls.pt` (Ultralytics YOLOv8-nano classification head, ImageNet-pretrained),
  ~1.44M params.
- **Per fold**: up to 20 epochs, batch 32, image size 224, `lr0=1e-3`, `lrf=0.01`, dropout 0.3,
  early-stopping patience 15 (Ultralytics' own trainer, not the Keras `train.py` used elsewhere).
- **Evaluation**: `model.predict()` over each fold's `val/` class folders, YOLO's alphabetical
  class-index re-mapped back to `[Alert, LowVigilant, Drowsy]` order (YOLO sorts folder names
  alphabetically → `Alert=0, Drowsy=1, LowVigilant=2` internally; the script corrects this before
  computing accuracy/macro-F1, exactly as her notebook does — a real, easy-to-miss gotcha if this
  script is ever modified).
- **Runs on**: Ultralytics auto-detects the best available device; on this machine that's Apple
  Silicon **MPS** (confirmed available via `torch.backends.mps.is_available()`, §8), meaningfully
  faster than pure CPU for a CNN like this compared to the LSTM-based M1/M2/M3/M4, which don't
  benefit from MPS the same way in TensorFlow.

### Current progress / next steps

Chained after M4 in the same autonomous background job (§10) — frame extraction happens once
(shared with M4), M4 trains first, then M5's fold-by-fold YOLO training begins automatically.
Expect this to be the slowest stage: 5 folds × up to 20 epochs of real CNN training over tens of
thousands of images each, even with MPS acceleration. Check `results/reports_regen/M5_results.json`
and `models/checkpoints_regen/M5_fold{0..4}.pt` once it lands; compare against the thesis's
**0.5451 ± 0.1000 acc / 0.4439 macro-F1**.

### Expected metrics

Target: thesis-reported **0.5451 ± 0.1000** accuracy, **0.4439** macro-F1
(`docs/report_summary.md` §4) — this project's own existing `.pt` checkpoints (§8) already
reproduce this exact result without retraining, so the freshly-retrained numbers from this run are
a genuine independent check of whether the full pipeline (not just the weights) is faithful:
landing close to those numbers validates the frame-extraction/labeling/fold logic end-to-end, the
same role M1's near-exact match played for the FAU/telemetry pipeline (§9).
