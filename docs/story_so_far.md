# Story So Far

A running narrative log of this Master's project. Update this file whenever a milestone is
reached, a decision is made, or direction changes — see the standing rule in `CLAUDE.md`. Newest
entries at the top. Keep entries short and factual; link out to other docs for detail rather than
duplicating them here.

---

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
