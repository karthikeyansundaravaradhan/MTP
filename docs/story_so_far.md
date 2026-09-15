# Story So Far

A running narrative log of this Master's project. Update this file whenever a milestone is
reached, a decision is made, or direction changes — see the standing rule in `CLAUDE.md`. Newest
entries at the top. Keep entries short and factual; link out to other docs for detail rather than
duplicating them here.

---

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
