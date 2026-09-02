# Session Handoff — Driver Drowsiness Detection (UL-DD) Masters Project

Snapshot of a full working session, written so a **new Claude Code session in a new location**
can pick this project up cold. This is a one-time snapshot, not a living doc — for ongoing
narrative going forward, keep updating `docs/story_so_far.md` instead (see the standing rule in
`CLAUDE.md`).

## 1. What this project is

Master's project building a multimodal/hybrid/ensemble driver-drowsiness-detection model on the
**UL-DD dataset**, explicitly meant to reproduce and then improve on a prior MTech thesis project
by "Kavipriya" (`Kavipriya_report (2).html`, her actual code cloned into
`driver-drowsiness-detection-system/`). Read `CLAUDE.md` first — it's the standing project
onboarding doc and points to everything else. In priority order, the docs to read are:

1. `CLAUDE.md` — project map, environment notes, standing rules
2. `docs/story_so_far.md` — full chronological narrative, newest entries first (**this session's
   detailed history lives here — read it in full for anything this handoff compresses**)
3. `docs/repo_understanding.md` — deep understanding of Kavipriya's actual codebase: what's real
   vs. fabricated documentation, module map, per-model (M1/M4/M5) deep dives, known bugs
4. `docs/dataset_summary.md` / `docs/report_summary.md` — UL-DD dataset details and her written
   thesis's content, respectively

## 2. Where things stand right now (end of this session)

| Model | Status | Result | Notes |
|---|---|---|---|
| M1 (FAU BiLSTM) | ✅ Reproduced | acc 0.390±0.114, macro-F1 0.357±0.108 | thesis: 0.391±0.116/0.348 — **near-exact match** |
| M2 (Telemetry LSTM) | ✅ Reproduced | acc 0.324±0.069, macro-F1 0.259±0.074 | thesis: 0.387±0.040/0.339 — somewhat lower |
| M3 (FAU+CAN cross-attention) | ✅ Reproduced | acc 0.352±0.120, macro-F1 0.322±0.125 | thesis: 0.399±0.098/0.378 — somewhat lower |
| M4 (real-time BiLSTM, 10 features) | ✅ **Trained for the first time ever** | acc 0.353±0.056, macro-F1 0.275±0.058 | never trained by Kavipriya at all — no target to compare against, see `docs/repo_understanding.md` §10 |
| M5 (YOLOv8n-cls on IR frames) | ⏳ **Not yet retrained** | existing checkpoints reproduce thesis's 0.545/0.444 without retraining | full 5-fold retrain from raw video crashed once (see §4 below), safe to relaunch now, **not yet restarted** |

M2/M3's shortfall vs. thesis is concentrated in one fold (fold 3 = subjects L/M/N/O) — the worst
fold for all three FAU/CAN models, flagged as worth a closer look rather than dismissed as noise
(`docs/repo_understanding.md` §9 "Results").

## 3. Environment — 3 separate conda envs, each for a reason

| Env | Python | Purpose | Key packages |
|---|---|---|---|
| `uldd` | 3.11.11 | The **official UL-DD dataset notebooks** (`DriverDrowsinessDataset/Codes/Codes/`) | pandas, sklearn, opencv, dlib (via `conda-forge`, **not pip** — pip tries to compile from source and needs cmake+compiler, blocked by this Mac's unaccepted Xcode license) |
| `ddd-repo` | 3.11 | Training M1–M5 (TensorFlow/Keras for M1-M4, PyTorch+Ultralytics for M5/M6) | tensorflow 2.21 (Keras 3), torch 2.13 (**MPS/Apple-GPU available**), ultralytics 8.4, opencv |
| `mp-extract` | 3.11 | **Only** for MediaPipe-based feature/frame extraction (M4/M5 raw-video processing) | `mediapipe==0.10.21` (pinned old version — see §5 gotcha #3), opencv, numpy<2, pandas |

**Never install mediapipe into `ddd-repo`** — confirmed it breaks TensorFlow's import chain via a
protobuf/numpy version conflict. Keep extraction (`mp-extract`) and training (`ddd-repo`) fully
separate.

Interpreters: `/opt/anaconda3/envs/{uldd,ddd-repo,mp-extract}/bin/python`. This machine's
`/usr/bin/python3` is an unusable Xcode-CLT stub (unaccepted license) — always use the conda envs
above, never system python3.

Git: outer project repo initialized (`git init` done, `.gitignore` excludes
`DriverDrowsinessDataset/`), nothing committed yet — do that only when explicitly asked.
`driver-drowsiness-detection-system/` is a separate nested git repo (her actual GitHub repo,
checked out on branch `feature/final` — **not `main`, which is stale/pre-M5**).

## 4. This session's biggest detour: a disk-space / iCloud incident

Partway through M5's retraining, a filesystem `TimeoutError` crash led to discovering this Mac's
`~/Desktop` (this whole project lives under `~/Desktop/Masters Project`) is iCloud-synced
(confirmed: `~/Library/Mobile Documents/com~apple~CloudDocs/Desktop` is a symlink back to the real
`~/Desktop`), and was auto-evicting/re-fetching this project's large dataset files under disk
pressure (a single video-frame read took 11 seconds instead of milliseconds; disk was at 85–99%
full). Resolved this session by:
- Verifying (file-count + exact-size comparison against each zip's central directory — **not** a
  full checksum, since that would force iCloud to re-fetch everything) that the unzipped copies of
  `CSV_Files.zip`, `Extracted_Features.zip`, `Video_Data.zip` were byte-identical, then deleting
  those 3 zips (~10.9 GB freed; disk usage dropped from 85% → 29%, 40 GB free now).
  `Codes.zip` had one 3-byte mismatch in one notebook file — left untouched (harmless, tiny).
- User was given manual steps to disable iCloud Desktop & Documents sync via System Settings (no
  safe scriptable/terminal path exists for this) — **whether this was actually done is unknown to
  this handoff; check with the user**.

**If dataset reads ever become mysteriously slow again on this Mac, suspect this exact issue
first** before assuming a code bug.

## 5. Gotchas / lessons learned this session (read before repeating any of this work)

1. **Background training jobs die silently if the Mac sleeps.** The very first retraining attempt
   (M1/M2/M3) died mid-fold when the machine went to idle sleep, killing the unprotected
   background process — no error, just a log that goes silent. **Fix used throughout the rest of
   the session**: always launch long jobs as
   `nohup caffeinate -ims <command> > log 2>&1 < /dev/null & disown` — fully detached (survives
   Claude Code session restarts) and sleep-proofed (survives the Mac going idle). Verify with
   `ps aux | grep caffeinate` after launching.
2. **Checkpoint files in `driver-drowsiness-detection-system/models/checkpoints/` for M1/M2/M3 are
   Git LFS pointer stubs, not real weights** — `git-lfs` wasn't installed at clone time, so the
   clone only pulled 132-byte pointer text. Installed via `brew install git-lfs` — but this
   session ultimately **retrained from scratch instead of pulling LFS** once it became clear the
   old checkpoints and the currently-available dataset copy were internally inconsistent (see
   `docs/repo_understanding.md` §8 for the fold-size-mismatch evidence). M5's `.pt` checkpoints
   and M6's results, by contrast, were confirmed real/consistent — LFS pull was still worth doing
   for those.
3. **MediaPipe's modern Tasks API (`FaceLandmarker`) hard-crashes on this Mac** (Apple Silicon,
   mediapipe 1.0.1) with a Metal-GPU `Check failed: service_ Service is unavailable` error,
   regardless of CPU-delegate settings. Fix: use the older `mp.solutions.face_mesh.FaceMesh` API
   instead (works fine), which required pinning `mediapipe==0.10.21` in its own env (`mp-extract`)
   since that API was removed in mediapipe 1.x and its dependency pins conflict with TensorFlow's.
4. **`src/data/preprocess.py` assumes triple-nested dataset folders**
   (`Extracted_Features/Extracted_Features/Extracted_Features/...`), but this project's actual
   local copy is only **double-nested**. Always override `proc.fau_root`/`proc.tele_root` by one
   level when instantiating `ULDDProcessor` against this project's dataset — see
   `scripts/regen_and_train.py` for the exact pattern (construct normally, then reassign
   `.fau_root`/`.tele_root` before calling `.run()`).
5. **No random seed is set anywhere** in her training code (`preprocess.py`, `architecture.py`,
   `train.py`) — data pipeline is deterministic, but weight init/dropout are not. Don't expect
   bit-exact reproduction of her numbers; "same ballpark" is the right bar (and M1 hit it almost
   exactly, which is itself validation that the pipeline is faithful).
6. **M5's evaluation has a real class-index gotcha**: YOLO sorts class folder names alphabetically
   internally (`Alert=0, Drowsy=1, LowVigilant=2`), which does **not** match this project's
   `CLASS_NAMES = ['Alert', 'LowVigilant', 'Drowsy']` ordering — `m5_train.py` remaps this
   explicitly before computing metrics; don't lose that remap if this script is ever modified.
7. **Root-level docs in `driver-drowsiness-detection-system/` contain fabricated content.**
   `PROJECT_COMPLETION_REPORT.md` and `MULTI_MODAL_FUSION_SUMMARY.md` describe a model taxonomy
   and a 74.2%-accuracy result with zero corresponding code anywhere in the repo — never cite
   numbers from these two files. Full canonical-vs-ignore breakdown in
   `docs/repo_understanding.md` §1.

## 6. Scripts written this session (persisted to `scripts/`, not scratchpad)

These were originally written to a session-specific ephemeral scratchpad directory that would not
have survived a session restart — copied here specifically so they'd persist. **They all contain
absolute paths to this exact machine's project location
(`/Users/karthikeyansundaravaradhan/Desktop/Masters Project/...`) — if the project folder moves
(e.g. to somewhere under `~/Downloads`), these paths need updating before re-running anything.**

| Script | Purpose |
|---|---|
| `scripts/regen_and_train.py` | Regenerates `datasets/processed/ul_dd_regen/` from raw dataset (with the path-nesting fix, §5.4) and trains M1/M2/M3 via `cross_validate` |
| `scripts/m4_m5_extract.py` | Combined single-decode-pass extraction of M4's 10-feature windows and M5's classification frames from raw IR video (run in `mp-extract` env) |
| `scripts/m4_train.py` | Trains M4 via `cross_validate` against the extracted fold data (run in `ddd-repo` env) |
| `scripts/m5_train.py` | Full 5-fold YOLOv8n-cls training/eval, headless port of her notebook 09's cells 9–13 (run in `ddd-repo` env) |
| `scripts/chain_m4_m5.sh` | Waits for extraction to finish (polls the log for a `Done.` sentinel), then runs `m4_train.py` then `m5_train.py` automatically — this is what got interrupted by the disk-space crash partway into M5 |
| `scripts/verify_zips.py` | The zip-vs-unzipped-copy integrity check used in §4 (file count + exact size, no full checksum) |
| `scripts/reproduce_m123.py` | **Superseded** — an early attempt to evaluate the *original* (LFS-pointer-broken) checkpoints; kept only for history, not needed going forward |

## 7. Immediate next steps

1. **Relaunch M5 training.** Disk space is healthy now; the crash was disk-space-specific, not a
   code bug. Fold 0's partial `datasets/yolo_cls_regen/fold_0/` from the crashed run should be
   deleted first (it's incomplete) before re-running `scripts/m5_train.py` (chained after nothing
   this time — M4 is already done, so just run `m5_train.py` directly, no need to re-run
   `chain_m4_m5.sh` or re-do extraction). Launch it the safe way (§5.1: `caffeinate` + `nohup` +
   `disown`), expect it to be the slowest stage (5 folds × up to 20 epochs of real CNN training).
2. **Confirm with the user** whether they disabled iCloud Desktop sync (§4) — if not, keep an eye
   on disk space during the M5 run and consider re-flagging if it gets tight again.
3. Once M5 is reproduced, decide with the user: dig into fold 3's consistently weak performance
   across M1/M2/M3, or move on to the M6 alignment-fix rebuild (`docs/repo_understanding.md` §7.1)
   as the project's first genuinely new contribution beyond reproduction.
4. Update `docs/story_so_far.md` and `docs/repo_understanding.md` §11 with the final M5 numbers
   once that run completes (this handoff was written before that finished).
