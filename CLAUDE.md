# Masters Project — Driver Drowsiness Detection (UL-DD)

## What this project is

A Master's project building a **multimodal / hybrid / ensemble model for driver drowsiness
detection**, using the UL-DD dataset, in the spirit of (and explicitly meant to improve on)
`Kavipriya_report (2).html` — a prior project on the same dataset.

**Status (2026-11-06):**
- M1–M7 are reproduced or rebuilt; see `docs/models_summary.md`.
- The final phase runs in `m8/`, `scripts/m8/` and `results/m8/`:
  - per-variant results and architecture diagrams are in `docs/m8_models.md`;
  - the schedule is `model_research/plan_14day.md`; round 2 is `model_research/round2_research.md`.
- M8 (VIGIL MoE: IR-112 DINOv2 + telemetry/clock fallback) is frozen.
- **Deliverables are drafted (2026-11-08):**
  - thesis narrative and claims: `docs/thesis_story.md`;
  - report: `report/report.md` → `report/thesis_report.docx` via `report/build_docx.py`;
  - slides: claude.ai Slides artifact, sources in `report/slides/`;
  - deployment models in `models/final/`, inference in `m8/inference.py`;
  - Gradio demo: `app/demo.py`;
  - figures: `scripts/m8/make_figures.py` → `results/figures/`.
- `docs/story_so_far.md` is always the most current status.

Read these before doing anything substantive:
- **`docs/story_so_far.md`** — chronological narrative log of the project: what's been done, what
  was decided, and why, newest first. Read this first to get oriented on where things stand.
- **`docs/dataset_summary.md`** — full breakdown of UL-DD: modalities, file formats, sample
  rates, folder layout, known per-subject data gaps, the dataset authors' own baseline results.
- **`docs/report_summary.md`** — full breakdown of Kavipriya's prior *written report*: models
  tried (M1–M7), results, what she did *not* try (biometric signals, grip pressure, pose/depth
  video). Where `docs/repo_understanding.md` (her actual code) disagrees with this, the repo wins.
- **`docs/repo_understanding.md`** — full breakdown of Kavipriya's actual cloned codebase
  (`driver-drowsiness-detection-system/`): canonical vs. superseded files, module/notebook map,
  reproduction steps, data-expectation checks, known issues.
- **`Project_Onboarding.txt`** — the original task brief/roadmap (replicate baselines → design →
  build → run on mini then full dataset with wandb logging → experiments → results → conclusion).
- **`Literature/ULDD Dataset.pdf.md`** — full text of the UL-DD dataset paper (markdown-converted).

## Standing rule: keep `docs/story_so_far.md` current

`docs/story_so_far.md` is the project's running narrative log — a chronological record of what
was done, what was decided, and why, newest entries first. **Update it whenever a milestone is
reached, a decision is made, or the direction changes** (e.g. a new doc is produced, an
environment/setup change is made, results are reproduced, a modeling approach is chosen or
dropped, scope is clarified with the user). Keep entries short and factual, and link to the
relevant doc/file for detail instead of duplicating it inline. This is a durable rule for the
whole project, not a one-off task — apply it in every future session without being asked again.

## Repo layout

```
Masters Project/
├── CLAUDE.md                      # this file
├── Project_Onboarding.txt         # original task brief
├── docs/
│   ├── story_so_far.md            # chronological log (newest first)
│   ├── dataset_summary.md
│   ├── report_summary.md
│   ├── repo_understanding.md
│   ├── literature_review.md
│   ├── models_summary.md          # M1–M7 (reproduction phase) + diagrams
│   ├── m8_models.md               # final phase: R*, C1–C12, M8, round 2 + diagrams
│   └── thesis_story.md            # thesis narrative, novelty claims C1–C6, what is not claimed
├── model_research/                # plan_14day.md, round2_research.md, ablation_plan.md, surveys
├── m8/                            # final-phase package (windows, extract, data, heads, train, vigil, events, inference)
├── scripts/                       # per-model entry points; scripts/m8/ = final-phase runs, queues, make_figures.py
├── models/final/                  # deployment models refit on all 16 video subjects (vstar, vstar_calA, m8, events)
├── app/demo.py                    # Gradio demo (replay UL-DD drives, upload video, results)
├── report/                        # report.md, build_docx.py → thesis_report.docx; slides/ = deck sources
├── results/m8/                    # per-run JSON + runs/<name>/preds.npz
├── results/figures/               # report/slide figures (f0–f9), from out-of-fold predictions
├── cache/                         # gitignored: windows, embeddings, logs
├── Literature/
│   ├── ULDD Dataset.pdf(.md)      # dataset paper + markdown conversion
│   └── Kavipriya_Ramasamy_..._Final_Phase (2).pdf   # prior report, PDF
├── Kavipriya_report (2).html      # prior report, HTML (same content as the PDF above)
├── ULDD Dataset.pdf.html          # dataset paper, HTML render (same content as Literature/ copy)
├── driver-drowsiness-detection-system/   # Kavipriya's actual code, cloned as its own nested git repo (in progress)
└── DriverDrowsinessDataset/
    ├── Info.xlsx                  # modality catalogue + per-subject/session availability matrix
    ├── Labels.csv                 # KSS labels, all subjects/sessions
    ├── CSV_Files/CSV_Files/{A-S}/{A,D}/          # raw biometric+behavioral time series
    ├── Extracted_Features/Extracted_Features/{A-S}/{A,D}/   # FL/PL/FAU csvs
    ├── Video_Data/Video_Data/{A-S}/{A,D}/        # IR/3D/Pose mp4s (missing for B, I, M)
    ├── Codes/Codes/                # official notebooks: preprocessing, feature extraction, baselines
    ├── ULDD Dataset/               # empty leftover folder, no data — ignore
    └── *.zip                       # original zipped downloads, redundant with unzipped copies
```

Subject IDs are letters **A–S** (19 subjects). Session suffix **A = Alert**, **D = Drowsy**.
File naming is always `<Subject>_<Signal>_<Session>.<ext>`. Full modality/format details are in
`docs/dataset_summary.md` — don't re-derive them from scratch, read that file first.

## Environment

- This machine's `/usr/bin/python3` is the **Xcode Command Line Tools stub** and refuses to run
  ("You have not agreed to the Xcode license agreements... sudo xcodebuild -license") until that
  license is accepted with `sudo`. Left as-is (never run `sudo xcodebuild -license` without
  explicit user confirmation) — worked around entirely via conda instead, see below.
- No `node`, no working system Python outside conda, no `pandoc`. `perl` and macOS's built-in
  `textutil` do work and were used to parse `Info.xlsx` (raw XML via `perl`) and convert the HTML
  report to text (`textutil -convert txt`) during initial dataset exploration.
- **A working project environment now exists**: conda env **`uldd`** (Python 3.11.11, matching
  `DriverDrowsinessDataset/Codes/Codes/Readme.md`'s stated version), created via the machine's
  existing Anaconda install at `/opt/anaconda3` (activate with `conda activate uldd`, or call the
  interpreter directly at `/opt/anaconda3/envs/uldd/bin/python`). All packages from
  `DriverDrowsinessDataset/Codes/Codes/requirements.txt` are installed and import-verified
  (pandas, scikit-learn, opencv-python, statsmodels, scipy, fastparquet, pyarrow, matplotlib,
  seaborn, jupyterlab, etc.). **`dlib` was installed via `conda install -c conda-forge dlib`**
  instead of pip — `pip install dlib==19.24.6` tries to compile from source and needs `cmake` +
  a C++ compiler, and the system compiler is blocked by the same Xcode license issue above;
  conda-forge ships a prebuilt binary (resolved to `19.24.2`) that avoids all of that. If any
  other package in the requirements file also fails to build from source in the future, prefer
  `conda install -c conda-forge <pkg>` over fighting the compiler toolchain.
- Git is initialized at the project root (`git init` done, `.gitignore` added excluding
  `DriverDrowsinessDataset/` — tens of GB, see `docs/dataset_summary.md` §7 — plus OS/editor/
  Python/notebook-checkpoint/wandb cruft). Nothing has been staged or committed yet; do that only
  when the user explicitly asks.
- **`driver-drowsiness-detection-system/`** — Kavipriya's actual codebase
  (`github.com/kavipriyariisc/driver-drowsiness-detection-system`) is being cloned directly into
  the project root by the user as a **separate nested git repo** (it has its own `.git`, remote
  `origin` already pointed at that URL). It is *not* a submodule of the outer repo — git will
  just see it as an untracked directory. Do not run `git add`/commit on it from the outer repo
  without checking with the user first (nested repos are usually meant to stay independent or be
  added deliberately as a submodule, not silently absorbed). Once the clone finishes, treat that
  folder's own README/code as the authoritative source for exactly what Kavipriya built — more
  authoritative than the prose descriptions in `docs/report_summary.md`, which were inferred from
  her write-up only.

## Known open questions / decisions not yet made

These were raised when this file was created and should be resolved with the user before or
during early project work — see chat history for the original ask, or just ask again if unclear:
- *Resolved 2026-11-05/06:*
  - **Scope:** video / visual cues, posture, telemetry and time. Physiological CSVs are not model
    inputs; BVP may only validate rPPG.
  - **Protocol:** her 5-fold subject-independent folds and 3-class KSS binning are kept, and LOSO
    is added as co-primary.
  - **Deliverables** (14–21 days): report, PPT, codebase and a Gradio demo.
  - See `docs/story_so_far.md` 2026-11-05.
- Compute: **local machine is primary**; Google Colab and, if needed, a rented Azure VM are the
  fallbacks for anything GPU-heavy (the user's own words) — don't assume a GPU is available by
  default, and don't provision cloud compute without being asked.
- Experiment tracking: wandb is set up (see the section below).

## Working conventions

- Modeling code follows a clear split mirroring the dataset's own
  modality families (biometric / facial / behavioral) plus a fusion layer, consistent with both
  the dataset paper's pipeline (Fig. 5: Data Collection → Individual Pre-processing → Feature
  Extraction → Multimodal Pre-processing → Fusion) and Kavipriya's model family naming (M1, M2,
  M3...) if this project continues that numbering for comparability.
- Treat `docs/dataset_summary.md` and `docs/report_summary.md` as living references — update them
  if the dataset folder structure changes (e.g. after running the official notebooks) or if new
  literature is added to `Literature/`.

## Experiment tracking (wandb)

Entity `m25ai2065-iitj`, project `drowsiness_project`. Authenticated locally via `wandb login`
(credentials live in this machine's `~/.netrc`, **never in a tracked file** — two API keys were
briefly pasted into this file and into chat and are treated as burned; if wandb auth ever needs
resetting, generate a fresh key at wandb.ai/authorize and run `wandb login <key>` again, don't
paste the raw key into any file in this repo). `wandb` is installed in the `ddd-repo` conda env.
See `docs/story_so_far.md` for which runs have been backfilled/logged so far.

## GitHub

Intended remote: `https://github.com/karthikeyansundaravaradhan/MTP.git`. Not yet pushed as of
this writing — decide with the user what should actually go there (docs/scripts/code, not the
multi-GB `DriverDrowsinessDataset/` which is gitignored, and not `driver-drowsiness-detection-system/`
which is its own separate nested repo with its own remote).