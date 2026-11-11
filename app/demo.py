"""Gradio demo for the UL-DD drowsiness models.

    /opt/anaconda3/envs/ddd-repo/bin/python app/demo.py          # http://127.0.0.1:7860

Tabs
  1. Replay a UL-DD drive: per-window predictions over a 40-min drive next to the true KSS,
     the IR frame and face crop at any minute, and a simulated camera blackout for VIGIL.
     "Out-of-fold" uses the cross-validation predictions (the driver was held out from training,
     so this is the honest view); "deployment" runs the models in models/final/, which were
     trained on all 16 drivers (in-sample for UL-DD, shown for completeness).
  2. Upload a video: any IR face video -> V* (and calibrated V* when an alert reference video is
     given) through m8.inference.encode_video.
  3. Results: every model's scores, one sub-tab per evaluation method (Phase 1 protocol, 5-fold,
     LOSO, calibrated P4, camera blackout, binary B2, drive vs within-drive, known-driver splits,
     hypotheses), plus figures from results/figures/.
"""
import json
import sys
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "m8")]
from m8.config import CLASS_NAMES, RESULTS, VIDEO_ROOT  # noqa: E402
from m8.extract import fl_boxes, square_crop  # noqa: E402
from m8.inference import Predictor  # noqa: E402
from m8.windows import load_windows  # noqa: E402
from final_metrics import load  # noqa: E402

W = load_windows()
PRED = Predictor()
FIG = ROOT / "results" / "figures"
COL = {"vstar": "#2f6db5", "calA": "#138a72", "events": "#c2477a", "fusion": "#0b5d4b", "m8": "#7b4fb3"}
LABEL = {"vstar": "V* (IR DINOv2)", "calA": "Calibrated V*", "events": "Events LR",
         "fusion": "Calibrated V* + events", "m8": "VIGIL (fallback MoE)"}
OOF = {"vstar": "ir_112_mean_cls_mean_video_y3", "calA": "ir_112_mean_cls_mean_video_y3_ref240A",
       "events": "events_lr_y3", "fusion": "fusion_calA_events_y3"}
SESSIONS = sorted({f"{s} / {'Alert' if se == 'A' else 'Drowsy'}" for s, se in
                   W[W.avail_video == 1][["subject", "session"]].itertuples(index=False)})
_cache = {}


def _parse(choice):
    s, se = choice.split(" / ")
    return s, se[0]


def session_frame(subj, sess, source, blackout):
    key = (subj, sess, source, blackout)
    if key in _cache:
        return _cache[key]
    if source.startswith("Deployment"):
        df = PRED.predict_session(subj, sess, blackout_s=blackout)
    else:
        m = np.where((W.subject.values == subj) & (W.session.values == sess) & (W.avail_video.values == 1))[0]
        df = W.iloc[m][["subject", "session", "start_s", "epoch_idx", "kss", "y3", "y_b2"]].reset_index(drop=True)
        specs = dict(OOF, m8=f"vigil:C9d_ir112_moe_telclock:moe_telclock:{int(blackout)}")
        for name, spec in specs.items():
            lp = load(spec, W, "y3", 3)
            if lp is None:
                continue
            p = np.exp(lp[m])
            p /= p.sum(1, keepdims=True)
            PRED._put(df, name, p)
        d = np.load(RESULTS / "runs" / "C9d_ir112_moe_telclock" / "preds.npz")
        mi, ci = list(d["modes"]).index("moe_telclock"), list(d["conditions"]).index(int(blackout))
        g = np.full(len(W), np.nan)
        g[d["rows"]] = d["gate"][mi, ci]
        df["m8_gate"] = g[m]
    _cache[key] = df
    return df


def timeline(df, models, minute):
    fig, ax = plt.subplots(figsize=(9, 3.3))
    t = (df.start_s.values + 30) / 60
    ax.step(t, df.y3, where="mid", color="#1f2430", lw=2.5, label="True KSS class")
    for k in models:
        cols = [f"p_{k}_{c}" for c in range(3)]
        if cols[0] in df:
            ax.plot(t, df[cols].values @ np.arange(3), color=COL[k], lw=1.6, label=LABEL[k])
    if "m8_gate" in df and "m8" in models and np.isfinite(df.m8_gate).any():
        ax.fill_between(t, 0, 2 * df.m8_gate.values, color=COL["m8"], alpha=0.08, label="VIGIL gate × 2 (fallback weight)")
    ax.axvline(minute, color="#9ca3af", ls="--", lw=1)
    ax.set_yticks([0, 1, 2], ["Alert", "Low-vig.", "Drowsy"])
    ax.set_xlabel("Minutes into drive")
    ax.set_ylabel("Expected class")
    ax.set_xlim(0, 40)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def frame_at(subj, sess, minute):
    path = VIDEO_ROOT / subj / sess / f"{subj}_IR_{sess}.mp4"
    cap = cv2.VideoCapture(str(path))
    f = int(minute * 60 * 60)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, im = cap.read()
    if not ok:
        return None, None
    b = fl_boxes(subj, sess, np.array([f]))[0]
    crop = square_crop(im, *b, 112) if np.isfinite(b).all() else None
    if crop is not None:
        cx, cy, sd = b
        cv2.rectangle(im, (int(cx - sd / 2), int(cy - sd / 2)), (int(cx + sd / 2), int(cy + sd / 2)), (80, 200, 120), 3)
    return im[..., ::-1], (cv2.resize(crop, (224, 224), interpolation=cv2.INTER_NEAREST)[..., ::-1]
                           if crop is not None else None)


def replay(choice, source, blackout, models, minute):
    subj, sess = _parse(choice)
    df = session_frame(subj, sess, source, float(blackout))
    i = int(np.clip(np.searchsorted(df.start_s.values + 30, minute * 60), 0, len(df) - 1))
    row = df.iloc[i]
    lines = [f"**Window {int(row.start_s) // 60}:{int(row.start_s) % 60:02d}–{int(row.start_s + 60) // 60}:"
             f"{int(row.start_s + 60) % 60:02d}**: true KSS **{row.kss:.0f}** ({CLASS_NAMES[int(row.y3)]})", "",
             "| Model | Alert | Low-vigilant | Drowsy | Prediction |", "|---|---|---|---|---|"]
    for k in models:
        if f"p_{k}_0" in df:
            p = [row[f"p_{k}_{c}"] for c in range(3)]
            lines.append(f"| {LABEL[k]} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} | **{CLASS_NAMES[int(np.argmax(p))]}** |")
    if "m8_gate" in df and np.isfinite(row.get("m8_gate", np.nan)):
        lines.append(f"\nVIGIL gate (weight on telemetry + clock expert): **{row.m8_gate:.2f}**")
    from sklearn.metrics import f1_score
    present = sorted(set(df.y3))
    f1s = [f"{LABEL[k]} {f1_score(df.y3[df[f'pred_{k}'] >= 0], df[f'pred_{k}'][df[f'pred_{k}'] >= 0], average='macro', labels=present, zero_division=0):.2f}"
           for k in models if f"pred_{k}" in df]
    note = ("out-of-fold: this driver was not in training" if source.startswith("Out")
            else "deployment models saw this drive in training, so these numbers are in-sample")
    lines.append(f"\nWhole-drive window macro-F1 over the classes present ({note}): " + " · ".join(f1s))
    im, crop = frame_at(subj, sess, minute)
    return timeline(df, models, minute), "\n".join(lines), im, crop


def upload(video, ref, max_min, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload an IR face video first.")
    progress(0, desc="Detecting faces and encoding frames (DINOv2, 2 fps)")
    df = PRED.predict_video(video, ref, max_s=max_min * 60 if max_min else None,
                            progress=lambda f: progress(f * 0.9, desc="Encoding"))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    t = (df.start_s.values + 30) / 60
    for k in ["vstar", "calA"]:
        if f"p_{k}_0" in df:
            ax.plot(t, df[[f"p_{k}_{c}" for c in range(3)]].values @ np.arange(3), color=COL[k], lw=1.8, label=LABEL[k])
    ax.set_yticks([0, 1, 2], ["Alert", "Low-vig.", "Drowsy"])
    ax.set_ylim(-0.1, 2.1)
    ax.set_xlabel("Minutes into video")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = ROOT / "results" / "inference" / "upload_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    share = df.pred_vstar.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    msg = (f"{len(df)} windows · face not detected in {df.face_missing.mean():.0%} of sampled frames · V* predicted: "
           + ", ".join(f"{CLASS_NAMES[k]} {share[k]:.0%}" for k in range(3))
           + ("" if ref else " · add an alert reference video to run the calibrated model"))
    return fig, msg, str(out)


ABOUT = """
### What the models are
- **V\\***: frozen DINOv2-S/14 (with registers) on 112 px IR face crops at 2 fps, then a mean-pool linear head (3,843 trainable parameters).
- **Calibrated V\\***: the same head on tokens minus the driver's mean token over the first 4 minutes of their *Alert* drive.
- **Events LR**: 27 blink, PERCLOS, yawn, nod and posture features from UL-DD's 60 fps landmarks.
- **VIGIL (M8)**: V\\* plus a telemetry + clock expert, mixed by a gate driven by vision quality. It falls back when the camera fails.

### Headline numbers (subject-independent, 16 drivers, 3-class KSS)
| Model | 5-fold pooled F1 | LOSO pooled F1 |
|---|---|---|
| Clock only (time in drive) | 0.366 | — |
| M5 end-to-end CNN on IR frames (YOLOv8n-cls, R4s) | 0.362 | — |
| V\\* | 0.446 | 0.471 |
| VIGIL, clean / 60 s camera blackout | 0.428 / 0.349 (vision-only: 0.186) | — |
| Calibrated V\\* | 0.521 | 0.518 |
| Calibrated V\\* + events | **0.536** | **0.525** |

### Caveats
- In UL-DD, the drowsiness label largely coincides with *which drive* it is. Calibrated models separate the drives well (AUC 0.90), but within a single drive, time-in-drive orders drowsiness better than any vision model.
- Videos uploaded here are face-cropped with OpenCV's Haar detector instead of UL-DD's landmarks. On UL-DD video, the two crop methods agree on 69% of V\\* window decisions.
- This is a research demo, not a safety system.
"""


def _j(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def _rows(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


NICE = {"R1 clock": "R1 clock (time in drive)", "R4s honest M5": "M5 end-to-end CNN (R4s, inner-val checkpoints)",
        "V* IR-112 mean": "V* (DINOv2 IR-112, mean-pool)", "Pose MIL": "Pose DINOv2, MIL head",
        "IR+pose equal (exploratory)": "IR + pose, equal weights (exploratory)", "Events LR (27)": "Events LR (27 features)",
        "M8 clean": "VIGIL (M8), clean", "Vision-only (ModDrop) clean": "Vision-only (ModDrop), clean",
        "M8 60 s blackout": "VIGIL (M8), 60 s blackout", "Vision-only 60 s blackout": "Vision-only, 60 s blackout",
        "Calibrated V* (P4)": "Calibrated V*", "Calibrated V* + events (P4)": "**Calibrated V* + events**"}


def results_tables():
    fm = _j("final_metrics")
    t = {}
    # Phase 1 protocol
    t["Phase 1 protocol (M1–M7)"] = (
        "**Protocol:** subject-independent 5-fold (ABCD / EFGH / IJK / LMNO / PQRS); 60 s windows at 4 Hz indexed by "
        "FAU row; validation = 15% of training windows (M5: the test drivers, see note); no seeds. "
        "**Metric:** fold-mean ± SD over the 5 folds.\n\n" + _rows(
            ["Model", "Signals", "Architecture", "Accuracy", "Macro-F1"],
            [["M1", "30 FAU, 4 Hz", "BiLSTM 128 → BiLSTM 64 → Dense", "0.390 ± 0.114", "0.357 ± 0.108"],
             ["M2", "5 telemetry channels, 4 Hz", "LSTM 64 → LSTM 32 → Dense", "0.324 ± 0.069", "0.259 ± 0.074"],
             ["M3", "FAU + telemetry", "BiLSTM + LSTM, bidirectional cross-attention", "0.352 ± 0.120", "0.322 ± 0.125"],
             ["M4", "10 geometric features (EAR, MAR, PERCLOS, blinks, head pose)", "BiLSTM 64 → BiLSTM 32 → Dense",
              "0.353 ± 0.056", "0.275 ± 0.058"],
             ["M5 (as first run)", "IR frames, 1 fps, 224 px", "YOLOv8n-cls fine-tuned", "0.541 ± 0.080", "0.447 ± 0.071 †"],
             ["M6 Lite", "Frame embeddings + telemetry", "BiLSTM + BiLSTM, concat", "0.332 ± 0.032", "0.303 ± 0.022 ‡"],
             ["M6 Full", "Frame embeddings + telemetry", "Transformer + BiLSTM, cross-attention", "0.318 ± 0.058",
              "0.292 ± 0.044 ‡"],
             ["M7", "16 IR frames per window", "EfficientNet-B0 → BiLSTM 256 → attention pool", "0.452 ± 0.052",
              "0.355 ± 0.063"]])
        + "\n\n† Checkpoint and early stopping were chosen on the test drivers. Re-run with inner-validation selection "
          "under the final protocol it scores 0.333 fold-mean / 0.362 pooled (next tab, R4s).  \n"
          "‡ Frame embeddings came from one fold's backbone, so folds 1–4 are optimistic.")
    # Final 5-fold
    r0, r6, r7 = _j("R0_R1_baselines")["y3"], _j("R6_tabular"), _j("R7_image_stats_probe")["y3"]
    base = [["R0 majority", "—", _f(r0["R0_majority"]["fold_mean_f1"]), _f(r0["R0_majority"]["pooled"]["macro_f1"]),
             "—", "—"],
            ["R0 stratified random", "—", _f(r0["R0_stratified"]["fold_mean_f1"]),
             _f(r0["R0_stratified"]["pooled"]["macro_f1"]), "—", "—"],
            ["R1b clock + session label (shortcut bound)", "—", _f(r0["R1b_clock_session"]["fold_mean_f1"]),
             _f(r0["R1b_clock_session"]["pooled"]["macro_f1"]), "—", "—"],
            ["R7 image statistics", "—", _f(r7["fold_mean_f1"]), _f(r7["pooled"]["macro_f1"]), "—", "—"]]
    for k, lab in [("face_video", "R6 face-geometry stats"), ("tel_video", "R6 telemetry stats"),
                   ("face_tel_video", "R6 face + telemetry stats")]:
        sm = r6[k]["summary"]
        base.append([lab, "—", _f(sm["fold_mean_f1"]), _f(sm["pooled"]["macro_f1"]), "—", "—"])
    main = []
    for k, v in fm["y3"].items():
        if "LOSO" in k or "P4" in k or "blackout" in k:
            continue
        main.append([NICE.get(k, k), f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]", _f(v["fold_mean_f1"]),
                     f"**{v['pooled_f1']:.3f}**", _f(v["epoch_f1"]), _f(v["qwk"], 2)])
    abl = [["IR 112, MIL head", "0.410", "0.411"], ["IR 168, MIL head", "0.424", "0.410"],
           ["IR 112, temporal transformer", "0.411", "0.404"],
           ["IR 112, 240 s context (mean / MIL / transformer)", "0.455 / 0.444 / 0.470", "0.451 / 0.443 / 0.458"],
           ["L3D face 168 px (mean / MIL)", "0.332 / 0.326", "0.378 / 0.373"],
           ["Pose 224 px (mean / MIL)", "0.414 / 0.433", "0.470 / 0.499"],
           ["Landmark / AU series (mean / MIL)", "0.318 / 0.311", "0.386 / 0.355"],
           ["IR eye crops 10 fps (mean / MIL)", "0.360 / 0.367", "0.385 / 0.409"],
           ["DINOv3-S/16 instead of DINOv2", "0.329", "0.325"], ["V* with ordinal loss", "0.399", "0.404"],
           ["Telemetry expert alone", "0.362", "—"], ["IR + telemetry, learned late fusion", "—", "0.306"],
           ["IR + pose, learned weights", "0.405", "0.433"]]
    t["Final protocol: 5-fold (P0)"] = (
        "**Protocol:** same 5 subject-independent folds; canonical 60 s windows on the video clock (4,532 windows, 16 "
        "drivers); all selection on 4 subject-wise inner folds; 3 seeds. **Metrics:** pooled macro-F1 over all test "
        "windows with a 95% driver-bootstrap CI; fold-mean; epoch-level F1; QWK.\n\n### Baselines and controls\n"
        + _rows(["Model", "95% CI", "Fold-mean", "Pooled", "Epoch F1", "QWK"], base)
        + "\n\n### Main models\n" + _rows(["Model", "95% CI", "Fold-mean", "Pooled", "Epoch F1", "QWK"], main)
        + "\n\n### Single-stream and fusion ablations (none beats V*)\n" + _rows(["Variant", "Fold-mean", "Pooled"], abl))
    loso = [[NICE.get(k.replace(" LOSO", ""), k.replace(" LOSO", "")), f"**{v['pooled_f1']:.3f}**",
             f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]", _f(v["epoch_f1"]), _f(v["qwk"], 2)]
            for k, v in fm["y3"].items() if "LOSO" in k] + [["Pose DINOv2, MIL head", "0.409", "—", "—", "—"]]
    t["Final protocol: LOSO (V1)"] = (
        "**Protocol:** leave-one-subject-out over the 16 video drivers; same windows and heads as 5-fold.\n\n"
        + _rows(["Model", "Pooled", "95% CI", "Epoch F1", "QWK"], loso)
        + "\n\nPose drops from 0.499 (5-fold) to 0.409; IR-based models hold or improve.")
    p4 = [[NICE.get(k, k), f"**{v['pooled_f1']:.3f}**", f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]", _f(v["epoch_f1"]),
           _f(v["qwk"], 2), _f(v["partial_spearman_kss_given_time"], 2)]
          for k, v in fm["y3"].items() if "P4" in k]
    t["Calibrated (P4)"] = (
        "**Protocol:** P0 or LOSO, plus the first 240 s of the test driver's Alert drive, unlabelled, at test time "
        "(subtract its mean DINOv2 token). Fusion with events = mean log-probability, fixed in advance.\n\n"
        + _rows(["Model", "Pooled", "95% CI", "Epoch F1", "QWK", "Partial ρ (KSS | time)"], p4)
        + "\n\nFailed calibration variants: same-drive reference (C12) 0.368; absolute + residual tokens 0.405.")
    rob = []
    for f, lab in [("C9_ir112", "C9"), ("C9b_ir112_moe", "C9b"), ("C9c_ir112_fastgate", "C9c"),
                   ("C9d_ir112_moe_telclock", "C9d = VIGIL (M8)")]:
        s = _j(f)["summary"]
        for m in sorted({k.split("@")[0] for k in s}):
            vals = [s.get(f"{m}@{x}") for x in (0, 30, 45, 60)]
            g = [v.get("gate_mean") for v in vals if v]
            rob.append([f"{lab}: {m}"] + [_f(v["pooled"]["macro_f1"]) if v else "—" for v in vals]
                       + [f"{g[0]:.2f} → {g[-1]:.2f}" if g and g[0] is not None and m not in ("vision", "late") else "—"])
    t["Camera blackout"] = (
        "**Protocol:** 5-fold P0; each test window gets a contiguous camera blackout of L seconds at the same random "
        "position for every model and seed. Modes: vision = vision-only with the same ModDrop training; late = "
        "ungated telemetry correction; gate = residual gate; clock = clock-only residual; moe / moe_clock / "
        "moe_telclock = mixtures of experts.\n\n"
        + _rows(["Model: mode", "0 s", "30 s", "45 s", "60 s (camera off)", "Gate mean 0 → 60 s"], rob))
    b2 = [[NICE.get(k, k), f"**{v['pooled_f1']:.3f}**", f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]", _f(v.get("auc"), 2)]
          for k, v in fm["b2"].items()]
    t["Binary B2"] = (
        "**Protocol:** KSS ≤ 5 vs ≥ 6, 5-fold P0 unless marked LOSO / P4. B2 equals 'which drive' for 91% of windows, "
        "so read high scores together with the drive vs within-drive tab.\n\n"
        + _rows(["Model", "Pooled F1", "95% CI", "AUC"], b2))
    t["Drive vs within-drive"] = (
        "**Protocol:** out-of-fold 5-fold scores. Drive-AUC separates Drowsy-drive from Alert-drive windows; "
        "within-drive AUC orders windows only against others from the same driver and drive.\n\n" + _rows(
            ["Model", "B2 within-drive", "B2 drive-AUC", "KSS ≥ 7 within-drive", "KSS ≥ 7 drive-AUC"],
            [["Clock", "**0.81**", "0.50", "**0.90**", "0.50"], ["V*", "0.50", "0.66", "0.64", "0.79"],
             ["Pose MIL", "0.60", "0.62", "0.63", "0.67"], ["Events", "0.56", "0.59", "0.68", "0.59"],
             ["VIGIL (M8)", "0.50", "0.68", "0.66", "0.80"], ["Calibrated V*", "0.63", "**0.90**", "0.66", "**0.91**"],
             ["Calibrated V* + events", "0.64", "0.90", "0.70", "0.89"]]))
    sd = [["Unseen drivers (P0)", "0.446", "—", "—", "0.366"]]
    for nm, lab, extra in [("random", "Random windows", ("0.576", "0.376")),
                           ("epoch", "Held-out 4-min epochs, 60 s purge", ("0.563", "0.363")),
                           ("drive", "Known driver, new drive", ("0.093", "0.370"))]:
        s = _j(f"SD_{nm}")
        sd.append([lab, f"**{s['vstar']['pooled']['macro_f1']:.3f}**", _f(s["shortcut"]["pooled"]["macro_f1"]), *extra])
    t["Known-driver splits"] = (
        "**Protocol:** V* retrained with the test driver's other data in training; compared with no-video logistic "
        "regressions.\n\n" + _rows(["Split", "V*", "No video: who + drive + time", "Who + time", "Time only"], sd))
    t["Hypotheses (H1–H4)"] = (
        "**Protocol:** fixed before the final evaluation; exact subject-level paired permutation test on pooled "
        "macro-F1 (2¹⁶ sign flips), Holm-corrected.\n\n" + _rows(
            ["Hypothesis", "Δ pooled F1", "p", "Holm p"],
            [[k.replace("M8", "VIGIL").replace("R4s", "M5 (R4s)"), f"{v['diff_pooled_f1']:+.3f}", _f(v["p"]),
              _f(v["p_holm"])] for k, v in fm["H"].items()]))
    return t


def build():
    with gr.Blocks(title="UL-DD drowsiness demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Driver drowsiness from IR face video: UL-DD research demo")
        with gr.Tab("Replay a UL-DD drive"):
            with gr.Row():
                sess = gr.Dropdown(SESSIONS, value="K / Drowsy", label="Driver / drive")
                source = gr.Radio(["Out-of-fold (honest)", "Deployment models (in-sample)"],
                                  value="Out-of-fold (honest)", label="Prediction source")
                black = gr.Radio([0, 30, 45, 60], value=0, label="Camera blackout per 60 s window (s), VIGIL")
            models = gr.CheckboxGroup(list(LABEL), value=["vstar", "fusion", "m8"], label="Models",
                                      info="m8 = VIGIL")
            minute = gr.Slider(0.5, 39.5, value=20, step=0.25, label="Minute")
            plot = gr.Plot(label="Expected class over the drive")
            with gr.Row():
                info = gr.Markdown()
                frame = gr.Image(label="IR frame (landmark face box)", height=260)
                crop = gr.Image(label="112 px model input", height=260)
            inputs = [sess, source, black, models, minute]
            for c in inputs:
                c.change(replay, inputs, [plot, info, frame, crop])
            demo.load(replay, inputs, [plot, info, frame, crop])
        with gr.Tab("Upload a video"):
            gr.Markdown("Upload an IR (or grayscale) driver-face video. Optional: an alert reference video of the same "
                        "driver for the calibrated model (its first 4 minutes are used).")
            with gr.Row():
                vid = gr.Video(label="Drive video", sources=["upload"])
                ref = gr.Video(label="Alert reference video (optional)", sources=["upload"])
            max_min = gr.Slider(1, 40, value=10, step=1, label="Analyse the first N minutes")
            go = gr.Button("Run V*", variant="primary")
            up_plot = gr.Plot()
            up_msg = gr.Markdown()
            up_csv = gr.File(label="Per-window predictions (CSV)")
            go.click(upload, [vid, ref, max_min], [up_plot, up_msg, up_csv])
        with gr.Tab("Results"):
            with gr.Tabs():
                with gr.Tab("Overview"):
                    gr.Markdown(ABOUT)
                for name, md in results_tables().items():
                    with gr.Tab(name):
                        gr.Markdown(md)
                with gr.Tab("Figures"):
                    with gr.Row():
                        gr.Image(str(FIG / "f9_progression.png"), label="Baselines built up")
                        gr.Image(str(FIG / "f1_model_comparison.png"), label="Model comparison")
                    with gr.Row():
                        gr.Image(str(FIG / "f4_robustness.png"), label="Camera-failure robustness")
                        gr.Image(str(FIG / "f5_session_confound.png"), label="Drive vs within-drive signal")
                    with gr.Row():
                        gr.Image(str(FIG / "f2_confusion_y3.png"), label="Confusion matrices, 3-class")
                        gr.Image(str(FIG / "f3b_confusion_loso.png"), label="Confusion matrices, LOSO")
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7860)
