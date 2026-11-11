"""Report / slide figures from the saved CV predictions (no training). Writes results/figures/*.png.

All model numbers are out-of-fold test predictions (5-fold P0 or LOSO), never the deployment
models in models/final/ (those are in-sample on UL-DD).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8.config import RESULTS, ROOT  # noqa: E402
from m8.windows import load_windows  # noqa: E402
from final_metrics import load  # noqa: E402

OUT = ROOT / "results" / "figures"
INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
C = dict(clock="#9ca3af", cnn="#c27c2c", feat="#b8a27a", seq="#d9a05b", vstar="#2f6db5", calA="#138a72", fusion="#0b5d4b",
         m8="#7b4fb3", events="#c2477a", pose="#5b8bd0", vision="#d1495b", shortcut="#9ca3af")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
CLS3, CLS2 = ["Alert\n(KSS≤3)", "Low-vigilant\n(4–6)", "Drowsy\n(≥7)"], ["KSS≤5", "KSS≥6"]
W = load_windows()
FM = json.loads((RESULTS / "final_metrics.json").read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def rows_of(lp):
    return np.where(~np.isnan(lp[:, 0]) & (W.avail_video.values == 1))[0]


# ---------------------------------------------------------------- 1. model comparison (y3)
def fig_comparison():
    items = [("R1 clock (time in drive)", "R1 clock", None, "clock"),
             ("M5 end-to-end CNN (YOLOv8n-cls, R4s)", "R4s honest M5", None, "cnn"),
             ("Events LR (27 blink/yawn/posture)", "Events LR (27)", "Events LR LOSO", "events"),
             ("VIGIL fallback MoE (clean input)", "M8 clean", None, "m8"),
             ("V* IR-112 DINOv2 mean-pool", "V* IR-112 mean", "V* LOSO", "vstar"),
             ("Calibrated V* (Alert reference)", "Calibrated V* (P4)", "Calibrated V* LOSO (P4)", "calA"),
             ("Calibrated V* + events fusion", "Calibrated V* + events (P4)", "Calibrated V* + events LOSO (P4)",
              "fusion")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (lab, k5, kl, col) in enumerate(items):
        r = FM["y3"][k5]
        ax.barh(i, r["pooled_f1"], color=C[col], height=0.62)
        ax.errorbar(r["pooled_f1"], i, xerr=[[r["pooled_f1"] - r["ci"][0]], [r["ci"][1] - r["pooled_f1"]]],
                    color=INK, lw=1, capsize=3)
        ax.text(0.012, i, f"{r['pooled_f1']:.3f}", va="center", color="white", fontsize=9, fontweight="bold")
        if kl:
            ax.plot(FM["y3"][kl]["pooled_f1"], i, "D", color="white", mec=INK, ms=6, zorder=5)
    ax.plot([], [], "D", color="white", mec=INK, label="LOSO (16 folds)")
    ax.axvline(FM["y3"]["R1 clock"]["pooled_f1"], color=C["clock"], ls="--", lw=1)
    ax.axvline(0.333, color=MUTED, ls=":", lw=1)
    ax.text(0.333, len(items) - 0.4, "chance ", ha="right", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(items)), [i[0] for i in items])
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Pooled macro-F1, 3-class KSS (bars: 5-fold, whiskers: subject-bootstrap 95% CI)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Subject-independent 3-class performance, 16 drivers", loc="left", color=INK)
    save(fig, "f1_model_comparison")


# ---------------------------------------------------------------- 2/3. confusion matrices
def cm_panel(ax, y, pred, labels, title, nc):
    cm = confusion_matrix(y, pred, labels=list(range(nc)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(nc):
        for j in range(nc):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.55 else INK)
    f1 = f1_score(y, pred, average="macro", labels=list(range(nc)))
    ax.set_title(f"{title}\nmacro-F1 {f1:.3f}", fontsize=9.5, color=INK)
    ax.set_xticks(range(nc), labels, fontsize=7.5)
    ax.set_yticks(range(nc), labels, fontsize=7.5)
    ax.grid(False)


def fig_confusions():
    specs = [("R1 clock", "R1"), ("M5 end-to-end CNN (R4s)", "frames:R4_seed0+R4_seed1+R4_seed2"),
             ("V* (IR-112 DINOv2)", "ir_112_mean_cls_mean_video_y3"),
             ("VIGIL (clean)", "vigil:C9d_ir112_moe_telclock:moe_telclock:0"),
             ("Calibrated V*", "ir_112_mean_cls_mean_video_y3_ref240A"),
             ("Calibrated V* + events", "fusion_calA_events_y3")]
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (t, s) in zip(axs.ravel(), specs):
        lp = load(s, W, "y3", 3)
        r = rows_of(lp)
        cm_panel(ax, W.y3.values[r], lp[r].argmax(1), CLS3, t, 3)
    for ax in axs[:, 0]:
        ax.set_ylabel("True KSS class")
    for ax in axs[-1]:
        ax.set_xlabel("Predicted")
    fig.suptitle("Confusion matrices, 3-class, 5-fold pooled test windows (row-normalised, counts in brackets)",
                 x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "f2_confusion_y3")
    specs = [("R1 clock", "R1"), ("V*", "ir_112_mean_cls_mean_video_y_b2"),
             ("VIGIL (clean)", "vigil:C9d_b2:moe_telclock:0"),
             ("Calibrated V*", "ir_112_mean_cls_mean_video_y_b2_ref240A")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (t, s) in zip(axs, specs):
        lp = load(s, W, "y_b2", 2)
        r = rows_of(lp)
        cm_panel(ax, W.y_b2.values[r], lp[r].argmax(1), CLS2, t, 2)
    axs[0].set_ylabel("True")
    fig.suptitle("Confusion matrices, binary KSS≤5 vs ≥6, 5-fold pooled", x=0.02, ha="left", color=INK,
                 fontsize=11)
    fig.tight_layout()
    save(fig, "f3_confusion_b2")
    lp = load("fusion_calA_events_y3_loso", W, "y3", 3)
    lv = load("ir_112_mean_cls_mean_video_y3_loso", W, "y3", 3)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.4))
    for ax, l, t in [(axs[0], lv, "V* (LOSO)"), (axs[1], lp, "Calibrated V* + events (LOSO)")]:
        r = rows_of(l)
        cm_panel(ax, W.y3.values[r], l[r].argmax(1), CLS3, t, 3)
    fig.tight_layout()
    save(fig, "f3b_confusion_loso")


# ---------------------------------------------------------------- 4. robustness (VIGIL)
def fig_robustness():
    m8 = json.loads((RESULTS / "C9d_ir112_moe_telclock.json").read_text())["summary"]
    c9 = json.loads((RESULTS / "C9_ir112.json").read_text())["summary"]
    L = [0, 30, 45, 60]
    lines = [("VIGIL: MoE, telemetry + clock fallback", [m8[f"moe_telclock@{x}"] for x in L], C["m8"], "o"),
             ("Residual telemetry gate (C9)", [c9[f"gate@{x}"] for x in L], C["pose"], "s"),
             ("Vision only (same ModDrop training)", [c9[f"vision@{x}"] for x in L], C["vision"], "^")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    for lab, s, col, mk in lines:
        a.plot(L, [v["pooled"]["macro_f1"] for v in s], marker=mk, color=col, lw=2, ms=7, label=lab)
    a.axhline(FM["y3"]["R1 clock"]["pooled_f1"], color=C["clock"], ls="--", lw=1)
    a.text(1, FM["y3"]["R1 clock"]["pooled_f1"] + 0.008, "clock baseline", color=MUTED, fontsize=8)
    a.set_xticks(L, ["0 (clean)", "30", "45", "60 (camera fully off)"])
    a.set_xlabel("Contiguous camera blackout inside each 60 s test window (s)")
    a.set_ylabel("Pooled macro-F1 (3-class)")
    a.set_ylim(0.15, 0.5)
    a.legend(loc="lower left", fontsize=8.5)
    a.set_title("Graceful degradation under camera failure", loc="left", color=INK)
    b.plot(L, [m8[f"moe_telclock@{x}"]["gate_mean"] for x in L], marker="o", color=C["m8"], lw=2, ms=7)
    b.plot(L, [c9[f"gate@{x}"]["gate_mean"] for x in L], marker="s", color=C["pose"], lw=2, ms=7)
    b.set_ylim(0, 1)
    b.set_xticks(L)
    b.set_xlabel("Blackout (s)")
    b.set_ylabel("Mean gate g (weight on fallback expert)")
    b.set_title("VIGIL's gate opens only when vision fails", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, "f4_robustness")


# ---------------------------------------------------------------- 5. session confound
def strat_auc(y, s, g):
    from sklearn.metrics import roc_auc_score
    num = den = 0
    for k in np.unique(g):
        m = g == k
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pos) and len(neg):
            n = len(pos) * len(neg)
            num += roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]) * n
            den += n
    return num / den


def fig_confound():
    from sklearn.metrics import roc_auc_score
    models = [("Clock", "R1", "clock"), ("V*", "ir_112_mean_cls_mean_video_{y}", "vstar"),
              ("Pose MIL", "pose_224_mil_cls_mean_video_{y}", "pose"), ("Events", "events_lr_{y}", "events"),
              ("VIGIL", None, "m8"), ("Calibrated V*", "ir_112_mean_cls_mean_video_{y}_ref240A", "calA"),
              ("Cal. V* + events", "fusion_calA_events_{y}", "fusion")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (y, nc, title) in zip(axs, [("y_b2", 2, "Binary KSS≥6"), ("y3", 3, "Drowsy KSS≥7 vs rest")]):
        out = {}
        for lab, spec, col in models:
            if spec is None:
                spec = "vigil:C9d_b2:moe_telclock:0" if y == "y_b2" else "vigil:C9d_ir112_moe_telclock:moe_telclock:0"
            lp = load(spec.format(y=y), W, y, nc)
            r = rows_of(lp)
            p = np.exp(lp[r])
            p = p[:, nc - 1] / p.sum(1)
            yy = (W[y].values[r] == nc - 1).astype(int)
            g = (W.subject.astype(str) + W.session.astype(str)).values[r]
            sess = (W.session.values[r] == "D").astype(int)
            out[lab] = (roc_auc_score(sess, p), strat_auc(yy, p, g))
            ax.scatter(*out[lab], s=90, color=C[col], edgecolor="white", lw=1.5, zorder=3)
            off, ha = {("y_b2", "V*"): ((-8, -3), "right"), ("y_b2", "Calibrated V*"): ((6, -13), "left"),
                       ("y_b2", "Cal. V* + events"): ((-6, 7), "right")}.get((y, lab), ((7, -3), "left"))
            ax.annotate(lab, out[lab], xytext=off, textcoords="offset points", fontsize=8.5, color=INK, ha=ha)
        ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.axvline(0.5, color=MUTED, ls=":", lw=1)
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Drive-AUC: separates the Drowsy drive from the Alert drive")
        ax.set_title(title, loc="left", color=INK)
    axs[0].set_ylabel("Within-drive AUC: orders windows of the same drive")
    fig.suptitle("What each model actually detects: the drive (x) vs drowsiness within a drive (y)", x=0.02,
                 ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "f5_session_confound")


# ---------------------------------------------------------------- 6. subject-dependent splits
def fig_subject_dependent():
    splits = [("Unseen drivers\n(5-fold P0)", FM["y3"]["V* IR-112 mean"]["pooled_f1"], None, None,
               FM["y3"]["R1 clock"]["pooled_f1"]),
              ("Known driver,\nnew drive", 0.413, 0.635, 0.093, 0.370),
              ("Held-out 4-min\nKSS epochs", 0.802, 0.867, 0.563, 0.363),
              ("Random windows\n(leaky)", 0.910, 0.926, 0.576, 0.376)]
    for nm in ["drive", "epoch", "random"]:
        s = json.loads((RESULTS / f"SD_{nm}.json").read_text())
        i = {"drive": 1, "epoch": 2, "random": 3}[nm]
        splits[i] = (splits[i][0], s["vstar"]["pooled"]["macro_f1"], s["shortcut"]["pooled"]["macro_f1"],
                     splits[i][3], splits[i][4])
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bw = 0.2
    series = [("V* (sees video)", 1, C["vstar"]), ("No video: who + which drive + time", 2, "#4b5563"),
              ("No video: who + time", 3, "#9ca3af"), ("No video: time only", 4, "#d1d5db")]
    for j, (lab, k, col) in enumerate(series):
        vals = np.array([s[k] if s[k] is not None else np.nan for s in splits], float)
        ax.bar(x + (j - 1.5) * bw, np.nan_to_num(vals), bw * 0.92, color=col, label=lab)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi + (j - 1.5) * bw, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [s[0] for s in splits])
    ax.set_ylabel("Pooled macro-F1 (3-class)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Seeing the test driver in training: high scores come from recognising the recording",
                 loc="left", color=INK, fontsize=10.5)
    save(fig, "f6_subject_dependent")


# ---------------------------------------------------------------- 7. timelines (out-of-fold)
def fig_timeline(subject="K"):
    specs = [("V*", "ir_112_mean_cls_mean_video_y3", "vstar"),
             ("Calibrated V* + events", "fusion_calA_events_y3", "fusion"),
             ("VIGIL (clean)", "vigil:C9d_ir112_moe_telclock:moe_telclock:0", "m8")]
    lps = {lab: load(s, W, "y3", 3) for lab, s, _ in specs}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, sess in zip(axs, ["A", "D"]):
        m = np.where((W.subject.values == subject) & (W.session.values == sess) & (W.avail_video.values == 1))[0]
        t = (W.start_s.values[m] + 30) / 60
        ax.step(t, W.y3.values[m], where="mid", color=INK, lw=2.5, label="True KSS class")
        for lab, _, col in specs:
            p = np.exp(lps[lab][m])
            p /= p.sum(1, keepdims=True)
            ax.plot(t, p @ np.arange(3), color=C[col], lw=1.5, alpha=0.9, label=f"{lab}: E[class]")
        ax.set_title(f"Driver {subject}, {'Alert' if sess == 'A' else 'Drowsy'} drive", loc="left", color=INK)
        ax.set_xlabel("Minutes into drive")
        ax.set_yticks([0, 1, 2], ["Alert", "Low-vig.", "Drowsy"])
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Out-of-fold predictions over two drives (driver held out from training)", x=0.02, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, f"f7_timeline_{subject}")


# ---------------------------------------------------------------- 8. per-subject LOSO
def fig_loso_subjects():
    a, b = (load(s, W, "y3", 3) for s in ["ir_112_mean_cls_mean_video_y3_loso", "fusion_calA_events_y3_loso"])
    r = rows_of(b)
    subs = sorted(set(W.subject.values[r]))
    va, vb = [], []
    for s in subs:
        m = r[W.subject.values[r] == s]
        y = W.y3.values[m]
        va.append(f1_score(y, a[m].argmax(1), average="macro", labels=sorted(set(y))))
        vb.append(f1_score(y, b[m].argmax(1), average="macro", labels=sorted(set(y))))
    o = np.argsort(vb)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(subs))
    for i, j in enumerate(o):
        ax.plot([i, i], [va[j], vb[j]], color=GRID, lw=2, zorder=1)
    ax.scatter(x, np.array(va)[o], color=C["vstar"], s=40, label="V*", zorder=2)
    ax.scatter(x, np.array(vb)[o], color=C["fusion"], s=40, label="Calibrated V* + events", zorder=3)
    ax.set_xticks(x, np.array(subs)[o])
    ax.set_xlabel("Held-out driver (LOSO fold), sorted")
    ax.set_ylabel("Macro-F1 over present classes")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Per-driver LOSO performance: driver variance dominates", loc="left", color=INK)
    save(fig, "f8_loso_per_subject")


# ---------------------------------------------------------------- 9. baseline ladder
# M1-M4, M6, M7 fold-mean F1 from their own runs (docs/models_summary.md); they use FAU-indexed windows.
SEQ = [("M1 FAU BiLSTM", 0.357), ("M2 Telemetry LSTM", 0.259), ("M3 FAU+Tel attention", 0.322),
       ("M4 Geometry BiLSTM", 0.275), ("M6 Emb.+Tel fusion", 0.303), ("M7 CNN-BiLSTM", 0.355)]


def fig_progression():
    fm = lambda k: FM["y3"][k]["fold_mean_f1"]  # noqa: E731
    r0 = json.loads((RESULTS / "R0_R1_baselines.json").read_text())["y3"]
    r6 = json.loads((RESULTS / "R6_tabular.json").read_text())
    r7 = json.loads((RESULTS / "R7_image_stats_probe.json").read_text())["y3"]
    stages = [("Floors", [("Majority (R0)", r0["R0_majority"]["fold_mean_f1"], "clock"),
                          ("Stratified random (R0)", r0["R0_stratified"]["fold_mean_f1"], "clock")]),
              ("No\nsensor", [("Clock (R1)", fm("R1 clock"), "clock")]),
              ("Hand-built\nfeatures", [("Image statistics (R7)", r7["fold_mean_f1"], "feat"),
                                        ("Face geometry stats (R6)", r6["face_video"]["summary"]["fold_mean_f1"], "feat"),
                                        ("Telemetry stats (R6)", r6["tel_video"]["summary"]["fold_mean_f1"], "feat")]),
              ("Sequence models", [(lab, v, "seq") for lab, v in SEQ]),
              ("End-to-end\nCNN", [("M5 YOLOv8n-cls (R4s)", fm("R4s honest M5"), "cnn")]),
              ("Frozen\nfoundation", [("V*", fm("V* IR-112 mean"), "vstar"), ("VIGIL / M8 (clean)", fm("M8 clean"), "m8")]),
              ("Events", [("Events LR (27)", fm("Events LR (27)"), "events")]),
              ("Calibration\n+ fusion", [("Calibrated V*", fm("Calibrated V* (P4)"), "calA"),
                                         ("Cal. V* + events", fm("Calibrated V* + events (P4)"), "fusion")])]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    i, ticks = 0, []
    for g, (stage, items) in enumerate(stages):
        x0 = i
        for lab, v, col in items:
            ax.bar(i, v, 0.7, color=C[col], edgecolor="white")
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
            ticks.append(lab)
            i += 1
        ax.text((x0 + i - 1) / 2, 0.6, stage, ha="center", va="center", color=MUTED, fontsize=8)
        if g < len(stages) - 1:
            ax.axvline(i - 0.5, color=GRID, lw=1)
    ax.axhline(fm("R1 clock"), color=C["clock"], ls="--", lw=1)
    ax.set_xticks(range(len(ticks)), ticks, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Fold-mean macro-F1 (3-class)")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="x", visible=False)
    ax.set_title("Baselines built up stage by stage (same 5-fold subject-independent split; "
                 "sequence models use FAU-indexed windows)", loc="left", color=INK, fontsize=10.5)
    save(fig, "f9_progression")


# ---------------------------------------------------------------- 0. architecture
def fig_architecture():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f4f6", ec=MUTED, bold=False, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec,
                                    lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.3)

    def arrow(x0, y0, x1, y1, col=MUTED, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color=col, lw=1.2,
                                     shrinkA=2, shrinkB=2))

    blue, teal, purple, pink = "#dbe7f6", "#d5efe8", "#e9e0f5", "#f6dde7"
    # shared vision pipeline
    box(0.1, 4.3, 1.6, 1.2, "IR face video\n240×220, 60 fps", bold=True)
    box(2.0, 4.3, 1.7, 1.2, "Face crop, 112 px\n2 fps\n(landmark box)")
    box(4.0, 4.3, 1.9, 1.2, "Frozen DINOv2\nViT-S/14-reg\n(not trained)", fc=blue)
    box(6.2, 4.3, 1.8, 1.2, "Token per frame\n[CLS ‖ mean-patch]\n768-d")
    for x0, x1 in [(1.7, 2.0), (3.7, 4.0), (5.9, 6.2)]:
        arrow(x0, 4.9, x1, 4.9)
    # V* and calibrated
    box(8.4, 5.05, 2.2, 0.9, "V*: mean over 60 s →\nLayerNorm → Linear", fc=blue)
    box(8.4, 3.95, 2.2, 0.9, "Calibrated V*: token minus\ndriver's alert-drive\nreference (first 4 min)",
        fc=teal)
    arrow(8.0, 5.1, 8.4, 5.5)
    arrow(8.0, 4.7, 8.4, 4.4)
    box(11.0, 5.05, 1.9, 0.9, "P(Alert, Low-vig.,\nDrowsy)", bold=True)
    arrow(10.6, 5.5, 11.0, 5.5)
    # events + fusion
    box(0.1, 2.3, 1.6, 1.1, "UL-DD landmarks\nFL + PL\n60 fps")
    box(2.0, 2.3, 3.9, 1.1, "27 events per 60 s: blink rate and duration,\nPERCLOS, long closures, eyelid velocity,\n"
                            "yawns, nods, posture motion, head drop", fc=pink, fs=8)
    box(6.2, 2.3, 1.8, 1.1, "Standardise →\nbalanced logistic\nregression", fc=pink)
    arrow(1.7, 2.85, 2.0, 2.85)
    arrow(5.9, 2.85, 6.2, 2.85)
    box(8.4, 2.45, 2.2, 0.85, "Fusion, fixed in advance:\nmean log-probability", fc=teal, bold=True)
    arrow(8.0, 2.85, 8.4, 2.85)
    arrow(9.5, 3.95, 9.5, 3.3)
    box(11.0, 2.45, 1.9, 0.85, "Best model:\n0.536 / LOSO 0.525", bold=True)
    arrow(10.6, 2.87, 11.0, 2.87)
    # M8
    box(0.1, 0.2, 1.6, 1.4, "Telemetry (14 ch)\n+ clock features")
    box(2.0, 0.2, 1.9, 1.4, "Fallback expert\nLayerNorm → MLP(64)\n→ z_t", fc=purple)
    box(4.2, 0.2, 1.7, 1.4, "Vision quality u:\nframes observed,\nface missing,\nno-frame flag")
    box(6.2, 0.2, 1.8, 1.4, "Gate\ng = σ(w·u + b)\nb = −3 at init", fc=purple)
    box(8.4, 0.2, 2.2, 1.4, "VIGIL (fallback MoE)\nz = (1−g)·z_v + g·z_t\nModDrop: 15–60 s\nblackouts",
        fc=purple, bold=True)
    box(11.0, 0.2, 1.9, 1.4, "Clean 0.428\n60 s blackout 0.349\n(vision only 0.186)", bold=True)
    arrow(1.7, 0.9, 2.0, 0.9)
    arrow(5.9, 0.9, 6.2, 0.9)
    arrow(8.0, 0.9, 8.4, 0.9)
    arrow(10.6, 0.9, 11.0, 0.9)
    ax.add_patch(FancyArrowPatch((3.0, 1.6), (8.6, 1.6), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>",
                                 mutation_scale=11, color=C["m8"], lw=1.2))
    ax.text(5.8, 1.92, "z_t", color=C["m8"], fontsize=8)
    ax.add_patch(FancyArrowPatch((7.1, 4.3), (9.2, 1.6), connectionstyle="arc3,rad=0.15", arrowstyle="-|>",
                                 mutation_scale=11, color=C["vstar"], lw=1.2, ls="--"))
    ax.text(7.25, 2.05, "z_v", fontweight="bold", color=C["vstar"], fontsize=8)
    ax.text(0.1, 6.0, "System overview: one frozen encoder, three ways to use it", fontsize=12, color=INK,
            fontweight="bold")
    save(fig, "f0_architecture")


if __name__ == "__main__":
    fig_architecture()
    fig_comparison()
    fig_confusions()
    fig_robustness()
    fig_confound()
    fig_subject_dependent()
    fig_timeline("K")
    fig_timeline("E")
    fig_loso_subjects()
    fig_progression()
