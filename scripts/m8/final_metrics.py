"""Item 3 / Day 8 scoring: one table of every headline model under the extra validation checks,
plus the pre-registered H1-H4 tests with Holm correction. Scoring only; no training.

Per model (window-level test predictions, video cohort):
  pooled macro-F1 [subject-bootstrap CI], fold-mean
  V3 epoch-level macro-F1: mean of the window probabilities in each 4-min KSS epoch -> one decision
  V6 ordinal: QWK and MAE of E[y_hat] against the class index (y3 only)
  V4a between-session: share of two-session subjects whose mean E[y_hat] is higher in D than in A
  V4b within-session: mean Spearman(E[y_hat], KSS) over sessions whose KSS varies
  V5 beats the clock: paired pooled-F1 test vs R1; partial Spearman(E[y_hat], KSS | time)
  B2 adds window-level ROC-AUC of P(KSS >= 6).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8.config import RESULTS  # noqa: E402
from m8.evaluate import holm, paired_permutation_pooled, summarize  # noqa: E402
from m8.windows import load_windows  # noqa: E402
from compare_runs import window_logp  # noqa: E402
from r0_r1_baselines import feats  # noqa: E402

Y3 = [  # (label, spec, protocol)
    ("R1 clock", "R1", "5fold"),
    ("R4s honest M5", "frames:R4_seed0+R4_seed1+R4_seed2", "5fold"),
    ("V* IR-112 mean", "ir_112_mean_cls_mean_video_y3", "5fold"),
    ("Pose MIL", "pose_224_mil_cls_mean_video_y3", "5fold"),
    ("IR+pose equal (exploratory)", "avg:ir_112_mean_cls_mean_video_y3+pose_224_mil_cls_mean_video_y3", "5fold"),
    ("Events LR (27)", "events_lr_y3", "5fold"),
    ("M8 clean", "vigil:C9d_ir112_moe_telclock:moe_telclock:0", "5fold"),
    ("Vision-only (ModDrop) clean", "vigil:C9_ir112:vision:0", "5fold"),
    ("M8 60 s blackout", "vigil:C9d_ir112_moe_telclock:moe_telclock:60", "5fold"),
    ("Vision-only 60 s blackout", "vigil:C9_ir112:vision:60", "5fold"),
    ("Calibrated V* (P4)", "ir_112_mean_cls_mean_video_y3_ref240A", "5fold"),
    ("Calibrated V* + events (P4)", "fusion_calA_events_y3", "5fold"),
    ("V* LOSO", "ir_112_mean_cls_mean_video_y3_loso", "loso"),
    ("Calibrated V* LOSO (P4)", "ir_112_mean_cls_mean_video_y3_ref240A_loso", "loso"),
    ("Calibrated V* + events LOSO (P4)", "fusion_calA_events_y3_loso", "loso"),
    ("Events LR LOSO", "events_lr_y3_loso", "loso"),
]
B2 = [
    ("R1 clock", "R1", "5fold"),
    ("V* IR-112 mean", "ir_112_mean_cls_mean_video_y_b2", "5fold"),
    ("Pose MIL", "pose_224_mil_cls_mean_video_y_b2", "5fold"),
    ("Events LR (27)", "events_lr_y_b2", "5fold"),
    ("M8 clean", "vigil:C9d_b2:moe_telclock:0", "5fold"),
    ("M8 60 s blackout", "vigil:C9d_b2:moe_telclock:60", "5fold"),
    ("Vision-only 60 s blackout", "vigil:C9d_b2:vision:60", "5fold"),
    ("Calibrated V* (P4)", "ir_112_mean_cls_mean_video_y_b2_ref240A", "5fold"),
    ("Calibrated V* + events (P4)", "fusion_calA_events_y_b2", "5fold"),
    ("V* LOSO", "ir_112_mean_cls_mean_video_y_b2_loso", "loso"),
    ("Calibrated V* LOSO (P4)", "ir_112_mean_cls_mean_video_y_b2_ref240A_loso", "loso"),
    ("Calibrated V* + events LOSO (P4)", "fusion_calA_events_y_b2_loso", "loso"),
]


def load(spec, w, y_col, nc):
    out = np.full((len(w), nc), np.nan, np.float32)
    if spec == "R1":
        for k in range(5):
            tr, te = w.fold.values != k, w.fold.values == k
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats(w[tr], False), w[y_col][tr])
            out[te] = m.predict_log_proba(feats(w[te], False))
        return out
    if spec.startswith("vigil:"):
        _, run, mode, L = spec.split(":")
        p = RESULTS / "runs" / run / "preds.npz"
        if not p.exists():
            return None
        d = np.load(p)
        mi, ci = list(d["modes"]).index(mode), list(d["conditions"]).index(int(L))
        out[d["rows"]] = np.log(np.exp(d["lp"][mi, ci]).mean(0) + 1e-9)
        return out
    if spec.startswith("avg:"):
        parts = [load(s, w, y_col, nc) for s in spec[4:].split("+")]
        return None if any(p is None for p in parts) else np.mean(parts, 0)
    if spec.startswith("frames:"):
        return window_logp(spec, w) if y_col == "y3" else None
    p = RESULTS / "runs" / spec / "preds.npz"
    if not p.exists():
        return None
    d = np.load(p)
    out[d["rows"]] = np.log(np.exp(d["test_lp"]).mean(0) + 1e-9)
    return out


def partial_spearman(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    res = lambda r: r - np.polyval(np.polyfit(rc, r, 1), rc)  # noqa: E731
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def score(w, rows, lp, y_col, nc, protocol, clock_pred):
    win = w.iloc[rows].reset_index(drop=True)
    if protocol == "loso":
        subs = sorted(set(win.subject))
        win["fold"] = win.subject.map({s: i for i, s in enumerate(subs)})
    l = lp[rows]
    p = np.exp(l)
    p = p / p.sum(1, keepdims=True)
    y = win[y_col].values
    s = summarize(win, l, y_col, nc)
    ey = p @ np.arange(nc)
    g = win.assign(**{f"p{k}": p[:, k] for k in range(nc)}).groupby(["subject", "session", "epoch_idx"])
    ep = g[[f"p{k}" for k in range(nc)]].mean().values
    ey_ep = g[y_col].agg(lambda v: np.bincount(v).argmax()).values
    out = dict(n=len(rows), pooled_f1=s["pooled"]["macro_f1"], ci=s["pooled_f1_ci95"],
               fold_mean_f1=s["fold_mean_f1"] if protocol == "5fold" else None,
               per_class_f1=s["pooled"]["per_class_f1"],
               epoch_f1=float(f1_score(ey_ep, ep.argmax(1), average="macro", labels=list(range(nc)), zero_division=0)),
               n_epochs=int(len(ep)))
    if nc == 3:
        out["qwk"] = float(cohen_kappa_score(y, p.argmax(1), weights="quadratic"))
        out["mae_expected"] = float(np.abs(ey - y).mean())
    else:
        out["auc"] = float(roc_auc_score(y, p[:, 1]))
    sess = win.assign(ey=ey).groupby(["subject", "session"]).ey.mean().unstack()
    both = sess.dropna()
    out["between_session_D_gt_A"] = float(((both["D"] > both["A"]) + 0.5 * (both["D"] == both["A"])).mean())
    out["n_two_session_subjects"] = int(len(both))
    rho = [spearmanr(ey[i], win.kss.values[i])[0] for i in win.groupby(["subject", "session"]).indices.values()
           if win.kss.values[i].std() > 0]
    out["within_session_spearman"] = float(np.nanmean(rho))
    out["partial_spearman_kss_given_time"] = partial_spearman(ey, win.kss.values, win.start_s.values)
    if clock_pred is not None:
        t = paired_permutation_pooled(win.subject.values, y, p.argmax(1), clock_pred[rows], n_classes=nc)
        out["vs_clock"] = t
    return out


def table(w, models, y_col, nc):
    clock = load("R1", w, y_col, nc).argmax(1)
    res, lps = {}, {}
    for label, spec, proto in models:
        lp = load(spec, w, y_col, nc)
        if lp is None:
            print(f"  [skip] {label}: {spec} missing")
            continue
        rows = np.where(~np.isnan(lp[:, 0]) & (w.avail_video.values == 1))[0]
        lps[label] = lp
        res[label] = score(w, rows, lp, y_col, nc, proto, None if spec == "R1" else clock)
        r = res[label]
        extra = f"QWK {r['qwk']:.2f} MAE {r['mae_expected']:.2f}" if nc == 3 else f"AUC {r['auc']:.2f}"
        vc = f"vs clock {r['vs_clock']['diff_pooled_f1']:+.3f} p={r['vs_clock']['p']:.2f}" if "vs_clock" in r else ""
        print(f"  {label:34s} pooled {r['pooled_f1']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}] epoch {r['epoch_f1']:.3f} "
              f"{extra} | D>A {r['between_session_D_gt_A']:.2f} within ρ {r['within_session_spearman']:+.2f} "
              f"partial ρ {r['partial_spearman_kss_given_time']:+.2f} | {vc}", flush=True)
    return res, lps


def h_tests(w, lps):
    pairs = {"H1 V* vs R4s (P0)": ("V* IR-112 mean", "R4s honest M5"),
             "H2 M8 vs vision-only, clean (do-no-harm)": ("M8 clean", "Vision-only (ModDrop) clean"),
             "H3 M8 vs vision-only, 60 s blackout": ("M8 60 s blackout", "Vision-only 60 s blackout"),
             "H4 calibrated vs uncalibrated V* (P4)": ("Calibrated V* (P4)", "V* IR-112 mean")}
    out = {}
    for h, (a, b) in pairs.items():
        rows = np.where(~np.isnan(lps[a][:, 0]) & ~np.isnan(lps[b][:, 0]) & (w.avail_video.values == 1))[0]
        out[h] = paired_permutation_pooled(w.subject.values[rows], w.y3.values[rows], lps[a][rows].argmax(1),
                                           lps[b][rows].argmax(1))
    adj = holm([v["p"] for v in out.values()])
    for (h, v), pa in zip(out.items(), adj):
        v["p_holm"] = pa
        print(f"  {h:45s} Δ pooled F1 {v['diff_pooled_f1']:+.3f}  p {v['p']:.3f}  Holm {pa:.3f}", flush=True)
    return out


def main():
    w = load_windows()
    print("== y3 (3-class) ==")
    r3, lps = table(w, Y3, "y3", 3)
    print("== B2 (KSS <= 5 vs >= 6) ==")
    rb, _ = table(w, B2, "y_b2", 2)
    print("== Pre-registered H1-H4 (subject-level paired permutation on pooled F1, Holm) ==")
    h = h_tests(w, lps)
    (RESULTS / "final_metrics.json").write_text(json.dumps(dict(y3=r3, b2=rb, H=h), indent=1, default=float))


if __name__ == "__main__":
    main()
