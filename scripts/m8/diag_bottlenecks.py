"""Round-2 research diagnostics: where does V* (IR-112 mean-pool) lose, and what limits it?

D1 label structure     KSS variance split: between-subject / between-session / within-session
D2 within-session      Spearman(E[y_hat], KSS) inside each session vs across sessions
D3 oracle offset       per-subject bias correction of E[y_hat] (oracle upper bound on calibration)
D4 boundary windows    error on windows straddling two KSS epochs vs clean windows
D5 learning curve      linear probe F1 vs number of training subjects (data bottleneck)
D6 telemetry ~ clock   R^2 of time-in-session from window telemetry stats (held-out subjects)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8.config import RESULTS  # noqa: E402
from m8.data import video_bank  # noqa: E402
from m8.windows import load_windows  # noqa: E402
from probe_session_identity import window_means  # noqa: E402
from r6_tabular import window_stats  # noqa: E402

out = {}
w = load_windows()
v = w[w.avail_video == 1]

# D1 -------------------------------------------------------------------------------------------
k = v.kss.values
tot = k.var()
subj_mean = v.groupby("subject").kss.transform("mean").values
sess_mean = v.groupby(["subject", "session"]).kss.transform("mean").values
out["D1"] = dict(between_subject=float(subj_mean.var() / tot),
                 between_session_within_subject=float(((sess_mean - subj_mean) ** 2).mean() / tot),
                 within_session=float(((k - sess_mean) ** 2).mean() / tot),
                 labels_per_session=v.groupby(["subject", "session"]).y3.nunique().value_counts().to_dict())

# V* predictions --------------------------------------------------------------------------------
d = np.load(RESULTS / "runs" / "ir_112_mean_cls_mean_video_y3" / "preds.npz")
rows = d["rows"]
p = np.exp(d["test_lp"]).mean(0)
win = w.iloc[rows].reset_index(drop=True)
ey = p @ np.arange(3)
y = win.y3.values
pred = p.argmax(1)

# D2 -------------------------------------------------------------------------------------------
within = []
for (s, se), idx in win.groupby(["subject", "session"]).indices.items():
    if win.kss.values[idx].std() > 0:
        within.append(spearmanr(ey[idx], win.kss.values[idx])[0])
sess_level = win.assign(ey=ey).groupby(["subject", "session"])[["ey", "kss"]].mean()
out["D2"] = dict(within_session_spearman_mean=float(np.nanmean(within)), n_sessions=len(within),
                 within_session_spearman_frac_positive=float(np.mean(np.array(within) > 0)),
                 session_level_spearman=float(spearmanr(sess_level.ey, sess_level.kss)[0]),
                 overall_spearman=float(spearmanr(ey, win.kss)[0]))

# D3 -------------------------------------------------------------------------------------------
# Oracle: shift each subject's E[y_hat] so its mean equals the subject's mean label, then cut at
# 0.5 / 1.5. Compares with the same cut on the uncorrected E[y_hat].
cut = lambda e: np.digitize(e, [0.5, 1.5])  # noqa: E731
shift = win.assign(ey=ey, y=y).groupby("subject").apply(lambda g: (g.y - g.ey).mean())
ey_or = ey + win.subject.map(shift).values
out["D3"] = dict(f1_argmax=float(f1_score(y, pred, average="macro")),
                 f1_expected_cut=float(f1_score(y, cut(ey), average="macro")),
                 f1_oracle_subject_offset=float(f1_score(y, cut(ey_or), average="macro")))

# D4 -------------------------------------------------------------------------------------------
e0 = (win.start_s.values // 240).astype(int)
e1 = ((win.start_s.values + 59.999) // 240).astype(int)
straddle = e0 != e1
lab = w.set_index(["subject", "session", "epoch_idx"]).kss
out["D4"] = dict(frac_straddle=float(straddle.mean()),
                 acc_straddle=float((pred == y)[straddle].mean()),
                 acc_clean=float((pred == y)[~straddle].mean()))

# D5 -------------------------------------------------------------------------------------------
X, ok = window_means(video_bank(w, "ir_112"))
vid = np.where(ok & (w.avail_video.values == 1))[0]
subs = np.array(sorted(w.subject.values[vid].astype(str).tolist() and set(w.subject.values[vid])))
rng = np.random.default_rng(0)
curve = {}
for n_tr in [4, 6, 8, 10, 12, 15]:
    scores = []
    for rep in range(20 if n_tr < 15 else 16):
        if n_tr == 15:
            test_s = [subs[rep]]
            train_s = [s for s in subs if s != subs[rep]]
        else:
            perm = rng.permutation(subs)
            train_s, test_s = perm[:n_tr], perm[n_tr:]
        tr = vid[np.isin(w.subject.values[vid], train_s)]
        te = vid[np.isin(w.subject.values[vid], test_s)]
        if len(set(w.y3.values[tr])) < 3:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000))
        pr = clf.fit(X[tr], w.y3.values[tr]).predict(X[te])
        if n_tr == 15:
            scores.append((w.y3.values[te], pr))
        else:
            scores.append(f1_score(w.y3.values[te], pr, average="macro"))
    if n_tr == 15:  # LOSO: pool predictions over the 16 held-out subjects
        yy = np.concatenate([a for a, _ in scores])
        pp = np.concatenate([b for _, b in scores])
        curve[n_tr] = float(f1_score(yy, pp, average="macro"))
    else:
        curve[n_tr] = float(np.mean(scores))
out["D5_linear_probe_f1_vs_train_subjects"] = curve

# D6 -------------------------------------------------------------------------------------------
tel = window_stats(w, "tel", list(range(14)))
tr_rows = np.where(w.avail_tel.values == 1)[0]
pred_t = np.zeros(len(w))
for kf in range(5):
    a, b = tr_rows[w.fold.values[tr_rows] != kf], tr_rows[w.fold.values[tr_rows] == kf]
    pred_t[b] = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(tel[a], w.start_s.values[a]).predict(tel[b])
out["D6"] = dict(r2_time_from_telemetry=float(r2_score(w.start_s.values[tr_rows], pred_t[tr_rows])))

print(json.dumps(out, indent=1, default=str))
(RESULTS / "diag_bottlenecks.json").write_text(json.dumps(out, indent=1, default=str))
