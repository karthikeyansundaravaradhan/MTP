"""Shared scoring for every M8-phase run: window aggregation, causal smoothing, metrics,
subject-bootstrap CIs and paired subject-level permutation tests."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)

from m8.windows import load_windows


def metrics(y, p, n_classes=3):
    labels = list(range(n_classes))
    return dict(
        macro_f1=float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
        acc=float(accuracy_score(y, p)),
        bal_acc=float(balanced_accuracy_score(y, p)),
        per_class_f1=[float(v) for v in f1_score(y, p, average=None, labels=labels, zero_division=0)],
        qwk=float(cohen_kappa_score(y, p, weights="quadratic", labels=labels)) if n_classes > 2 else None,
        confusion=confusion_matrix(y, p, labels=labels).tolist(),
        n=int(len(y)),
    )


def frames_to_windows(subject, session, t_s, probs, windows=None, eps=1e-6):
    """Average per-frame log-probabilities over each canonical window that has video.
    Returns the windows DataFrame (rows with >=1 frame) plus a (W,C) log-prob array."""
    w = load_windows() if windows is None else windows
    frame_df = pd.DataFrame({"subject": subject, "session": session, "t": t_s})
    assert not frame_df.duplicated().any(), "duplicate (subject, session, t): score one model at a time"
    logp = np.log(probs + eps)
    rows, out = [], []
    for (s, se), idx in frame_df.groupby(["subject", "session"]).groups.items():
        idx = np.asarray(idx)
        t = frame_df.t.values[idx]
        order = np.argsort(t)
        t, lp = t[order], logp[idx][order]
        ws = w[(w.subject == s) & (w.session == se) & (w.avail_video == 1)]
        for r in ws.itertuples():
            lo, hi = np.searchsorted(t, r.start_s), np.searchsorted(t, r.start_s + 60)
            if hi > lo:
                rows.append(r.Index)
                out.append(lp[lo:hi].mean(0))
    return w.loc[rows].reset_index(drop=True), np.stack(out)


def causal_ema(win_df, logp, beta):
    """Per-session causal EMA over window order on log-probabilities (beta=0 -> no smoothing)."""
    if beta <= 0:
        return logp
    out = logp.copy()
    for _, idx in win_df.groupby(["subject", "session"]).indices.items():
        idx = idx[np.argsort(win_df.win_idx.values[idx])]
        acc = logp[idx[0]]
        for j, i in enumerate(idx):
            acc = logp[i] if j == 0 else beta * acc + (1 - beta) * logp[i]
            out[i] = acc
    return out


def subject_bootstrap(win_df, y, pred, n_boot=2000, seed=0, n_classes=3):
    """95% CI of pooled macro-F1, resampling test subjects with replacement."""
    rng = np.random.default_rng(seed)
    subs = win_df.subject.values
    uniq = np.unique(subs)
    by = {s: np.where(subs == s)[0] for s in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[s] for s in rng.choice(uniq, len(uniq))])
        vals.append(f1_score(y[idx], pred[idx], average="macro", labels=list(range(n_classes)),
                             zero_division=0))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def per_subject_balacc(win_df, y, pred):
    out = {}
    for s in np.unique(win_df.subject.values):
        m = win_df.subject.values == s
        out[s] = float(balanced_accuracy_score(y[m], pred[m]))
    return out


def paired_permutation(a, b, n_perm=20000, seed=0):
    """Two-sided sign-flip test on per-subject differences; a, b: dict subject -> score.
    Exact enumeration for <=20 subjects, otherwise Monte Carlo with the +1 correction."""
    subs = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in subs])
    obs = abs(d.mean()) - 1e-12
    if len(d) <= 20:
        flips = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
        p = float((np.abs((flips * d).mean(1)) >= obs).mean())
    else:
        flips = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, len(d)))
        p = float(((np.abs((flips * d).mean(1)) >= obs).sum() + 1) / (n_perm + 1))
    return dict(mean_diff=float(d.mean()), p=p, n_subjects=len(d))


def causal_hmm(win_df, logp, stay, prior):
    """Per-session HMM forward filter over window order. Emissions are scaled likelihoods
    p(y|x)/p(y); transitions keep the state with probability `stay`, else move uniformly.
    Causal: window i only sees windows <= i of its own session."""
    nc = logp.shape[1]
    log_t = np.log(np.full((nc, nc), (1 - stay) / (nc - 1)) + np.eye(nc) * (stay - (1 - stay) / (nc - 1)))
    lprior = np.log(np.asarray(prior) + 1e-9)
    out = logp.copy()
    for _, idx in win_df.groupby(["subject", "session"]).indices.items():
        idx = idx[np.argsort(win_df.win_idx.values[idx])]
        a = logp[idx[0]]
        for j, i in enumerate(idx):
            if j:
                pred = np.logaddexp.reduce(a[:, None] + log_t, 0)
                a = pred + logp[i] - lprior
                a = a - np.logaddexp.reduce(a)
            out[i] = a
    return out


def _macro_f1_from_conf(c):
    """c: (..., K, K) confusion (rows = truth). Macro-F1 over all K classes, 0 for empty classes."""
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    denom = c.sum(-1) + c.sum(-2)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0).mean(-1)


def paired_permutation_pooled(subjects, y, pred_a, pred_b, n_classes=3, n_perm=20000, seed=0):
    """Subject-level paired permutation test on the primary metric (pooled macro-F1).

    Under H0 the two models are exchangeable within each subject, so each subject's predictions
    are swapped between A and B; the statistic is pooled-F1(A side) - pooled-F1(B side). Unlike
    per-subject balanced accuracy, it stays meaningful for subjects with one or two classes.
    Exact enumeration for <=20 subjects, otherwise Monte Carlo with the +1 correction."""
    subs = np.unique(subjects)
    labels = list(range(n_classes))
    ca = np.stack([confusion_matrix(y[subjects == s], pred_a[subjects == s], labels=labels) for s in subs])
    cb = np.stack([confusion_matrix(y[subjects == s], pred_b[subjects == s], labels=labels) for s in subs])
    obs = float(_macro_f1_from_conf(ca.sum(0)) - _macro_f1_from_conf(cb.sum(0)))
    n = len(subs)
    if n <= 20:
        keep = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    else:
        keep = np.random.default_rng(seed).integers(0, 2, size=(n_perm, n)).astype(float)
    flat_a, flat_b = ca.reshape(n, -1), cb.reshape(n, -1)
    side_a = (keep @ flat_a + (1 - keep) @ flat_b).reshape(-1, n_classes, n_classes)
    side_b = (keep @ flat_b + (1 - keep) @ flat_a).reshape(-1, n_classes, n_classes)
    null = np.abs(_macro_f1_from_conf(side_a) - _macro_f1_from_conf(side_b))
    hits = (null >= abs(obs) - 1e-12).sum()
    p = float(hits / len(keep)) if n <= 20 else float((hits + 1) / (len(keep) + 1))
    return dict(diff_pooled_f1=obs, p=p, n_subjects=n)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def summarize(win_df, logp, y_col="y3", n_classes=3):
    """Fold-mean + pooled metrics, bootstrap CI, per-subject balanced accuracy."""
    y = win_df[y_col].values
    pred = logp.argmax(1)
    folds = {int(k): metrics(y[g], pred[g], n_classes)
             for k, g in win_df.groupby("fold").indices.items()}
    f1s = [m["macro_f1"] for m in folds.values()]
    pooled = metrics(y, pred, n_classes)
    return dict(
        fold_mean_f1=float(np.mean(f1s)), fold_std_f1=float(np.std(f1s)),
        fold_mean_acc=float(np.mean([m["acc"] for m in folds.values()])),
        pooled=pooled, pooled_f1_ci95=subject_bootstrap(win_df, y, pred, n_classes=n_classes),
        per_fold=folds, per_subject_balacc=per_subject_balacc(win_df, y, pred),
    )
