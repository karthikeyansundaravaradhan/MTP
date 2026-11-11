"""Per-window access to cached streams without materialising overlapping windows.

A Bank holds every cached frame of one stream (all sessions concatenated) plus, for each
canonical window, the indices of its frames. Windows are gathered at batch time.
"""
from dataclasses import dataclass

import numpy as np
import torch

from m8.config import CACHE



@dataclass
class Bank:
    name: str
    feats: torch.Tensor      # (N_frames, D) float16/32
    index: np.ndarray        # (W, T) int64 row into feats, -1 = missing frame
    quality: np.ndarray      # (W, T, Q) float32 per-frame quality (zeros if none)
    mu: torch.Tensor = None  # per-fold normalisation set by the trainer (outer-train stats)
    sd: torch.Tensor = None

    @property
    def dim(self):
        return self.feats.shape[1]

    _index_t: torch.Tensor = None

    def to(self, device):
        self.feats = self.feats.to(device)
        self._index_t = torch.from_numpy(self.index).to(device)
        return self

    def set_norm(self, rows):
        idx = np.unique(self.index[rows].ravel())
        idx = torch.from_numpy(idx[idx >= 0]).to(self.feats.device)
        x = self.feats[idx].float()
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6

    def gather(self, rows):
        if self._index_t is not None:
            idx = self._index_t[torch.as_tensor(rows, device=self._index_t.device)]
        else:
            idx = torch.from_numpy(self.index[rows])
        mask = idx >= 0
        x = self.feats[idx.clamp(min=0)].float()
        if self.mu is not None:
            x = (x - self.mu) / self.sd
        return x * mask[..., None], mask


def _video_tokens(tok, mode):
    if mode == "cls_mean":
        return np.concatenate([tok[:, 0], tok[:, 1]], 1)
    if mode == "cls":
        return tok[:, 0]
    if mode == "all":
        return tok.reshape(len(tok), -1)
    raise ValueError(mode)


def video_bank(windows, tag, mode="cls_mean", context_s=60, stride=1, ref_s=0, ref_session="same",
               ref_concat=False):
    """tag like 'ir_168'. Frames with t in [start_s+60-context_s, start_s+60) belong to a window:
    context_s=60 is the window itself, larger values add causal look-back (left-aligned, so
    windows near the session start have fewer frames). stride=k keeps every k-th frame, so
    context_s=240, stride=4 costs the same steps as the plain 60 s window.
    ref_s>0 (C12): subtract the session's own mean token over its first ref_s seconds
    (start-of-drive calibration; removes person/session appearance).
    ref_session="A" (R2-7): the reference is the first ref_s seconds of the subject's *Alert*
    session for both sessions -- a fresh-driver calibration that keeps between-session level.
    ref_concat (R2-8): return [absolute, absolute - reference] instead of the difference only."""
    refs = {}
    if ref_s > 0 and ref_session == "A":
        for s in windows.subject.unique():
            f = CACHE / "emb" / tag / f"{s}_A.npz"
            if f.exists():
                d = np.load(f)
                refs[s] = _video_tokens(d["tok"], mode).astype(np.float32)[d["t"] < ref_s].mean(0, keepdims=True)
    feats, quals, index, offset = [], [], None, 0
    rows_by_session = windows.groupby(["subject", "session"]).indices
    steps = None
    for (s, se), rows in rows_by_session.items():
        f = CACHE / "emb" / tag / f"{s}_{se}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        t = d["t"]
        if steps is None:
            fps = int(round(1 / np.median(np.diff(t))))
            steps = int(round(fps * context_s / stride))
            index = np.full((len(windows), steps), -1, np.int64)
            qual = np.zeros((len(windows), steps, d["q"].shape[1]), np.float32)
        for r in rows:
            lo = np.searchsorted(t, windows.start_s.values[r] + 60 - context_s)
            hi = np.searchsorted(t, windows.start_s.values[r] + 60)
            sel = np.arange(hi - 1, lo - 1, -stride)[::-1][-steps:]  # anchored on the newest frame
            n = len(sel)
            if n > 0:
                index[r, :n] = offset + sel
                qual[r, :n] = d["q"][sel]
        tok = _video_tokens(d["tok"], mode).astype(np.float32)
        if ref_s > 0:
            ref = refs[s] if ref_session == "A" else tok[t < ref_s].mean(0, keepdims=True)
            tok = np.concatenate([tok, tok - ref], 1) if ref_concat else tok - ref
        feats.append(tok)
        offset += len(t)
    feats = torch.from_numpy(np.concatenate(feats).astype(np.float16))
    return Bank(tag, feats, index, qual)


def series_bank(windows, kind, cols=None):
    """kind 'tab' (2 Hz, 120 steps) or 'tel' (4 Hz, 240 steps). Raw values; the trainer
    calls set_norm() with outer-train rows so z-scoring never sees test subjects."""
    rate = {"tab": 2, "tel": 4}[kind]
    steps = 60 * rate
    feats, offset = [], 0
    index = np.full((len(windows), steps), -1, np.int64)
    for (s, se), rows in windows.groupby(["subject", "session"]).indices.items():
        d = np.load(CACHE / kind / f"{s}_{se}.npz")
        x = d["x"] if cols is None else d["x"][:, cols]
        for r in rows:
            lo = int(windows.start_s.values[r] * rate)
            index[r] = offset + lo + np.arange(steps)
        feats.append(x)
        offset += len(x)
    feats = np.concatenate(feats).astype(np.float32)
    return Bank(kind, torch.from_numpy(feats), index, np.zeros((len(windows), steps, 0), np.float32))
