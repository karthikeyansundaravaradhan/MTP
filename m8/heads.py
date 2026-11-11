"""Trainable heads on top of frozen per-frame features. All take (x, mask) with
x (B, T, D) and mask (B, T) bool, and return (logits (B, C), frame attention (B, T))."""
import math

import torch
import torch.nn as nn


class GatedAttnPool(nn.Module):
    """Gated attention-MIL pooling (Ilse et al. 2018)."""

    def __init__(self, d, h=32):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1)

    def forward(self, h, mask):
        a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)
        a = a.masked_fill(~mask, -1e4).softmax(-1)
        return (a.unsqueeze(-1) * h).sum(1), a


class MILHead(nn.Module):
    """H-std: LayerNorm -> Linear(D, d) -> GELU -> Dropout -> gated attention-MIL -> Linear."""

    def __init__(self, d_in, n_classes=3, d=64, drop=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(drop))
        self.pool = GatedAttnPool(d)
        self.out = nn.Linear(d, n_classes)

    def embed(self, x, mask):
        return self.pool(self.proj(x), mask)

    def forward(self, x, mask):
        h, a = self.embed(x, mask)
        return self.out(h), a


class MeanPoolHead(nn.Module):
    """Linear-probe control: masked mean over time -> LayerNorm -> Linear."""

    def __init__(self, d_in, n_classes=3, **_):
        super().__init__()
        self.out = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(0.2), nn.Linear(d_in, n_classes))

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).float()
        h = (x * m).sum(1) / m.sum(1).clamp(min=1)
        a = mask.float() / mask.float().sum(1, keepdim=True).clamp(min=1)
        return self.out(h), a


def sinusoid(t, d):
    """t (B, T) seconds -> (B, T, d) sinusoidal time encoding."""
    freqs = torch.exp(torch.arange(0, d, 2, device=t.device) * (-math.log(1000.0) / d))
    ang = t.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


class TemporalTransformerHead(nn.Module):
    """Linear -> + time PE -> [CLS] + 2 pre-LN layers -> CLS -> Linear (C5)."""

    def __init__(self, d_in, n_classes=3, d=64, layers=2, heads=4, drop=0.3, fps=2):
        super().__init__()
        self.fps = fps
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_classes)

    def forward(self, x, mask):
        b, t, _ = x.shape
        secs = torch.arange(t, device=x.device).float().expand(b, t) / self.fps
        h = self.proj(x) + sinusoid(secs, self.cls.shape[-1])
        h = torch.cat([self.cls.expand(b, -1, -1), h], 1)
        keep = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), mask], 1)
        z = self.enc(h, src_key_padding_mask=~keep)
        a = mask.float() / mask.float().sum(1, keepdim=True).clamp(min=1)
        return self.out(self.norm(z[:, 0])), a


HEADS = {"mil": MILHead, "mean": MeanPoolHead, "tf": TemporalTransformerHead}
