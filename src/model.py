"""
Stage 3 — LeadAcidBiMamba.

A bidirectional selective state-space model for lead-acid remaining useful life.
Written from scratch; it mirrors the shape of the lithium BiMamba-APF but shares
no code with it, because the inputs and the failure physics are different.

    (B, 30, F) daily features
      → input projection + learned positional encoding
      → 3 × BiMambaBlock            forward and reverse selective scans
      → DegradationAnchorAttention  learned prototypes of degradation state
      → attention pooling over time
      → RUL head, predicting mean and log-variance

Two departures from the lithium model, both driven by this dataset:

  * NO CHEMISTRY EMBEDDING. One chemistry, one form factor. The lithium model
    carries chemistry and shape tokens because it spans LFP/NMC/LCO/NCA; here
    that capacity would only add parameters to overfit with.

  * HETEROSCEDASTIC OUTPUT. The head emits a log-variance alongside the mean,
    so the model can say when it does not know. That matters more here than for
    lithium: half the training labels are censored, so uncertainty is not a
    nicety but a property of the supervision itself.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Dropout):
    """Dropout that stays on in eval, so repeated passes sample the posterior."""

    def forward(self, x):
        return F.dropout(x, self.p, training=True, inplace=self.inplace)


class SelectiveSSM(nn.Module):
    """Selective state-space scan — the Mamba core.

    Input-dependent Δ, B and C make the state transition selective: the model
    learns which parts of the history to retain and which to forget, rather
    than applying one fixed decay as a linear RNN would.

        h_t = exp(Δ_t·A)·h_{t−1} + Δ_t·B_t·x_t
        y_t = C_t·h_t + D·x_t
    """

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        # A is kept in log space so it stays negative after exp — a stable decay
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(d_inner))
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)   # small initial Δ

    def forward(self, x):                       # (B, L, d_inner)
        B, L, _ = x.shape
        S = self.d_state
        proj = self.x_proj(x)
        B_m, C_m, dt_r = proj[..., :S], proj[..., S:2 * S], proj[..., 2 * S:]
        dt = F.softplus(self.dt_proj(dt_r))                       # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                        # (d_inner, S)

        dA = torch.exp(dt.unsqueeze(-1) * A)                      # (B, L, d_inner, S)
        dBx = dt.unsqueeze(-1) * B_m.unsqueeze(2) * x.unsqueeze(-1)

        h = torch.zeros(B, self.d_inner, S, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bds,bs->bd", h, C_m[:, t]))
        return torch.stack(ys, dim=1) + x * self.D


class MambaBlock(nn.Module):
    """Gated Mamba block: norm → project → depthwise conv → SSM → gate → out."""

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        d_inner = int(expand * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1,
                                groups=d_inner, bias=True)
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        y = self.ssm(F.silu(u)) * F.silu(z)
        return res + self.dropout(self.out_proj(y))


class BiMambaBlock(nn.Module):
    """Forward and reverse scans, concatenated and projected back.

    Degradation is not causal in a fixed window: the meaning of an early day
    depends on what followed it inside the same 30-day view. A reverse scan
    lets the block use that, and it stays leakage-free because the window
    itself never extends past the prediction day.
    """

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.fwd = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.rev = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        f = self.fwd(x)
        r = torch.flip(self.rev(torch.flip(x, dims=[1])), dims=[1])
        return self.norm(self.proj(torch.cat([f, r], dim=-1)))


class DegradationAnchorAttention(nn.Module):
    """Cross-attention onto learned degradation prototypes.

    The anchors are free parameters, not data: the model learns a small set of
    canonical degradation states and each day attends to whichever it resembles.
    It gives the network somewhere to put "this looks like a sulfating cell"
    without needing that label.
    """

    def __init__(self, d_model=128, n_heads=4, n_anchors=4, dropout=0.1):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.anchors = nn.Parameter(torch.randn(n_anchors, d_model) * 0.02)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        res = x
        x = self.norm(x)
        a = self.anchors.unsqueeze(0).expand(B, -1, -1)
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(a).view(B, -1, self.h, self.dk).transpose(1, 2)
        v = self.v(a).view(B, -1, self.h, self.dk).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1)
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, L, D)
        return res + self.drop(self.o(out))


class LeadAcidBiMamba(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, n_blocks: int = 3,
                 d_state: int = 16, n_anchors: int = 4, seq_len: int = 30,
                 dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model), nn.LayerNorm(d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            BiMambaBlock(d_model, d_state, dropout=dropout) for _ in range(n_blocks))
        self.anchor_attn = DegradationAnchorAttention(
            d_model, n_anchors=n_anchors, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)
        # attention pooling — later days usually matter more, but let it decide
        self.pool = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), MCDropout(dropout))
        self.mu = nn.Linear(64, 1)
        self.logvar = nn.Linear(64, 1)

    def forward(self, x):
        h = self.input_proj(x) + self.pos[:, :x.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(self.anchor_attn(h))
        w = torch.softmax(self.pool(h), dim=1)
        z = self.head((h * w).sum(dim=1))
        # softplus keeps RUL non-negative without saturating like a sigmoid
        return F.softplus(self.mu(z)).squeeze(-1), \
            self.logvar(z).squeeze(-1).clamp(-7.0, 4.0)

    @torch.no_grad()
    def predict_with_uncertainty(self, x, passes: int = 30):
        """MC-dropout mean, plus epistemic and aleatoric standard deviations."""
        self.eval()
        mus, varz = [], []
        for _ in range(passes):
            m, lv = self(x)
            mus.append(m)
            varz.append(torch.exp(lv))
        mus = torch.stack(mus)
        return mus.mean(0), mus.std(0), torch.stack(varz).mean(0).sqrt()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
