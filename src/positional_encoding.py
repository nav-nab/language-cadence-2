"""
positional_encoding.py -- fixed (non-learned) positional encodings for the
language-cadence-2 small GPT.

Three kinds, matching the frozen Stage 4 design:
  - sinusoidal_pe: the base PE (Vaswani-style, fixed, no learned params).
  - periodic_pe: the periodic component added on top of the base, reusing
    language-cadence-1's folded-harmonic construction exactly. H = D/2
    row-pairs, folded harmonic order k_j = min(j mod P, P - (j mod P)) for
    j = 1..H, rows sin/cos of n * 2*pi*k_j / P.
  - RoPE (rotary): NOT injected additively/via concat like the other two --
    it rotates query/key vectors inside attention instead. See
    RotaryEmbedding / apply_rotary in model.py, which is where it's used.

combined_pe(N, D, period) builds base + periodic (period=None means the
no-periodic-component baseline: base only).
"""
from __future__ import annotations

import math

import torch


def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def periodic_pe(seq_len: int, d_model: int, period: int) -> torch.Tensor:
    assert d_model % 2 == 0, "d_model must be even (H = D/2 row-pairs)"
    H = d_model // 2
    n = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    j = torch.arange(1, H + 1, dtype=torch.float32).unsqueeze(0)  # (1, H)
    j_mod_p = torch.remainder(j, period)
    k = torch.minimum(j_mod_p, period - j_mod_p)  # folded harmonic order, (1, H)
    angle = n * 2 * math.pi * k / period  # (N, H)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


def combined_pe(seq_len: int, d_model: int, period: int | None) -> torch.Tensor:
    """The base PE, plus the periodic component on top of it (unless period
    is None, which is the no-periodic-component baseline: base only)."""
    pe = sinusoidal_pe(seq_len, d_model)
    if period is not None:
        pe = pe + periodic_pe(seq_len, d_model, period)
    return pe
