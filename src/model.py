"""
model.py -- small GPT for language-cadence-2, supporting three positional
encoding modes (Stage 4 / Stage 6 design):

  pe_mode="periodic": base sinusoidal PE + periodic component (period=P),
      injected via concat: x = Linear(concat(token_emb, base+periodic_PE)).
  pe_mode="baseline": base sinusoidal PE only (the no-periodic-component
      control), same concat injection.
  pe_mode="rope": no additive/concat PE at all -- rotary position embedding
      applied to queries/keys inside attention (the RoPE control).

Architecture (frozen): D=128, 4 layers, 4 heads (head_dim=32), context
N=1024, vocab 8000, pre-norm transformer blocks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encoding import combined_pe


def rotary_freqs(head_dim: int, seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE cos/sin tables, shape (seq_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, T, head_dim)
        if self.use_rope:
            cos, sin = rope_cos_sin
            q, k = apply_rotary(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, use_rope=use_rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope_cos_sin=rope_cos_sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    pe_mode: "periodic" (needs `period`), "baseline", or "rope".
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        seq_len: int,
        pe_mode: str,
        period: int | None = None,
    ):
        super().__init__()
        assert pe_mode in ("periodic", "baseline", "rope")
        if pe_mode == "periodic":
            assert period is not None, "pe_mode='periodic' requires a period"
        self.pe_mode = pe_mode
        self.seq_len = seq_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        use_rope = pe_mode == "rope"
        if not use_rope:
            # base (+ periodic, unless baseline) PE, fixed (not a learned parameter)
            period_for_pe = period if pe_mode == "periodic" else None
            pe = combined_pe(seq_len, d_model, period_for_pe)
            self.register_buffer("pe", pe, persistent=False)
            self.concat_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.register_buffer("pe", None, persistent=False)
            self.concat_proj = None
            head_dim = d_model // n_heads
            cos, sin = rotary_freqs(head_dim, seq_len)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList([Block(d_model, n_heads, use_rope=use_rope) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok = self.tok_emb(idx)  # (B, T, D)

        rope_cos_sin = None
        if self.pe_mode == "rope":
            x = tok
            rope_cos_sin = (self.rope_cos[:T].to(x.dtype), self.rope_sin[:T].to(x.dtype))
        else:
            pe = self.pe[:T].unsqueeze(0).expand(B, -1, -1).to(tok.dtype)
            x = self.concat_proj(torch.cat([tok, pe], dim=-1))

        for blk in self.blocks:
            x = blk(x, rope_cos_sin=rope_cos_sin)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
