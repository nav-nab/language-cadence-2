"""
benchmark_throughput.py -- laptop throughput benchmark for the language-cadence-2
small GPT, to check feasibility of the 48-hour total compute budget.

Runs a tiny nanoGPT-style model (config chosen to match the likely sweep
architecture: D=64, 4 layers, 4 heads, context 384, vocab ~8000) for a fixed
wall-clock duration and reports steps/sec, tokens/sec, and time estimates for
plausible sweep sizes.
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# --- config (matches the likely small-GPT sweep architecture) ---
D = 64
N_LAYERS = 4
N_HEADS = 4
SEQ_LEN = 384
VOCAB = 8000
BATCH_SIZE = 16
DURATION_SEC = 100  # steady-state measurement window


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, D_ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D_)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, targets):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    model = TinyGPT(VOCAB, D, N_LAYERS, N_HEADS, SEQ_LEN)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randint(0, VOCAB, (BATCH_SIZE, SEQ_LEN))
    y = torch.randint(0, VOCAB, (BATCH_SIZE, SEQ_LEN))

    # warmup
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        opt.step()

    steps = 0
    t0 = time.time()
    while time.time() - t0 < DURATION_SEC:
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        opt.step()
        steps += 1
    elapsed = time.time() - t0

    sec_per_step = elapsed / steps
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    tokens_per_sec = tokens_per_step / sec_per_step

    print(f"params: {n_params:,}")
    print(f"config: D={D} layers={N_LAYERS} heads={N_HEADS} seq_len={SEQ_LEN} vocab={VOCAB} batch_size={BATCH_SIZE}")
    print(f"steps: {steps} in {elapsed:.1f}s -> {sec_per_step*1000:.1f} ms/step, {tokens_per_sec:,.0f} tokens/sec")

    # feasibility projections
    for total_hours in (48,):
        budget_sec = total_hours * 3600
        max_steps_total = budget_sec / sec_per_step
        print(f"\n--- {total_hours}h total compute budget ---")
        print(f"max total training steps across ALL runs: {max_steps_total:,.0f}")
        for n_runs, steps_each in [(1, 3000), (6, 3000), (12, 3000), (20, 3000), (30, 2000)]:
            total = n_runs * steps_each
            hours = total * sec_per_step / 3600
            print(f"  {n_runs} runs x {steps_each} steps = {total:,} steps -> {hours:.2f}h")


if __name__ == "__main__":
    main()
