"""
benchmark_real_arch.py -- benchmark the ACTUAL Stage 4 model (src/model.py)
at the frozen architecture, to replace the analytical FLOP estimate with a
real measurement, and check whether batch_size=16 fits in the VM's ~3.8GB
RAM at context length 1024.

Usage:
    python3 scripts/benchmark_real_arch.py
"""
import os
import resource
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import GPT, count_params  # noqa: E402

VOCAB = 8000
D = 128
LAYERS = 4
HEADS = 4
SEQ_LEN = 1024
BATCH_SIZE = 16
DURATION_SEC = 30


def run_benchmark(pe_mode: str, period):
    torch.manual_seed(0)
    model = GPT(VOCAB, D, LAYERS, HEADS, SEQ_LEN, pe_mode=pe_mode, period=period)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randint(0, VOCAB, (BATCH_SIZE, SEQ_LEN))
    y = torch.randint(0, VOCAB, (BATCH_SIZE, SEQ_LEN))

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()

    steps = 0
    t0 = time.time()
    while time.time() - t0 < DURATION_SEC:
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        steps += 1
    elapsed = time.time() - t0

    sec_per_step = elapsed / steps
    tokens_per_sec = (BATCH_SIZE * SEQ_LEN) / sec_per_step
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux

    print(f"[{pe_mode}{'' if period is None else f' P={period}'}] "
          f"params={n_params:,}  steps={steps} in {elapsed:.1f}s -> "
          f"{sec_per_step*1000:.0f} ms/step, {tokens_per_sec:,.0f} tok/s, peak_rss={peak_rss_mb:.0f}MB")

    hours_per_3000 = sec_per_step * 3000 / 3600
    print(f"   -> a 3000-step run at this config: {hours_per_3000:.2f}h")
    return sec_per_step, peak_rss_mb


if __name__ == "__main__":
    print(f"Config: D={D} layers={LAYERS} heads={HEADS} seq_len={SEQ_LEN} vocab={VOCAB} batch={BATCH_SIZE}\n")
    mode = sys.argv[1] if len(sys.argv) > 1 else "periodic"
    if mode == "periodic":
        run_benchmark("periodic", 45)
    elif mode == "baseline":
        run_benchmark("baseline", None)
    elif mode == "rope":
        run_benchmark("rope", None)
    else:
        raise ValueError(mode)
