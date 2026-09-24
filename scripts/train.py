"""
train.py -- resumable training loop for a single language-cadence-2 run
(one pe_mode/period/seed combination).

Designed to run on Navid's own machine, not the sandbox. Checkpoints on a
WALL-CLOCK timer (not a step count), so an interruption loses at most
`--checkpoint_every_seconds` of compute regardless of how fast or slow a
given machine/config is. Re-running the same command resumes from the last
checkpoint automatically -- no separate "resume" flag needed.

Data loading: non-overlapping (seq_len+1)-token windows over the
concatenated, <eos>-separated train_ids.npy (the "more standard" choice per
language-cadence-1's methods note, vs sampling-with-replacement). Windows
are shuffled each epoch via a seed that's a pure function of
(--seed, epoch), so the shuffle never needs to be saved in the checkpoint --
resuming just needs (epoch, position_in_epoch), which IS saved.

Usage:
    python3 scripts/train.py --pe_mode periodic --period 45 --seed 0
    python3 scripts/train.py --pe_mode baseline --seed 1
    python3 scripts/train.py --pe_mode rope --seed 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model import GPT, count_params  # noqa: E402
from syllable_tokenizer.vocab import Vocab  # noqa: E402


def run_id_for(args) -> str:
    if args.pe_mode == "periodic":
        return f"periodic_P{args.period}_seed{args.seed}"
    return f"{args.pe_mode}_seed{args.seed}"


def make_windows(ids: np.ndarray, chunk_len: int) -> np.ndarray:
    """Non-overlapping (chunk_len)-token windows, dropping any remainder."""
    n_chunks = len(ids) // chunk_len
    usable = ids[: n_chunks * chunk_len]
    return usable.reshape(n_chunks, chunk_len)


def epoch_permutation(n_chunks: int, seed: int, epoch: int) -> np.ndarray:
    """Deterministic shuffle order for a given epoch -- a pure function of
    (seed, epoch), so it never needs to be stored in the checkpoint."""
    rng = np.random.RandomState(seed * 100_000 + epoch)
    return rng.permutation(n_chunks)


def get_batch(windows: np.ndarray, perm: np.ndarray, batch_idx: int, batch_size: int, device):
    idx = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    batch = windows[idx].astype(np.int64)  # (B, chunk_len)
    x = torch.from_numpy(batch[:, :-1]).to(device)
    y = torch.from_numpy(batch[:, 1:]).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, windows: np.ndarray, n_batches: int, batch_size: int, device, seed: int = 12345) -> float:
    """Cheap eval on a fixed-size, deterministic random subset of windows."""
    model.eval()
    rng = np.random.RandomState(seed)
    n_chunks = len(windows)
    total_loss, total_batches = 0.0, 0
    for _ in range(n_batches):
        idx = rng.choice(n_chunks, size=batch_size, replace=False)
        batch = windows[idx].astype(np.int64)
        x = torch.from_numpy(batch[:, :-1]).to(device)
        y = torch.from_numpy(batch[:, 1:]).to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / total_batches


@torch.no_grad()
def full_eval_loss(model, windows: np.ndarray, batch_size: int, device) -> float:
    """Full pass over every window in `windows` (used once, at the end, on
    val/test -- not for the periodic in-training curve)."""
    model.eval()
    n_chunks = len(windows)
    n_batches = n_chunks // batch_size
    total_loss = 0.0
    for b in range(n_batches):
        batch = windows[b * batch_size : (b + 1) * batch_size].astype(np.int64)
        x = torch.from_numpy(batch[:, :-1]).to(device)
        y = torch.from_numpy(batch[:, 1:]).to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / n_batches if n_batches else float("nan")


def save_checkpoint(path: str, state: dict):
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pe_mode", choices=["periodic", "baseline", "rope"], required=True)
    p.add_argument("--period", type=int, default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vocab_path", default="configs/vocab.json")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--checkpoint_every_seconds", type=float, default=600.0)  # 10 min, under the 30-min ceiling
    p.add_argument("--eval_every_steps", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=10)
    args = p.parse_args()

    if args.pe_mode == "periodic" and args.period is None:
        p.error("--period is required when --pe_mode periodic")

    torch.manual_seed(args.seed)

    run_id = run_id_for(args)
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    vocab = Vocab.load(args.vocab_path)
    vocab_size = len(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = np.load(os.path.join(args.data_dir, "train_ids.npy"))
    val_ids = np.load(os.path.join(args.data_dir, "val_ids.npy"))
    chunk_len = args.seq_len + 1
    train_windows = make_windows(train_ids, chunk_len)
    val_windows = make_windows(val_ids, chunk_len)
    n_chunks = len(train_windows)
    steps_per_epoch = n_chunks // args.batch_size

    model = GPT(
        vocab_size, args.d_model, args.n_layers, args.n_heads, args.seq_len,
        pe_mode=args.pe_mode, period=args.period,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Only these args affect the model architecture, data, or training
    # determinism -- changing them mid-run would silently corrupt the
    # resumed state. Everything else (--steps, --eval_every_steps,
    # --eval_batches, --checkpoint_every_seconds, --out_dir) is safe to
    # change between runs (e.g. extending the step budget) and is
    # deliberately NOT checked here.
    IDENTITY_KEYS = [
        "pe_mode", "period", "seed", "batch_size", "seq_len",
        "d_model", "n_layers", "n_heads", "lr", "vocab_path", "data_dir",
    ]

    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        saved_cfg = ckpt["config"]
        current_cfg = vars(args)
        mismatches = {
            k: (saved_cfg.get(k), current_cfg.get(k))
            for k in IDENTITY_KEYS
            if saved_cfg.get(k) != current_cfg.get(k)
        }
        if mismatches:
            raise ValueError(f"Checkpoint config mismatch for {run_id}: {mismatches}. "
                              f"Refusing to resume with different settings -- use a different --out_dir or fix the args.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_step = ckpt["step"]
        print(f"[{run_id}] resuming from checkpoint at step {start_step}/{args.steps}")
    else:
        print(f"[{run_id}] starting fresh: {count_params(model):,} params, "
              f"{n_chunks} train windows, {steps_per_epoch} steps/epoch")

    if start_step >= args.steps:
        print(f"[{run_id}] already complete ({start_step} >= {args.steps} steps).")
    else:
        last_ckpt_time = time.time()
        step = start_step
        while step < args.steps:
            epoch = step // steps_per_epoch
            pos_in_epoch = step % steps_per_epoch
            perm = epoch_permutation(n_chunks, args.seed, epoch)

            x, y = get_batch(train_windows, perm, pos_in_epoch, args.batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.eval_every_steps == 0 or step == args.steps:
                val_loss = eval_loss(model, val_windows, args.eval_batches, args.batch_size, device)
                rec = {
                    "step": step, "epoch": epoch, "train_loss": loss.item(),
                    "val_loss_subset": val_loss, "wall_time": time.time(),
                }
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{run_id}] step {step}/{args.steps} (epoch {epoch})  "
                      f"train_loss={loss.item():.4f}  val_loss(subset)={val_loss:.4f}")

            if time.time() - last_ckpt_time >= args.checkpoint_every_seconds or step == args.steps:
                save_checkpoint(ckpt_path, {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "config": vars(args),
                })
                last_ckpt_time = time.time()
                print(f"[{run_id}] checkpoint saved at step {step}")

    # final full-test-set eval (only once, at the very end -- per the protocol,
    # the periodic in-training numbers above are a cheap subset for the curve,
    # this is the number that should be reported/plotted)
    test_ids = np.load(os.path.join(args.data_dir, "test_ids.npy"))
    test_windows = make_windows(test_ids, chunk_len)
    final_test_loss = full_eval_loss(model, test_windows, args.batch_size, device)
    result = {"run_id": run_id, "final_test_loss": final_test_loss, "total_steps": args.steps}
    with open(os.path.join(run_dir, "final_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{run_id}] DONE. final_test_loss={final_test_loss:.4f}")


if __name__ == "__main__":
    main()
