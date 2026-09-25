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

Stage 5 protocol additions:
  - Best-val-loss checkpoint (checkpoint_best.pt): tracked throughout
    training and used for the reported test loss, instead of the final
    step's weights. At N=1024, 3000 steps is ~7-8 epochs over the small
    (6.5M-token) training set, so late-training overfitting is a real
    possibility -- this avoids rewarding/penalizing a PE choice based on
    where it happens to land after a fixed step count. best_val_loss/
    best_step survive resume (stored in the main checkpoint).
  - Hard divergence check: if the training loss ever goes non-finite, the
    run halts immediately (before backward/optimizer.step() can further
    corrupt the model) and writes divergence.json for inspection. The
    last saved checkpoint is untouched, so the fix is to restart that run
    (fresh --out_dir, or delete its checkpoint.pt) with a lower --lr --
    not to resume it as-is.
  - Soft spike warning: a step whose loss is far above the recent
    exponential moving average is logged to warnings.log and stderr but
    does not halt the run (early training is naturally noisy).

Usage:
    python3 scripts/train.py --pe_mode periodic --period 45 --seed 0
    python3 scripts/train.py --pe_mode baseline --seed 1
    python3 scripts/train.py --pe_mode rope --seed 2
"""
from __future__ import annotations

import argparse
import json
import math
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


def format_duration(seconds) -> str:
    """Human-readable duration for ETA display. None/NaN/negative -> "?"."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def compute_eta(step: int, start_step: int, args_steps: int, invocation_start_time: float,
                 eta_avg_run_seconds, eta_runs_remaining) -> str:
    """Build the " run ETA: ... | sweep ETA: ..." suffix for a progress line.

    The run ETA is measured live from THIS invocation's own step rate (steps
    since it was started/resumed, not the whole run's history), so it
    reflects current machine conditions rather than a stale average. The
    sweep ETA additionally projects the remaining runs using the average
    duration of runs already completed in this sweep (passed in by
    run_sweep.py from real measured timings); if none have completed yet, it
    falls back to this run's own live rate as a stand-in estimate."""
    steps_this_invocation = step - start_step
    if steps_this_invocation <= 0:
        return ""
    elapsed = time.time() - invocation_start_time
    sec_per_step = elapsed / steps_this_invocation
    current_run_eta = (args_steps - step) * sec_per_step

    suffix = f"  run ETA: {format_duration(current_run_eta)}"
    if eta_runs_remaining is not None:
        future_run_seconds = eta_avg_run_seconds if eta_avg_run_seconds is not None else sec_per_step * args_steps
        sweep_eta = current_run_eta + future_run_seconds * eta_runs_remaining
        suffix += f" | sweep ETA: {format_duration(sweep_eta)} ({eta_runs_remaining} run(s) left after this)"
    return suffix


def check_finite_loss(loss_val: float, step: int, run_dir: str, run_id: str, sweep_tag: str = "") -> None:
    """Halt immediately (exit code 1) if the loss has gone non-finite, before
    it can further corrupt the model via backward()/optimizer.step(). Writes
    a diagnostic file so the run can be inspected and retried (e.g. with a
    lower --lr) without losing the last valid checkpoint."""
    if math.isfinite(loss_val):
        return
    diag = {"step": step, "run_id": run_id, "train_loss": loss_val}
    with open(os.path.join(run_dir, "divergence.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(f"{sweep_tag}[{run_id}] DIVERGED at step {step}: train_loss={loss_val} is not finite. "
          f"The last saved checkpoint is unaffected -- restart this run (fresh --out_dir, "
          f"or delete its checkpoint.pt) with a lower --lr rather than resuming as-is.",
          file=sys.stderr)
    sys.exit(1)


def check_loss_spike(loss_val: float, loss_ema, step: int, run_dir: str, run_id: str,
                      sweep_tag: str = "", warmup_steps: int = 50, spike_ratio: float = 10.0,
                      ema_alpha: float = 0.02):
    """Soft warning (logged, not fatal) if a step's loss is far above the
    recent exponential moving average. Returns the updated EMA."""
    if loss_ema is None:
        return loss_val
    if step > warmup_steps and loss_val > spike_ratio * loss_ema:
        msg = (f"{sweep_tag}[{run_id}] WARNING: loss spike at step {step}: train_loss={loss_val:.4f} "
               f"vs recent avg {loss_ema:.4f}")
        print(msg, file=sys.stderr)
        with open(os.path.join(run_dir, "warnings.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return (1 - ema_alpha) * loss_ema + ema_alpha * loss_val


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
    p.add_argument("--checkpoint_every_seconds", type=float, default=60.0)  # 1 min -- checkpoint I/O (~tens of MB) is negligible next to ~1.8s/step
    p.add_argument("--eval_every_steps", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--print_every_steps", type=int, default=10,
                    help="Print step/train_loss this often (cheap, no extra compute -- "
                         "the loss is already computed for backprop). Independent of "
                         "--eval_every_steps, which additionally runs a real val-loss "
                         "pass and is deliberately less frequent to limit that overhead.")
    p.add_argument("--sweep_index", type=int, default=None,
                    help="This run's 1-based position in the sweep (display only, set by run_sweep.py).")
    p.add_argument("--sweep_total", type=int, default=None,
                    help="Total number of runs in the sweep (display only, set by run_sweep.py).")
    p.add_argument("--eta_avg_run_seconds", type=float, default=None,
                    help="Average wall-clock duration of a completed run in this sweep so far "
                         "(display only, set by run_sweep.py from real completed-run timings). "
                         "Used to project the total sweep ETA beyond this run.")
    p.add_argument("--eta_runs_remaining", type=int, default=None,
                    help="Number of runs left in the sweep AFTER this one (display only, set by "
                         "run_sweep.py).")
    args = p.parse_args()

    if args.pe_mode == "periodic" and args.period is None:
        p.error("--period is required when --pe_mode periodic")

    torch.manual_seed(args.seed)

    run_id = run_id_for(args)
    sweep_tag = (f"[run {args.sweep_index}/{args.sweep_total}] "
                 if args.sweep_index is not None and args.sweep_total is not None else "")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    best_ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
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
    best_val_loss = float("inf")
    best_step = 0
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
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_step = ckpt.get("best_step", 0)
        print(f"{sweep_tag}[{run_id}] resuming from checkpoint at step {start_step}/{args.steps} "
              f"(best val_loss so far: {best_val_loss:.4f} at step {best_step})", flush=True)
    else:
        print(f"{sweep_tag}[{run_id}] starting fresh: {count_params(model):,} params, "
              f"{n_chunks} train windows, {steps_per_epoch} steps/epoch", flush=True)

    if start_step >= args.steps:
        print(f"{sweep_tag}[{run_id}] already complete ({start_step} >= {args.steps} steps).", flush=True)
    else:
        last_ckpt_time = time.time()
        invocation_start_time = time.time()
        step = start_step
        loss_ema = None
        while step < args.steps:
            epoch = step // steps_per_epoch
            pos_in_epoch = step % steps_per_epoch
            perm = epoch_permutation(n_chunks, args.seed, epoch)

            x, y = get_batch(train_windows, perm, pos_in_epoch, args.batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss_val = loss.item()
            check_finite_loss(loss_val, step, run_dir, run_id, sweep_tag)  # halts here if non-finite
            loss_ema = check_loss_spike(loss_val, loss_ema, step, run_dir, run_id, sweep_tag)
            loss.backward()
            optimizer.step()
            step += 1

            is_eval_step = step % args.eval_every_steps == 0 or step == args.steps
            is_print_step = step % args.print_every_steps == 0 or step == args.steps

            if is_eval_step:
                val_loss = eval_loss(model, val_windows, args.eval_batches, args.batch_size, device)
                rec = {
                    "step": step, "epoch": epoch, "train_loss": loss_val,
                    "val_loss_subset": val_loss, "wall_time": time.time(),
                }
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                eta_suffix = compute_eta(step, start_step, args.steps, invocation_start_time,
                                          args.eta_avg_run_seconds, args.eta_runs_remaining)
                print(f"{sweep_tag}[{run_id}] step {step}/{args.steps} (epoch {epoch})  "
                      f"train_loss={loss_val:.4f}  val_loss(subset)={val_loss:.4f}{eta_suffix}", flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    save_checkpoint(best_ckpt_path, {
                        "step": step,
                        "val_loss": val_loss,
                        "model_state_dict": model.state_dict(),
                        "config": vars(args),
                    })
                    print(f"{sweep_tag}[{run_id}]   new best val_loss={val_loss:.4f} at step {step} -- checkpoint_best.pt updated", flush=True)
            elif is_print_step:
                eta_suffix = compute_eta(step, start_step, args.steps, invocation_start_time,
                                          args.eta_avg_run_seconds, args.eta_runs_remaining)
                print(f"{sweep_tag}[{run_id}] step {step}/{args.steps} (epoch {epoch})  "
                      f"train_loss={loss_val:.4f}{eta_suffix}", flush=True)

            if time.time() - last_ckpt_time >= args.checkpoint_every_seconds or step == args.steps:
                save_checkpoint(ckpt_path, {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "config": vars(args),
                })
                last_ckpt_time = time.time()
                print(f"{sweep_tag}[{run_id}] checkpoint saved at step {step}", flush=True)

    # Final eval uses the BEST-val-loss checkpoint, not the final step's
    # weights (Stage 5 protocol) -- see module docstring for why.
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
    else:
        print(f"{sweep_tag}[{run_id}] WARNING: no checkpoint_best.pt found (val loss never improved, or "
              f"--eval_every_steps > --steps) -- using final-step weights for test eval instead.",
              file=sys.stderr)

    test_ids = np.load(os.path.join(args.data_dir, "test_ids.npy"))
    test_windows = make_windows(test_ids, chunk_len)
    final_test_loss = full_eval_loss(model, test_windows, args.batch_size, device)
    result = {
        "run_id": run_id,
        "pe_mode": args.pe_mode,
        "period": args.period,
        "seed": args.seed,
        "final_test_loss": final_test_loss,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "total_steps": args.steps,
    }
    with open(os.path.join(run_dir, "final_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"{sweep_tag}[{run_id}] DONE. final_test_loss={final_test_loss:.4f} "
          f"(from best-val checkpoint at step {best_step}, val_loss={best_val_loss:.4f})", flush=True)


if __name__ == "__main__":
    main()
