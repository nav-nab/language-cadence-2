"""
run_sweep.py -- launches the full Stage 5 sweep (8 periods x 3 seeds +
3 baseline seeds + 3 RoPE seeds = 30 runs) by repeatedly invoking
scripts/train.py.

Safe to interrupt and re-run: each individual run is itself resumable
(see train.py), and this launcher additionally skips any run whose
runs/<run_id>/final_result.json already exists, so re-running the whole
sweep after a partial interruption just picks up where it left off
without re-touching completed runs.

A run that diverges (train.py exits with a non-finite loss) is reported
and the sweep continues with the next config rather than aborting --
inspect runs/<run_id>/divergence.json and retry that one run by hand
(fresh --out_dir, or a lower --lr for just that run) once the rest of
the sweep is done.

Uses train.py's defaults for the frozen architecture (D=128, 4 layers,
4 heads, N=1024, vocab=8000, batch=16, lr=3e-4) -- only pe_mode, period,
seed, steps, and out_dir are passed explicitly.

Usage:
    python3 scripts/run_sweep.py
    python3 scripts/run_sweep.py --dry_run
    python3 scripts/run_sweep.py --only periodic
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PERIODS = [2, 3, 6, 9, 16, 45, 128, 1024]
SEEDS = [0, 1, 2]
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")


def build_configs() -> list[dict]:
    configs = []
    for period in PERIODS:
        for seed in SEEDS:
            configs.append({"pe_mode": "periodic", "period": period, "seed": seed})
    for seed in SEEDS:
        configs.append({"pe_mode": "baseline", "period": None, "seed": seed})
    for seed in SEEDS:
        configs.append({"pe_mode": "rope", "period": None, "seed": seed})
    return configs


def run_id_for(cfg: dict) -> str:
    if cfg["pe_mode"] == "periodic":
        return f"periodic_P{cfg['period']}_seed{cfg['seed']}"
    return f"{cfg['pe_mode']}_seed{cfg['seed']}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["periodic", "baseline", "rope"], default=None,
                    help="Run only this pe_mode's configs.")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--dry_run", action="store_true", help="Print planned commands without running them.")
    args = p.parse_args()

    configs = build_configs()
    if args.only:
        configs = [c for c in configs if c["pe_mode"] == args.only]

    print(f"Sweep: {len(configs)} run(s) planned.\n")

    completed, skipped, failed = [], [], []
    for i, cfg in enumerate(configs, 1):
        run_id = run_id_for(cfg)
        result_path = os.path.join(args.out_dir, run_id, "final_result.json")
        if os.path.exists(result_path):
            print(f"[{i}/{len(configs)}] {run_id}: already complete, skipping.")
            skipped.append(run_id)
            continue

        cmd = [sys.executable, TRAIN_SCRIPT,
               "--pe_mode", cfg["pe_mode"],
               "--seed", str(cfg["seed"]),
               "--steps", str(args.steps),
               "--out_dir", args.out_dir,
               "--sweep_index", str(i),
               "--sweep_total", str(len(configs))]
        if cfg["period"] is not None:
            cmd += ["--period", str(cfg["period"])]

        print(f"[{i}/{len(configs)}] {run_id}: {' '.join(cmd)}")
        if args.dry_run:
            continue

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[{i}/{len(configs)}] {run_id}: FAILED (exit {proc.returncode}) -- "
                  f"see runs/{run_id}/divergence.json if present. Continuing with next run.\n")
            failed.append(run_id)
        else:
            completed.append(run_id)

    if not args.dry_run:
        print("\n--- Sweep summary ---")
        print(f"  completed this session: {len(completed)}")
        print(f"  already done (skipped): {len(skipped)}")
        print(f"  failed: {len(failed)}")
        if failed:
            print(f"  failed run_ids: {failed}")


if __name__ == "__main__":
    main()
