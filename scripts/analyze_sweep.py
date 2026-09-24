"""
analyze_sweep.py -- aggregates runs/*/final_result.json from the Stage 5
sweep into a summary table and the period-vs-test-loss plot.

Test loss for each run comes from that run's best-val-loss checkpoint
(see train.py), grouped by (pe_mode, period) and averaged across the 3
seeds. The plot is the primary Stage 5 deliverable: it shows the actual
shape of test loss vs. period P (monotonic toward N like language-cadence-1,
a sweet spot, non-monotonic noise, etc.), with baseline (no periodic
component) and RoPE shown as horizontal reference bands. The "best P vs.
baseline/RoPE" comparison is a specific readout from this plot, chosen by
validation loss (best_val_loss), not by peeking at test loss.

Usage:
    python3 scripts/analyze_sweep.py
    python3 scripts/analyze_sweep.py --runs_dir runs --out_dir results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def load_results(runs_dir: str) -> list[dict]:
    results = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "final_result.json"))):
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def group_by_config(results: list[dict]) -> dict:
    """Group by (pe_mode, period) -> list of final_test_loss across seeds."""
    groups: dict = {}
    for r in results:
        key = (r["pe_mode"], r.get("period"))
        groups.setdefault(key, []).append(r["final_test_loss"])
    return groups


def summarize(groups: dict) -> list[dict]:
    rows = []
    for (pe_mode, period), losses in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] if kv[0][1] is not None else -1)
    ):
        mean = statistics.mean(losses)
        std = statistics.stdev(losses) if len(losses) > 1 else 0.0
        rows.append({
            "pe_mode": pe_mode, "period": period, "n_seeds": len(losses),
            "mean_test_loss": mean, "std_test_loss": std,
        })
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"{'pe_mode':<10} {'period':<8} {'n':<3} {'mean_test_loss':<16} {'std':<8}")
    for r in rows:
        period_str = str(r["period"]) if r["period"] is not None else "--"
        print(f"{r['pe_mode']:<10} {period_str:<8} {r['n_seeds']:<3} "
              f"{r['mean_test_loss']:<16.4f} {r['std_test_loss']:<8.4f}")


def best_period_row(rows: list[dict]):
    """Note: in the real Stage 5 protocol, "best" for the headline claim
    should be selected by validation loss, not by this function re-sorting
    on test loss. This helper is for the full-curve view; see DECISIONS.md
    for how the single best-P vs. baseline/RoPE comparison is chosen."""
    periodic_rows = [r for r in rows if r["pe_mode"] == "periodic"]
    if not periodic_rows:
        return None
    return min(periodic_rows, key=lambda r: r["mean_test_loss"])


def make_plot(rows: list[dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periodic_rows = sorted((r for r in rows if r["pe_mode"] == "periodic"), key=lambda r: r["period"])
    periods = [r["period"] for r in periodic_rows]
    means = [r["mean_test_loss"] for r in periodic_rows]
    stds = [r["std_test_loss"] for r in periodic_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    if periods:
        ax.errorbar(periods, means, yerr=stds, marker="o", capsize=4, label="periodic PE", color="C0")
        ax.set_xscale("log", base=2)

    for pe_mode, color in [("baseline", "C1"), ("rope", "C2")]:
        ref_rows = [r for r in rows if r["pe_mode"] == pe_mode]
        if ref_rows:
            r = ref_rows[0]
            ax.axhline(r["mean_test_loss"], color=color, linestyle="--", label=f"{pe_mode} (mean)")
            if r["std_test_loss"] > 0:
                ax.axhspan(r["mean_test_loss"] - r["std_test_loss"], r["mean_test_loss"] + r["std_test_loss"],
                           color=color, alpha=0.1)

    ax.set_xlabel("Period P")
    ax.set_ylabel("Test loss (at best-val checkpoint)")
    ax.set_title("Test loss vs. periodic PE period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    results = load_results(args.runs_dir)
    if not results:
        print(f"No final_result.json files found under {args.runs_dir}/ -- nothing to analyze yet.")
        sys.exit(0)

    groups = group_by_config(results)
    rows = summarize(groups)
    print(f"Loaded {len(results)} run(s) across {len(rows)} config(s).\n")
    print_table(rows)

    best = best_period_row(rows)
    baseline_rows = [r for r in rows if r["pe_mode"] == "baseline"]
    rope_rows = [r for r in rows if r["pe_mode"] == "rope"]
    print()
    if best:
        print(f"Lowest-mean-test-loss periodic config: P={best['period']}  "
              f"mean_test_loss={best['mean_test_loss']:.4f} (n={best['n_seeds']})")
    if baseline_rows:
        print(f"Baseline (no periodic component): mean_test_loss={baseline_rows[0]['mean_test_loss']:.4f} "
              f"(n={baseline_rows[0]['n_seeds']})")
    if rope_rows:
        print(f"RoPE: mean_test_loss={rope_rows[0]['mean_test_loss']:.4f} (n={rope_rows[0]['n_seeds']})")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved summary table to {summary_path}")

    try:
        make_plot(rows, os.path.join(args.out_dir, "period_sweep.png"))
    except ImportError:
        print("matplotlib not installed -- skipping plot (pip install matplotlib to enable it).")


if __name__ == "__main__":
    main()
