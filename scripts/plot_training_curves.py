"""
plot_training_curves.py -- plots val_loss_subset (periodic in-training eval,
scripts/train.py's --eval_every_steps=100) against training step for every
config in the sweep, to look for qualitative differences/anomalies in HOW
loss evolves during training, not just where it ends up (which is what
analyze_sweep.py / stats_sweep.py cover).

Each config's 3 seeds are averaged (mean +/- std band) at each step -- the
step grid is identical across all runs (eval_every_steps=100, steps=3000),
so no interpolation is needed.

Usage:
    python3 scripts/plot_training_curves.py
    python3 scripts/plot_training_curves.py --runs_dir runs --out_dir results
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# Sequential blue ramp (ordinal use: step >=250 nearest-surface, per the
# project's dataviz palette), light->dark mapped to increasing period.
PERIOD_RAMP = {
    2:    "#86b6ef",  # step 250
    3:    "#5598e7",  # step 350
    6:    "#3987e5",  # step 400
    9:    "#2a78d6",  # step 450
    16:   "#256abf",  # step 500
    45:   "#1c5cab",  # step 550
    128:  "#104281",  # step 650
    1024: "#0d366b",  # step 700
}
BASELINE_COLOR = "#898781"  # muted ink (reference line)
ROPE_COLOR = "#eb6834"      # categorical slot 2, orange (distinct accent)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"


def load_curves(runs_dir: str) -> dict:
    """Return {(pe_mode, period): {seed: (steps_array, val_loss_array)}}."""
    curves: dict = {}
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        final_path = os.path.join(run_dir, "final_result.json")
        if not (os.path.exists(metrics_path) and os.path.exists(final_path)):
            continue
        with open(final_path) as f:
            final = json.load(f)
        # metrics.jsonl is append-only; a run that was interrupted and
        # resumed mid-run can log the same step twice (once before the
        # interruption, once again after resuming from the last
        # checkpoint). Dedupe by step, keeping the LAST record for each
        # step (the most recent write), then sort ascending.
        by_step = {}
        with open(metrics_path) as f:
            for line in f:
                rec = json.loads(line)
                by_step[rec["step"]] = rec["val_loss_subset"]
        steps = sorted(by_step)
        vals = [by_step[s] for s in steps]
        key = (final["pe_mode"], final.get("period"))
        curves.setdefault(key, {})[final["seed"]] = (np.array(steps), np.array(vals))
    return curves


def aggregate(curves: dict) -> dict:
    """Return {(pe_mode, period): (steps, mean, std)}, assuming all seeds of
    a config share the same step grid (true here: eval_every_steps=100,
    steps=3000 fixed across the whole sweep)."""
    out = {}
    for key, by_seed in curves.items():
        seed_arrays = list(by_seed.values())
        steps = seed_arrays[0][0]
        for s, _ in seed_arrays:
            assert np.array_equal(s, steps), f"step grid mismatch for {key}"
        stacked = np.stack([v for _, v in seed_arrays])  # n_seeds x n_steps
        out[key] = (steps, stacked.mean(axis=0), stacked.std(axis=0))
    return out


def make_plot(agg: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
    for ax in (ax_full, ax_zoom):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    periods = sorted(p for (m, p) in agg if m == "periodic")

    def plot_series(ax):
        for p in periods:
            steps, mean, std = agg[("periodic", p)]
            color = PERIOD_RAMP[p]
            ax.plot(steps, mean, color=color, linewidth=1.8, label=f"P={p}", zorder=3)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12, zorder=2)
        if ("baseline", None) in agg:
            steps, mean, std = agg[("baseline", None)]
            ax.plot(steps, mean, color=BASELINE_COLOR, linewidth=2.2, linestyle="--",
                     label="baseline (no periodic)", zorder=4)
            ax.fill_between(steps, mean - std, mean + std, color=BASELINE_COLOR, alpha=0.12, zorder=2)
        if ("rope", None) in agg:
            steps, mean, std = agg[("rope", None)]
            ax.plot(steps, mean, color=ROPE_COLOR, linewidth=2.2, linestyle="--",
                     label="RoPE", zorder=4)
            ax.fill_between(steps, mean - std, mean + std, color=ROPE_COLOR, alpha=0.12, zorder=2)

    plot_series(ax_full)
    ax_full.set_title("Full training run", color=INK_PRIMARY, fontsize=11)
    ax_full.set_xlabel("Training step", color=INK_SECONDARY)
    ax_full.set_ylabel("Validation loss (eval subset, mean ± 1 std across 3 seeds)", color=INK_SECONDARY)

    plot_series(ax_zoom)
    ax_zoom.set_xlim(1500, 3000)
    # Autoscale y to the zoomed x-range only, with a little headroom.
    all_vals = []
    for key, (steps, mean, std) in agg.items():
        mask = steps >= 1500
        all_vals.append(mean[mask] - std[mask])
        all_vals.append(mean[mask] + std[mask])
    all_vals = np.concatenate(all_vals)
    pad = 0.03 * (all_vals.max() - all_vals.min())
    ax_zoom.set_ylim(all_vals.min() - pad, all_vals.max() + pad)
    ax_zoom.set_title("Zoomed: late training (step ≥ 1500)", color=INK_PRIMARY, fontsize=11)
    ax_zoom.set_xlabel("Training step", color=INK_SECONDARY)

    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, labelcolor=INK_SECONDARY)

    fig.suptitle("Validation loss vs. training step, all 10 configs (mean of 3 paired seeds)",
                 color=INK_PRIMARY, fontsize=13, y=1.02)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def make_periodic_only_plot(agg: dict, out_path: str) -> None:
    """Periodic configs only, on their own y-axis (not compressed by
    baseline/RoPE sharing the scale), so period-to-period structure during
    training is actually legible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = sorted(p for (m, p) in agg if m == "periodic")

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
    for ax in (ax_full, ax_zoom):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    def plot_series(ax, mask_fn=None):
        for p in periods:
            steps, mean, std = agg[("periodic", p)]
            if mask_fn is not None:
                m = mask_fn(steps)
                steps, mean, std = steps[m], mean[m], std[m]
            color = PERIOD_RAMP[p]
            ax.plot(steps, mean, color=color, linewidth=2.0, label=f"P={p}", zorder=3)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, zorder=2)
        if ("baseline", None) in agg:
            steps, mean, std = agg[("baseline", None)]
            if mask_fn is not None:
                m = mask_fn(steps)
                steps, mean, std = steps[m], mean[m], std[m]
            ax.plot(steps, mean, color=BASELINE_COLOR, linewidth=2.4, linestyle="--",
                     label="baseline (no periodic)", zorder=4)

    plot_series(ax_full)
    ax_full.set_title("Full training run (periodic configs + baseline only)", color=INK_PRIMARY, fontsize=11)
    ax_full.set_xlabel("Training step", color=INK_SECONDARY)
    ax_full.set_ylabel("Validation loss (eval subset, mean across 3 seeds)", color=INK_SECONDARY)

    plot_series(ax_zoom, mask_fn=lambda s: s >= 2000)
    ax_zoom.set_xlim(2000, 3000)
    ax_zoom.set_title("Zoomed: step ≥ 2000, own y-scale", color=INK_PRIMARY, fontsize=11)
    ax_zoom.set_xlabel("Training step", color=INK_SECONDARY)

    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, labelcolor=INK_SECONDARY)
    fig.suptitle("Periodic configs vs. baseline only, during training (own y-scale)",
                 color=INK_PRIMARY, fontsize=13, y=1.02)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    curves = load_curves(args.runs_dir)
    if not curves:
        print(f"No metrics.jsonl files found under {args.runs_dir}/ -- nothing to plot yet.")
        return
    agg = aggregate(curves)

    os.makedirs(args.out_dir, exist_ok=True)
    make_plot(agg, os.path.join(args.out_dir, "training_curves.png"))
    make_periodic_only_plot(agg, os.path.join(args.out_dir, "training_curves_periodic_only.png"))


if __name__ == "__main__":
    main()
