"""
stats_sweep.py -- formal significance testing for the Stage 5 period sweep.

Two questions, both requested explicitly by the user:

1. periodic-vs-baseline gap: for each of the 8 periods, is the periodic
   config's test loss significantly different from baseline? Seeds 0,1,2
   are identical (same init, same data order) across every config by
   protocol design, so this is a PAIRED comparison (scipy.stats.ttest_rel),
   not an independent two-sample test -- pairing removes seed-to-seed noise
   that is shared between the two arms and gives much more power with only
   n=3 per group. With 8 simultaneous comparisons we control the family-wise
   error rate with Holm-Bonferroni correction, and report Cohen's d_z
   (paired effect size: mean difference / std of the paired differences)
   alongside each test since a tiny n=3 p-value alone is easy to misread.

2. period-vs-period gap ("is the spread across periods real, or an artifact
   of how we scaled the plot?"): statistical significance is the ratio of
   an effect to its measurement noise (standard error) -- it does not
   depend on how a y-axis happens to be drawn or zoomed, so "is it an
   artifact of scale" is actually the question a significance test is
   built to answer. We run two omnibus tests across all 8 periods, treating
   period as the within-subject factor and seed as the subject/blocking
   variable (the pairing again): a repeated-measures ANOVA
   (statsmodels.stats.anova.AnovaRM) and a Friedman test
   (scipy.stats.friedmanchisquare) as a non-parametric cross-check given the
   very small n=3 per cell. We also directly compare the best (lowest mean
   test loss) vs. worst periodic config with a paired t-test, since that is
   the largest period-to-period gap in the data and the most favorable case
   for finding a real difference.

Usage:
    python3 scripts/stats_sweep.py
    python3 scripts/stats_sweep.py --runs_dir runs --out_dir results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sstats


def load_results(runs_dir: str) -> list[dict]:
    results = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "final_result.json"))):
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def paired_by_seed(results: list[dict], metric: str = "final_test_loss") -> dict:
    """Return {(pe_mode, period): {seed: metric_value}}."""
    out: dict = {}
    for r in results:
        key = (r["pe_mode"], r.get("period"))
        out.setdefault(key, {})[r["seed"]] = r[metric]
    return out


def aligned_arrays(groups: dict, key_a, key_b) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (a_vals, b_vals, seeds) aligned by shared seed, sorted by seed."""
    a_map = groups[key_a]
    b_map = groups[key_b]
    shared_seeds = sorted(set(a_map) & set(b_map))
    a_vals = np.array([a_map[s] for s in shared_seeds], dtype=float)
    b_vals = np.array([b_map[s] for s in shared_seeds], dtype=float)
    return a_vals, b_vals, shared_seeds


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d_z for paired samples: mean(diff) / std(diff, ddof=1)."""
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("inf") if diff.mean() != 0 else 0.0
    return float(diff.mean() / sd)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original input order, each capped at 1.0 and monotonically non-decreasing
    as required by the procedure."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw_adj = (n - rank) * pvals[idx]
        running_max = max(running_max, raw_adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def periodic_vs_baseline(groups: dict, periods: list[int]) -> list[dict]:
    rows = []
    pvals = []
    for p in periods:
        a, b, seeds = aligned_arrays(groups, ("periodic", p), ("baseline", None))
        t, pval = sstats.ttest_rel(a, b)
        d = cohens_d_paired(a, b)
        rows.append({
            "period": p, "n": len(seeds), "mean_periodic": float(a.mean()),
            "mean_baseline": float(b.mean()), "mean_diff": float(a.mean() - b.mean()),
            "t_stat": float(t), "p_value": float(pval), "cohens_d": d,
        })
        pvals.append(pval)
    adjusted = holm_bonferroni(pvals)
    for row, adj in zip(rows, adjusted):
        row["p_value_holm"] = float(adj)
        row["significant_at_0.05_holm"] = bool(adj < 0.05)
    return rows


def omnibus_period_effect(groups: dict, periods: list[int]) -> dict:
    """RM-ANOVA + Friedman test across all 8 periodic configs, period as the
    within-subject factor, seed as the subject. Only uses periodic configs
    (not baseline/RoPE) -- this directly answers 'is the period-to-period
    spread distinguishable from noise'."""
    seeds = sorted(set.intersection(*[set(groups[("periodic", p)]) for p in periods]))
    matrix = np.array([[groups[("periodic", p)][s] for p in periods] for s in seeds])  # n_seeds x n_periods

    friedman_stat, friedman_p = sstats.friedmanchisquare(*[matrix[:, i] for i in range(len(periods))])

    anova_result = None
    anova_error = None
    try:
        import pandas as pd
        from statsmodels.stats.anova import AnovaRM

        long_rows = []
        for si, s in enumerate(seeds):
            for pi, p in enumerate(periods):
                long_rows.append({"seed": s, "period": str(p), "loss": matrix[si, pi]})
        df = pd.DataFrame(long_rows)
        aov = AnovaRM(df, depvar="loss", subject="seed", within=["period"]).fit()
        row = aov.anova_table.iloc[0]
        anova_result = {
            "F_stat": float(row["F Value"]),
            "df_num": float(row["Num DF"]),
            "df_denom": float(row["Den DF"]),
            "p_value": float(row["Pr > F"]),
        }
    except Exception as e:  # pragma: no cover -- exercised only if statsmodels/pandas unavailable
        anova_error = str(e)

    return {
        "n_seeds": len(seeds), "n_periods": len(periods),
        "friedman_stat": float(friedman_stat), "friedman_p": float(friedman_p),
        "rm_anova": anova_result, "rm_anova_error": anova_error,
    }


def all_pairwise_period_comparisons(groups: dict, periods: list[int]) -> list[dict]:
    """Paired t-test for every one of the C(n,2) period-vs-period pairs,
    Holm-Bonferroni corrected across the whole family. This is the family
    a "best vs. worst" comparison is implicitly drawn from once the pair is
    selected by looking at the data (picking the single lowest- and
    highest-mean periods out of 8 is exactly the "testing the winner"
    problem: Fisher's-LSD-style single-pair t-tests are only valid when the
    pair was chosen before seeing the data; for >3 groups you need a
    correction that accounts for the full comparison space the extreme
    pair could have come from -- see Tukey's HSD / the studentized range
    literature for the uncorrected-pairs version of this mistake)."""
    import itertools
    rows = []
    pvals = []
    for p1, p2 in itertools.combinations(periods, 2):
        a, b, seeds = aligned_arrays(groups, ("periodic", p1), ("periodic", p2))
        t, pval = sstats.ttest_rel(a, b)
        d = cohens_d_paired(a, b)
        rows.append({
            "period_a": p1, "period_b": p2, "n": len(seeds),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(a.mean() - b.mean()),
            "t_stat": float(t), "p_value": float(pval), "cohens_d": d,
        })
        pvals.append(pval)
    adjusted = holm_bonferroni(pvals)
    for row, adj in zip(rows, adjusted):
        row["p_value_holm"] = float(adj)
        row["significant_at_0.05_holm"] = bool(adj < 0.05)
    return rows


def best_vs_worst(groups: dict, periods: list[int], pairwise_family: list[dict] | None = None) -> dict:
    """Identify the lowest- and highest-mean periodic configs and report
    their comparison's Holm-corrected p-value FROM WITHIN the full
    all-pairs family (pairwise_family), not an uncorrected single test --
    since best/worst were selected by looking at the data, an uncorrected
    test here would be the "testing the winner" bias. If pairwise_family
    is not given, it is computed here (at the cost of running all C(n,2)
    tests)."""
    means = {p: np.mean(list(groups[("periodic", p)].values())) for p in periods}
    best_p = min(means, key=means.get)
    worst_p = max(means, key=means.get)
    a, b, seeds = aligned_arrays(groups, ("periodic", best_p), ("periodic", worst_p))
    t, pval = sstats.ttest_rel(a, b)
    d = cohens_d_paired(a, b)

    if pairwise_family is None:
        pairwise_family = all_pairwise_period_comparisons(groups, periods)
    p_holm = None
    for row in pairwise_family:
        pair = {row["period_a"], row["period_b"]}
        if pair == {best_p, worst_p}:
            p_holm = row["p_value_holm"]
            break

    return {
        "best_period": best_p, "worst_period": worst_p, "n": len(seeds),
        "mean_best": float(a.mean()), "mean_worst": float(b.mean()),
        "mean_diff": float(b.mean() - a.mean()), "t_stat": float(t),
        "p_value_uncorrected": float(pval),
        "p_value_holm_within_all_pairs": p_holm,
        "significant_at_0.05_holm": bool(p_holm < 0.05) if p_holm is not None else None,
        "cohens_d": d,
        "note": ("best/worst were selected by looking at the data (min/max of 8 means), so "
                 "p_value_uncorrected alone is invalid for this comparison (the 'testing the "
                 "winner' problem) -- p_value_holm_within_all_pairs is the correct number to "
                 "read, Holm-corrected across all C(8,2)=28 period-vs-period pairs."),
    }


def print_report(pvb_rows: list[dict], omnibus: dict, bw: dict) -> None:
    print("=" * 78)
    print("1. Periodic vs. baseline (paired t-test, Holm-Bonferroni corrected)")
    print("=" * 78)
    print(f"{'P':<6} {'mean_periodic':<14} {'mean_base':<11} {'diff':<9} "
          f"{'t':<7} {'p_raw':<9} {'p_holm':<9} {'d':<7} sig?")
    for r in pvb_rows:
        sig = "yes" if r["significant_at_0.05_holm"] else "no"
        print(f"{r['period']:<6} {r['mean_periodic']:<14.4f} {r['mean_baseline']:<11.4f} "
              f"{r['mean_diff']:<+9.4f} {r['t_stat']:<7.2f} {r['p_value']:<9.4f} "
              f"{r['p_value_holm']:<9.4f} {r['cohens_d']:<7.2f} {sig}")

    print()
    print("=" * 78)
    print("2. Omnibus test: is the period-to-period spread distinguishable from noise?")
    print("=" * 78)
    print(f"Friedman chi-square = {omnibus['friedman_stat']:.3f}, p = {omnibus['friedman_p']:.4f} "
          f"(n={omnibus['n_seeds']} seeds x {omnibus['n_periods']} periods)")
    if omnibus["rm_anova"]:
        a = omnibus["rm_anova"]
        print(f"Repeated-measures ANOVA: F({a['df_num']:.0f},{a['df_denom']:.0f}) = {a['F_stat']:.3f}, "
              f"p = {a['p_value']:.4f}")
    else:
        print(f"Repeated-measures ANOVA: unavailable ({omnibus['rm_anova_error']})")

    print()
    print("=" * 78)
    print(f"3. Best (P={bw['best_period']}) vs. worst (P={bw['worst_period']}) periodic config")
    print("=" * 78)
    print(f"mean P={bw['best_period']}: {bw['mean_best']:.4f}   mean P={bw['worst_period']}: {bw['mean_worst']:.4f}   "
          f"diff: {bw['mean_diff']:+.4f}")
    print(f"paired t = {bw['t_stat']:.2f}, uncorrected p = {bw['p_value_uncorrected']:.4f}, "
          f"Cohen's d = {bw['cohens_d']:.2f}")
    sig = "yes" if bw["significant_at_0.05_holm"] else "no"
    p_holm_str = f"{bw['p_value_holm_within_all_pairs']:.4f}" if bw['p_value_holm_within_all_pairs'] is not None else "n/a"
    print(f"Holm-corrected p (within all 28 period-vs-period pairs) = {p_holm_str}  "
          f"significant at 0.05? {sig}")
    print("Note: best/worst were picked by looking at the data, so the uncorrected p above is NOT")
    print("a valid significance test on its own ('testing the winner' bias) -- the Holm-corrected")
    print("value, drawn from the full 28-pair family, is the one to trust.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--metric", default="final_test_loss",
                    choices=["final_test_loss", "best_val_loss"])
    args = p.parse_args()

    results = load_results(args.runs_dir)
    if not results:
        print(f"No final_result.json files found under {args.runs_dir}/ -- nothing to analyze yet.")
        sys.exit(0)

    groups = paired_by_seed(results, metric=args.metric)
    periods = sorted(p for (mode, p) in groups if mode == "periodic")
    if not periods or ("baseline", None) not in groups:
        print("Need both periodic configs and a baseline config to run significance tests.")
        sys.exit(1)

    pvb_rows = periodic_vs_baseline(groups, periods)
    omnibus = omnibus_period_effect(groups, periods)
    pairwise_rows = all_pairwise_period_comparisons(groups, periods)
    bw = best_vs_worst(groups, periods, pairwise_family=pairwise_rows)

    print_report(pvb_rows, omnibus, bw)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "stats_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metric": args.metric,
            "periodic_vs_baseline": pvb_rows,
            "omnibus_period_effect": omnibus,
            "all_pairwise_period_comparisons": pairwise_rows,
            "best_vs_worst": bw,
        }, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
