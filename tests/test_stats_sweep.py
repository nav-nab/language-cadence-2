"""Tests for scripts/stats_sweep.py, using synthetic data with known
properties so we trust the statistical machinery before pointing it at the
real sweep results."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import stats_sweep as ss


def make_result(pe_mode, period, seed, final_test_loss, best_val_loss=None):
    return {
        "run_id": f"{pe_mode}_{period}_seed{seed}",
        "pe_mode": pe_mode,
        "period": period,
        "seed": seed,
        "final_test_loss": final_test_loss,
        "best_val_loss": best_val_loss if best_val_loss is not None else final_test_loss + 0.03,
        "best_step": 1000,
        "total_steps": 3000,
    }


def test_paired_by_seed_groups_correctly():
    results = [
        make_result("baseline", None, 0, 1.0),
        make_result("baseline", None, 1, 1.1),
        make_result("periodic", 2, 0, 1.2),
        make_result("periodic", 2, 1, 1.3),
    ]
    groups = ss.paired_by_seed(results)
    assert groups[("baseline", None)] == {0: 1.0, 1: 1.1}
    assert groups[("periodic", 2)] == {0: 1.2, 1: 1.3}


def test_aligned_arrays_matches_by_seed_even_if_unordered():
    groups = {
        ("periodic", 2): {1: 2.0, 0: 1.0, 2: 3.0},
        ("baseline", None): {0: 1.5, 2: 3.5, 1: 2.5},
    }
    a, b, seeds = ss.aligned_arrays(groups, ("periodic", 2), ("baseline", None))
    assert seeds == [0, 1, 2]
    assert list(a) == [1.0, 2.0, 3.0]
    assert list(b) == [1.5, 2.5, 3.5]


def test_cohens_d_paired_known_value():
    # diff = [1, 1, 1] -> mean 1, sd 0 -> infinite effect (perfectly consistent)
    a = np.array([2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0])
    d = ss.cohens_d_paired(a, b)
    assert d == float("inf")

    # diff = [1, 2, 0] -> mean 1, sd(ddof=1) = 1 -> d = 1
    a2 = np.array([2.0, 4.0, 2.0])
    b2 = np.array([1.0, 2.0, 2.0])
    d2 = ss.cohens_d_paired(a2, b2)
    assert d2 == pytest.approx(1.0)


def test_cohens_d_paired_zero_diff_and_zero_sd():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    assert ss.cohens_d_paired(a, b) == 0.0


def test_holm_bonferroni_matches_hand_computation():
    # Raw p-values; Holm-Bonferroni: sort ascending, multiply by (n - rank),
    # take cumulative max, cap at 1.
    pvals = [0.01, 0.04, 0.03, 0.50]
    adj = ss.holm_bonferroni(pvals)
    # sorted order: 0.01(idx0), 0.03(idx2), 0.04(idx1), 0.50(idx3)
    # idx0: 4*0.01=0.04 -> running_max=0.04
    # idx2: 3*0.03=0.09 -> running_max=0.09
    # idx1: 2*0.04=0.08 -> running_max stays 0.09 (monotonic)
    # idx3: 1*0.50=0.50 -> running_max=0.50
    assert adj[0] == pytest.approx(0.04)
    assert adj[2] == pytest.approx(0.09)
    assert adj[1] == pytest.approx(0.09)
    assert adj[3] == pytest.approx(0.50)


def test_holm_bonferroni_caps_at_one():
    adj = ss.holm_bonferroni([0.9, 0.9])
    assert all(v <= 1.0 for v in adj)


def test_periodic_vs_baseline_detects_strong_consistent_effect():
    # Periodic is always exactly 1.0 higher than baseline, same across seeds
    # -> this should be a very significant, large-effect difference even
    # with only n=3.
    results = []
    for seed in range(3):
        results.append(make_result("baseline", None, seed, 3.0 + 0.01 * seed))
        results.append(make_result("periodic", 5, seed, 4.0 + 0.01 * seed))
    groups = ss.paired_by_seed(results)
    rows = ss.periodic_vs_baseline(groups, periods=[5])
    row = rows[0]
    assert row["mean_diff"] == pytest.approx(1.0, abs=0.02)
    assert row["p_value"] < 0.01
    assert row["cohens_d"] > 2.0  # huge, consistent effect


def test_periodic_vs_baseline_no_effect_when_identical():
    results = []
    for seed in range(3):
        val = 3.0 + 0.1 * seed  # varies across seeds but identical between configs
        results.append(make_result("baseline", None, seed, val))
        results.append(make_result("periodic", 5, seed, val))
    groups = ss.paired_by_seed(results)
    rows = ss.periodic_vs_baseline(groups, periods=[5])
    row = rows[0]
    assert row["mean_diff"] == pytest.approx(0.0, abs=1e-9)
    assert row["cohens_d"] == 0.0


def test_periodic_vs_baseline_applies_holm_correction_across_periods():
    # 3 periods, all with a real but modest, noisy effect -- just check the
    # Holm-adjusted p-values are >= the raw p-values (correction only makes
    # things harder to call significant, never easier).
    results = []
    rng = np.random.default_rng(0)
    for seed in range(3):
        base = 3.0 + rng.normal(0, 0.05)
        results.append(make_result("baseline", None, seed, base))
        for period in [2, 3, 4]:
            results.append(make_result("periodic", period, seed, base + 0.02 + rng.normal(0, 0.01)))
    groups = ss.paired_by_seed(results)
    rows = ss.periodic_vs_baseline(groups, periods=[2, 3, 4])
    for r in rows:
        assert r["p_value_holm"] >= r["p_value"] - 1e-12


def test_omnibus_period_effect_detects_real_period_differences():
    # Periods have systematically different means, small within-period noise
    # -> Friedman and RM-ANOVA should both report small p-values.
    results = []
    period_means = {2: 3.0, 3: 3.5, 4: 4.0, 5: 4.5}
    rng = np.random.default_rng(1)
    for seed in range(3):
        for period, mean in period_means.items():
            results.append(make_result("periodic", period, seed, mean + rng.normal(0, 0.01)))
    groups = ss.paired_by_seed(results)
    omnibus = ss.omnibus_period_effect(groups, periods=sorted(period_means))
    assert omnibus["friedman_p"] < 0.1
    if omnibus["rm_anova"] is not None:
        assert omnibus["rm_anova"]["p_value"] < 0.05


def test_omnibus_period_effect_no_signal_when_periods_identical_in_distribution():
    # Period adds no systematic effect -- each (seed, period) cell is pure
    # noise uncorrelated with period, so the rank order within each seed's
    # row is effectively random. (Exactly-tied values within a row are a
    # degenerate case for Friedman's tie-correction term, so we use distinct
    # noise per cell rather than literal ties.) With n=3 seeds x 4 periods
    # this is a weak test of power, but it checks the plumbing and that
    # nothing spuriously reports strong significance.
    results = []
    rng = np.random.default_rng(2)
    for seed in range(3):
        for period in [2, 3, 4, 5]:
            results.append(make_result("periodic", period, seed, 3.0 + rng.normal(0, 0.05)))
    groups = ss.paired_by_seed(results)
    omnibus = ss.omnibus_period_effect(groups, periods=[2, 3, 4, 5])
    assert omnibus["friedman_p"] > 0.05


def test_best_vs_worst_identifies_correct_periods_and_sign():
    results = []
    for seed in range(3):
        results.append(make_result("periodic", 2, seed, 3.0 + 0.001 * seed))   # best (lowest)
        results.append(make_result("periodic", 9, seed, 3.5 + 0.001 * seed))   # middle
        results.append(make_result("periodic", 128, seed, 4.0 + 0.001 * seed))  # worst (highest)
    groups = ss.paired_by_seed(results)
    bw = ss.best_vs_worst(groups, periods=[2, 9, 128])
    assert bw["best_period"] == 2
    assert bw["worst_period"] == 128
    assert bw["mean_diff"] > 0  # worst - best should be positive


def test_load_results_reads_final_result_json_files(tmp_path):
    run_dir = tmp_path / "runs" / "baseline_seed0"
    run_dir.mkdir(parents=True)
    data = make_result("baseline", None, 0, 3.9)
    with open(run_dir / "final_result.json", "w") as f:
        json.dump(data, f)
    results = ss.load_results(str(tmp_path / "runs"))
    assert len(results) == 1
    assert results[0]["pe_mode"] == "baseline"
    assert results[0]["seed"] == 0
    assert results[0]["final_test_loss"] == pytest.approx(3.9)


def test_main_runs_end_to_end_on_synthetic_sweep(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "results"
    rng = np.random.default_rng(3)
    configs = [("baseline", None)] + [("periodic", p) for p in [2, 3, 6, 9, 16, 45, 128, 1024]] + [("rope", None)]
    for pe_mode, period in configs:
        for seed in range(3):
            tag = f"{pe_mode}_{period}_seed{seed}" if period is not None else f"{pe_mode}_seed{seed}"
            d = runs_dir / tag
            d.mkdir(parents=True)
            base = 3.6 if pe_mode == "rope" else 3.9
            data = make_result(pe_mode, period, seed, base + rng.normal(0, 0.02))
            with open(d / "final_result.json", "w") as f:
                json.dump(data, f)

    old_argv = sys.argv
    try:
        sys.argv = ["stats_sweep.py", "--runs_dir", str(runs_dir), "--out_dir", str(out_dir)]
        ss.main()
    finally:
        sys.argv = old_argv

    out_path = out_dir / "stats_sweep.json"
    assert out_path.exists()
    with open(out_path) as f:
        saved = json.load(f)
    assert len(saved["periodic_vs_baseline"]) == 8
    assert "friedman_p" in saved["omnibus_period_effect"]
    assert "best_period" in saved["best_vs_worst"]

    captured = capsys.readouterr()
    assert "Periodic vs. baseline" in captured.out
    assert "Omnibus test" in captured.out
