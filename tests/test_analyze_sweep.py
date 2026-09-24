import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from analyze_sweep import load_results, group_by_config, summarize, best_period_row  # noqa: E402


def _write_result(runs_dir, run_id, pe_mode, period, seed, test_loss):
    d = os.path.join(runs_dir, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "final_result.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id, "pe_mode": pe_mode, "period": period, "seed": seed,
            "final_test_loss": test_loss, "best_step": 100, "best_val_loss": test_loss - 0.05,
            "total_steps": 200,
        }, f)


def test_group_and_summarize(tmp_path):
    runs_dir = str(tmp_path)
    _write_result(runs_dir, "periodic_P8_seed0", "periodic", 8, 0, 3.0)
    _write_result(runs_dir, "periodic_P8_seed1", "periodic", 8, 1, 3.2)
    _write_result(runs_dir, "periodic_P16_seed0", "periodic", 16, 0, 2.5)
    _write_result(runs_dir, "baseline_seed0", "baseline", None, 0, 3.5)

    results = load_results(runs_dir)
    assert len(results) == 4

    groups = group_by_config(results)
    assert groups[("periodic", 8)] == [3.0, 3.2]
    assert groups[("periodic", 16)] == [2.5]
    assert groups[("baseline", None)] == [3.5]

    rows = summarize(groups)
    row_p8 = next(r for r in rows if r["pe_mode"] == "periodic" and r["period"] == 8)
    assert row_p8["mean_test_loss"] == pytest.approx(3.1)
    assert row_p8["n_seeds"] == 2

    best = best_period_row(rows)
    assert best["period"] == 16  # lowest mean among periodic configs


def test_best_period_row_none_when_no_periodic_configs(tmp_path):
    runs_dir = str(tmp_path)
    _write_result(runs_dir, "baseline_seed0", "baseline", None, 0, 3.5)
    results = load_results(runs_dir)
    rows = summarize(group_by_config(results))
    assert best_period_row(rows) is None


def test_load_results_empty_dir_returns_empty_list(tmp_path):
    assert load_results(str(tmp_path)) == []
