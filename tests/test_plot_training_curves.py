"""Tests for scripts/plot_training_curves.py -- mainly the metrics.jsonl
dedup logic, since a real resumed run can log the same step twice."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import plot_training_curves as pc


def write_run(tmp_path, name, pe_mode, period, seed, steps_and_losses):
    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "metrics.jsonl", "w") as f:
        for step, loss in steps_and_losses:
            f.write(json.dumps({"step": step, "epoch": 0, "train_loss": loss,
                                 "val_loss_subset": loss, "wall_time": 0.0}) + "\n")
    final = {"run_id": name, "pe_mode": pe_mode, "period": period, "seed": seed,
              "final_test_loss": steps_and_losses[-1][1], "best_step": steps_and_losses[-1][0],
              "best_val_loss": steps_and_losses[-1][1], "total_steps": steps_and_losses[-1][0]}
    with open(run_dir / "final_result.json", "w") as f:
        json.dump(final, f)


def test_load_curves_dedupes_resumed_duplicate_steps_keeping_last(tmp_path):
    # Simulates a run interrupted and resumed: step 100 and 200 logged once
    # before the interruption (with stale/slightly-different values), then
    # re-logged after resuming from checkpoint.
    write_run(tmp_path, "periodic_P2_seed0", "periodic", 2, 0, [
        (100, 5.9), (200, 5.3),             # before interruption
        (100, 5.91), (200, 5.31), (300, 4.8),  # after resume
    ])
    curves = pc.load_curves(str(tmp_path))
    steps, vals = curves[("periodic", 2)][0]
    assert list(steps) == [100, 200, 300]
    # last-written record for each step wins
    assert vals[0] == 5.91
    assert vals[1] == 5.31
    assert vals[2] == 4.8


def test_load_curves_groups_by_config_and_seed(tmp_path):
    write_run(tmp_path, "baseline_seed0", "baseline", None, 0, [(100, 5.0), (200, 4.5)])
    write_run(tmp_path, "baseline_seed1", "baseline", None, 1, [(100, 5.1), (200, 4.6)])
    write_run(tmp_path, "rope_seed0", "rope", None, 0, [(100, 4.9), (200, 4.0)])
    curves = pc.load_curves(str(tmp_path))
    assert set(curves[("baseline", None)].keys()) == {0, 1}
    assert set(curves[("rope", None)].keys()) == {0}


def test_aggregate_computes_mean_and_std_across_seeds():
    curves = {
        ("baseline", None): {
            0: (np.array([100, 200]), np.array([5.0, 4.0])),
            1: (np.array([100, 200]), np.array([6.0, 4.0])),
        }
    }
    agg = pc.aggregate(curves)
    steps, mean, std = agg[("baseline", None)]
    assert list(steps) == [100, 200]
    assert mean[0] == 5.5
    assert mean[1] == 4.0
    assert std[0] == 0.5
    assert std[1] == 0.0


def test_aggregate_raises_on_genuine_step_grid_mismatch():
    curves = {
        ("periodic", 5): {
            0: (np.array([100, 200]), np.array([5.0, 4.0])),
            1: (np.array([100, 300]), np.array([5.0, 4.0])),  # different grid, not a dedupe case
        }
    }
    try:
        pc.aggregate(curves)
        assert False, "expected an assertion error for mismatched step grids"
    except AssertionError:
        pass
