import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from run_sweep import build_configs, run_id_for, PERIODS, SEEDS  # noqa: E402


def test_build_configs_count_and_shape():
    configs = build_configs()
    assert len(configs) == len(PERIODS) * len(SEEDS) + len(SEEDS) + len(SEEDS)

    periodic = [c for c in configs if c["pe_mode"] == "periodic"]
    assert len(periodic) == len(PERIODS) * len(SEEDS)
    assert len(set((c["period"], c["seed"]) for c in periodic)) == len(periodic)  # all unique

    baseline = [c for c in configs if c["pe_mode"] == "baseline"]
    rope = [c for c in configs if c["pe_mode"] == "rope"]
    assert len(baseline) == len(SEEDS)
    assert len(rope) == len(SEEDS)
    assert all(c["period"] is None for c in baseline + rope)


def test_run_id_for_formatting():
    assert run_id_for({"pe_mode": "periodic", "period": 45, "seed": 1}) == "periodic_P45_seed1"
    assert run_id_for({"pe_mode": "baseline", "period": None, "seed": 2}) == "baseline_seed2"
    assert run_id_for({"pe_mode": "rope", "period": None, "seed": 0}) == "rope_seed0"


def test_dry_run_does_not_invoke_train(tmp_path, capsys):
    import subprocess
    script = os.path.join(REPO_ROOT, "scripts", "run_sweep.py")
    result = subprocess.run(
        [sys.executable, script, "--only", "baseline", "--out_dir", str(tmp_path), "--dry_run"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "3 run(s) planned" in result.stdout
    assert not os.listdir(tmp_path)  # nothing was actually run


import json  # noqa: E402

from run_sweep import measure_completed_run_seconds, average_completed_run_seconds  # noqa: E402


def _write_completed_run(runs_dir, run_id, total_steps, step_records):
    """step_records: list of (step, wall_time) tuples written to metrics.jsonl."""
    d = os.path.join(runs_dir, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        for step, wall_time in step_records:
            f.write(json.dumps({"step": step, "epoch": 0, "train_loss": 1.0,
                                 "val_loss_subset": 1.0, "wall_time": wall_time}) + "\n")
    with open(os.path.join(d, "final_result.json"), "w") as f:
        json.dump({"run_id": run_id, "pe_mode": "baseline", "period": None, "seed": 0,
                   "final_test_loss": 1.0, "best_step": total_steps, "best_val_loss": 1.0,
                   "total_steps": total_steps}, f)


def test_measure_completed_run_seconds_extrapolates_to_full_run(tmp_path):
    runs_dir = str(tmp_path)
    # 1 second/step measured between step 100 and step 1000 -> 3000-step run should be ~3000s
    _write_completed_run(runs_dir, "baseline_seed0", total_steps=3000,
                          step_records=[(100, 1000.0), (1000, 1900.0)])
    d = measure_completed_run_seconds(os.path.join(runs_dir, "baseline_seed0"))
    assert d == pytest.approx(3000.0, rel=0.01)


def test_measure_completed_run_seconds_none_when_incomplete_or_missing(tmp_path):
    runs_dir = str(tmp_path)
    # no final_result.json at all
    os.makedirs(os.path.join(runs_dir, "in_progress"), exist_ok=True)
    with open(os.path.join(runs_dir, "in_progress", "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 100, "wall_time": 1.0}) + "\n")
    assert measure_completed_run_seconds(os.path.join(runs_dir, "in_progress")) is None

    # only one metrics record -- not enough to measure a rate
    _write_completed_run(runs_dir, "one_record", total_steps=3000, step_records=[(100, 1000.0)])
    assert measure_completed_run_seconds(os.path.join(runs_dir, "one_record")) is None


def test_average_completed_run_seconds_averages_across_runs(tmp_path):
    runs_dir = str(tmp_path)
    _write_completed_run(runs_dir, "run_a", total_steps=1000, step_records=[(0, 0.0), (1000, 1000.0)])
    _write_completed_run(runs_dir, "run_b", total_steps=1000, step_records=[(0, 0.0), (1000, 3000.0)])
    # in-progress run with no final_result.json should be ignored, not error
    os.makedirs(os.path.join(runs_dir, "run_c_in_progress"), exist_ok=True)

    avg = average_completed_run_seconds(runs_dir)
    assert avg == pytest.approx(2000.0)  # mean of 1000s and 3000s


def test_average_completed_run_seconds_none_when_nothing_completed(tmp_path):
    runs_dir = str(tmp_path)
    os.makedirs(os.path.join(runs_dir, "in_progress"), exist_ok=True)
    assert average_completed_run_seconds(runs_dir) is None
