import os
import sys

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
