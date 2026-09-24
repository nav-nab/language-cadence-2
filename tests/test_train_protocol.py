"""
Tests for the Stage 5 protocol additions to scripts/train.py:
  - hard divergence halt on a non-finite loss (check_finite_loss)
  - best-val-loss checkpoint tracking and its use for the reported test loss
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "scripts", "train.py")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from train import check_finite_loss  # noqa: E402


def test_check_finite_loss_halts_on_nan(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        check_finite_loss(float("nan"), step=5, run_dir=str(tmp_path), run_id="test_run")
    assert exc_info.value.code == 1
    diag_path = tmp_path / "divergence.json"
    assert diag_path.exists()
    diag = json.loads(diag_path.read_text())
    assert diag["step"] == 5
    assert diag["run_id"] == "test_run"


def test_check_finite_loss_halts_on_inf(tmp_path):
    with pytest.raises(SystemExit):
        check_finite_loss(float("inf"), step=1, run_dir=str(tmp_path), run_id="test_run")
    assert (tmp_path / "divergence.json").exists()


def test_check_finite_loss_noop_on_finite_value(tmp_path):
    check_finite_loss(1.23, step=5, run_dir=str(tmp_path), run_id="test_run")
    assert not (tmp_path / "divergence.json").exists()


def _make_tiny_corpus(data_dir, vocab_path, n_tokens=2000, vocab_size=24, seed=0):
    rng = np.random.RandomState(seed)
    for name in ("train", "val", "test"):
        ids = rng.randint(4, vocab_size, size=n_tokens).astype(np.uint16)
        np.save(os.path.join(data_dir, f"{name}_ids.npy"), ids)

    id_to_piece = ["<pad>", "<unk>", "<bos>", "<eos>"] + [f"tok{i}" for i in range(vocab_size - 4)]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"id_to_piece": id_to_piece, "piece_freq": {}}, f)


def _run(run_dir, steps, overrides=None):
    default_args = {
        "--pe_mode": "baseline",
        "--seed": "0",
        "--steps": str(steps),
        "--batch_size": "4",
        "--seq_len": "4",
        "--d_model": "8",
        "--n_layers": "1",
        "--n_heads": "2",
        "--vocab_path": os.path.join(run_dir, "vocab.json"),
        "--data_dir": run_dir,
        "--out_dir": os.path.join(run_dir, "runs"),
        "--checkpoint_every_seconds": "0",
        "--eval_every_steps": "1",
        "--eval_batches": "1",
    }
    if overrides:
        default_args.update(overrides)
    cmd = [sys.executable, TRAIN_SCRIPT]
    for k, v in default_args.items():
        cmd += [k, v]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result


@pytest.mark.slow
def test_best_checkpoint_tracks_lowest_logged_val_loss(tmp_path):
    data_dir = str(tmp_path)
    vocab_path = os.path.join(data_dir, "vocab.json")
    _make_tiny_corpus(data_dir, vocab_path, n_tokens=2000, vocab_size=24)

    _run(data_dir, steps=10)

    run_dir = os.path.join(data_dir, "runs", "baseline_seed0")
    metrics = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    assert len(metrics) == 10  # eval_every_steps=1

    best_logged = min(metrics, key=lambda r: r["val_loss_subset"])
    result = json.load(open(os.path.join(run_dir, "final_result.json")))

    assert result["best_step"] == best_logged["step"]
    assert result["best_val_loss"] == pytest.approx(best_logged["val_loss_subset"], abs=1e-6)
    assert os.path.exists(os.path.join(run_dir, "checkpoint_best.pt"))
    assert result["pe_mode"] == "baseline"
    assert result["period"] is None
    assert result["seed"] == 0


@pytest.mark.slow
def test_best_checkpoint_survives_resume(tmp_path):
    data_dir = str(tmp_path)
    vocab_path = os.path.join(data_dir, "vocab.json")
    _make_tiny_corpus(data_dir, vocab_path, n_tokens=2000, vocab_size=24)

    _run(data_dir, steps=5)
    run_dir = os.path.join(data_dir, "runs", "baseline_seed0")
    metrics_after_5 = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    best_after_5 = min(metrics_after_5, key=lambda r: r["val_loss_subset"])

    _run(data_dir, steps=10)  # resumes from step 5
    metrics_after_10 = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    best_overall = min(metrics_after_10, key=lambda r: r["val_loss_subset"])

    result = json.load(open(os.path.join(run_dir, "final_result.json")))
    assert result["best_step"] == best_overall["step"]
    assert result["best_val_loss"] == pytest.approx(best_overall["val_loss_subset"], abs=1e-6)
    # sanity: the pre-resume best is still one of the logged 10 entries
    assert any(m["step"] == best_after_5["step"] for m in metrics_after_10)


@pytest.mark.slow
def test_sweep_tag_appears_in_progress_output(tmp_path):
    data_dir = str(tmp_path)
    vocab_path = os.path.join(data_dir, "vocab.json")
    _make_tiny_corpus(data_dir, vocab_path, n_tokens=2000, vocab_size=24)

    result = _run(data_dir, steps=3, overrides={"--sweep_index": "3", "--sweep_total": "30"})
    assert "[run 3/30]" in result.stdout

    # and the no-tag path (a lone train.py invocation) stays untagged
    other_dir = os.path.join(data_dir, "untagged")
    os.makedirs(other_dir)
    _make_tiny_corpus(other_dir, os.path.join(other_dir, "vocab.json"), n_tokens=2000, vocab_size=24)
    result2 = _run(other_dir, steps=3)
    assert "[run" not in result2.stdout
