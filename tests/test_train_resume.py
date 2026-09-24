"""
Functional test for scripts/train.py's checkpoint/resume behavior: training
in two interrupted halves must reach the exact same final state as training
straight through in one go (same seed, same data order, no skipped or
duplicated batches).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "scripts", "train.py")


def _make_tiny_corpus(data_dir, vocab_path, n_tokens=2000, vocab_size=24, seed=0):
    rng = np.random.RandomState(seed)
    for name in ("train", "val", "test"):
        ids = rng.randint(4, vocab_size, size=n_tokens).astype(np.uint16)  # skip special-token ids 0-3
        np.save(os.path.join(data_dir, f"{name}_ids.npy"), ids)

    id_to_piece = ["<pad>", "<unk>", "<bos>", "<eos>"] + [f"tok{i}" for i in range(vocab_size - 4)]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"id_to_piece": id_to_piece, "piece_freq": {}}, f)


def _run(run_dir, steps, extra_args=()):
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--pe_mode", "baseline",
        "--seed", "0",
        "--steps", str(steps),
        "--batch_size", "4",
        "--seq_len", "4",
        "--d_model", "8",
        "--n_layers", "1",
        "--n_heads", "2",
        "--vocab_path", os.path.join(run_dir, "vocab.json"),
        "--data_dir", run_dir,
        "--out_dir", os.path.join(run_dir, "runs"),
        "--checkpoint_every_seconds", "0",
        "--eval_every_steps", "2",
        "--eval_batches", "1",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result


@pytest.mark.slow
def test_interrupted_resume_matches_straight_through(tmp_path):
    data_dir = str(tmp_path)
    vocab_path = os.path.join(data_dir, "vocab.json")
    _make_tiny_corpus(data_dir, vocab_path, n_tokens=2000, vocab_size=24)

    # straight-through run
    straight_dir = os.path.join(data_dir, "straight")
    os.makedirs(straight_dir)
    _make_tiny_corpus(straight_dir, os.path.join(straight_dir, "vocab.json"), n_tokens=2000, vocab_size=24)
    _run(straight_dir, steps=6)
    straight_ckpt = torch.load(os.path.join(straight_dir, "runs", "baseline_seed0", "checkpoint.pt"), map_location="cpu")
    straight_result = json.load(open(os.path.join(straight_dir, "runs", "baseline_seed0", "final_result.json")))

    # interrupted run: 0->3, then 3->6
    interrupted_dir = os.path.join(data_dir, "interrupted")
    os.makedirs(interrupted_dir)
    _make_tiny_corpus(interrupted_dir, os.path.join(interrupted_dir, "vocab.json"), n_tokens=2000, vocab_size=24)
    _run(interrupted_dir, steps=3)
    _run(interrupted_dir, steps=6)  # should resume from step 3
    interrupted_ckpt = torch.load(os.path.join(interrupted_dir, "runs", "baseline_seed0", "checkpoint.pt"), map_location="cpu")
    interrupted_result = json.load(open(os.path.join(interrupted_dir, "runs", "baseline_seed0", "final_result.json")))

    assert straight_ckpt["step"] == interrupted_ckpt["step"] == 6

    for k in straight_ckpt["model_state_dict"]:
        assert torch.allclose(straight_ckpt["model_state_dict"][k], interrupted_ckpt["model_state_dict"][k], atol=1e-6), \
            f"model weights diverged for {k}"

    assert straight_result["final_test_loss"] == pytest.approx(interrupted_result["final_test_loss"], abs=1e-5)

    # metrics.jsonl from the interrupted run should have no duplicate step entries
    metrics_path = os.path.join(interrupted_dir, "runs", "baseline_seed0", "metrics.jsonl")
    steps_logged = [json.loads(line)["step"] for line in open(metrics_path)]
    assert len(steps_logged) == len(set(steps_logged)), f"duplicate steps in metrics: {steps_logged}"


@pytest.mark.slow
def test_resume_refuses_mismatched_config(tmp_path):
    data_dir = str(tmp_path)
    _make_tiny_corpus(data_dir, os.path.join(data_dir, "vocab.json"), n_tokens=2000, vocab_size=24)
    _run(data_dir, steps=3)

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--pe_mode", "baseline",
        "--seed", "0",
        "--steps", "6",
        "--batch_size", "4",
        "--seq_len", "4",
        "--d_model", "16",  # mismatched on purpose
        "--n_layers", "1",
        "--n_heads", "2",
        "--vocab_path", os.path.join(data_dir, "vocab.json"),
        "--data_dir", data_dir,
        "--out_dir", os.path.join(data_dir, "runs"),
        "--checkpoint_every_seconds", "0",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "mismatch" in result.stderr.lower()
