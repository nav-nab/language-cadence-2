"""
finalize_vocab.py -- from the precomputed train/val/test piece-frequency
counters (see tokenize_corpus_chunk.py), build a coverage-vs-vocab-size
curve and save the chosen vocabulary.

Usage:
    python3 scripts/finalize_vocab.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syllable_tokenizer.vocab import Vocab, SPECIAL_TOKENS  # noqa: E402

CANDIDATE_SIZES = [1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000, 19286]
CHOSEN_SIZE = 8000


def load_counter(path: str) -> Counter:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Counter(data["counts"])


def main():
    train_counts = load_counter("data/processed/tmp/train_counts.json")
    val_counts = load_counter("data/processed/tmp/val_counts.json")
    test_counts = load_counter("data/processed/tmp/test_counts.json")

    ranked_pieces = [p for p, _ in train_counts.most_common()]
    total_train_tokens = sum(train_counts.values())
    total_val_tokens = sum(val_counts.values())
    total_test_tokens = sum(test_counts.values())

    curve = []
    cum_train = 0
    cum_val = 0
    prev_n = 0
    for size in CANDIDATE_SIZES:
        size = min(size, len(ranked_pieces))
        chunk = ranked_pieces[prev_n:size]
        cum_train += sum(train_counts[p] for p in chunk)
        cum_val += sum(val_counts.get(p, 0) for p in chunk)
        prev_n = size
        curve.append(
            {
                "vocab_size": size,
                "train_coverage": cum_train / total_train_tokens,
                "val_coverage": cum_val / total_val_tokens,
            }
        )
        print(f"size={size:6d}  train_coverage={curve[-1]['train_coverage']:.4%}  val_coverage={curve[-1]['val_coverage']:.4%}")

    chosen_pieces = ranked_pieces[:CHOSEN_SIZE]
    chosen_set = set(chosen_pieces)
    chosen_test_coverage = sum(test_counts.get(p, 0) for p in chosen_set) / total_test_tokens

    id_to_piece = list(SPECIAL_TOKENS) + chosen_pieces[: CHOSEN_SIZE - len(SPECIAL_TOKENS)]
    vocab = Vocab(
        piece_to_id={p: i for i, p in enumerate(id_to_piece)},
        id_to_piece=id_to_piece,
        piece_freq={p: train_counts.get(p, 0) for p in id_to_piece},
    )
    os.makedirs("configs", exist_ok=True)
    vocab.save("configs/vocab.json")

    os.makedirs("results", exist_ok=True)
    with open("results/vocab_coverage.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_train_tokens": total_train_tokens,
                "total_val_tokens": total_val_tokens,
                "total_test_tokens": total_test_tokens,
                "n_distinct_pieces_in_train": len(ranked_pieces),
                "candidate_sizes_curve": curve,
                "chosen_size": len(id_to_piece),
                "chosen_val_coverage": next(c["val_coverage"] for c in curve if c["vocab_size"] == CHOSEN_SIZE),
                "chosen_train_coverage": next(c["train_coverage"] for c in curve if c["vocab_size"] == CHOSEN_SIZE),
                "chosen_test_coverage": chosen_test_coverage,
            },
            f,
            indent=2,
        )
    print(f"\nChosen vocab size {len(id_to_piece)}: test coverage = {chosen_test_coverage:.4%}")
    print("Saved configs/vocab.json and results/vocab_coverage.json")


if __name__ == "__main__":
    main()
