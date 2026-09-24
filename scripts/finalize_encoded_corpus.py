"""
finalize_encoded_corpus.py -- concatenate encode_corpus_chunk.py's chunks (in
order) into the final token-id array for a split.

Usage:
    python3 scripts/finalize_encoded_corpus.py <chunk_dir> <out_path>
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np


def main(chunk_dir, out_path):
    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npy")))
    arrays = [np.load(p) for p in chunk_paths]
    full = np.concatenate(arrays)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, full)
    print(f"concatenated {len(chunk_paths)} chunks -> {len(full)} ids -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
