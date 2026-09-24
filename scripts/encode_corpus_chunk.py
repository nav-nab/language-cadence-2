"""
encode_corpus_chunk.py -- encode a slice of a jsonl split into token ids
(using the frozen configs/vocab.json) and save as a .npy chunk. Chunked for
the same reason as tokenize_corpus_chunk.py: a big split can be processed
across several short shell calls. Talks are separated by a single <eos>
token so the model doesn't see spurious continuity across talk boundaries.

Usage:
    python3 scripts/encode_corpus_chunk.py <split.jsonl> <chunk_dir> <start> <end>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syllable_tokenizer.vocab import Vocab, EOS  # noqa: E402


def main(in_path, chunk_dir, start, end):
    vocab = Vocab.load("configs/vocab.json")
    eos_id = vocab.piece_to_id[EOS]

    texts = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["content"])

    end = min(end, len(texts))
    ids: list[int] = []
    for t in texts[start:end]:
        ids.extend(vocab.encode(t))
        ids.append(eos_id)

    arr = np.array(ids, dtype=np.uint16)
    os.makedirs(chunk_dir, exist_ok=True)
    out_path = os.path.join(chunk_dir, f"chunk_{start:06d}_{end:06d}.npy")
    np.save(out_path, arr)
    print(f"encoded [{start}:{end}) of {len(texts)} texts -> {len(arr)} ids -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
