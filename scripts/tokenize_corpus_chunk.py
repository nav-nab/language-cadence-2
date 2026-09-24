"""
tokenize_corpus_chunk.py -- tokenize a slice of a jsonl split and merge piece
counts into a running counter file. Chunked so a big split (e.g. the ~3.9M-
word train set) can be processed across several short calls on a slow
laptop-scale shell, rather than one long call that might get killed.

Usage:
    python3 scripts/tokenize_corpus_chunk.py <split.jsonl> <counter.json> <start> <end>
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syllable_tokenizer.syllabify import tokenize  # noqa: E402


def main(in_path, counter_path, start, end):
    texts = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["content"])

    end = min(end, len(texts))
    counter = Counter()
    if os.path.exists(counter_path):
        with open(counter_path, encoding="utf-8") as f:
            saved = json.load(f)
        counter.update(saved.get("counts", {}))
        done_ranges = saved.get("done_ranges", [])
    else:
        done_ranges = []

    total_tokens_this_chunk = 0
    for t in texts[start:end]:
        toks = tokenize(t)
        counter.update(toks)
        total_tokens_this_chunk += len(toks)

    done_ranges.append([start, end])
    with open(counter_path, "w", encoding="utf-8") as f:
        json.dump({"counts": counter, "done_ranges": done_ranges, "n_texts_total": len(texts)}, f)

    print(f"processed [{start}:{end}) of {len(texts)} texts, {total_tokens_this_chunk} tokens this chunk, "
          f"{sum(counter.values())} total tokens so far, {len(counter)} distinct pieces so far")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
