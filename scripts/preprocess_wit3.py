"""
preprocess_wit3.py -- clean WIT3 English TED talks and split train/val/test
by speaker.

Frozen preprocessing decisions (checked with Navid, 2026-09-24):
  1. Cue stripping: remove parenthetical audience/audio cues like
     "(Laughter)", "(Applause)", "(Music)", "(In Chinese)" -- drop, don't
     replace with a placeholder token.
  2. Minimum talk length: drop talks under 100 words (measured AFTER cue
     stripping, i.e. actual spoken-content word count).
  3. Dedupe by talkid (kept as a safety check; the corpus already has none).
  4. Casing: keep original case.
  5. Punctuation: keep as-is (kept deliberately -- punctuation may carry
     clause/sentence-boundary "rhythm" information; see DECISIONS.md).
  6. Digits: keep as-is (the tokenizer already splits them into single-digit
     tokens).
  7. Train/val/test split: by SPEAKER (a speaker's talks all land in one
     split, so no speaker crosses a split boundary), greedily bin-packed to
     hit ~90/5/5 by WORD count (not talk count, since talk length varies a
     lot). Deterministic given a fixed speaker ordering.

Usage:
    python3 scripts/preprocess_wit3.py data/processed/talks.jsonl data/processed/
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

CUE_RE = re.compile(r"\([A-Za-z][A-Za-z '\-]{1,40}\)")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

MIN_WORDS = 100
SPLIT_RATIOS = {"train": 0.90, "val": 0.05, "test": 0.05}


def clean_content(text: str) -> str:
    text = CUE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_talks(path: str) -> list[dict]:
    talks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                talks.append(json.loads(line))
    return talks


def dedupe_by_talkid(talks: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for t in talks:
        if t["talkid"] in seen:
            continue
        seen.add(t["talkid"])
        out.append(t)
    return out


def assign_splits_by_speaker(talks: list[dict]) -> dict[int, str]:
    """Greedy bin-packing: group talks by speaker, sort speaker groups by
    total word count descending, and assign each group whole to whichever
    split is currently furthest below its target share of total words."""
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for t in talks:
        by_speaker[t.get("speaker") or f"UNKNOWN_{t['talkid']}"].append(t)

    speaker_words = {
        spk: sum(t["clean_wordnum"] for t in group) for spk, group in by_speaker.items()
    }
    total_words = sum(speaker_words.values())
    targets = {split: ratio * total_words for split, ratio in SPLIT_RATIOS.items()}
    current = {split: 0 for split in SPLIT_RATIOS}

    # largest speakers first -- classic greedy bin-packing for balance
    ordered_speakers = sorted(speaker_words, key=lambda s: -speaker_words[s])

    talkid_to_split: dict[int, str] = {}
    for spk in ordered_speakers:
        # pick the split with the largest remaining deficit (target - current)
        deficits = {split: targets[split] - current[split] for split in SPLIT_RATIOS}
        chosen = max(deficits, key=lambda s: deficits[s])
        current[chosen] += speaker_words[spk]
        for t in by_speaker[spk]:
            talkid_to_split[t["talkid"]] = chosen

    return talkid_to_split


def main(in_path: str, out_dir: str):
    talks = load_talks(in_path)
    n_before = len(talks)

    talks = dedupe_by_talkid(talks)
    n_after_dedupe = len(talks)

    for t in talks:
        t["clean_content"] = clean_content(t["content"])
        t["clean_wordnum"] = len(WORD_RE.findall(t["clean_content"]))

    talks = [t for t in talks if t["clean_wordnum"] >= MIN_WORDS]
    n_after_length_filter = len(talks)

    splits = assign_splits_by_speaker(talks)

    os.makedirs(out_dir, exist_ok=True)
    split_files = {split: open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") for split in SPLIT_RATIOS}
    split_stats = {
        split: {"n_talks": 0, "n_words": 0, "n_speakers": set()} for split in SPLIT_RATIOS
    }

    for t in talks:
        split = splits[t["talkid"]]
        rec = {
            "talkid": t["talkid"],
            "speaker": t.get("speaker"),
            "date": t.get("date"),
            "title": t.get("title"),
            "content": t["clean_content"],
            "wordnum": t["clean_wordnum"],
        }
        split_files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        split_stats[split]["n_talks"] += 1
        split_stats[split]["n_words"] += t["clean_wordnum"]
        split_stats[split]["n_speakers"].add(t.get("speaker"))

    for f in split_files.values():
        f.close()

    total_words_final = sum(s["n_words"] for s in split_stats.values())
    report = {
        "n_talks_before": n_before,
        "n_talks_after_dedupe": n_after_dedupe,
        "n_duplicates_removed": n_before - n_after_dedupe,
        "n_talks_after_min_length_filter": n_after_length_filter,
        "n_talks_dropped_for_length": n_after_dedupe - n_after_length_filter,
        "min_words_threshold": MIN_WORDS,
        "total_words_final": total_words_final,
        "splits": {
            split: {
                "n_talks": s["n_talks"],
                "n_words": s["n_words"],
                "word_share": s["n_words"] / total_words_final,
                "n_distinct_speakers": len(s["n_speakers"]),
            }
            for split, s in split_stats.items()
        },
    }
    return report


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/talks.jsonl"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    report = main(in_path, out_dir)
    print(json.dumps(report, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/wit3_preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
