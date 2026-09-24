"""
inspect_corpus.py -- report word counts, cue frequency, talk-length
distribution, speaker overlap, and CMUdict coverage for the parsed WIT3
English TED corpus (Stage 2 inspection, before any preprocessing decisions
are frozen).

Usage:
    python3 scripts/inspect_corpus.py data/processed/talks.jsonl results/wit3_corpus_stats.json
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syllable_tokenizer.syllabify import cmudict_syllable_count  # noqa: E402

CUE_RE = re.compile(r"\(([A-Za-z][A-Za-z '\-]{1,30})\)")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def load_talks(path: str) -> list[dict]:
    talks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                talks.append(json.loads(line))
    return talks


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main(in_path: str, out_path: str, cmudict_sample_talks: int = 400):
    talks = load_talks(in_path)
    report: dict = {"n_talks": len(talks)}

    # --- dedupe check ---
    talkids = [t["talkid"] for t in talks]
    report["n_distinct_talkids"] = len(set(talkids))
    report["n_duplicate_talkid_rows"] = len(talkids) - len(set(talkids))

    # --- word counts (both self-reported wordnum, and our own whitespace-word count) ---
    reported_wordnums = [t["wordnum"] for t in talks if t.get("wordnum") is not None]
    own_wordcounts = [len(WORD_RE.findall(t["content"])) for t in talks]
    report["total_words_reported_wordnum"] = sum(reported_wordnums)
    report["total_words_own_count"] = sum(own_wordcounts)

    sorted_own = sorted(own_wordcounts)
    report["talk_length_words_own_count"] = {
        "min": sorted_own[0],
        "p10": percentile(sorted_own, 0.10),
        "median": percentile(sorted_own, 0.50),
        "mean": statistics.mean(sorted_own),
        "p90": percentile(sorted_own, 0.90),
        "max": sorted_own[-1],
    }
    report["talks_under_50_words"] = sum(1 for w in own_wordcounts if w < 50)
    report["talks_under_100_words"] = sum(1 for w in own_wordcounts if w < 100)
    report["talks_under_200_words"] = sum(1 for w in own_wordcounts if w < 200)

    # --- cue frequency ---
    cue_counter: Counter = Counter()
    talks_with_any_cue = 0
    for t in talks:
        cues = CUE_RE.findall(t["content"])
        if cues:
            talks_with_any_cue += 1
        for c in cues:
            cue_counter[c.strip().lower()] += 1
    report["talks_with_at_least_one_cue"] = talks_with_any_cue
    report["total_cue_occurrences"] = sum(cue_counter.values())
    report["top_30_cues"] = cue_counter.most_common(30)

    # --- speaker overlap (relevant for the train/val/test split decision) ---
    speakers = [t.get("speaker") or "UNKNOWN" for t in talks]
    speaker_counter = Counter(speakers)
    report["n_distinct_speakers"] = len(speaker_counter)
    report["n_speakers_with_multiple_talks"] = sum(1 for c in speaker_counter.values() if c > 1)
    report["talks_by_repeat_speakers"] = sum(c for c in speaker_counter.values() if c > 1)
    report["top_20_speakers_by_talk_count"] = speaker_counter.most_common(20)

    # --- date range ---
    dates = sorted(t["date"] for t in talks if t.get("date"))
    report["date_range"] = [dates[0], dates[-1]] if dates else None

    # --- CMUdict coverage (sampled: full corpus is slow via per-word dict lookup+lru_cache,
    #     but lookup itself is O(1) once cached; use a subsample of talks for the word-token pass) ---
    sample = talks[:cmudict_sample_talks]
    all_words = []
    for t in sample:
        all_words.extend(w.lower() for w in WORD_RE.findall(t["content"]))
    in_dict = sum(1 for w in all_words if cmudict_syllable_count(w) is not None)
    report["cmudict_coverage_sample"] = {
        "n_talks_sampled": len(sample),
        "n_word_tokens": len(all_words),
        "coverage_word_tokens": in_dict / len(all_words) if all_words else None,
        "n_distinct_words": len(set(all_words)),
        "coverage_distinct_words": (
            sum(1 for w in set(all_words) if cmudict_syllable_count(w) is not None) / len(set(all_words))
            if all_words
            else None
        ),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/talks.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/wit3_corpus_stats.json"
    main(in_path, out_path)
