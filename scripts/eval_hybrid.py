"""
eval_hybrid.py -- evaluate the hybrid syllable tokenizer on the Brown corpus.

Reproduces (approximately -- this is a from-notes rebuild, not the original
code) the class of diagnostics described in project notes:
  - token stream stats: syllables/word, chars/syllable-token, distinct piece
    types, and cumulative coverage of the top-N most frequent pieces.
  - structural sanity checks: pieces always rejoin to the original word; every
    piece contains a vowel letter except an allow-listed syllabic-consonant
    residue (e.g. "n't", "bl", "-ble" endings).
  - fallback quality: syllable-count accuracy of the OOV fallback estimator,
    measured by hiding CMUdict for a sample of in-dictionary words and
    checking the fallback's count against the true (CMUdict-majority) count.
  - agreement: hybrid cut points vs. plain pyphen cut points, restricted to
    words where the two methods agree on syllable count (an apples-to-apples
    comparison of *placement* given the same count).

Writes results to results/eval_hybrid_report.json (repo-root-relative).
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk

for pkg, path in [("brown", "corpora/brown")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import brown  # noqa: E402

import syllable_tokenizer.syllabify as syl  # noqa: E402
from syllable_tokenizer.syllabify import (  # noqa: E402
    cmudict_syllable_count,
    fallback_syllable_count,
    syllabify_word,
    tokenize,
    _pyphen_count,
    WORD_MARKER,
)

VOWEL_RE = re.compile(r"[aeiouy]")
SYLLABIC_RESIDUE_OK = ("n't", "'s", "'d", "'ll", "'re", "'ve", "'m", "'t", "bl", "cl", "dl", "fl", "gl", "kl", "pl", "tl", "zl")


def word_tokens_from_brown(sample_size: int, seed: int = 0) -> list[str]:
    words = [w.lower() for w in brown.words() if re.fullmatch(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", w)]
    rng = random.Random(seed)
    if sample_size and sample_size < len(words):
        words = rng.sample(words, sample_size)
    return words


def eval_report(sample_size: int = 200_000, oov_probe_size: int = 3000, seed: int = 0) -> dict:
    report: dict = {}

    all_words = [w.lower() for w in brown.words() if re.fullmatch(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", w)]
    report["brown_word_token_count"] = len(all_words)
    report["brown_distinct_word_count"] = len(set(all_words))

    sample_words = word_tokens_from_brown(sample_size, seed=seed)

    # --- CMUdict coverage ---
    in_dict = sum(1 for w in sample_words if cmudict_syllable_count(w) is not None)
    report["cmudict_coverage_word_tokens"] = in_dict / len(sample_words)
    distinct_sample = list(set(sample_words))
    in_dict_distinct = sum(1 for w in distinct_sample if cmudict_syllable_count(w) is not None)
    report["cmudict_coverage_distinct_words"] = in_dict_distinct / len(distinct_sample)

    # --- structural sanity checks + syllable stats (over the sample) ---
    rejoin_ok = 0
    vowel_in_every_piece = 0
    total_syllables = 0
    total_chars_in_syllable_tokens = 0
    total_syllable_tokens = 0
    piece_counter: Counter = Counter()

    for w in sample_words:
        pieces = syllabify_word(w)
        if "".join(pieces) == w:
            rejoin_ok += 1
        if all(VOWEL_RE.search(p) or p in SYLLABIC_RESIDUE_OK for p in pieces):
            vowel_in_every_piece += 1
        total_syllables += len(pieces)
        for i, p in enumerate(pieces):
            tok = (WORD_MARKER if i == 0 else "") + p
            piece_counter[tok] += 1
            total_chars_in_syllable_tokens += len(p)
            total_syllable_tokens += 1

    n = len(sample_words)
    report["pieces_rejoin_word_pct"] = rejoin_ok / n
    report["vowel_in_every_piece_pct"] = vowel_in_every_piece / n
    report["syllables_per_word"] = total_syllables / n
    report["chars_per_syllable_token"] = total_chars_in_syllable_tokens / total_syllable_tokens

    # --- full token-stream stats over a chunk of running text (captures punctuation/digits too) ---
    text_sample = " ".join(brown.words()[:400_000])
    stream = tokenize(text_sample)
    stream_counter = Counter(stream)
    report["stream_token_count"] = len(stream)
    report["stream_distinct_token_types"] = len(stream_counter)

    ranked = [c for _, c in stream_counter.most_common()]
    total_tok = sum(ranked)
    cum = 0
    coverage_at = {}
    for k in (1000, 2000, 4000, 8000, 16000):
        cum = sum(ranked[:k])
        coverage_at[str(k)] = cum / total_tok
    report["coverage_by_vocab_size"] = coverage_at
    report["distinct_piece_types_in_word_sample"] = len(piece_counter)

    # --- fallback estimator accuracy (hide CMUdict on a probe of in-dict words) ---
    in_dict_words = [w for w in distinct_sample if cmudict_syllable_count(w) is not None]
    rng = random.Random(seed + 1)
    probe = rng.sample(in_dict_words, min(oov_probe_size, len(in_dict_words)))
    fallback_correct = 0
    pyphen_correct = 0
    for w in probe:
        true_k = cmudict_syllable_count(w)
        fb_k = fallback_syllable_count(w)
        py_k = _pyphen_count(w)
        if fb_k == true_k:
            fallback_correct += 1
        if py_k == true_k:
            pyphen_correct += 1
    report["fallback_count_accuracy_on_oov_probe"] = fallback_correct / len(probe)
    report["pyphen_alone_count_accuracy_on_oov_probe"] = pyphen_correct / len(probe)

    # --- hybrid cut-point agreement with pyphen, restricted to same-count words ---
    same_count, cuts_match = 0, 0
    for w in probe:
        true_k = cmudict_syllable_count(w)
        py_k = _pyphen_count(w)
        if py_k != true_k or true_k <= 1:
            continue
        same_count += 1
        hybrid_pieces = syllabify_word(w)
        py_hyph = syl._PYPHEN.inserted(w)
        py_pieces = py_hyph.split("-")
        if hybrid_pieces == py_pieces:
            cuts_match += 1
    report["hybrid_vs_pyphen_cut_agreement_when_counts_match"] = (cuts_match / same_count) if same_count else None
    report["hybrid_vs_pyphen_agreement_sample_n"] = same_count

    report["sample_size"] = n
    report["oov_probe_size"] = len(probe)
    report["seed"] = seed
    return report


if __name__ == "__main__":
    rep = eval_report()
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "eval_hybrid_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
