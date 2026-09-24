"""
syllabify.py -- hybrid English syllable tokenizer.

Design (rebuilt from project notes, language-cadence-2):
  - CMUdict gives the syllable COUNT for a word (number of vowel phones in
    a pronunciation). When a word has multiple CMUdict pronunciations with
    disagreeing counts, we take the majority-vote count across pronunciations
    (ties broken by the first-listed pronunciation's count).
  - Spelling rules place the k-1 cut points inside the word: a DP over
    candidate cut positions (every inter-letter boundary), scored by
    agreement with pyphen's hyphenation points, agreement with NLTK's
    Sonority Sequencing Principle (SSP) syllable splitter, a legal-English-
    onset bonus (the next syllable's leading consonant cluster must be a
    licensed English onset), and small bonuses for common suffix boundaries.
  - Words not in CMUdict (OOV, ~1-2% of tokens) get their syllable count k
    from a fallback: majority vote of {vowel-group heuristic, pyphen count,
    NLTK SSP count}, then the same DP cut-placement is used to place k-1 cuts.
  - Syllabic-consonant words (e.g. "didn't" -> did-n't, "enabling" ->
    en-a-bl-ing) are special-cased: contractions split the "'t"/"'s"/"'d"/
    "'ll"/"'re"/"'ve"/"'m" clitic off as its own syllable, and a word-final
    consonant+"le" (-ble, -cle, -dle, -fle, -gle, -kle, -ple, -tle, -zle)
    is treated as a syllable nucleus on its own (no vowel required).
  - Output token stream: word-syllable pieces (first syllable of each word
    marked with a leading "▁"), plus punctuation characters and single
    digit characters as standalone tokens. Everything is lowercased before
    syllabification.

Public API:
    tokenize(text: str) -> list[str]
    syllabify_word(word: str) -> list[str]   # pieces, no "▁" marker, no case change assumed already lowercase
"""

from __future__ import annotations

import re
import string
from collections import Counter
from functools import lru_cache

import pyphen
import nltk

# ---------------------------------------------------------------------------
# Setup: NLTK data (cmudict corpus, SSP syllable tokenizer needs nothing extra)
# ---------------------------------------------------------------------------

def _ensure_nltk_data():
    for pkg, path in [("cmudict", "corpora/cmudict")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

from nltk.corpus import cmudict as _nltk_cmudict  # noqa: E402
from nltk.tokenize import SyllableTokenizer  # noqa: E402

_CMUDICT = _nltk_cmudict.dict()  # word (lowercase) -> list[list[phone_str]]
_SSP = SyllableTokenizer()
_PYPHEN = pyphen.Pyphen(lang="en_US")

VOWELS = set("aeiouy")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")  # word incl. one internal apostrophe-suffix (don't, it's)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]|[^\sA-Za-z0-9]")

WORD_MARKER = "▁"  # "▁", marks the first syllable/piece of a new word

CLITICS = ("n't", "'s", "'d", "'ll", "'re", "'ve", "'m", "'t")
CONS_LE_SUFFIXES = ("ble", "cle", "dle", "fle", "gle", "kle", "ple", "tle", "zle")

SUFFIX_BONUS = (
    "tion", "sion", "ment", "ness", "ful", "less", "ing", "ed", "er", "est",
    "ly", "able", "ible", "al", "ive", "ous", "ity", "ize", "ise",
)

# Legal English syllable onsets (consonant clusters that may begin a syllable).
# Not exhaustive linguistics -- a practical allow-list for scoring cut points.
LEGAL_ONSETS = set(
    list("bcdfghjklmnpqrstvwxyz")
    + [
        "bl", "br", "ch", "cl", "cr", "dr", "dw", "fl", "fr", "gl", "gr",
        "gw", "kl", "kr", "kw", "ph", "pl", "pr", "qu", "sc", "sh", "sk",
        "sl", "sm", "sn", "sp", "st", "sw", "th", "tr", "tw", "wh", "wr",
        "scr", "shr", "spl", "spr", "squ", "str", "thr",
        "gh", "ng",  # loose extras for coverage
    ]
)


# ---------------------------------------------------------------------------
# CMUdict syllable count (with multi-pronunciation fix)
# ---------------------------------------------------------------------------

def _phones_syllable_count(phones: list[str]) -> int:
    return sum(1 for p in phones if p[-1].isdigit())


@lru_cache(maxsize=None)
def cmudict_syllable_count(word: str) -> int | None:
    """Majority-vote syllable count across all CMUdict pronunciations, or None if OOV."""
    prons = _CMUDICT.get(word.lower())
    if not prons:
        return None
    counts = [_phones_syllable_count(p) for p in prons]
    if len(set(counts)) == 1:
        return counts[0]
    vote = Counter(counts).most_common()
    top_count, top_n = vote[0]
    # tie -> first-listed pronunciation's count wins
    if sum(1 for _, n in vote if n == top_n) > 1:
        return counts[0]
    return top_count


# ---------------------------------------------------------------------------
# Fallback syllable-count estimators for OOV words
# ---------------------------------------------------------------------------

def _vowel_group_count(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    # silent trailing e: "-e" or "-le" (not after another vowel) drops a group,
    # unless that would take us to zero.
    if word.endswith("e") and not word.endswith("le") and n > 1:
        # crude silent-e correction: only if the e-group is exactly the final "e"
        if re.search(r"[^aeiouy]e$", word):
            n -= 1
    return max(n, 1)


def _pyphen_count(word: str) -> int:
    hyph = _PYPHEN.inserted(word)
    return hyph.count("-") + 1


def _ssp_count(word: str) -> int:
    try:
        return max(len(_SSP.tokenize(word)), 1)
    except Exception:
        return 1


@lru_cache(maxsize=None)
def fallback_syllable_count(word: str) -> int:
    counts = [_vowel_group_count(word), _pyphen_count(word), _ssp_count(word)]
    vote = Counter(counts).most_common()
    top_count, top_n = vote[0]
    if sum(1 for _, n in vote if n == top_n) > 1:
        # no majority: fall back to pyphen (empirically most reliable alone)
        return counts[1]
    return top_count


# ---------------------------------------------------------------------------
# Cut-point placement: DP over candidate split positions
# ---------------------------------------------------------------------------

def _pyphen_cut_positions(word: str) -> set[int]:
    hyph = _PYPHEN.inserted(word)
    positions = set()
    idx = 0
    for ch in hyph:
        if ch == "-":
            positions.add(idx)
        else:
            idx += 1
    return positions


def _ssp_cut_positions(word: str) -> set[int]:
    try:
        pieces = _SSP.tokenize(word)
    except Exception:
        return set()
    positions = set()
    idx = 0
    for piece in pieces[:-1]:
        idx += len(piece)
        positions.add(idx)
    return positions


def _onset_score(next_syll_start: str) -> float:
    """Score how legal the consonant cluster at the start of next_syll_start is as an onset."""
    m = re.match(r"[^aeiouy]+", next_syll_start)
    if not m:
        return 0.5  # starts with a vowel: neutral, always fine
    cluster = m.group(0)
    if cluster in LEGAL_ONSETS:
        return 1.0
    if len(cluster) == 1:
        return 0.8  # single consonant onsets are essentially always fine
    return -1.0  # illegal-looking cluster: penalize this cut point


def _suffix_bonus(word: str, pos: int) -> float:
    tail = word[pos:]
    for suf in SUFFIX_BONUS:
        if tail == suf or tail.startswith(suf):
            return 0.5
    return 0.0


def _score_cutset(word: str, cuts: tuple[int, ...], pyphen_pos: set[int], ssp_pos: set[int]) -> float:
    score = 0.0
    for c in cuts:
        if c in pyphen_pos:
            score += 1.0
        if c in ssp_pos:
            score += 1.0
        score += _onset_score(word[c:])
        score += _suffix_bonus(word, c)
    return score


def _candidate_positions(word: str) -> list[int]:
    # every inter-letter boundary except position 0 and len(word)
    return list(range(1, len(word)))


def _best_cutset(word: str, k: int) -> tuple[int, ...]:
    """Choose k-1 cut positions (1 <= pos <= len(word)-1, strictly increasing) maximizing score."""
    n = k - 1
    if n <= 0:
        return tuple()
    candidates = _candidate_positions(word)
    if n > len(candidates):
        # not enough room (very short word, oversized k): fall back to even spacing
        step = max(len(word) // k, 1)
        return tuple(sorted({min(step * i, len(word) - 1) for i in range(1, k)}))[:n]

    pyphen_pos = _pyphen_cut_positions(word)
    ssp_pos = _ssp_cut_positions(word)

    # Small n (almost always n <= 5 for real words): simple beam/DP by position index.
    # DP[i][j] = best score using first i candidates, having chosen j cuts, cuts must be
    # strictly increasing positions -- equivalent to "choose j positions out of candidates".
    # For tractability, use a greedy-with-backtrack DP over candidate list (len <= ~15).
    m = len(candidates)
    NEG = float("-inf")
    # dp[j][i] = best total score choosing j cuts from first i candidates, i-th candidate is the j-th cut (last used)
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    choice = [[-1] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for j in range(1, n + 1):
        for i in range(j, m + 1):
            pos = candidates[i - 1]
            single = _score_cutset(word, (pos,), pyphen_pos, ssp_pos)
            best = NEG
            best_prev = -1
            for ip in range(j - 1, i):
                if dp[j - 1][ip] > NEG:
                    val = dp[j - 1][ip] + single
                    if val > best:
                        best = val
                        best_prev = ip
            dp[j][i] = best
            choice[j][i] = best_prev

    # pick best end index for n cuts
    best_i, best_val = -1, NEG
    for i in range(n, m + 1):
        if dp[n][i] > best_val:
            best_val = dp[n][i]
            best_i = i
    cuts = []
    j, i = n, best_i
    while j > 0:
        cuts.append(candidates[i - 1])
        i = choice[j][i]
        j -= 1
    return tuple(sorted(cuts))


# ---------------------------------------------------------------------------
# Special-casing: clitics and syllabic consonant (-Cle) endings
# ---------------------------------------------------------------------------

def _split_clitic(word: str) -> tuple[str, str | None]:
    for cl in CLITICS:
        if word.endswith(cl) and len(word) > len(cl):
            return word[: -len(cl)], word[-len(cl):]
    return word, None


def syllabify_word(word: str) -> list[str]:
    """Split a single lowercase word (letters/apostrophes only) into syllable pieces."""
    if not word:
        return []
    if len(word) == 1:
        return [word]

    stem, clitic = _split_clitic(word)

    def _syllabify_stem(w: str) -> list[str]:
        if len(w) <= 1:
            return [w] if w else []
        k = cmudict_syllable_count(w)
        oov = k is None
        if oov:
            k = fallback_syllable_count(w)
        k = max(1, min(k, len(w)))  # can't have more syllables than letters

        # syllabic-consonant special case: word-final consonant+"le"
        le_pieces: list[str] = []
        remainder = w
        if w.endswith(CONS_LE_SUFFIXES) and k >= 2:
            le_start = len(w) - 3  # "-Cle" is 3 chars: consonant + l + e
            remainder = w[:le_start]
            le_pieces = [w[le_start:]]
            k -= 1
            if not remainder:
                return le_pieces
            if k <= 0:
                return [remainder] + le_pieces
            if k == 1:
                return [remainder] + le_pieces

        cuts = _best_cutset(remainder, k)
        pieces = []
        prev = 0
        for c in cuts:
            pieces.append(remainder[prev:c])
            prev = c
        pieces.append(remainder[prev:])
        return pieces + le_pieces

    pieces = _syllabify_stem(stem)
    if clitic:
        pieces = pieces + [clitic]
    return pieces


# ---------------------------------------------------------------------------
# Full text tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize raw text into syllable/punctuation/digit tokens (lowercased)."""
    text = text.lower()
    tokens: list[str] = []
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isalpha() or ("'" in tok and re.fullmatch(r"[a-z]+(?:'[a-z]+)*", tok)):
            pieces = syllabify_word(tok)
            for i, p in enumerate(pieces):
                tokens.append((WORD_MARKER if i == 0 else "") + p)
        elif tok.isdigit():
            for ch in tok:
                tokens.append(ch)
        else:
            tokens.append(tok)
    return tokens


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or "Understanding didn't stop them from enabling cooking, cleverer options."
    print(tokenize(sample))
