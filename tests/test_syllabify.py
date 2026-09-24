import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from syllabify import (
    tokenize,
    syllabify_word,
    cmudict_syllable_count,
    fallback_syllable_count,
    WORD_MARKER,
)
from tokenizer import Vocab, UNK, PAD, BOS, EOS


WORDS_WITH_KNOWN_COUNTS = [
    ("cat", 1),
    ("dog", 1),
    ("table", 2),
    ("banana", 3),
    ("understanding", 4),
    ("communication", 5),
    ("a", 1),
    ("i", 1),
]


@pytest.mark.parametrize("word,expected_k", WORDS_WITH_KNOWN_COUNTS)
def test_cmudict_count_matches_expected(word, expected_k):
    k = cmudict_syllable_count(word)
    assert k == expected_k, f"{word}: got {k}, expected {expected_k}"


@pytest.mark.parametrize("word,_", WORDS_WITH_KNOWN_COUNTS)
def test_pieces_rejoin_word(word, _):
    pieces = syllabify_word(word)
    assert "".join(pieces) == word


@pytest.mark.parametrize("word,expected_k", WORDS_WITH_KNOWN_COUNTS)
def test_piece_count_matches_cmudict_count(word, expected_k):
    pieces = syllabify_word(word)
    assert len(pieces) == expected_k


def test_clitic_split():
    pieces = syllabify_word("didn't")
    assert pieces[-1] == "n't"
    assert "".join(pieces) == "didn't"


def test_oov_word_gets_fallback_count():
    # a word almost certainly absent from CMUdict
    made_up = "flibbertigibbetizing"
    assert cmudict_syllable_count(made_up) is None
    k = fallback_syllable_count(made_up)
    assert k >= 1


def test_tokenize_marks_word_starts():
    toks = tokenize("cat dog")
    assert toks[0].startswith(WORD_MARKER)
    # second word's first piece should also carry the marker
    second_word_start = [i for i, t in enumerate(toks) if t.startswith(WORD_MARKER)]
    assert len(second_word_start) == 2


def test_tokenize_splits_digits_and_punctuation():
    toks = tokenize("I have 42 cats.")
    assert "4" in toks and "2" in toks
    assert "." in toks
    assert "42" not in toks


def test_tokenize_lowercases():
    toks = tokenize("HELLO World")
    assert all(t == t.lower() for t in toks)


def test_tokenize_is_deterministic():
    text = "The quick brown fox jumps over the lazy dog, again and again."
    assert tokenize(text) == tokenize(text)


def test_vocab_fit_and_encode_roundtrip_ids_are_valid():
    corpus = [
        "Understanding didn't stop them from enabling cooking.",
        "The quick brown fox jumps over the lazy dog 42 times.",
    ]
    vocab = Vocab.fit(corpus, vocab_size=200)
    assert PAD in vocab.piece_to_id and UNK in vocab.piece_to_id
    for text in corpus:
        ids = vocab.encode(text)
        assert all(0 <= i < len(vocab) for i in ids)


def test_vocab_oov_falls_back_to_characters():
    corpus = ["a b c"]
    vocab = Vocab.fit(corpus, vocab_size=10)  # tiny vocab, most things will be OOV
    ids = vocab.encode("supercalifragilisticexpialidocious")
    assert len(ids) > 1  # exploded into fallback char tokens, not a single <unk>


def test_vocab_save_load_roundtrip(tmp_path):
    corpus = ["cats and dogs run fast"]
    vocab = Vocab.fit(corpus, vocab_size=100)
    path = tmp_path / "vocab.json"
    vocab.save(str(path))
    loaded = Vocab.load(str(path))
    assert loaded.id_to_piece == vocab.id_to_piece
    assert loaded.encode("cats and dogs") == vocab.encode("cats and dogs")


def test_single_letter_words():
    assert syllabify_word("a") == ["a"]
    assert syllabify_word("i") == ["i"]


def test_every_word_has_at_least_one_piece():
    for w in ["strengths", "rhythm", "queue", "though"]:
        pieces = syllabify_word(w)
        assert len(pieces) >= 1
        assert "".join(pieces) == w
