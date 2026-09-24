"""
tokenizer.py -- vocabulary + character-fallback layer over syllabify.tokenize.

A piece produced by syllabify.tokenize() is looked up in a fixed vocabulary.
If present, it becomes one token id. If absent (rare, since the vocabulary is
built to cover the large majority of the training-corpus piece distribution),
it falls back to a sequence of per-character tokens, each character prefixed
with a fallback marker so the model can tell "this is a raw character, not a
learned syllable piece" apart in the embedding table. The word-start marker
("▁") on a piece's first character is preserved through the fallback path
(it becomes part of the first character token) so word boundaries are never
lost, whether or not a piece hit the vocabulary.

Special tokens: <pad>, <unk>, <bos>, <eos>.

Usage:
    vocab = Vocab.fit(corpus_iter_of_strings, vocab_size=8000)
    vocab.save("configs/vocab.json")
    vocab = Vocab.load("configs/vocab.json")
    ids = vocab.encode(text)
    pieces = vocab.encode_pieces(text)   # token strings, before id lookup
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from syllabify import tokenize, WORD_MARKER

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
CHAR_FALLBACK_PREFIX = "\u0000"  # internal marker distinguishing a fallback char-token from a real 1-char piece


def _char_fallback_tokens(piece: str) -> list[str]:
    """Explode an OOV piece into per-character fallback tokens, preserving a leading word marker."""
    out = []
    for ch in piece:
        if ch == WORD_MARKER:
            continue
        out.append(CHAR_FALLBACK_PREFIX + ch)
    if piece.startswith(WORD_MARKER) and out:
        out[0] = WORD_MARKER + out[0]
    return out or [CHAR_FALLBACK_PREFIX + piece]


@dataclass
class Vocab:
    piece_to_id: dict = field(default_factory=dict)
    id_to_piece: list = field(default_factory=list)
    piece_freq: dict = field(default_factory=dict)  # informational, kept for coverage analysis

    @classmethod
    def fit(cls, texts, vocab_size: int, min_freq: int = 1) -> "Vocab":
        """Build a vocabulary of `vocab_size` entries (including special tokens) from raw texts."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        id_to_piece = list(SPECIAL_TOKENS)
        budget = vocab_size - len(SPECIAL_TOKENS)
        most_common = [p for p, c in counter.most_common() if c >= min_freq][:budget]
        id_to_piece.extend(most_common)

        piece_to_id = {p: i for i, p in enumerate(id_to_piece)}
        piece_freq = {p: counter.get(p, 0) for p in id_to_piece}
        return cls(piece_to_id=piece_to_id, id_to_piece=id_to_piece, piece_freq=piece_freq)

    def __len__(self) -> int:
        return len(self.id_to_piece)

    def encode_pieces(self, text: str) -> list[str]:
        """Tokenize text into vocabulary pieces, exploding OOV pieces into char-fallback tokens."""
        out = []
        for piece in tokenize(text):
            if piece in self.piece_to_id:
                out.append(piece)
            else:
                out.extend(_char_fallback_tokens(piece))
        return out

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.piece_to_id.get(p, self.piece_to_id[UNK]) for p in self.encode_pieces(text)]
        if add_bos:
            ids = [self.piece_to_id[BOS]] + ids
        if add_eos:
            ids = ids + [self.piece_to_id[EOS]]
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces = [self.id_to_piece[i] if 0 <= i < len(self.id_to_piece) else UNK for i in ids]
        words = []
        cur = ""
        for p in pieces:
            if p in SPECIAL_TOKENS:
                continue
            raw = p.lstrip(CHAR_FALLBACK_PREFIX)
            starts_word = raw.startswith(WORD_MARKER) or p.startswith(WORD_MARKER)
            raw = raw.lstrip(WORD_MARKER)
            if starts_word:
                if cur:
                    words.append(cur)
                cur = raw
            else:
                cur += raw
        if cur:
            words.append(cur)
        return " ".join(words)

    def coverage(self, texts) -> float:
        """Fraction of tokenized pieces (not char-fallback-exploded) present directly in the vocab."""
        total, hit = 0, 0
        for text in texts:
            for piece in tokenize(text):
                total += 1
                if piece in self.piece_to_id:
                    hit += 1
        return hit / total if total else 0.0

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"id_to_piece": self.id_to_piece, "piece_freq": self.piece_freq},
                f,
                ensure_ascii=False,
                indent=1,
            )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        id_to_piece = data["id_to_piece"]
        piece_to_id = {p: i for i, p in enumerate(id_to_piece)}
        return cls(piece_to_id=piece_to_id, id_to_piece=id_to_piece, piece_freq=data.get("piece_freq", {}))


if __name__ == "__main__":
    demo_corpus = [
        "Understanding didn't stop them from enabling cooking, cleverer options.",
        "The quick brown fox jumps over the lazy dog 42 times.",
    ]
    v = Vocab.fit(demo_corpus, vocab_size=50)
    print("vocab size:", len(v))
    ids = v.encode(demo_corpus[0])
    print("ids:", ids)
    print("decoded:", v.decode(ids))
