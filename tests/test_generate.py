import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from generate import load_model, sample_next, IncrementalDecoder  # noqa: E402
from model import GPT  # noqa: E402
from syllable_tokenizer.vocab import Vocab, EOS  # noqa: E402

# Small sizes throughout -- this is testing generation logic, not training
# a real model, so there's no reason to pay the full D=128/seq_len=1024
# config's cost.
D = 16
LAYERS = 2
HEADS = 2
SEQ_LEN = 32

DEMO_CORPUS = [
    "Understanding didn't stop them from enabling cooking, cleverer options.",
    "The quick brown fox jumps over the lazy dog 42 times.",
    "Navid trained a tiny transformer on TED talks about mathematics.",
]


@pytest.fixture
def vocab() -> Vocab:
    return Vocab.fit(DEMO_CORPUS, vocab_size=120)


@pytest.fixture
def model(vocab) -> GPT:
    torch.manual_seed(0)
    m = GPT(len(vocab), D, LAYERS, HEADS, SEQ_LEN, pe_mode="rope")
    m.eval()
    return m


def test_incremental_decoder_matches_vocab_decode_for_real_text(vocab):
    ids = vocab.encode(DEMO_CORPUS[0])
    reference = vocab.decode(ids)

    decoder = IncrementalDecoder(vocab)
    words = []
    for tid in ids:
        w = decoder.feed(tid)
        if w:
            words.append(w)
    tail = decoder.flush()
    if tail:
        words.append(tail)

    assert "".join(words) == reference


def test_incremental_decoder_matches_vocab_decode_for_generated_ids(model, vocab):
    # Generated ids can include char-fallback pieces (OOV words), special
    # tokens, etc. -- exercise the decoder on actual sampled output, not
    # just a clean encoded sentence.
    device = torch.device("cpu")
    context = vocab.encode("The quick brown fox")
    torch.manual_seed(0)
    for _ in range(25):
        context.append(sample_next(model, context, device, temperature=1.0, top_k=0))

    reference = vocab.decode(context)

    decoder = IncrementalDecoder(vocab)
    words = []
    for tid in context:
        w = decoder.feed(tid)
        if w:
            words.append(w)
    tail = decoder.flush()
    if tail:
        words.append(tail)

    assert "".join(words).strip() == reference


def test_incremental_decoder_skips_special_tokens(vocab):
    decoder = IncrementalDecoder(vocab)
    eos_id = vocab.piece_to_id[EOS]
    # <eos> alone should never surface as visible text.
    assert decoder.feed(eos_id) is None
    assert decoder.flush() == ""


def test_sample_next_returns_a_valid_token_id(model, vocab):
    device = torch.device("cpu")
    context = [vocab.piece_to_id[EOS]]
    for _ in range(10):
        nid = sample_next(model, context, device, temperature=0.8, top_k=10)
        assert 0 <= nid < len(vocab)
        context.append(nid)


def test_sample_next_top_k_1_is_deterministic_argmax(model, vocab):
    device = torch.device("cpu")
    context = vocab.encode("Navid trained a tiny")

    with torch.no_grad():
        logits, _ = model(torch.tensor([context[-model.seq_len:]]))
    expected = logits[0, -1, :].argmax().item()

    # top_k=1 collapses the distribution to a single point mass, so every
    # draw (regardless of random seed / temperature) must land on it.
    for temperature in (0.3, 1.0, 1.7):
        got = sample_next(model, context, device, temperature=temperature, top_k=1)
        assert got == expected


def test_sample_next_only_uses_the_last_seq_len_tokens(model, vocab):
    device = torch.device("cpu")
    short_context = vocab.encode("The quick brown fox")
    # Pad a long, irrelevant prefix in front -- sample_next should produce
    # the exact same result either way, since it windows to model.seq_len.
    long_context = [vocab.piece_to_id[EOS]] * 500 + short_context

    torch.manual_seed(42)
    next_from_short = sample_next(model, short_context, device, temperature=1.0, top_k=5)
    torch.manual_seed(42)
    next_from_long = sample_next(model, long_context, device, temperature=1.0, top_k=5)

    assert next_from_short == next_from_long


def test_load_model_round_trips_a_checkpoint(tmp_path, vocab):
    vocab_path = tmp_path / "vocab.json"
    vocab.save(str(vocab_path))

    torch.manual_seed(0)
    original = GPT(len(vocab), D, LAYERS, HEADS, SEQ_LEN, pe_mode="rope")
    ckpt_path = tmp_path / "checkpoint_best.pt"
    torch.save(
        {
            "step": 100,
            "val_loss": 1.23,
            "model_state_dict": original.state_dict(),
            "config": {
                "pe_mode": "rope",
                "period": None,
                "d_model": D,
                "n_layers": LAYERS,
                "n_heads": HEADS,
                "seq_len": SEQ_LEN,
                "vocab_path": str(vocab_path),
            },
        },
        ckpt_path,
    )

    device = torch.device("cpu")
    loaded_model, loaded_vocab, cfg, info = load_model(str(ckpt_path), device)

    assert cfg["pe_mode"] == "rope"
    assert len(loaded_vocab) == len(vocab)
    assert not loaded_model.training  # eval() was called
    assert info["val_loss"] == 1.23
    assert info["step"] == 100

    context = vocab.encode("The quick brown fox")
    x = torch.tensor([context])
    with torch.no_grad():
        original.eval()
        expected_logits, _ = original(x)
        got_logits, _ = loaded_model(x)
    assert torch.allclose(expected_logits, got_logits)
