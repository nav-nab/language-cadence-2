import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from positional_encoding import sinusoidal_pe, periodic_pe, combined_pe  # noqa: E402
from model import GPT, count_params  # noqa: E402

VOCAB = 8000
D = 128
LAYERS = 4
HEADS = 4
SEQ_LEN = 1024


def test_sinusoidal_pe_shape_and_position_zero():
    pe = sinusoidal_pe(16, 8)
    assert pe.shape == (16, 8)
    # at position 0: sin(0)=0 on even cols, cos(0)=1 on odd cols
    assert torch.allclose(pe[0, 0::2], torch.zeros(4))
    assert torch.allclose(pe[0, 1::2], torch.ones(4))


def test_periodic_pe_is_actually_periodic_in_position():
    P = 5
    pe = periodic_pe(3 * P + 1, 8, P)
    # pe(n) should equal pe(n + P) since k_j are integers and the angle
    # advances by an integer multiple of 2*pi every P steps
    assert torch.allclose(pe[0], pe[P], atol=1e-5)
    assert torch.allclose(pe[3], pe[3 + P], atol=1e-5)
    assert torch.allclose(pe[3], pe[3 + 2 * P], atol=1e-5)


def test_periodic_pe_folded_harmonic_matches_formula():
    # D=4 -> H=2 -> j=1,2. P=3: k_1=min(1%3,3-1%3)=min(1,2)=1; k_2=min(2%3,3-2%3)=min(2,1)=1
    P = 3
    pe = periodic_pe(4, 4, P)
    n = torch.arange(4, dtype=torch.float32)
    import math
    expected_col0 = torch.sin(n * 2 * math.pi * 1 / P)  # j=1, k=1
    expected_col2 = torch.sin(n * 2 * math.pi * 1 / P)  # j=2, k=1
    assert torch.allclose(pe[:, 0], expected_col0, atol=1e-5)
    assert torch.allclose(pe[:, 2], expected_col2, atol=1e-5)


def test_combined_pe_baseline_equals_sinusoidal_only():
    pe_base = sinusoidal_pe(20, 16)
    pe_combined_none = combined_pe(20, 16, None)
    assert torch.allclose(pe_base, pe_combined_none)


def test_combined_pe_with_period_differs_from_baseline():
    pe_baseline = combined_pe(20, 16, None)
    pe_periodic = combined_pe(20, 16, 7)
    assert not torch.allclose(pe_baseline, pe_periodic)


@pytest.mark.parametrize("pe_mode,period", [("periodic", 45), ("baseline", None), ("rope", None)])
def test_gpt_forward_backward_all_pe_modes(pe_mode, period):
    torch.manual_seed(0)
    model = GPT(VOCAB, D, LAYERS, HEADS, SEQ_LEN, pe_mode=pe_mode, period=period)
    B, T = 2, 32  # small T for a fast test, well under SEQ_LEN
    idx = torch.randint(0, VOCAB, (B, T))
    targets = torch.randint(0, VOCAB, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, VOCAB)
    assert loss.item() > 0 and torch.isfinite(loss)
    loss.backward()
    # gradients should flow into the embedding and at least one block
    assert model.tok_emb.weight.grad is not None
    assert any(p.grad is not None for p in model.blocks[0].parameters())


def test_gpt_periodic_requires_period():
    with pytest.raises(AssertionError):
        GPT(VOCAB, D, LAYERS, HEADS, SEQ_LEN, pe_mode="periodic", period=None)


def test_frozen_architecture_param_count_in_expected_ballpark():
    model = GPT(VOCAB, D, LAYERS, HEADS, SEQ_LEN, pe_mode="periodic", period=45)
    n = count_params(model)
    # expected ~2.87M from the Stage 4 estimate; allow a reasonable band
    assert 2.5e6 < n < 3.3e6, f"param count {n} outside expected band"


def test_rope_and_periodic_have_different_param_counts():
    # rope has no concat_proj layer, periodic/baseline do
    m_rope = GPT(VOCAB, D, LAYERS, HEADS, SEQ_LEN, pe_mode="rope")
    m_periodic = GPT(VOCAB, D, LAYERS, HEADS, SEQ_LEN, pe_mode="periodic", period=45)
    assert count_params(m_rope) < count_params(m_periodic)
