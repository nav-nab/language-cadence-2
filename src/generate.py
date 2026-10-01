"""
generate.py -- checkpoint loading, autoregressive sampling, and incremental
piece-to-word decoding, shared by scripts/generate_gui.py.

Split out from the GUI script (rather than living inside it) so this logic
can be unit-tested (tests/test_generate.py) without a tkinter/display
dependency, and reused by any other entry point later (a CLI version, say).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model import GPT
from syllable_tokenizer.vocab import Vocab, SPECIAL_TOKENS, CHAR_FALLBACK_PREFIX
from syllable_tokenizer.syllabify import WORD_MARKER


class IncrementalDecoder:
    """Turns a stream of token ids into a stream of whole words, mirroring
    Vocab.decode()'s piece-to-word logic but incrementally -- so a caller
    can display generated text as it's produced instead of re-decoding the
    whole sequence from scratch after every single token."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self._cur = ""

    def feed(self, token_id: int) -> str | None:
        """Consume one token id. Returns a completed word (with a trailing
        space) if this token started a new word and a previous word was
        sitting in the buffer, else None."""
        if not (0 <= token_id < len(self.vocab.id_to_piece)):
            return None
        piece = self.vocab.id_to_piece[token_id]
        if piece in SPECIAL_TOKENS:
            return None
        raw = piece.lstrip(CHAR_FALLBACK_PREFIX)
        starts_word = raw.startswith(WORD_MARKER) or piece.startswith(WORD_MARKER)
        raw = raw.lstrip(WORD_MARKER)
        out = None
        if starts_word:
            if self._cur:
                out = self._cur + " "
            self._cur = raw
        else:
            self._cur += raw
        return out

    def flush(self) -> str:
        """Call once generation stops, to emit whatever's left in the buffer."""
        out = self._cur
        self._cur = ""
        return out


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPT, Vocab, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    vocab = Vocab.load(cfg["vocab_path"])
    model = GPT(
        vocab_size=len(vocab),
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        seq_len=cfg["seq_len"],
        pe_mode=cfg["pe_mode"],
        period=cfg.get("period"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, cfg


@torch.no_grad()
def sample_next(model: GPT, context_ids: list[int], device: torch.device,
                 temperature: float, top_k: int) -> int:
    """Sample one next-token id given the (possibly long) context so far.
    Only the last model.seq_len tokens are actually fed to the model, since
    that's the longest context it has positional information for."""
    window = context_ids[-model.seq_len:]
    x = torch.tensor([window], dtype=torch.long, device=device)
    logits, _ = model(x)
    logits = logits[0, -1, :] / max(temperature, 1e-5)
    if top_k and top_k > 0:
        top_vals, top_idx = torch.topk(logits, min(top_k, logits.size(-1)))
        masked = torch.full_like(logits, float("-inf"))
        masked[top_idx] = top_vals
        logits = masked
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
