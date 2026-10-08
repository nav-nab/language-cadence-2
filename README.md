# Language Cadence 2

<img src="assets/avatar-2.svg" alt="Tempo, the Language Cadence 2 mascot" align="right" width="110" height="103">

**Does giving a transformer an explicit periodic "clock" help it learn language?**
Short answer: no, but different periods produce statistically distinguishable
results. Along the way we also built a custom syllable tokenizer that
stands on its own.

📄 **[Read the full write-up →](writeup/paper.pdf)**

## TL;DR

We trained small GPTs (2.87M params) from scratch on English TED-talk
transcripts, comparing a standard positional encoding against one with an
added periodic component tuned to 8 different periods, plus a RoPE control.
30 runs, 3 seeds each, zero divergence.

- **Every single periodic configuration underperformed the plain baseline.**
  Consistently, in the same direction, across all 8 periods.
- Yet no individual period's disadvantage survives correction for multiple
  comparisons — the effect is a group signal, not a property of any one
  period (omnibus tests reject "no effect" decisively: ANOVA *p* < 10⁻⁵,
  Friedman *p* = 0.0082).
- RoPE beats both by roughly 10× the margin of any periodic-vs-baseline gap
  — a sanity check that the setup can detect a real effect when one exists.
- The most plausible explanation: a rigid, exact-period assumption fighting
  soft, irregular data. Real clause and sentence lengths don't repeat on a
  fixed schedule, so the periodic signal doesn't just add noise — it gives
  the model a specific, confident, *wrong* answer to "where am I in the
  cycle," which the model then has to spend effort unlearning.

## A tokenizer that turned out to matter too

Getting periods to mean anything linguistically required ditching
byte-pair tokenization for real syllables. The hybrid tokenizer built for
this project — CMUdict counts, a spelling-rule cut-point algorithm scored
against two independent syllabifiers, and a voted fallback for unseen
words — hits **98.4%** syllable-count accuracy on out-of-vocabulary words
(vs. 79.3% for the best single off-the-shelf tool it was checked against)
and rejoins every input word exactly. It's a standalone component
(`syllable_tokenizer/`) and we think it's useful beyond this project to
anyone who needs a well-behaved, linguistically grounded syllable-level
unit for English text.

## Try the trained model yourself

`scripts/generate_gui.py` is a small desktop app that loads the best RoPE
checkpoint from the sweep and generates text indefinitely from an optional
start string, streaming it into a window until you hit Stop:

```
python3 scripts/generate_gui.py
```

Just a toy for poking at the model, not a research artifact — trained
checkpoints aren't committed to the repo (see `.gitignore`), so this only
works after you've run the sweep yourself (`scripts/run_sweep.py`).

## Repo layout

| Path | What's there |
|---|---|
| [`writeup/paper.pdf`](writeup/paper.pdf) | The full paper — methodology, stats, discussion, limitations |
| `syllable_tokenizer/` | The hybrid syllable tokenizer (library code + tests) |
| `src/` | Model, positional-encoding, and generation (sampling/decoding) implementation |
| `scripts/` | Data pipeline, tokenizer evaluation, sweep orchestration, and the generator GUI |
| `results/` | Metrics, coverage curves, corpus stats, sweep outputs |
| `configs/` | Frozen vocabulary and run configs |
| [`DECISIONS.md`](DECISIONS.md) | Running log of every implementation decision made while executing |

## Data

English source side of WIT3/IWSLT TED talks (2016 release) — 2,085 talks,
~4.3M words, split by speaker with zero overlap across train/val/test.
CC BY-NC-ND; please cite Cettolo et al. (EAMT 2012) and credit TED if you
reuse it. The raw archive and processed corpus aren't committed — see
[`DECISIONS.md`](DECISIONS.md) for the rebuild pipeline.

## Context

Sequel to `language-cadence-1` (a prior null result on a character-level
tokenizer, which this project traces to a specific boundary effect and
fixes). Design history lives in the associated Claude Project chat; this
repo is the execution — code, configs, metrics, and the final write-up.
