# Language Cadence 2

Small ML research project: does adding a periodic component to a standard
positional encoding make a small GPT train more efficiently (lower test
loss after a fixed number of training steps) on English text? Sequel to
`language-cadence-1` (finished, kept separate) — English-only, syllable
tokenizer instead of characters, question reframed around training
efficiency rather than cross-language "cadence" comparison.

Full design history lives in the "Language Cadence" Claude Project chat.
This repo is the execution: code, configs, metrics, results. Design is
final; implementation choices made while executing are logged in
[`DECISIONS.md`](DECISIONS.md).

## Data

Corpus: WIT3 / IWSLT TED talks, English source side, 2016 release (Hugging
Face repo `IWSLT/ted_talks_iwslt`, `data/XML_releases.tgz`). CC BY-NC-ND;
please cite Cettolo et al. (EAMT 2012) and credit TED if you reuse this.
The raw archive, extracted XML, and any processed/tokenized corpus are
**not** committed — rebuild with `scripts/download_and_prepare_data.py`
(coming in the data step).

## Tokenizer

True syllables, not BPE: CMUdict gives each word's syllable count, spelling
rules (scored against pyphen and NLTK's SSP syllable splitter) place the
cut points. See `syllabify.py` / `tokenizer.py` and `DECISIONS.md` for the
rebuild note. Run `pytest tests/` before trusting it. Reference stats on
Brown are in `eval_hybrid_report.json`; the deployed vocabulary is rebuilt
from WIT3 coverage, not the provisional Brown-based one.

## Status

See `DECISIONS.md` for the running log of what's settled vs. still open.
