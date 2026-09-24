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
[`DECISIONS.md`](DECISIONS.md), which is the authoritative record of what's
settled vs. still open.

## Data

Corpus: WIT3 / IWSLT TED talks, English source side, 2016 release (Hugging
Face repo `IWSLT/ted_talks_iwslt`, `data/XML_releases.tgz`, 1.67GB). CC
BY-NC-ND; please cite Cettolo et al. (EAMT 2012) and credit TED if you
reuse this. The raw archive, extracted XML, and processed/tokenized corpus
are **not** committed. Rebuild pipeline:

```
python3 scripts/parse_wit3.py data/extracted/ted_en-20160408.xml data/processed/talks.jsonl
python3 scripts/preprocess_wit3.py data/processed/talks.jsonl data/processed
```

(the raw archive and extracted XML must be downloaded/unzipped from the
Hugging Face path above first). `preprocess_wit3.py` strips audience/audio
cues like `(Laughter)`/`(Applause)`, drops talks under 100 words, and
splits by speaker into `data/processed/{train,val,test}.jsonl` at roughly
90/5/5 by word count with zero speaker overlap across splits. Corpus
inspection and preprocessing reports: `results/wit3_corpus_stats.json`,
`results/wit3_preprocess_report.json`.

## Tokenizer

True syllables, not BPE: CMUdict gives each word's syllable count, spelling
rules (scored against pyphen and NLTK's SSP syllable splitter) place the
cut points. Library code lives in `syllable_tokenizer/` (`syllabify.py`,
`vocab.py`); see `DECISIONS.md` for the rebuild note. Run `pytest tests/`
before trusting it, and `python3 scripts/eval_hybrid.py` to regenerate
`results/eval_hybrid_report.json` (reference stats on Brown).

The deployed vocabulary (`configs/vocab.json`, 8000 pieces) is rebuilt from
real WIT3 coverage, not the provisional Brown-based one:
`scripts/tokenize_corpus_chunk.py` + `scripts/finalize_vocab.py` (chunked
because tokenizing the ~3.9M-word train split in one shot risks the local
shell's time limit). Coverage curve and the frozen choice are in
`results/vocab_coverage.json`.

## Compute

`scripts/benchmark_throughput.py` measures laptop throughput for the
likely sweep architecture. Result and 48h-budget feasibility check are in
`DECISIONS.md`.

## Status

Stages 1-3 (setup, data, vocabulary) are done and frozen. Stage 4 (model +
positional encoding) is next. See `DECISIONS.md` for the full running log,
every frozen decision, and what's still open.
