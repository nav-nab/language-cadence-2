# Prompt history: write-up vs. codebase consistency check

- **Date range:** 2026-10-01 to 2026-10-02
- **Session type:** Cowork execution session (Claude, desktop app, linked to the user's computer, attached to the "Language Cadence" Project). It audited, edited and committed files in the language-cadence-2 repo.
- **Session link/ID:** https://claude.ai/code/session_01FGyL5T9xXvLMeiJQ41r4Mj
- **Model configured for the session:** claude-sonnet-5-5 (the serving model may differ)
- **Record status:** The user prompts are verbatim. The session was never auto-compacted or summarized, so nothing here is reconstructed from a summary. Claude's replies are condensed or lightly paraphrased where the original was long. Raw tool output (file dumps, JSON, test output) is omitted and summarized in brackets. System reminders and injected context are omitted. Two short progress messages sent mid-task are included.

---

## Turn 1

### User

Look inside the following path:

[local path to the language-cadence-2 repo on Navid's computer]

Here you will find an ML research project. There is a codebase and an accompany write-up.

Your task is to check that the information contained in the write-up is consistent with the codebase, i.e. that the write-up does not contain hallucinations. Ask me before making any changes.

### Claude

[Read the repo layout, `writeup/paper.tex`, `src/model.py`, `src/positional_encoding.py`, `syllable_tokenizer/syllabify.py`, `scripts/train.py`, `results/*.json`, and every `runs/*/metrics.jsonl` and `final_result.json`. Recomputed the training-curve gaps. Viewed the training-curve figure.]

Mid-task progress messages sent to the user:

> Still checking the write-up against the code and results files. I've read the paper, model, tokenizer and training code, and the results JSONs. Next I'm checking the remaining claims: training-curve numbers, best-step selection, corpus stats and the companion-paper details. I won't change anything until you've seen the findings.

> Nearly done. I've verified the results numbers, stats tables and training-curve figures against the data, and found a few discrepancies. I'm now running the tokenizer on a couple of the paper's examples (e.g. "enabling"), then I'll send you the full list before changing anything.

Final findings message (condensed):

I checked the write-up against the code, `results/*.json`, `runs/*/metrics.jsonl` and the figures. Nothing was changed. The numbers mostly match, but I found three errors, several claims the code contradicts or that are stated too strongly, and a few smaller points.

**Verified as consistent**
- The Brown tokenizer table matches `eval_hybrid_report.json`, including the 3.7-point fallback-vs-pyphen gap.
- Vocabulary figures match: 6,398,658 training pieces, 19,286 distinct pieces, 99.5% / 99.2% coverage at 8,000.
- Corpus numbers match: 14,631 cues, 221 of 1,767 speakers, and the three splits.
- Table 1 matches `sweep_summary.json`, and Table 2 matches `stats_sweep.json`.
- The Friedman, ANOVA and best-vs-worst figures (p = 0.0155, Holm 0.34) are correct.
- The PE formulas match `positional_encoding.py`. Concat injection, pre-norm, the fixed learning rate of 3e-4 and checkpoint selection match the code.
- The U-shape numbers match (0.055 → 0.006 at step 1500 → 0.030 by step 2900; spread 0.047 → 0.008 → 0.037).
- The parameter count is about 2.87M (computed by hand).
- All runs completed 3,000 steps, and there are no `warnings.log` or `divergence.json` files.

**Errors**
1. Epoch position of the gap minimum. The paper says step 1,500 is "roughly 2–3 of the 7–8 epochs". The code gives 397 steps per epoch, so step 1,500 is about epoch 3.8.
2. "The four smallest periods" lists five values (2, 3, 6, 9, 16).
3. The `enabling` example. The paper says the syllabic-*le* special case turns *enabling* into *en-a-bl-ing*. That rule only fires on words ending in consonant + *le*, and *enabling* ends in *-ing*. If that output is produced, the cut-point DP produces it. This was checked by reading the code only, because CMUdict was not available in the sandbox. No test covers it.

**Contradicted by the code, or overstated**
4. `best_step` is 3000 in all 30 runs, so the best-validation checkpoint was always the final one. The checkpoint rule never changed any result, and the validation curves are still falling at step 3000. This weakens the discussion's "late-epoch overfitting" and "model starts relying on the prior" stories, and the matching future-work item.
5. "Identical parameter initialization across all 10 configurations": the seed is the same, but RoPE has no `concat_proj` layer, so its random draws for the blocks differ. Shuffle order is identical.
6. "Concatenation for every configuration in the sweep": RoPE does not use concat, and has about 2.84M parameters, not 2.87M.
7. "All 30 runs completed without incident": DECISIONS.md says two runs were interrupted and resumed.
8. RoPE gap "roughly an order of magnitude larger": the RoPE gap is 0.33 and the periodic gaps are 0.016–0.055, so 6× to 20×.
9. The code breaks three-way fallback ties in favour of pyphen. The paper calls it a plain majority vote.
10. Onset scoring is slightly more than a bonus: illegal clusters are penalised (−1) and single consonants score 0.8.

**Not verifiable from the repo**
- Companion-paper claims: H=32, "P=128 is its distinguishability optimum", "Section 2.1". The user's notes call P=128 "2H" for v1, which would imply H=64 there. This needs checking.
- The 2026 references (Liao & Shi, Nguyen et al. arXiv 2609.21362), the Atuhurra arXiv number and the Ghosh details were not checked.
- The training stream is 6.52M tokens, versus the 6.40M pieces cited. The extra tokens come from the character-level fallback for unknown pieces and the end-of-talk markers. This is not an error, but the paper does not say so.

Claude asked whether to apply fixes 1–3 and 5–10 directly in `paper.tex`, and said that for item 4 it needed the user's choice on rewording the overfitting discussion versus adding a note only.

---

## Turn 2

### User

Make the changes

### Claude

[Backed up `paper.tex` to `paper.tex.bak`. Applied 13 exact-string replacements in `writeup/paper.tex`, each asserted to match exactly once. Compiled a scratch copy with pdflatex.]

I made the edits to `writeup/paper.tex`, and it compiles cleanly (17 pages, tested in a scratch copy). I did not regenerate `paper.pdf`. The original is saved as `writeup/paper.tex.bak`.

- **Epoch count:** the U-shape minimum at step 1,500 now reads "roughly 3.8 of the run's 7–8 epochs (397 steps per epoch)".
- **"Four smallest periods":** now "five".
- **`enabling` example:** the special-case sentence now uses *table* (ta-ble) as the example. A parenthetical says words ending in *-ing* are not affected and are split by the general scoring.
- **Best-validation checkpoint:** I added a sentence saying the best step was 3,000 in all 30 runs, so no overfitting was observed. In the training-dynamics section I added that validation loss was still falling at the end of every run. In the discussion I added that the "misleading prior" story does not rest on observed overfitting. In the follow-up wording (training dynamics and conclusion), "overfitting" is now "train/validation divergence".
- **Initialization:** seeds and shuffle order are shared across all 10 configurations. Initial parameters are identical for baseline and periodic. RoPE is matched only in the seed.
- **Concat injection:** stated for the baseline and periodic configurations only. The RoPE control does not use it.
- **Parameter count:** 2.87M, with 2.84M for RoPE.
- **"Without incident":** now "completed their full budget", plus a note that two runs were interrupted and resumed.
- **RoPE gap:** now "between 6 and 20 times larger".
- **Fallback tie-break:** added the tie-break to pyphen.
- **Onset scoring:** reworded to say it is a score, with illegal clusters penalised.

Still open: the companion-paper claims (H=32 and the "P=128 optimum") were not touched, and the 2026 references were not checked.

---

## Turn 3

### User

Commit these to Git

### Claude

[Ran `git status`. Found `paper.tex` modified, plus two untracked files: `pyphen-0.18.1-py3-none-any.whl` and `writeup/paper.tex.bak`. The wheel was created by a stray `pip download` that Claude had run inside the repo folder by mistake. Committed only `writeup/paper.tex`. Git printed warnings about stale lock and temp files in `.git` that could not be deleted because deletion was blocked.]

The changes to `writeup/paper.tex` are committed as `b4d05fe`. Git printed warnings about stale lock and temp files in `.git` that it could not delete. The commit itself went through.

Two files are left untracked and uncommitted:
- `pyphen-0.18.1-py3-none-any.whl`: downloaded by mistake while testing. It could not be deleted because deletion is blocked here.
- `writeup/paper.tex.bak`: the backup of the original paper.

Claude offered to ask for delete permission to remove both, and noted that `.git` might still hold a leftover `index.lock` or `HEAD.lock` that the user would need to delete if git complained.

---

## Turn 4

### User

Delete both. Then compile the tex into pdf and then commit again.

### Claude

[Requested and received delete permission for the repo folder. Deleted the wheel file, `paper.tex.bak`, `.git/index.lock`, `.git/HEAD.lock` and the stale `tmp_obj_*` files. Ran pdflatex twice in `writeup/`. Removed the build files (`.aux`, `.log`, `.out`, `.toc`). Committed `writeup/paper.pdf`.]

I deleted the stray `pyphen` wheel file, `paper.tex.bak` and the stale git lock files. I then recompiled `paper.tex` into `paper.pdf` (17 pages, no errors or undefined references). That is committed as `20835eb`, and the working tree is clean.

The LaTeX build files (`.aux`, `.log`, `.out`, `.toc`) were removed after compiling.

Commits made in this session:
- `b4d05fe`: Fix write-up inconsistencies with codebase and results
- `20835eb`: Rebuild paper.pdf from corrected paper.tex

---

## Turn 5 (archiving request)

### User

This conversation is being wound down as part of archiving the project's
full prompt history. Before finishing, please do the following:

1. Write out this conversation's history as a clean, readable Markdown
file: my prompts verbatim (don't paraphrase or summarize them), and
your responses included too (these can be lightly condensed if a reply
was dominated by raw tool output or long file dumps, but keep the
substance — don't drop actual explanations, decisions, or reasoning).
2. Start the file with a short header noting:
   * the approximate date range of this conversation
   * what kind of agent/session this is (e.g. "claude.ai Project design
chat", "Cowork/Claude Code execution session")
   * this session's own link/ID if you have one (e.g. a Claude-Session
URL), so it can be traced back later
   * whether this record is a complete verbatim transcript or, for any
part of it, a reconstruction from a prior summary (this matters: if
this conversation was ever auto-compacted/summarized earlier in its
own history, say so explicitly and note which parts are summary
rather than verbatim)
3. Save that file into the language-cadence-2 repo, in a folder called
prompt_history/ at the repo root (create it if it doesn't exist), named
like:
YYYY-MM-DD_short-label.md
using the start date of this conversation and a short label of your
choosing for what it covers (e.g. 2026-09-18_design-chat.md).
   * If you have file/repo access (direct file tools, or a connected
device/shell), write the file there directly.
   * If you do NOT have file or repo access, instead output the complete
Markdown file content directly in your reply, clearly marked as
such, so I can save it into that folder myself.
4. Do NOT run `git add`/`git commit` — just write the file (or give me
the content). I'll handle committing once all sessions have exported.

Once you've done this, confirm what you wrote (or output) and where.

### Claude

The connection to the user's computer had dropped, so the file could not be written to the repo. Claude output this file's full content in its reply instead, for the user to save as `prompt_history/2026-10-01_writeup-consistency-check.md`. No `git add` or `git commit` was run.