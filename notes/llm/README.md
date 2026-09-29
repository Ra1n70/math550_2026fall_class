# LLM lecture notes

The 26-page `lecture_notes.pdf` is a classroom companion to Sebastian Raschka's
*Build a Large Language Model (From Scratch)* (Manning, 2024). The revised
pretraining sequence follows chapters 2–5, while explicitly distinguishing the
course implementation from the book and from released GPT-2.

## Reading route

- Sections 2–4: tokenization and shifted windows, attention, and the GPT decoder
  (book chapters 2–4).
- Sections 5–7: next-token likelihood, cross-entropy and perplexity, evaluation,
  the first AdamW loop, sampling, and checkpoints (book chapter 5).
- Sections 8–10: course extensions for Gutenberg books, custom BPE, hardware,
  scheduling, and tokenizer/checkpoint contracts. Warmup, cosine decay, and
  clipping are associated with the book's Appendix D, not its first loop.
- Part II: retained instruction tuning, reward modeling, and PPO-style RLHF,
  followed by evidence, diagnostics, lab commands, and source records.

The notes contain original explanations and diagrams, not reproduced book
pages. Book references are pinned to the same repository revision as the
course's GPT-2 slides:
[`f49cad747931fe6af0a7d2802cd480d1c1a0ec8e`](https://github.com/rasbt/LLMs-from-scratch/tree/f49cad747931fe6af0a7d2802cd480d1c1a0ec8e).
The exact reading-data source for *The Verdict* is
[this pinned plain-text file](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/f49cad747931fe6af0a7d2802cd480d1c1a0ec8e/ch02/01_main-chapter-code/the-verdict.txt).

## Editable sources and build

- `lecture_notes.tex`: main document, book-to-course map, attention figures,
  architecture, course extensions, and references.
- `pretraining.tex`: the detailed chapter-5 teaching sequence and four Python
  examples using the local `GPTOutput.logits` interface.
- `figures/gpt_block.tex` and `figures/pretraining_flow.tex`: native TikZ diagrams.
- Existing figures in `output/llm/figures/` are referenced, not regenerated.

From the repository root, with the project's LaTeX dependencies installed:

```bash
make notes-llm
```

The PDF has a linked contents page, selectable equations, embedded fonts, and
code examples kept together on the page. The revision changes teaching
materials only; it does not change the model or training drivers.

## Numerical checks and evidence boundaries

With the project's `llm` Python dependencies already installed:

```bash
uv run --no-sync --extra llm python notes/llm/validation/check_pretraining_examples.py
```

The checker extracts and executes the four Python listings directly from
`pretraining.tex`. It tests the loss, token-weighted evaluation, dropout and
gradient modes, exception handling, sampling bounds, causality, and learning
on a repeated synthetic batch. `validation/results.json` records the source
hash, versions, seeds, settings, and numerical results. Use `--output PATH` to
save a new record elsewhere without replacing the checked-in one.

These are small CPU correctness checks, not language-quality or GPU benchmarks.
The public-book plots and reported 30-update results are explicitly labeled as
archived measurements. No new corpus training was performed for this revision.
The notes also document current seed-initialization and exact-resume limitations
in the course driver; documenting them does not claim they have been repaired.
