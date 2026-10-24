# Data for the four-problem LLM homework

Problem 3 uses one novel for next-token pretraining; Problem 4 uses a fixed
instruction dataset for a paired loss-masking experiment. Both use the supplied
`ByteTokenizer` unchanged: 256 UTF-8 byte IDs and five control-token IDs
(`V=261`). No tokenizer fitting, pretrained weights, or additional book corpus
is needed. Run commands below from the repository root after installing the
project's `llm` extra.

## 1. Pretraining: *The Time Machine*

H. G. Wells's *The Time Machine* is
[Project Gutenberg eBook 35](https://www.gutenberg.org/ebooks/35), an English
novel listed as public domain in the USA. Follow the item's embedded terms,
[Project Gutenberg's terms of use](https://www.gutenberg.org/policy/terms_of_use.html),
and its [permission guidance](https://www.gutenberg.org/policy/permission);
users outside the USA must check their local copyright rules.

```bash
uv run --no-sync --extra llm python -m experiments.llm.download_data --dataset homework
```

This command reuses a nonempty `data/llm/raw/homework/pg35.txt` if present.
It does not enlarge or replace the pretraining books corpus. On a new download,
the current downloader uses the registered Project Gutenberg mirror, beginning
with `https://gutenberg.pglaf.org/cache/epub/35/pg35.txt`, and records the actual
requested/resolved URL and hashes in `data/llm/download_manifest.json`.

The checked-in manifest's previously downloaded snapshot resolved from
`https://www.gutenberg.org/ebooks/35.txt.utf-8` to
`https://www.gutenberg.org/cache/epub/35/pg35.txt`. Its hashes are:

| Payload | SHA-256 |
| --- | --- |
| Original download | `2892e919000e17c83e1dac51b30f4675db50536b644d7579fe8a89bb399a9bdc` |
| Cleaned UTF-8 text | `4e94ac81aaa7ac2680590d63b480d7720cd9a509689a6617d16c8c99ec00f2bd` |

Cleaning removes the distribution header/footer, normalizes line endings,
trims line-end whitespace, and reduces repeated blank lines. The verified local
cleaned snapshot has 179,630 Python characters (181,406 UTF-8 bytes). These are
observed snapshot hashes, not promises that a future upstream revision will be
identical. Keep the local snapshot fixed throughout the assignment.

Split the cleaned text at `int(0.9 * len(text))` **before encoding**: 161,667
training characters and 17,963 validation characters for this snapshot. Encode
the two pieces separately and append one EOS to each. No window may cross the
split. This is a within-book suffix holdout, not an independent-book test set;
nearby sections share characters, style, and plot.

Keep Problem 3's protocol unchanged: the specified two-block GPT with context
256, seed 550, 100 updates, batch size 4, and the fixed evaluation windows.
Save the update-100 weights; those same weights initialize both Problem 4
models. Do not substitute a checkpoint trained on a different book collection.

## 2. Instruction following: the book's Chapter 7 records

Source: Sebastian Raschka, *Build a Large Language Model (From Scratch)*,
Chapter 7, at repository revision
`f49cad747931fe6af0a7d2802cd480d1c1a0ec8e`.

- [Pinned instruction JSON (raw download)](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/f49cad747931fe6af0a7d2802cd480d1c1a0ec8e/ch07/01_main-chapter-code/instruction-data.json):
  1,100 records with `instruction`, optional-context `input`, and reference
  `output` strings.
- SHA-256:
  `7e09a9af353722ca15c18bc580f093f9c39d3178335a75177319cb1e1d163944`.
- [Pinned repository license](https://github.com/rasbt/LLMs-from-scratch/blob/f49cad747931fe6af0a7d2802cd480d1c1a0ec8e/LICENSE.txt):
  Apache 2.0 terms, with an explicit exclusion for the book and related images.
  The JSON does not carry a separate license notice; retain the upstream license
  and attribution when redistributing repository material. This assignment does
  not reproduce book figures.

Prepare the fixed split:

```bash
uv run --no-sync --extra llm python -m homework.llm.prepare_instruction_data
```

The helper downloads only if
`data/llm/raw/homework/rasbt_instruction_data.json` is absent. It verifies the
pinned SHA-256 on every load, leaves an existing mismatching cache untouched,
and writes split indices and token lengths to
`homework/llm/instruction_manifest.json`. Use `--offline` to forbid downloads.
The helper does not train a model or contain the instructor's solution.

All indices below are zero-based positions in the source JSON:

| Split | Count | Source indices |
| --- | ---: | --- |
| Training | 32 | `0, 5, 6, 7, 12–39` |
| Validation | 8 | `40–47` |
| Test | 8 | `1, 2, 3, 4, 8, 9, 10, 11` |

The rule is fixed, not randomized: reserve the eight test prompts; normalize
each `(instruction, input)` pair by case folding and collapsing whitespace;
exclude duplicate normalized pairs; then take the first 40 eligible remaining
rows in source order, assigning the first 32 to training and the next eight to
validation. Eligibility requires at most 192 course-formatted prompt tokens
and at most 257 combined prompt/response tokens, including the response EOS.
After shifting, at most 256 input/target positions remain, so no selected row
is truncated. Maximum combined lengths are 245, 170, and 188 in the three splits.
Exact-prompt deduplication does not rule out related tasks or paraphrases.

Use the helper directly in your code:

```python
from homework.llm.prepare_instruction_data import load_instruction_splits
from math550.topics.llm import ByteTokenizer
from math550.topics.llm.data import format_instruction_prompt

splits, metadata = load_instruction_splits(download=False)
tokenizer = ByteTokenizer()
example = splits["train"][0]  # Already a course InstructionExample.
prompt_ids = tokenizer.encode(format_instruction_prompt(example))
response_ids = tokenizer.encode(example.response.strip(), add_eos=True)
```

For raw rows, use keyword arguments: the course class's positional ordering is
`instruction, response, context`, not the source JSON's field order.

```python
from math550.topics.llm import InstructionExample

example = InstructionExample(
    instruction=record["instruction"],
    context=record["input"],
    response=record["output"],
)
```

Keep the course prompt template, tokenizer, splits, and references fixed.
Train the full-sequence and response-only models from independent copies of the
same Problem 3 checkpoint with fresh identical optimizers and the same sequence
of sampled training indices (seed 552). Only the loss mask differs. Evaluate
both with response-only, token-weighted NLL; keep test references out of training
and model selection. Use the handout's eight task-specific criteria and separate
strict reference-match rule for generated answers.

The upstream [chapter notebook](https://github.com/rasbt/LLMs-from-scratch/blob/f49cad747931fe6af0a7d2802cd480d1c1a0ec8e/ch07/01_main-chapter-code/ch07.ipynb)
describes an author-prepared teaching dataset. Its
[dataset utilities](https://github.com/rasbt/LLMs-from-scratch/blob/f49cad747931fe6af0a7d2802cd480d1c1a0ec8e/ch07/02_dataset-utilities/README.md)
also document model-generated entries, and separate
[generation/refinement examples](https://github.com/rasbt/LLMs-from-scratch/blob/f49cad747931fe6af0a7d2802cd480d1c1a0ec8e/ch07/05_dataset-generation/README.md)
are available. We do not infer that every row of this JSON was produced by one
specific model. Treat the supplied outputs as weak teaching labels, not an
independently expert-verified benchmark: a valid response can differ from the
reference, and a reference may need scrutiny. No external generation service,
LLM judge, or new data generation is required.

This 32-example, byte-tokenized, from-scratch exercise is a small course
adaptation of Chapter 7 and Exercise 7.2, not a reproduction of the book's
larger pretrained-model experiment. It excludes the entire course prompt from
the response-only loss, including the assistant header. Low or zero task
success is compatible with a correct implementation.

## Instructor-only reproduction

After preparing both datasets, run the reference validator:

```bash
uv run --no-sync --extra llm python -m homework.llm.solutions.validate_solutions --device cpu
```

CPU is the default; `--device` also accepts `auto`, `cuda`, and `mps`.
The validator and `homework/llm/solutions/validation_results.json` are instructor
materials, not a student starter solution. Students implement and explain the
four problems in the handout; they are not assigned an evidence report or model
card. Retain exact settings and distinguish this short teaching run from claims
about general instruction-following ability.

The validator also regenerates the instructor's `solutions/results.tex` and
`solutions/pretraining_curve.png`. Build both PDFs with `make homework-llm`.
The separate `solutions/task_rubric_review.json` records a qualitative inspection
of the saved 24 continuations, with output fingerprints and per-row reasons.
It is not an automatic semantic metric or an independent human study. A rerun
with changed outputs requires a fresh rubric review; do not reuse stale scores.
