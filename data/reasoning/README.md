# Mathematical reasoning data

The classroom model uses the official GSM8K main split: 7,473 human-written
worked solutions for training and an untouched 1,319-problem test set. The
homework evaluates transfer on all 1,000 SVAMP challenge problems. These are
more useful than additional generic books for this module because the previous
LLM topic already teaches broad next-token pretraining on 16 Project Gutenberg
books; this topic starts from an already language-capable model and teaches
reasoning-specific behavior.

Run:

```bash
uv run --extra reasoning python -m experiments.reasoning.download_data
```

Every URL is pinned to an immutable repository commit in `sources.json`. The
downloader checks the expected row counts and records resolved URLs, byte sizes,
and SHA-256 hashes in `download_manifest.json`. GSM8K and SVAMP are MIT-licensed.
GSM8K solutions contain an explicit `####` numeric answer marker. The loader
removes `<<expression=result>>` calculator annotations from the rationale but
preserves both the written reasoning and final answer.

Do not tune on `gsm8k_test.jsonl`. The code takes a fixed 512-row validation set
from the official training split. SVAMP is reserved for transfer analysis and
homework rather than model selection.
