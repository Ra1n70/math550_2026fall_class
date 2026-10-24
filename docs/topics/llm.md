# Large language models from scratch

This three-week module builds a decoder-only language model and its post-training
pipeline without pretrained weights, `torch.nn.Transformer`, Hugging Face model
classes, or tokenizer libraries. PyTorch is used only for tensors, modules,
automatic differentiation, optimizers, and data loading. The conceptual route is
inspired by Sebastian Raschka's
[`LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch), while the code,
tests, datasets, course presets, and measured runs here are self-contained.

## Learning outcomes and sequence

Students finish one connected artifact across nine units:

1. UTF-8 bytes, byte-level BPE, round-trip tests, and compression tradeoffs.
2. Autoregressive factorization, shifted targets, cross-entropy, and perplexity.
3. Explicit multi-head causal self-attention and a behavioral causality test.
4. Pre-norm GPT blocks, weight tying, parameter counting, and a bigram baseline.
5. Whole-book corpus construction, memory-mapped batches, AdamW, warmup/cosine
   scheduling, gradient accumulation, clipping, evaluation, checkpoints, and an
   optional DistributedDataParallel extension.
6. Temperature/top-k generation and evidence-aware interpretation.
7. Supervised instruction fine-tuning with an assistant-only loss mask.
8. Bradley-Terry reward modeling from chosen/rejected response pairs.
9. Token-level KL rewards, generalized advantage estimation, a value head, and
   clipped PPO.

The core implementation lives in `src/math550/topics/llm/`; the scripts under
`experiments/llm/` are intentionally thin orchestration layers.

## Public data and exact download sources

Run `make download-llm-data`. The downloader strips Project Gutenberg headers,
records every resolved URL and SHA-256 in `data/llm/download_manifest.json`, and
verifies the two pinned Hugging Face files against the hashes below. Project
Gutenberg states that most of its collection is unrestricted under U.S.
copyright law; users outside the U.S. must check local law. See its
[terms](https://www.gutenberg.org/policy/terms_of_use.html) and
[permission guidance](https://www.gutenberg.org/policy/permission).

### Pre-training books

The downloader obtains its population from Project Gutenberg's [official weekly
machine-readable catalog](https://www.gutenberg.org/ebooks/offline_catalogs.html),
not a hand-written title list. It keeps English `Text` records, excludes the
separate homework title and duplicate normalized titles, and randomly samples N
works without replacement. About 10% are randomly held out as entire validation
books. Whole-book splitting avoids leakage from overlapping windows of the same
work.

The checked numerical evidence elsewhere in this module came from the earlier
16-book snapshot: 7,929,826 training tokens and 2,921,627 validation tokens at
vocabulary size 512, about 42 MB raw and 21 MB as `uint16` streams. Those numbers
describe that archived run, not every new random sample.

#### Enlarge the corpus

For 64 total books, run:

```bash
make download-llm-data LLM_BOOK_LIMIT=64
make prepare-llm-data \
  LLM_VOCAB_SIZE=1024 \
  LLM_TOKENIZER_CHARACTERS=5000000
```

`--book-limit N` has no hard-coded upper cap; N must be at least 2 and no larger
than the eligible population in the current catalog. On every run, cached
eligible `data/llm/raw/books/pg<ID>.txt` files are shuffled ahead of missing
candidates. Growing a cache from N to M therefore downloads only M-N new works;
repeated and smaller runs reuse files without deleting them. When more than N
eligible files are cached, the active N are randomly sampled from that cache.

A fresh 64-bit selection seed is recorded by default. Pass
`--selection-seed 2026` (or `LLM_SELECTION_SEED=2026` to `make`) when a class
needs a controlled sample. The cached catalog SHA-256, exact active book list,
random whole-book split, mirror URLs, and file hashes are all recorded in
`data/llm/download_manifest.json`; `prepare_data.py` treats that manifest as the
authoritative corpus and streams books into the token files.

New payloads are generated UTF-8 text files on `gutenberg.pglaf.org`, with
main-collection ZIP files as fallbacks. It is identified as a high-speed host in
Project Gutenberg's [registered mirror
list](https://www.gutenberg.org/MIRRORS.ALL), with a default two-second delay.
Gutenberg's [terms of
use](https://www.gutenberg.org/policy/terms_of_use.html) direct bulk automation
away from the human-facing site. For a very large run, point
`--book-url-template` at a private or authorized mirror; templates may use
`{id}` and `{id_path}` and may return ZIP or plain text. See the official
[mirroring instructions](https://www.gutenberg.org/help/mirroring.html) and
[robot harvest route](https://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en).
Add `--refresh-catalog` only when intentionally replacing the cached weekly catalog.

Increasing the corpus changes the data distribution but not the optimizer's
token budget. The MPS preset still processes about 24.6 million token positions
(6,000 updates times batch 16 times context 256). Read the new `train` token
count from `processed/metadata.json`, time an initial run, and increase steps if
the goal is broader corpus coverage rather than a fixed-compute comparison.

The homework transfer corpus is H. G. Wells's *The Time Machine*,
[Project Gutenberg 35](https://www.gutenberg.org/ebooks/35), downloaded through
the same approved text-archive mirror route and kept outside the training pool.

### Instruction and preference data

- SFT: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
  revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, CC BY-SA 3.0,
  [pinned JSONL](https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl),
  SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`.
- Preferences: [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf),
  revision `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`, MIT,
  [pinned harmless-base train gzip](https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/harmless-base/train.jsonl.gz),
  SHA-256 `bf3e1ecd3db61bf21d190f69351d26670c67aababa0fd282fa008c17a7202217`.

HH-RLHF intentionally includes harmful requests and unsafe rejected responses.
Do not project raw examples in class without prior review. Preference labels are
records of choices under an annotation protocol, not universal truth.

## Run the module

```bash
uv sync --locked --extra llm
make download-llm-data LLM_BOOK_LIMIT=64
make prepare-llm-data LLM_VOCAB_SIZE=1024 LLM_TOKENIZER_CHARACTERS=5000000
make test-llm
make smoke-llm
```

`LLM_VOCAB_SIZE`/`--vocab-size` is a requested BPE ceiling, not a promise that
all entries will be created. Training stops early when no adjacent pair reaches
the minimum frequency, and `data/llm/processed/metadata.json` records the actual
`vocab_size`. The standard pretraining drivers construct the GPT with that
actual size; they do not pad it automatically.

The smoke command uses a synthetic, stage-spanning correctness corpus so it can
exercise pre-training, SFT, reward modeling, and PPO quickly. The real-data path
starts with the prepared book streams:

```bash
# 0.12M-parameter, 30-step public-data verification
uv run --extra llm python -m experiments.llm.pretrain \
  --preset smoke --steps 30 --device auto \
  --checkpoint-root checkpoints/llm

# Consumer Mac/CUDA route: 10.9M parameters, context 256
uv run --extra llm python -m experiments.llm.pretrain \
  --preset mps --steps 6000 --device auto

# Colab route: 25.6M parameters, context 256
uv run --extra llm python -m experiments.llm.pretrain \
  --preset colab --steps 10000 --device cuda
```

`device=auto` selects CUDA, then Apple MPS, then CPU. The MPS and Colab presets
are designed as few-hour teaching runs, not wall-clock guarantees: speed depends
on the exact GPU, thermal state, PyTorch build, and evaluation interval. Start
with 100 timed steps, then set the final step count from the measured rate. A
full run was not used to make the included quality claims; the checked evidence
is the 30-step public-book smoke below.

Single-device training above is the default and is the clearest starting point.
On a machine with two or more CUDA GPUs, the optional DDP driver launches one
process per GPU:

```bash
# Two GPUs; set LLM_NPROC_PER_NODE to the number of visible CUDA devices.
make pretrain-llm-multi-gpu LLM_NPROC_PER_NODE=2

# Equivalent direct command, useful on a GPU server.
uv run --extra llm torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m experiments.llm.pretrain_distributed \
  --preset colab --steps 10000 --device cuda
```

The implementation uses PyTorch DistributedDataParallel (DDP), not
`DataParallel`: each worker owns one complete model replica and independently
samples token windows, while DDP averages gradients before the optimizer step.
Validation loss is averaged across workers, and only rank 0 writes checkpoints
and history. `batch_size` is per worker, so the effective sequences per update
are `batch_size × gradient_accumulation_steps × world_size`. Apple MPS remains a
single-device path; the multi-GPU lesson requires CUDA/NCCL. A CPU/Gloo route is
present only for correctness testing and is not a speed recommendation.

Generate from either checkpoint with fresh randomness by default:

```bash
uv run --extra llm python -m experiments.llm.generate \
  checkpoints/llm/pretrain-mps/best.pt \
  --prompt "Once upon a time" --temperature 0.8

# Reproduce exactly the same categorical draws.
uv run --extra llm python -m experiments.llm.generate \
  checkpoints/llm/pretrain-mps/best.pt \
  --prompt "Once upon a time" --temperature 0.8 --seed 2026

# If the checkpoint's sibling tokenizer.json is stale, select the original.
uv run --extra llm python -m experiments.llm.generate \
  checkpoints/llm/pretrain-mps/step_010000.pt \
  --tokenizer data/llm/processed/tokenizer.json
```

The command reports the actual sampling seed on standard error. Omitting
`--seed` creates a fresh 32-bit seed on every invocation; supplying one makes
the output reproducible. Temperature changes the categorical distribution, not
a promise that any particular draw must differ.

The tokenizer vocabulary defines the valid sampling IDs. If a customized model
head is larger for hardware-alignment padding, generation slices its logits to
the tokenizer's actual size before applying temperature and top-k. Conversely,
a tokenizer size or SHA-256 that conflicts with checkpoint metadata is rejected
early as the wrong sidecar file; it is not silently treated as padding.
Training stages `tokenizer.json` in the output directory before writing the
first checkpoint and refuses to reuse a directory whose existing checkpoints
belong to a different tokenizer. Use a fresh `--checkpoint-root` or
`--output-dir` when changing tokenizer training settings.

Continue post-training from the resulting checkpoint:

```bash
uv run --extra llm python -m experiments.llm.sft \
  checkpoints/llm/pretrain-mps/best.pt --device auto --epochs 2

uv run --extra llm python -m experiments.llm.reward_modeling \
  checkpoints/llm/sft/sft.pt --device auto --limit 5000

uv run --extra llm python -m experiments.llm.ppo \
  checkpoints/llm/sft/sft.pt checkpoints/llm/reward/reward_model.pt \
  --device auto --updates 20
```

## Verified evidence

`make test-llm` runs 30 unit tests covering random catalog selection, cache
reuse, large limits, and manifest-driven preparation in addition to BPE
multiplicity/round trips, causality, shapes, tied
weights, target shifting, assistant-only labels, checkpointing, pairwise
rewards, token masking, GAE, finite PPO gradients, sampling-seed behavior,
padded-vocabulary masking, tokenizer-contract validation, and `torchrun` rank
validation.

The integrated synthetic smoke produced these measured changes:

- GPT validation loss 5.655 → 4.404; perplexity 285.77 → 81.81.
- Bigram validation loss 6.498 → 4.999.
- Assistant-only SFT loss 4.503 → 3.258.
- Reward loss 0.717 → 0.159; pair accuracy 1.0 on the tiny training smoke, an
  intentional overfit check rather than a generalization claim.
- PPO total loss and every gradient remained finite; approximate KL was 0.00045.

The 30-step real-book run produced held-out loss 6.207 → 5.126 and perplexity
496.3 → 168.4. Its generated sample remained noise. This verifies the public
data, optimizer, checkpoint, evaluation, and generation paths; it does not
demonstrate a useful language model.

## Course artifacts

- Technical report: `output/pdf/llm_from_scratch_report.pdf`
- Lecture notes: `notes/llm/lecture_notes.pdf`
- Editable/PDF deck: `slides/llm/source/llm_from_scratch_teaching.pptx` and
  `slides/llm/llm_from_scratch_teaching.pdf`
- Homework and solutions: `homework/llm/homework.pdf` and
  `homework/llm/solutions.pdf`
- Homework validator: `homework/llm/solutions/validate_solutions.py`
- Numerical records: `output/llm/analysis/` and
  `homework/llm/solutions/validation_results.json`

The revised LLM homework has four problems (100 points): causal attention,
decoder parameter accounting, next-token pretraining/decoding, and a controlled
instruction fine-tuning experiment. It adapts Raschka Exercises 3.1–3.3, 4.1,
5.1–5.3, and 7.2, with pinned sources in `homework/llm/README_data.md`.
Students use a fixed byte tokenizer and *The Time Machine*, then a fixed
32/8/8 subset of the book's instruction dataset prepared by
`homework/llm/prepare_instruction_data.py`. Eight explicit task rubrics make
instruction-following scores distinct from response loss. Tokenizer transfer,
the Bigram-versus-GPT comparison, and the standalone model-card assignment are
not part of this homework. PPO is deferred to the reinforcement-learning
homework; the existing LLM module code and notes are retained.

## Interpretation limits

The module makes mechanisms observable at small scale. It does not reproduce
GPT-2 data volume, model size, training compute, safety evaluation, or capability.
Low loss is not sufficient evidence of instruction following; pair accuracy is
not sufficient evidence of alignment; PPO stability is not evidence that the
reward proxy is correct. Student write-ups must distinguish implementation
verification, held-out predictive performance, sampled behavior, and normative
claims.

## Primary references

- Sennrich, Haddow, and Birch, [Neural Machine Translation of Rare Words with
  Subword Units](https://aclanthology.org/P16-1162/).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Radford et al., [Language Models are Unsupervised Multitask
  Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Ouyang et al., [Training language models to follow instructions with human
  feedback](https://arxiv.org/abs/2203.02155).
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Bai et al., [Training a Helpful and Harmless Assistant with Reinforcement
  Learning from Human Feedback](https://arxiv.org/abs/2204.05862).
