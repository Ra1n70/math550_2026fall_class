# Reasoning models from a pretrained LLM

This four-week module turns a compact open language model into a mathematical
reasoning laboratory. It starts with revision-pinned
`Qwen/Qwen2.5-0.5B-Instruct`, then implements the reasoning-specific parts in
course code: worked-solution SFT, assistant-only masks, LoRA training, exact
numeric verification, inference-time sampling, self-consistency, group-relative
advantages, clipped GRPO, and a frozen-reference penalty.

"From scratch" here means implementing and exposing those mechanisms rather
than rebuilding the transformer a second time. The preceding LLM module already
constructs a GPT, tokenizer, pretraining loop, SFT, reward model, and PPO from
elementary PyTorch. This topic begins where foundation-model pretraining ends,
as does the reference progression in Sebastian Raschka's
[`reasoning-from-scratch`](https://github.com/rasbt/reasoning-from-scratch).

## Learning sequence

1. Define chain of thought (CoT) as an observable intermediate token sequence,
   not a training algorithm or privileged view of hidden computation.
2. Distinguish direct, zero-shot CoT, and few-shot CoT inputs; then separate CoT
   prompting, CoT SFT, generate-filter-SFT, and CoT RL.
3. Build a strict answer parser and evaluate greedy generations before training.
4. Fine-tune on human GSM8K worked solutions with prompt-masked cross-entropy.
5. Use LoRA to keep the trainable state small on MPS and consumer CUDA devices.
6. Sample several paths, compute pass@k, and select with majority vote.
7. Turn exact answers into transparent outcome rewards.
8. Derive REINFORCE, PPO token reweighting, and finite-group baseline scaling.
9. Factor RLVR into baseline, advantage normalizer, token-loss denominator,
   stale-rollout ratio, clipping rule, and reference regularization.
10. Compare GRPO, Dr. GRPO, RFT, MaxRL, PPO-style token clipping, and GSPO.
11. Run an online update against a frozen base-model reference.
12. Audit parser exploitation, unfaithful rationales, sparse rewards, leakage, and
   compute-quality tradeoffs.

## Public API

- `MathReasoningExample`, `load_gsm8k_jsonl`, `load_svamp_json`
- `ReasoningSFTDataset`, `ReasoningCollator`, `format_reasoning_prompt`
- prompt styles `direct` and `cot` for matched inference/RL prompting ablations
- `canonicalize_numeric_answer`, `extract_final_answer`
- `CompletionRecord`, `summarize_completion_records`, `verifiable_reward`
- `response_token_log_probs`, `group_relative_advantages`, `grpo_loss`
- shared `compute_group_advantages`, `policy_gradient_loss`, and
  `aggregate_token_losses` in `math550.core.alignment`
- `MODEL_PROFILES`, `model_spec`, `OpenModelSpec`, `load_open_model`,
  `load_reasoning_adapter`
- `SFTConfig`, `train_sft`, `generate_completion_groups`, `grpo_update`

## Data and exact downloads

Run `make download-reasoning-data`. The complete raw files are also included so
the lesson can begin offline after model weights are cached.

### Classroom: GSM8K

- Official repository: <https://github.com/openai/grade-school-math>
- Pinned revision: `3101c7d5072418e28b9008a6636bde82a006892c`
- Train URL: <https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl>
- Test URL: <https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl>
- Rows: 7,473 train; 1,319 official test
- License: MIT
- Paper: <https://arxiv.org/abs/2110.14168>

A fixed seed-550 split holds 512 rows out of the official training data for
validation. The official test data is untouched during training and model
selection. Training rationales have median 45 words; the three dataset files
occupy 5.26 MB.

### Homework transfer: SVAMP

- Official repository: <https://github.com/arkilpatel/SVAMP>
- Pinned revision: `689d7ccac74b9983a2ac7cc3b264f441b99e7c53`
- Exact file: <https://raw.githubusercontent.com/arkilpatel/SVAMP/689d7ccac74b9983a2ac7cc3b264f441b99e7c53/SVAMP.json>
- Rows: 1,000
- License: MIT
- Paper: <https://aclanthology.org/2021.naacl-main.168/>

The validated homework audit finds that 999 of 1,000 supplied equations match
their published answer. Record `chal-680` has equation `4 - 2 + 3`, which yields
5, but publishes answer 1. The assignment asks students to document this rather
than silently repair it.

### Open-model profiles

- Model: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775>
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Parameters: 494,032,768 in the base checkpoint
- Weight file: about 988 MB
- License: Apache-2.0

The code uses `trust_remote_code=False`. Model weights are downloaded into the
normal Hugging Face cache and are not committed to the course repository.

The default `qwen-0.5b` profile keeps smoke runs inexpensive. Two opt-in profiles
are revision pinned in `data/reasoning/sources.json`:

- `qwen-math-1.5b`: `Qwen/Qwen2.5-Math-1.5B-Instruct`, revision
  `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`, approximately 3.09 GB;
- `olmo2-1b-instruct`: `allenai/OLMo-2-0425-1B-Instruct`, revision
  `48d788eca847d4d7548f375ad03d3c9312f6139e`, approximately 2.98 GB.

Both are Apache-2.0. OLMo Instruct already includes SFT, DPO, and RLVR, and Qwen
Math is already math specialized. They are stronger comparison checkpoints, not
clean evidence that the course update created all observed capability.

## Commands

```bash
uv sync --locked --extra reasoning
make download-reasoning-data
make test-reasoning
make smoke-reasoning
make audit-rlvr
make validate-homework-reasoning

# Recommended consumer Mac/GPU track
make sft-reasoning

# Or start with the short track
uv run --extra reasoning python -m experiments.reasoning.sft \
  --preset quick --device auto --model-profile qwen-math-1.5b

# Evaluate a saved adapter on an untouched prefix or the full split
uv run --extra reasoning python -m experiments.reasoning.evaluate \
  --dataset gsm8k \
  --adapter checkpoints/reasoning/sft-quick/adapter \
  --device auto --limit 128 --samples 1 --output-name sft_quick_128

# Matched direct-vs-CoT prompting ablation on a frozen checkpoint
uv run --extra reasoning python -m experiments.reasoning.evaluate \
  --dataset gsm8k --device auto --limit 128 --samples 1 \
  --prompt-style direct --output-name base_direct_128
uv run --extra reasoning python -m experiments.reasoning.evaluate \
  --dataset gsm8k --device auto --limit 128 --samples 1 \
  --prompt-style cot --output-name base_cot_128

# Self-consistency
uv run --extra reasoning python -m experiments.reasoning.evaluate \
  --dataset gsm8k \
  --adapter checkpoints/reasoning/sft-quick/adapter \
  --device auto --limit 64 --samples 4 --output-name sft_quick_sc_64x4

# Online GRPO extension
uv run --extra reasoning python -m experiments.reasoning.grpo \
  --preset quick \
  --adapter checkpoints/reasoning/sft-quick/adapter \
  --device auto --updates 10 --prompts-per-update 2 --variant dr-grpo \
  --prompt-style cot
```

`auto` selects CUDA, then Apple MPS, then CPU. Explicit unavailable accelerators
raise a clear error. MPS runs float32. CUDA uses bfloat16 when supported and
float16 otherwise.

## Presets and realistic runtime

| Preset | Context | LoRA rank | Batch | Accumulation | Max SFT updates |
|---|---:|---:|---:|---:|---:|
| smoke | 160 | 4 | 1 | 2 | 2 |
| quick | 256 | 8 | 1 | 4 | 50 |
| mps | 384 | 16 | 1 | 8 | 800 |
| colab | 512 | 16 | 4 | 8 | 1,200 |

The MPS and Colab settings are few-hour teaching targets, not wall-clock
guarantees. Generation makes GRPO much slower than SFT. Time 20 updates on the
actual hardware, measure generated tokens per second, then set the final budget.

## Measured local evidence

The checked local run used Apple M2 MPS, rank-4 LoRA, 128 GSM8K training rows,
64 internal validation rows, context 160, and 20 optimizer steps. It took 14.24
seconds and reduced validation token loss from 1.174 to 0.740.

On the same first 16 official test rows:

| Condition | Exact | 95% Wilson interval | `####` format | Mean generated tokens |
|---|---:|---:|---:|---:|
| pinned base | 2/16 | 0.035-0.360 | 0/16 | 189.2 |
| 20-step adapter | 5/16 | 0.142-0.556 | 16/16 | 82.4 |

The intervals overlap, the prefix is tiny and non-random, and there is one seed.
This verifies the public-data/model/optimizer/generation path. It does not
establish a general accuracy improvement. The clean conclusion is that the
adapter learned the requested answer format and shorter stopping behavior.

Four sampled paths on eight rows produced first-sample accuracy 2/8,
majority-vote accuracy 2/8, and oracle pass@4 4/8. More compute found more good
candidates, but majority vote did not identify them. A real one-update GRPO
probe was finite; both rollouts received equal reward, so advantages were zero
and only the reference penalty contributed. This is retained as a sparse-reward
failure-mode example.

## Course artifacts

- Code: `src/math550/topics/reasoning/`
- Experiment drivers: `experiments/reasoning/`
- Numerical records: `output/reasoning/analysis/`
- Technical report: `output/pdf/reasoning_models_report.pdf`
- Lecture notes: `notes/reasoning/lecture_notes.pdf`
- Editable/PDF deck: `slides/reasoning/source/reasoning_models_teaching.pptx`
  and `slides/reasoning/reasoning_models_teaching.pdf`
- Homework and solutions: `homework/reasoning/homework.pdf` and
  `homework/reasoning/solutions.pdf`
- Homework validation: `homework/reasoning/solutions/validation_results.json`

## Interpretation limits

Exact final answers do not validate every rationale step. Written rationales do
not expose a privileged internal chain of thought. A verifier can be exploited.
Pass@k uses the gold answer and is not a deployment selector. GSM8K and SVAMP are
narrow English grade-school benchmarks. Stronger claims require the full held-out
sets, several seeds, transfer, error analysis, process checks, and compute-matched
comparisons.

## Where chain of thought belongs

CoT is a generated reasoning behavior or response format. In zero-shot CoT, the
input is an instruction plus a new question and the new rationale is output. In
few-shot CoT, example question-rationale-answer triples also appear in the input
context. Prompting changes no weights; CoT SFT imitates supplied traces;
generate-filter methods such as STaR turn successful sampled traces into new
supervised targets; RL samples traces and weights their policy gradients by
reward or advantage.

With outcome-only RLVR, one response advantage is broadcast across the sampled
rationale and answer tokens. PPO/GRPO token ratios can give those tokens
different magnitudes and clipping states, but a correct final answer does not
establish that every step was causally useful. Process checks, counterfactual
edits, or step-level rewards are separate additions.

## Primary references

- Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).
- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903).
- Kojima et al., [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916).
- Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171).
- Zelikman et al., [STaR](https://arxiv.org/abs/2203.14465).
- Lanham et al., [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702).
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300).
- DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
