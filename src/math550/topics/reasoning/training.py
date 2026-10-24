"""Explicit SFT, generation, and one-step GRPO training loops."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import MathReasoningExample, ReasoningCollator, format_reasoning_prompt
from .evaluation import verifiable_reward
from .modeling import adapter_disabled
from .objectives import grpo_loss, group_relative_advantages, response_token_log_probs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@dataclass(frozen=True)
class SFTConfig:
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int | None = 800
    log_interval: int = 20
    seed: int = 550

    def validate(self) -> None:
        if min(self.epochs, self.batch_size, self.gradient_accumulation_steps, self.log_interval) < 1:
            raise ValueError("SFT counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("invalid SFT optimizer setting")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")


@torch.no_grad()
def evaluate_sft_loss(
    model: Any,
    dataset: Dataset,
    collator: ReasoningCollator,
    *,
    device: torch.device,
    batch_size: int = 1,
    max_batches: int = 16,
) -> float:
    model.eval()
    losses: list[float] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        moved = {name: tensor.to(device) for name, tensor in batch.items()}
        with _autocast(device):
            loss = model(**moved).loss
        if not torch.isfinite(loss):
            raise RuntimeError("SFT evaluation produced non-finite loss")
        losses.append(float(loss.detach().cpu()))
    model.train()
    if not losses:
        raise ValueError("SFT validation dataset produced no batches")
    return float(np.mean(losses))


def train_sft(
    model: Any,
    training_dataset: Dataset,
    validation_dataset: Dataset,
    collator: ReasoningCollator,
    config: SFTConfig,
    *,
    device: torch.device,
    output_dir: Path,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """Fine-tune only trainable parameters with assistant-token cross-entropy."""

    config.validate()
    seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial_validation = evaluate_sft_loss(
        model,
        validation_dataset,
        collator,
        device=device,
        batch_size=config.batch_size,
    )
    history: list[dict[str, float | int]] = []
    optimizer_step = 0
    micro_step = 0
    accumulated = 0.0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    model.train()
    for epoch in range(config.epochs):
        for batch in loader:
            moved = {name: tensor.to(device) for name, tensor in batch.items()}
            with _autocast(device):
                loss = model(**moved).loss / config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite SFT loss at micro-step {micro_step}")
            loss.backward()
            accumulated += float(loss.detach().cpu())
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps != 0:
                continue
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step == 1 or optimizer_step % config.log_interval == 0:
                row = {
                    "step": optimizer_step,
                    "epoch": epoch + 1,
                    "train_loss": accumulated,
                    "gradient_norm": gradient_norm,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(row)
                print(
                    f"sft step={optimizer_step:4d} loss={accumulated:.4f} "
                    f"grad={gradient_norm:.3f}"
                )
            accumulated = 0.0
            if config.max_steps is not None and optimizer_step >= config.max_steps:
                stop = True
                break
        if stop:
            break
    final_validation = evaluate_sft_loss(
        model,
        validation_dataset,
        collator,
        device=device,
        batch_size=config.batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(adapter_dir)
    result: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset),
        "optimizer_steps": optimizer_step,
        "initial_validation_loss": initial_validation,
        "final_validation_loss": final_validation,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "adapter_path": str(adapter_dir),
    }
    (output_dir / "sft_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@torch.no_grad()
def generate_completion_groups(
    model: Any,
    tokenizer: Any,
    examples: Sequence[MathReasoningExample],
    *,
    device: torch.device,
    samples_per_problem: int = 1,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 550,
    prompt_style: str = "cot",
) -> tuple[list[list[str]], list[list[int]], list[list[float]]]:
    if samples_per_problem < 1 or max_new_tokens < 1:
        raise ValueError("generation counts must be positive")
    seed_everything(seed)
    model.eval()
    groups: list[list[str]] = []
    token_groups: list[list[int]] = []
    latency_groups: list[list[float]] = []
    for example in examples:
        prompt = format_reasoning_prompt(
            tokenizer, example.question, prompt_style=prompt_style
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        started = time.perf_counter()
        generation_arguments: dict[str, object] = {
            "do_sample": samples_per_problem > 1,
            "num_return_sequences": samples_per_problem,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if samples_per_problem > 1:
            generation_arguments.update({"temperature": temperature, "top_p": top_p})
        generated = model.generate(
            **encoded,
            **generation_arguments,
        )
        elapsed = time.perf_counter() - started
        prompt_length = int(encoded["input_ids"].shape[1])
        completions = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
        counts = [int(row.numel() - prompt_length) for row in generated]
        groups.append(completions)
        token_groups.append(counts)
        latency_groups.append([elapsed / samples_per_problem] * samples_per_problem)
    return groups, token_groups, latency_groups


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 4
    learning_rate: float = 5e-6
    clip_epsilon: float = 0.2
    beta: float = 0.04
    max_new_tokens: int = 160
    max_length: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    seed: int = 550
    baseline: str = "mean"
    advantage_normalizer: str = "std"
    loss_normalization: str = "sequence"
    normalization_constant: float | None = None
    importance_method: str = "grpo"
    prompt_style: str = "cot"

    def validate(self) -> None:
        if self.group_size < 2 or self.learning_rate <= 0:
            raise ValueError("GRPO needs group_size >= 2 and a positive learning rate")
        if not 0 < self.clip_epsilon < 1 or self.beta < 0:
            raise ValueError("GRPO needs clip_epsilon in (0, 1) and beta >= 0")
        if self.max_new_tokens < 1 or self.max_length < 2:
            raise ValueError("generation lengths must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature and top_p are outside their valid ranges")
        if self.baseline not in ("mean", "none"):
            raise ValueError("baseline must be mean or none")
        if self.advantage_normalizer not in ("std", "mean", "none"):
            raise ValueError("unknown advantage normalizer")
        if self.loss_normalization not in ("sequence", "token", "constant"):
            raise ValueError("unknown loss normalization")
        if self.normalization_constant is not None and self.normalization_constant <= 0:
            raise ValueError("normalization_constant must be positive")
        if self.importance_method not in ("none", "noclip", "grpo", "gspo", "cispo"):
            raise ValueError("unknown importance method")
        if self.prompt_style not in ("direct", "cot"):
            raise ValueError("prompt_style must be direct or cot")


def _rollout_tensors(
    tokenizer: Any,
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    rows: list[tuple[list[int], list[int]]] = []
    for prompt, completion in zip(prompts, completions, strict=True):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            completion_ids = completion_ids + [tokenizer.eos_token_id]
        prompt_ids = prompt_ids[: max_length - 2]
        completion_ids = completion_ids[: max_length - len(prompt_ids)]
        ids = prompt_ids + completion_ids
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        rows.append((ids, mask))
    maximum = max(len(ids) for ids, _ in rows)
    input_ids, attention_masks, response_masks = [], [], []
    for ids, mask in rows:
        padding = maximum - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * padding)
        attention_masks.append([1] * len(ids) + [0] * padding)
        response_masks.append(mask + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "response_mask": torch.tensor(response_masks, dtype=torch.bool),
    }


def grpo_update(
    model: Any,
    tokenizer: Any,
    examples: Sequence[MathReasoningExample],
    config: GRPOConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Generate one group per prompt and apply one transparent GRPO optimizer update."""

    config.validate()
    if not examples:
        raise ValueError("GRPO needs at least one prompt")
    seed_everything(config.seed)
    groups, _, _ = generate_completion_groups(
        model,
        tokenizer,
        examples,
        device=device,
        samples_per_problem=config.group_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        prompt_style=config.prompt_style,
    )
    prompts: list[str] = []
    flat_completions: list[str] = []
    rewards: list[float] = []
    for example, group in zip(examples, groups, strict=True):
        prompt = format_reasoning_prompt(
            tokenizer, example.question, prompt_style=config.prompt_style
        )
        for completion in group:
            prompts.append(prompt)
            flat_completions.append(completion)
            rewards.append(verifiable_reward(completion, example.answer))
    batch = _rollout_tensors(
        tokenizer, prompts, flat_completions, max_length=config.max_length
    )
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
    advantages = group_relative_advantages(
        reward_tensor,
        group_size=config.group_size,
        baseline=config.baseline,
        normalizer=config.advantage_normalizer,
    )
    # RL policy ratios assume a deterministic policy evaluation.  Keeping the
    # model in evaluation mode disables LoRA dropout while preserving autograd
    # for the live-policy forward pass below.
    model.eval()
    with torch.no_grad():
        old_logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        old_log_probs, action_mask = response_token_log_probs(
            old_logits, batch["input_ids"], batch["response_mask"]
        )
        with adapter_disabled(model):
            reference_logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        reference_log_probs, _ = response_token_log_probs(
            reference_logits, batch["input_ids"], batch["response_mask"]
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    new_logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).logits
    new_log_probs, _ = response_token_log_probs(
        new_logits, batch["input_ids"], batch["response_mask"]
    )
    normalization_constant = config.normalization_constant
    if config.loss_normalization == "constant" and normalization_constant is None:
        normalization_constant = len(examples) * config.group_size * config.max_new_tokens
    objective = grpo_loss(
        new_log_probs,
        old_log_probs.detach(),
        reference_log_probs.detach(),
        advantages,
        action_mask,
        clip_epsilon=config.clip_epsilon,
        beta=config.beta,
        importance_method=config.importance_method,
        normalization=config.loss_normalization,
        normalization_constant=normalization_constant,
    )
    optimizer.zero_grad(set_to_none=True)
    objective.total.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    optimizer.step()
    return {
        "prompts": len(examples),
        "group_size": config.group_size,
        "rewards": rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "mean_reward": float(reward_tensor.mean().cpu()),
        "reward_standard_deviation": float(reward_tensor.std(unbiased=False).cpu()),
        "baseline": config.baseline,
        "advantage_normalizer": config.advantage_normalizer,
        "loss_normalization": config.loss_normalization,
        "normalization_constant": normalization_constant,
        "importance_method": config.importance_method,
        "prompt_style": config.prompt_style,
        "loss": float(objective.total.detach().cpu()),
        "policy_loss": float(objective.policy.detach().cpu()),
        "reference_kl": float(objective.reference_kl.detach().cpu()),
        "clip_fraction": float(objective.clip_fraction.detach().cpu()),
        "gradient_norm": gradient_norm,
        "finite": bool(torch.isfinite(objective.total).item()),
    }
