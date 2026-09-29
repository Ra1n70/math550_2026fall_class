"""Explicit optimization loops shared by pretraining and supervised fine-tuning."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn

from .model import GPT, GPTConfig


SUPPORTED_TORCH_DEVICES = ("auto", "cuda", "mps", "cpu")


def resolve_torch_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested not in SUPPORTED_TORCH_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(SUPPORTED_TORCH_DEVICES)}")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available:
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 6_000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    checkpoint_interval: int = 1_000
    device: str = "auto"
    seed: int = 550
    dtype: str = "float32"

    def validate(self) -> None:
        if min(
            self.max_steps,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.eval_interval,
            self.eval_batches,
        ) < 1:
            raise ValueError("step and batch settings must be positive")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be nonnegative")
        if self.warmup_steps < 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("invalid optimizer setting")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        resolve_torch_device(self.device)


def learning_rate_at(step: int, config: TrainingConfig) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_learning_rate
    denominator = max(config.max_steps - config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / denominator
    coefficient = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )


def build_adamw(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )


def _autocast_context(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "float32":
        return nullcontext()
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


@torch.no_grad()
def estimate_next_token_loss(
    model: nn.Module,
    batch_function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    batches: int,
    device: torch.device,
    dtype: str,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        inputs, targets = batch_function()
        with _autocast_context(device, dtype):
            output = model(inputs, targets)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("evaluation produced a non-finite loss")
        losses.append(float(output.loss.cpu()))
    model.train()
    return float(np.mean(losses))


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    training_config: TrainingConfig,
    metadata: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "training_config": asdict(training_config),
            "step": int(step),
            "metadata": metadata or {},
        },
        path,
    )


def load_gpt_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[GPT, dict[str, object]]:
    resolved = resolve_torch_device(device) if isinstance(device, str) else device
    payload = torch.load(path, map_location=resolved, weights_only=False)
    model = GPT(GPTConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(resolved)
    return model, payload


def stage_tokenizer_for_checkpoints(source: Path, output_dir: Path) -> Path:
    """Pair a run directory with its tokenizer before any checkpoint is written."""

    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"tokenizer file does not exist: {source}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "tokenizer.json"
    existing_checkpoints = list(output_dir.glob("*.pt"))
    if existing_checkpoints and (
        not target.is_file() or target.read_bytes() != source.read_bytes()
    ):
        raise RuntimeError(
            f"{output_dir} already contains checkpoints paired with a different "
            "or missing tokenizer.json; use a fresh --checkpoint-root/output directory"
        )
    if not target.is_file() or target.read_bytes() != source.read_bytes():
        shutil.copyfile(source, target)
    return target


def train_language_model(
    model: GPT,
    training_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    validation_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    config: TrainingConfig,
    *,
    output_dir: Path,
    metadata: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Run next-token optimization with explicit accumulation and evaluation."""

    config.validate()
    seed_everything(config.seed)
    device = resolve_torch_device(config.device)
    model.to(device)
    optimizer = build_adamw(model, config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and config.dtype == "float16")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    best_validation = float("inf")

    for step in range(config.max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            inputs, targets = training_batch()
            with _autocast_context(device, config.dtype):
                output = model(inputs, targets)
                if output.loss is None:
                    raise RuntimeError("language model did not return a loss")
                loss = output.loss / config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
            accumulated_loss += float(loss.detach().cpu())

        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        rate = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        scaler.step(optimizer)
        scaler.update()

        should_evaluate = step == 0 or (step + 1) % config.eval_interval == 0 or step + 1 == config.max_steps
        if should_evaluate:
            validation_loss = estimate_next_token_loss(
                model,
                validation_batch,
                batches=config.eval_batches,
                device=device,
                dtype=config.dtype,
            )
            row = {
                "step": step + 1,
                "train_loss": accumulated_loss,
                "validation_loss": validation_loss,
                "validation_perplexity": float(math.exp(min(validation_loss, 20.0))),
                "learning_rate": rate,
                "gradient_norm": gradient_norm,
                "elapsed_seconds": time.perf_counter() - started,
                "device": str(device),
            }
            history.append(row)
            print(
                f"step={step + 1:6d} train={accumulated_loss:.4f} "
                f"valid={validation_loss:.4f} ppl={row['validation_perplexity']:.2f} "
                f"lr={rate:.2e}"
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    step=step + 1,
                    training_config=config,
                    metadata=metadata,
                )
        if (step + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                output_dir / f"step_{step + 1:06d}.pt",
                model,
                optimizer,
                step=step + 1,
                training_config=config,
                metadata=metadata,
            )

    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        step=config.max_steps,
        training_config=config,
        metadata=metadata,
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return history


def train_supervised_batches(
    model: GPT,
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train one SFT epoch; prompt and padding labels must already be -100."""

    model.train()
    total_loss = 0.0
    count = 0
    for batch in batches:
        inputs = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, targets)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("SFT produced a non-finite loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(output.loss.detach().cpu())
        count += 1
    if count == 0:
        raise ValueError("SFT batch iterable was empty")
    return total_loss / count
