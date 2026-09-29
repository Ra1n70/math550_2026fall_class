"""Small, explicit DistributedDataParallel helpers for GPT pretraining."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .model import GPT
from .training import (
    TrainingConfig,
    _autocast_context,
    build_adamw,
    estimate_next_token_loss,
    learning_rate_at,
    save_checkpoint,
    seed_everything,
)


@dataclass(frozen=True)
class DistributedContext:
    """The rank and device assigned to one process by ``torchrun``."""

    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def torchrun_coordinates(
    environ: Mapping[str, str] | None = None,
    *,
    require_multiple: bool = True,
) -> tuple[int, int, int]:
    """Read and validate the worker coordinates supplied by ``torchrun``."""

    values = os.environ if environ is None else environ
    missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in values]
    if missing:
        raise RuntimeError(
            "distributed pretraining must be launched with torchrun; missing "
            + ", ".join(missing)
        )
    try:
        rank = int(values["RANK"])
        local_rank = int(values["LOCAL_RANK"])
        world_size = int(values["WORLD_SIZE"])
    except ValueError as error:
        raise RuntimeError("torchrun rank variables must be integers") from error
    if world_size < (2 if require_multiple else 1):
        raise RuntimeError("multi-GPU pretraining requires at least two torchrun workers")
    if not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise RuntimeError("torchrun supplied invalid rank coordinates")
    return rank, local_rank, world_size


def initialize_distributed(
    *,
    device_type: str = "cuda",
    backend: str = "auto",
    require_multiple: bool = True,
) -> DistributedContext:
    """Initialize one DDP process using the environment created by ``torchrun``."""

    if not dist.is_available():
        raise RuntimeError("this PyTorch build does not include torch.distributed")
    if device_type not in {"cuda", "cpu"}:
        raise ValueError("distributed device must be cuda or cpu")
    if backend not in {"auto", "nccl", "gloo"}:
        raise ValueError("distributed backend must be auto, nccl, or gloo")

    rank, local_rank, world_size = torchrun_coordinates(require_multiple=require_multiple)
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("multi-GPU CUDA training was requested but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} has no matching CUDA device; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        selected_backend = "nccl" if backend == "auto" else backend
    else:
        device = torch.device("cpu")
        selected_backend = "gloo" if backend == "auto" else backend
        if selected_backend == "nccl":
            raise ValueError("the NCCL backend requires CUDA")

    dist.init_process_group(backend=selected_backend, init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=selected_backend,
        device=device,
    )


def cleanup_distributed() -> None:
    """Release the default process group if this process initialized one."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_mean(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / context.world_size).cpu())


def train_language_model_distributed(
    model: GPT,
    training_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    validation_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    config: TrainingConfig,
    context: DistributedContext,
    *,
    output_dir: Path,
    metadata: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Pretrain one GPT replica per worker and average gradients with DDP."""

    config.validate()
    if config.device != context.device.type:
        raise ValueError(
            f"training config device {config.device!r} does not match DDP device "
            f"{context.device.type!r}"
        )
    seed_everything(config.seed + context.rank)
    model.to(context.device)
    ddp_kwargs: dict[str, object] = {}
    if "forward_sync_buffers" in inspect.signature(DistributedDataParallel).parameters:
        ddp_kwargs["forward_sync_buffers"] = False
    else:
        ddp_kwargs["broadcast_buffers"] = False
    if context.device.type == "cuda":
        ddp_kwargs.update(device_ids=[context.local_rank], output_device=context.local_rank)
    distributed_model = DistributedDataParallel(model, **ddp_kwargs)
    optimizer = build_adamw(distributed_model, config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(context.device.type == "cuda" and config.dtype == "float16")
    )

    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    best_validation = float("inf")

    for step in range(config.max_steps):
        distributed_model.train()
        optimizer.zero_grad(set_to_none=True)
        local_train_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            inputs, targets = training_batch()
            synchronize = micro_step + 1 == config.gradient_accumulation_steps
            synchronization = nullcontext() if synchronize else distributed_model.no_sync()
            with synchronization:
                with _autocast_context(context.device, config.dtype):
                    output = distributed_model(inputs, targets)
                    if output.loss is None:
                        raise RuntimeError("language model did not return a loss")
                    loss = output.loss / config.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step} on rank {context.rank}")
                scaler.scale(loss).backward()
            local_train_loss += float(loss.detach().cpu())

        scaler.unscale_(optimizer)
        local_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), config.grad_clip)
        )
        rate = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        scaler.step(optimizer)
        scaler.update()

        should_evaluate = (
            step == 0
            or (step + 1) % config.eval_interval == 0
            or step + 1 == config.max_steps
        )
        if should_evaluate:
            local_validation = estimate_next_token_loss(
                distributed_model,
                validation_batch,
                batches=config.eval_batches,
                device=context.device,
                dtype=config.dtype,
            )
            train_loss = _distributed_mean(local_train_loss, context)
            validation_loss = _distributed_mean(local_validation, context)
            gradient_norm = _distributed_mean(local_gradient_norm, context)
            row = {
                "step": step + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_perplexity": float(math.exp(min(validation_loss, 20.0))),
                "learning_rate": rate,
                "gradient_norm": gradient_norm,
                "elapsed_seconds": time.perf_counter() - started,
                "device": str(context.device),
                "world_size": context.world_size,
            }
            history.append(row)
            if context.is_main:
                print(
                    f"step={step + 1:6d} train={train_loss:.4f} "
                    f"valid={validation_loss:.4f} ppl={row['validation_perplexity']:.2f} "
                    f"lr={rate:.2e} workers={context.world_size}"
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
            dist.barrier()

        if (step + 1) % config.checkpoint_interval == 0:
            if context.is_main:
                save_checkpoint(
                    output_dir / f"step_{step + 1:06d}.pt",
                    model,
                    optimizer,
                    step=step + 1,
                    training_config=config,
                    metadata=metadata,
                )
            dist.barrier()

    if context.is_main:
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
    dist.barrier()
    return history
