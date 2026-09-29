"""Pretrain the course GPT on multiple GPUs with PyTorch DDP and torchrun."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from math550.topics.llm import (
    GPT,
    TokenStream,
    cleanup_distributed,
    initialize_distributed,
    load_tokenizer,
    seed_everything,
    stage_tokenizer_for_checkpoints,
    tokenizer_sha256,
    train_language_model_distributed,
)

from .config import ExperimentConfig, with_step_override


def run(
    *,
    preset: str,
    data_root: Path,
    checkpoint_root: Path,
    device: str,
    backend: str,
    steps: int | None = None,
) -> Path:
    context = initialize_distributed(device_type=device, backend=backend)
    try:
        tokenizer_path = data_root / "processed" / "tokenizer.json"
        metadata_path = data_root / "processed" / "metadata.json"
        if not tokenizer_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "prepared token streams are missing; run experiments.llm.prepare_data"
            )

        tokenizer = load_tokenizer(tokenizer_path)
        tokenizer_fingerprint = tokenizer_sha256(tokenizer_path)
        experiment = ExperimentConfig(
            preset=preset, data_root=data_root, checkpoint_root=checkpoint_root
        )
        model_config = experiment.model(tokenizer.vocab_size)
        training_config = with_step_override(experiment.training(device), steps)
        train_stream = TokenStream(
            data_root / "processed" / "train.bin", model_config.block_size
        )
        validation_stream = TokenStream(
            data_root / "processed" / "validation.bin", model_config.block_size
        )
        train_generator = torch.Generator(device="cpu").manual_seed(
            experiment.seed + context.rank
        )
        validation_generator = torch.Generator(device="cpu").manual_seed(
            experiment.seed + 10_000 + context.rank
        )

        def training_batch():
            return train_stream.batch(
                training_config.batch_size,
                generator=train_generator,
                device=context.device,
            )

        def validation_batch():
            return validation_stream.batch(
                training_config.batch_size,
                generator=validation_generator,
                device=context.device,
            )

        # Construct identical initial replicas; DDP also broadcasts rank 0's
        # parameters when it wraps the model.
        seed_everything(experiment.seed)
        model = GPT(model_config)
        output_dir = checkpoint_root / f"pretrain-{preset}-ddp"
        if context.is_main:
            staged_tokenizer_path = stage_tokenizer_for_checkpoints(
                tokenizer_path, output_dir
            )
        else:
            staged_tokenizer_path = output_dir / "tokenizer.json"
        torch.distributed.barrier()
        corpus_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_language_model_distributed(
            model,
            training_batch,
            validation_batch,
            training_config,
            context,
            output_dir=output_dir,
            metadata={
                "stage": "pretraining",
                "training_mode": "distributed_data_parallel",
                "tokenizer_path": str(staged_tokenizer_path),
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer_sha256": tokenizer_fingerprint,
                "corpus_metadata": corpus_metadata,
                "parameter_count": model.parameter_count,
                "distributed": {
                    "backend": context.backend,
                    "world_size": context.world_size,
                    "batch_size_per_worker": training_config.batch_size,
                    "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                    "effective_sequences_per_update": (
                        training_config.batch_size
                        * training_config.gradient_accumulation_steps
                        * context.world_size
                    ),
                },
            },
        )
        return output_dir / "best.pt"
    finally:
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=("smoke", "cpu", "mps", "colab"), default="colab"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/llm"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/llm"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    checkpoint = run(**vars(args))
    if int(os.environ["RANK"]) == 0:
        print(checkpoint)


if __name__ == "__main__":
    main()
