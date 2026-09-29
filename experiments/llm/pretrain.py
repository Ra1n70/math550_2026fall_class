"""Pretrain the course GPT decoder on the prepared book token streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from math550.topics.llm import (
    GPT,
    TokenStream,
    load_tokenizer,
    resolve_torch_device,
    stage_tokenizer_for_checkpoints,
    tokenizer_sha256,
    train_language_model,
)

from .config import ExperimentConfig, with_step_override


def run(
    *,
    preset: str,
    data_root: Path,
    checkpoint_root: Path,
    device: str,
    steps: int | None = None,
) -> Path:
    tokenizer_path = data_root / "processed" / "tokenizer.json"
    metadata_path = data_root / "processed" / "metadata.json"
    if not tokenizer_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("prepared token streams are missing; run experiments.llm.prepare_data")
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_fingerprint = tokenizer_sha256(tokenizer_path)
    experiment = ExperimentConfig(preset=preset, data_root=data_root, checkpoint_root=checkpoint_root)
    model_config = experiment.model(tokenizer.vocab_size)
    training_config = with_step_override(experiment.training(device), steps)
    resolved = resolve_torch_device(device)
    train_stream = TokenStream(data_root / "processed" / "train.bin", model_config.block_size)
    validation_stream = TokenStream(data_root / "processed" / "validation.bin", model_config.block_size)
    train_generator = torch.Generator(device="cpu").manual_seed(experiment.seed)
    validation_generator = torch.Generator(device="cpu").manual_seed(experiment.seed + 1)

    def training_batch():
        return train_stream.batch(
            training_config.batch_size, generator=train_generator, device=resolved
        )

    def validation_batch():
        return validation_stream.batch(
            training_config.batch_size, generator=validation_generator, device=resolved
        )

    model = GPT(model_config)
    output_dir = checkpoint_root / f"pretrain-{preset}"
    staged_tokenizer_path = stage_tokenizer_for_checkpoints(tokenizer_path, output_dir)
    corpus_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_language_model(
        model,
        training_batch,
        validation_batch,
        training_config,
        output_dir=output_dir,
        metadata={
            "stage": "pretraining",
            "tokenizer_path": str(staged_tokenizer_path),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_sha256": tokenizer_fingerprint,
            "corpus_metadata": corpus_metadata,
            "parameter_count": model.parameter_count,
        },
    )
    return output_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "cpu", "mps", "colab"), default="mps")
    parser.add_argument("--data-root", type=Path, default=Path("data/llm"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/llm"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    print(run(**vars(args)))


if __name__ == "__main__":
    main()
