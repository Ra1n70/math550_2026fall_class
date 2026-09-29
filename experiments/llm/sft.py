"""Supervised fine-tune a pretrained course GPT on Databricks Dolly 15k."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from math550.topics.llm import (
    SFTDataset,
    TrainingConfig,
    build_adamw,
    deterministic_split,
    load_dolly_jsonl,
    load_gpt_checkpoint,
    load_tokenizer,
    resolve_generation_vocab_size,
    resolve_torch_device,
    save_checkpoint,
    seed_everything,
    stage_tokenizer_for_checkpoints,
    train_supervised_batches,
    tokenizer_sha256,
)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["targets"].to(device))
        if output.loss is None:
            raise RuntimeError("SFT model did not return a loss")
        losses.append(float(output.loss.cpu()))
    return sum(losses) / len(losses)


def run(
    checkpoint: Path,
    *,
    data_root: Path,
    output_dir: Path,
    device: str = "auto",
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    limit: int | None = None,
) -> Path:
    seed_everything(550)
    resolved = resolve_torch_device(device)
    model, source_payload = load_gpt_checkpoint(checkpoint, resolved)
    tokenizer_path = checkpoint.parent / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_fingerprint = tokenizer_sha256(tokenizer_path)
    resolve_generation_vocab_size(
        model_vocab_size=model.config.vocab_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
        checkpoint_metadata=source_payload.get("metadata"),
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    stage_tokenizer_for_checkpoints(tokenizer_path, output_dir)
    examples = load_dolly_jsonl(data_root / "raw" / "databricks-dolly-15k.jsonl", limit=limit)
    training_examples, validation_examples = deterministic_split(examples, 0.1, 550)
    max_length = model.config.block_size
    training_data = SFTDataset(training_examples, tokenizer, max_length)
    validation_data = SFTDataset(validation_examples, tokenizer, max_length)
    generator = torch.Generator().manual_seed(550)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    optimization = TrainingConfig(
        max_steps=max(1, epochs * len(training_loader)),
        batch_size=batch_size,
        learning_rate=learning_rate,
        min_learning_rate=learning_rate,
        warmup_steps=0,
        weight_decay=0.01,
        device=device,
        seed=550,
    )
    optimizer = build_adamw(model, optimization)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        training_loss = train_supervised_batches(
            model, training_loader, optimizer=optimizer, device=resolved
        )
        validation_loss = evaluate(model, validation_loader, resolved)
        history.append(
            {
                "epoch": epoch + 1,
                "train_assistant_loss": training_loss,
                "validation_assistant_loss": validation_loss,
                "validation_perplexity": math.exp(min(validation_loss, 20.0)),
            }
        )
        print(
            f"epoch={epoch + 1} train={training_loss:.4f} valid={validation_loss:.4f}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "sft.pt",
        model,
        optimizer,
        step=epochs * len(training_loader),
        training_config=optimization,
        metadata={
            "stage": "supervised instruction fine-tuning",
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_sha256": tokenizer_fingerprint,
            "dataset": "databricks/databricks-dolly-15k",
            "training_examples": len(training_data),
            "validation_examples": len(validation_data),
            "loss_scope": "assistant response tokens only",
        },
    )
    (output_dir / "sft_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output_dir / "sft.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/llm"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/llm/sft"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(run(**vars(args)))


if __name__ == "__main__":
    main()
