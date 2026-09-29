"""Train a pairwise reward model on Anthropic HH-RLHF harmless-base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from math550.topics.llm import (
    PreferenceDataset,
    RewardModel,
    deterministic_split,
    load_gpt_checkpoint,
    load_hh_preferences,
    load_tokenizer,
    preference_accuracy,
    preference_loss,
    resolve_torch_device,
    seed_everything,
)


@torch.no_grad()
def evaluate(model: RewardModel, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    for batch in loader:
        chosen = model(batch["chosen_ids"].to(device), batch["chosen_mask"].to(device))
        rejected = model(batch["rejected_ids"].to(device), batch["rejected_mask"].to(device))
        losses.append(float(preference_loss(chosen, rejected).cpu()))
        accuracies.append(float(preference_accuracy(chosen, rejected).cpu()))
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def run(
    checkpoint: Path,
    *,
    data_root: Path,
    output_dir: Path,
    device: str = "auto",
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    limit: int | None = 5_000,
) -> Path:
    seed_everything(550)
    resolved = resolve_torch_device(device)
    backbone, _ = load_gpt_checkpoint(checkpoint, resolved)
    tokenizer_path = checkpoint.parent / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    examples = load_hh_preferences(
        data_root / "raw" / "hh-rlhf-harmless-base-train.jsonl.gz", limit=limit
    )
    training_examples, validation_examples = deterministic_split(examples, 0.1, 550)
    training_data = PreferenceDataset(training_examples, tokenizer, backbone.config.block_size)
    validation_data = PreferenceDataset(validation_examples, tokenizer, backbone.config.block_size)
    generator = torch.Generator().manual_seed(550)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    model = RewardModel(backbone).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in training_loader:
            optimizer.zero_grad(set_to_none=True)
            chosen = model(batch["chosen_ids"].to(resolved), batch["chosen_mask"].to(resolved))
            rejected = model(batch["rejected_ids"].to(resolved), batch["rejected_mask"].to(resolved))
            loss = preference_loss(chosen, rejected)
            if not torch.isfinite(loss):
                raise RuntimeError("reward modeling produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
        validation_loss, validation_accuracy = evaluate(model, validation_loader, resolved)
        row = {
            "epoch": epoch + 1,
            "training_preference_loss": running_loss / len(training_loader),
            "validation_preference_loss": validation_loss,
            "validation_preference_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"epoch={epoch + 1} loss={row['training_preference_loss']:.4f} "
            f"valid_accuracy={validation_accuracy:.3f}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reward_model.pt"
    torch.save(
        {
            "backbone_config": backbone.config.to_dict(),
            "reward_model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "metadata": {
                "dataset": "Anthropic/hh-rlhf harmless-base",
                "training_pairs": len(training_data),
                "validation_pairs": len(validation_data),
                "content_warning": "contains harmful prompts and unsafe rejected responses",
            },
        },
        output_path,
    )
    shutil.copyfile(tokenizer_path, output_dir / "tokenizer.json")
    (output_dir / "reward_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/llm"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/llm/reward"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--limit", type=int, default=5_000)
    args = parser.parse_args()
    print(run(**vars(args)))


if __name__ == "__main__":
    main()
