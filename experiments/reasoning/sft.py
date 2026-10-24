"""LoRA supervised fine-tuning on human-written GSM8K worked solutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from math550.topics.reasoning import (
    ReasoningCollator,
    ReasoningSFTDataset,
    load_gsm8k_jsonl,
    load_open_model,
    parameter_summary,
    split_training_validation,
    train_sft,
)

from .config import ReasoningExperimentConfig


def run(
    *,
    preset: str,
    device: str,
    training_limit: int | None = None,
    max_steps: int | None = None,
    model_profile: str = "qwen-0.5b",
) -> Path:
    config = ReasoningExperimentConfig(preset=preset, model_profile=model_profile)
    examples = load_gsm8k_jsonl(config.data_root / "gsm8k_train.jsonl", split="train")
    training, validation = split_training_validation(examples, validation_size=512, seed=config.seed)
    limit = training_limit if training_limit is not None else config.training_limit()
    if limit is not None:
        training = training[:limit]
    model, tokenizer, resolved = load_open_model(config.model(), device=device, trainable_lora=True)
    training_dataset = ReasoningSFTDataset(
        training, tokenizer, max_length=config.max_length()
    )
    validation_dataset = ReasoningSFTDataset(
        validation[:64], tokenizer, max_length=config.max_length()
    )
    collator = ReasoningCollator(tokenizer.pad_token_id)
    sft_config = config.sft()
    if max_steps is not None:
        from dataclasses import replace

        sft_config = replace(sft_config, max_steps=max_steps)
    profile_suffix = "" if model_profile == "qwen-0.5b" else f"-{model_profile}"
    output_dir = config.checkpoint_root / f"sft-{preset}{profile_suffix}"
    result = train_sft(
        model,
        training_dataset,
        validation_dataset,
        collator,
        sft_config,
        device=resolved,
        output_dir=output_dir,
        tokenizer=tokenizer,
    )
    result["model"] = {**config.model().__dict__, **parameter_summary(model)}
    result["data"] = {
        "official_training_rows": len(examples),
        "internal_validation_rows": len(validation),
        "training_rows_used": len(training),
        "official_test_touched_during_training": False,
    }
    result_path = output_dir / "sft_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "quick", "mps", "colab"), default="mps")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--training-limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--model-profile",
        choices=("qwen-0.5b", "qwen-math-1.5b", "olmo2-1b-instruct"),
        default="qwen-0.5b",
    )
    args = parser.parse_args()
    run(
        preset=args.preset,
        device=args.device,
        training_limit=args.training_limit,
        max_steps=args.max_steps,
        model_profile=args.model_profile,
    )


if __name__ == "__main__":
    main()
