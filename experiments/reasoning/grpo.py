"""Run explicit online GRPO updates with exact-answer rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from math550.topics.reasoning import (
    grpo_update,
    load_gsm8k_jsonl,
    load_open_model,
    load_reasoning_adapter,
)

from .config import ReasoningExperimentConfig


def run(
    *,
    preset: str,
    device: str,
    adapter: Path | None,
    updates: int,
    prompts_per_update: int,
    model_profile: str = "qwen-0.5b",
    variant: str = "grpo",
    prompt_style: str = "cot",
) -> Path:
    config = ReasoningExperimentConfig(preset=preset, model_profile=model_profile)
    examples = load_gsm8k_jsonl(config.data_root / "gsm8k_train.jsonl", split="train")
    spec = config.model()
    if adapter is None:
        model, tokenizer, resolved = load_open_model(spec, device=device, trainable_lora=True)
    else:
        model, tokenizer, resolved = load_reasoning_adapter(
            adapter, spec, device=device, trainable=True
        )
    history: list[dict[str, object]] = []
    for update_index in range(updates):
        start = (update_index * prompts_per_update) % len(examples)
        prompts = examples[start : start + prompts_per_update]
        result = grpo_update(
            model,
            tokenizer,
            prompts,
            config.grpo(variant, prompt_style=prompt_style),
            device=resolved,
        )
        result["update"] = update_index + 1
        history.append(result)
        print(
            f"grpo update={update_index + 1:3d} reward={result['mean_reward']:.3f} "
            f"loss={result['loss']:.4f} kl={result['reference_kl']:.5f}"
        )
    profile_suffix = "" if model_profile == "qwen-0.5b" else f"-{model_profile}"
    output_dir = config.checkpoint_root / f"{variant}-{preset}{profile_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    result_path = output_dir / "grpo_results.json"
    result_path.write_text(
        json.dumps(
            {
                "preset": preset,
                "device": str(resolved),
                "model_profile": model_profile,
                "model_id": spec.model_id,
                "model_revision": spec.revision,
                "updates": updates,
                "prompts_per_update": prompts_per_update,
                "variant": variant,
                "prompt_style": prompt_style,
                "history": history,
                "interpretation": (
                    "A finite short run verifies the online objective. Capability claims require "
                    "held-out evaluation before and after a substantially longer run."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "quick", "mps", "colab"), default="quick")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--prompts-per-update", type=int, default=2)
    parser.add_argument(
        "--model-profile",
        choices=("qwen-0.5b", "qwen-math-1.5b", "olmo2-1b-instruct"),
        default="qwen-0.5b",
    )
    parser.add_argument(
        "--variant",
        choices=("grpo", "grpo-constant", "dr-grpo", "rft", "maxrl", "gspo"),
        default="grpo",
    )
    parser.add_argument("--prompt-style", choices=("direct", "cot"), default="cot")
    args = parser.parse_args()
    print(
        run(
            preset=args.preset,
            device=args.device,
            adapter=args.adapter,
            updates=args.updates,
            prompts_per_update=args.prompts_per_update,
            model_profile=args.model_profile,
            variant=args.variant,
            prompt_style=args.prompt_style,
        )
    )


if __name__ == "__main__":
    main()
