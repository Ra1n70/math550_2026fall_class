"""Evaluate greedy or self-consistent math answers on untouched benchmark rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from math550.topics.reasoning import (
    build_completion_records,
    generate_completion_groups,
    load_gsm8k_jsonl,
    load_open_model,
    load_reasoning_adapter,
    load_svamp_json,
    summarize_completion_records,
)

from .config import ReasoningExperimentConfig


def run(
    *,
    dataset: str,
    adapter: Path | None,
    device: str,
    limit: int,
    samples: int,
    max_new_tokens: int,
    output_name: str,
    model_profile: str = "qwen-0.5b",
    prompt_style: str = "cot",
) -> Path:
    config = ReasoningExperimentConfig(preset="mps", model_profile=model_profile)
    if dataset == "gsm8k":
        examples = load_gsm8k_jsonl(config.data_root / "gsm8k_test.jsonl", split="test")
    else:
        examples = load_svamp_json(config.data_root / "svamp.json")
    examples = examples[:limit]
    spec = config.model()
    if adapter is None:
        model, tokenizer, resolved = load_open_model(spec, device=device, trainable_lora=False)
        model_label = "base"
    else:
        model, tokenizer, resolved = load_reasoning_adapter(adapter, spec, device=device)
        model_label = str(adapter)
    completions, token_counts, latencies = generate_completion_groups(
        model,
        tokenizer,
        examples,
        device=resolved,
        samples_per_problem=samples,
        max_new_tokens=max_new_tokens,
        prompt_style=prompt_style,
    )
    records = build_completion_records(
        examples, completions, generated_tokens=token_counts, latencies=latencies
    )
    metrics = summarize_completion_records(records)
    metrics.update(
        {
            "dataset": dataset,
            "model": model_label,
            "model_profile": model_profile,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "device": str(resolved),
            "max_new_tokens": max_new_tokens,
            "prompt_style": prompt_style,
            "evaluation_prefix": f"first {limit} official rows",
        }
    )
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    result_path = config.analysis_dir / f"{output_name}.json"
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_path = config.analysis_dir / f"{output_name}_completions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].to_dict()))
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
    print(json.dumps(metrics, indent=2))
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gsm8k", "svamp"), default="gsm8k")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output-name", default="evaluation")
    parser.add_argument("--prompt-style", choices=("direct", "cot"), default="cot")
    parser.add_argument(
        "--model-profile",
        choices=("qwen-0.5b", "qwen-math-1.5b", "olmo2-1b-instruct"),
        default="qwen-0.5b",
    )
    args = parser.parse_args()
    run(
        dataset=args.dataset,
        adapter=args.adapter,
        device=args.device,
        limit=args.limit,
        samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        output_name=args.output_name,
        model_profile=args.model_profile,
        prompt_style=args.prompt_style,
    )


if __name__ == "__main__":
    main()
