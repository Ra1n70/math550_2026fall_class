"""Consolidate measured reasoning evidence and produce report-ready figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from math550.topics.reasoning import load_gsm8k_jsonl, load_svamp_json


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "output/reasoning/analysis"
FIGURES = ROOT / "output/reasoning/figures"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "maximum": ordered[-1],
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def run() -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    train = load_gsm8k_jsonl(ROOT / "data/reasoning/gsm8k_train.jsonl", split="train")
    test = load_gsm8k_jsonl(ROOT / "data/reasoning/gsm8k_test.jsonl", split="test")
    svamp = load_svamp_json(ROOT / "data/reasoning/svamp.json")
    sft = _read_json(ROOT / "checkpoints/reasoning/sft-smoke/sft_results.json")
    base = _read_json(ANALYSIS / "base_gsm8k_16.json")
    adapted = _read_json(ANALYSIS / "sft_smoke_gsm8k_16.json")
    self_consistency = _read_json(ANALYSIS / "sft_smoke_self_consistency_8x4.json")
    grpo = _read_json(ROOT / "checkpoints/reasoning/grpo-smoke/grpo_results.json")
    manifest = _read_json(ROOT / "data/reasoning/download_manifest.json")

    dataset_statistics = {
        "gsm8k_train": {
            "rows": len(train),
            "question_words": _distribution([len(row.question.split()) for row in train]),
            "rationale_words": _distribution([len(row.rationale.split()) for row in train]),
            "rationale_lines": _distribution([len(row.rationale.splitlines()) for row in train]),
        },
        "gsm8k_test": {
            "rows": len(test),
            "question_words": _distribution([len(row.question.split()) for row in test]),
            "rationale_words": _distribution([len(row.rationale.split()) for row in test]),
        },
        "svamp": {
            "rows": len(svamp),
            "question_words": _distribution([len(row.question.split()) for row in svamp]),
        },
        "download_bytes": sum(
            int(record["bytes"]) for record in manifest["files"].values()
        ),
    }
    (ANALYSIS / "dataset_statistics.json").write_text(
        json.dumps(dataset_statistics, indent=2), encoding="utf-8"
    )

    summary = {
        "scope": (
            "Short local M2 MPS verification. It validates the real model/data path; "
            "the small fixed test prefix is not a general benchmark."
        ),
        "dataset_statistics": dataset_statistics,
        "sft": sft,
        "greedy_base_16": base,
        "greedy_sft_16": adapted,
        "self_consistency_sft_8_by_4": self_consistency,
        "one_update_grpo": grpo["history"][0],
    }
    summary_path = ANALYSIS / "measured_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (ANALYSIS / "measured_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "condition",
                "problems",
                "samples_per_problem",
                "accuracy",
                "majority_accuracy",
                "pass_at_k",
                "format_rate",
                "mean_tokens",
            ],
        )
        writer.writeheader()
        for name, result in (
            ("base greedy", base),
            ("20-step SFT greedy", adapted),
            ("20-step SFT sample 4", self_consistency),
        ):
            writer.writerow(
                {
                    "condition": name,
                    "problems": result["problems"],
                    "samples_per_problem": result["samples_per_problem"],
                    "accuracy": result["first_sample_accuracy"],
                    "majority_accuracy": result["majority_vote_accuracy"],
                    "pass_at_k": result["pass_at_k_oracle_accuracy"],
                    "format_rate": result["marked_format_rate"],
                    "mean_tokens": result["mean_generated_tokens"],
                }
            )

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(
        ["Base", "20-step LoRA"],
        [base["first_sample_accuracy"], adapted["first_sample_accuracy"]],
        color=["#8795A1", "#2878B5"],
        width=0.58,
    )
    ax.errorbar(
        [0, 1],
        [base["first_sample_accuracy"], adapted["first_sample_accuracy"]],
        yerr=[
            [
                base["first_sample_accuracy"] - base["first_sample_accuracy_wilson_95_low"],
                adapted["first_sample_accuracy"] - adapted["first_sample_accuracy_wilson_95_low"],
            ],
            [
                base["first_sample_accuracy_wilson_95_high"] - base["first_sample_accuracy"],
                adapted["first_sample_accuracy_wilson_95_high"] - adapted["first_sample_accuracy"],
            ],
        ],
        fmt="none",
        ecolor="#18212A",
        capsize=5,
        linewidth=1.5,
    )
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Exact-answer accuracy")
    ax.set_title("Tiny held-out probe: 16 fixed GSM8K test problems")
    ax.text(
        0.5,
        -0.22,
        "Wide, overlapping 95% Wilson intervals: pipeline evidence, not a benchmark claim",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#4E5963",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "tiny_accuracy_probe.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axes[0].bar(
        ["Base", "20-step LoRA"],
        [base["marked_format_rate"], adapted["marked_format_rate"]],
        color=["#8795A1", "#2A9D8F"],
        width=0.58,
    )
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Responses ending in #### number")
    axes[0].set_title("Output contract")
    axes[1].bar(
        ["Base", "20-step LoRA"],
        [base["mean_generated_tokens"], adapted["mean_generated_tokens"]],
        color=["#8795A1", "#E07A5F"],
        width=0.58,
    )
    axes[1].set_ylabel("Mean generated tokens")
    axes[1].set_title("Compute per answer")
    fig.suptitle("SFT first learns a useful behavior and stopping format", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "format_and_length.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = ["First sample", "Majority vote", "Oracle pass@4"]
    values = [
        self_consistency["first_sample_accuracy"],
        self_consistency["majority_vote_accuracy"],
        self_consistency["pass_at_k_oracle_accuracy"],
    ]
    ax.bar(labels, values, color=["#8795A1", "#2878B5", "#E9A23B"], width=0.62)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Exact-answer rate")
    ax.set_title("Four samples help only if a selector can find the good path (n=8)")
    fig.tight_layout()
    fig.savefig(FIGURES / "inference_time_scaling.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    train_words = [len(row.rationale.split()) for row in train]
    ax.hist(train_words, bins=35, color="#2878B5", alpha=0.88)
    ax.axvline(statistics.median(train_words), color="#E07A5F", linewidth=2, label="median")
    ax.set_xlabel("Words in worked solution")
    ax.set_ylabel("GSM8K training problems")
    ax.set_title("Human rationales provide dense supervised reasoning text")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "rationale_length_distribution.png", bbox_inches="tight")
    plt.close(fig)
    return summary_path


if __name__ == "__main__":
    print(run())
