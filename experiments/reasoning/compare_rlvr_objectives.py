"""Create a deterministic audit of RLVR estimator and denominator choices."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from math550.core.alignment import compute_group_advantages, policy_gradient_loss


def _variant_record(
    name: str,
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    mask: torch.Tensor,
    *,
    baseline: str,
    normalizer: str,
    normalization: str,
) -> dict[str, object]:
    advantages = compute_group_advantages(
        rewards,
        baseline=baseline,  # type: ignore[arg-type]
        normalizer=normalizer,  # type: ignore[arg-type]
    ).advantages.reshape(-1)
    live = old_log_probs.clone().requires_grad_(True)
    result = policy_gradient_loss(
        live,
        advantages,
        mask,
        importance_method="none",
        normalization=normalization,  # type: ignore[arg-type]
        normalization_constant=float(mask.numel()) if normalization == "constant" else None,
    )
    result.loss.backward()
    return {
        "variant": name,
        "baseline": baseline,
        "advantage_normalizer": normalizer,
        "loss_normalization": normalization,
        "advantages": advantages.tolist(),
        "zero_advantage_fraction": float((advantages == 0).float().mean()),
        "loss": float(result.loss.detach()),
        "gradient_l2": float(live.grad.norm()),
    }


def run(output_root: Path = Path("output/reasoning")) -> Path:
    torch.manual_seed(550)
    # Three prompt groups: mixed/easy, mixed/hard, and all-wrong.  This makes
    # prompt-difficulty weighting and zero-signal pruning visible in one table.
    rewards = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    old_log_probs = torch.tensor(
        [
            [-0.2, -0.3, -0.4, -0.5],
            [-0.4, -0.5, -0.6, -0.7],
            [-0.3, -0.4, -0.5, -0.6],
            [-0.7, -0.8, -0.9, -1.0],
        ]
        * 3
    )
    lengths = torch.tensor([4, 3, 2, 1] * 3)
    mask = torch.arange(4)[None, :] < lengths[:, None]

    variants = [
        _variant_record(
            "GRPO",
            rewards,
            old_log_probs,
            mask,
            baseline="mean",
            normalizer="std",
            normalization="sequence",
        ),
        _variant_record(
            "GRPO constant",
            rewards,
            old_log_probs,
            mask,
            baseline="mean",
            normalizer="std",
            normalization="constant",
        ),
        _variant_record(
            "Dr. GRPO",
            rewards,
            old_log_probs,
            mask,
            baseline="mean",
            normalizer="none",
            normalization="constant",
        ),
        _variant_record(
            "RFT",
            rewards,
            old_log_probs,
            mask,
            baseline="none",
            normalizer="none",
            normalization="constant",
        ),
        _variant_record(
            "MaxRL",
            rewards,
            old_log_probs,
            mask,
            baseline="mean",
            normalizer="mean",
            normalization="constant",
        ),
    ]

    stale_log_probs = old_log_probs + torch.linspace(-0.35, 0.35, old_log_probs.numel()).reshape_as(
        old_log_probs
    )
    shared_advantages = compute_group_advantages(rewards, normalizer="std").advantages.reshape(-1)
    off_policy: list[dict[str, object]] = []
    for method in ("noclip", "grpo", "gspo", "cispo"):
        result = policy_gradient_loss(
            stale_log_probs,
            shared_advantages,
            mask,
            old_log_probs=old_log_probs,
            importance_method=method,  # type: ignore[arg-type]
            clip_epsilon=0.2,
            normalization="sequence",
        )
        selected_weights = result.importance_weight[mask]
        off_policy.append(
            {
                "method": method,
                "loss": float(result.loss),
                "clip_fraction": float(result.clip_fraction),
                "importance_mean": float(selected_weights.mean()),
                "importance_min": float(selected_weights.min()),
                "importance_max": float(selected_weights.max()),
            }
        )

    eta = torch.tensor([0.1, 0.5, 0.9])
    difficulty_weights = {
        "success_probability": eta.tolist(),
        "dr_grpo": torch.ones_like(eta).tolist(),
        "grpo_std": (eta * (1 - eta)).rsqrt().tolist(),
        "maxrl": eta.reciprocal().tolist(),
    }
    result = {
        "scope": (
            "Fixed-tensor behavior audit. It verifies estimator mechanics and induced weighting; "
            "it is not evidence that one RLVR algorithm trains a better language model."
        ),
        "seed": 550,
        "group_rewards": rewards.tolist(),
        "sequence_lengths": lengths.tolist(),
        "variants": variants,
        "off_policy": off_policy,
        "asymptotic_binary_prompt_weights": difficulty_weights,
        "reference": {
            "stanford_cs336_revision": "c2734a26308710949fe13226960a1e8cece94b7e",
            "handout_sha256": "c93d503a69aef4aa7519b7fc664abebbf2ebf5e5e88379a7ed92f4c209a2d26a",
        },
    }
    analysis_dir = output_root / "analysis"
    figure_dir = output_root / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / "rlvr_objective_audit.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (analysis_dir / "rlvr_objective_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "variant",
                "baseline",
                "advantage_normalizer",
                "loss_normalization",
                "zero_advantage_fraction",
                "loss",
                "gradient_l2",
            ],
        )
        writer.writeheader()
        for row in variants:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].bar(
        [str(row["variant"]) for row in variants],
        [float(row["gradient_l2"]) for row in variants],
        color=["#2F80ED", "#56CCF2", "#27AE60", "#F2994A", "#9B51E0"],
    )
    axes[0].set_ylabel("gradient $L_2$ norm")
    axes[0].set_title("Same rewards, different estimator scales")
    axes[0].tick_params(axis="x", rotation=28)
    for label, values, color in (
        ("Dr. GRPO", difficulty_weights["dr_grpo"], "#27AE60"),
        ("GRPO / std", difficulty_weights["grpo_std"], "#2F80ED"),
        ("MaxRL / mean", difficulty_weights["maxrl"], "#9B51E0"),
    ):
        axes[1].plot(eta, values, marker="o", label=label, color=color)
    axes[1].set_xlabel("prompt success probability $\\eta$")
    axes[1].set_ylabel("relative prompt weight")
    axes[1].set_title("Normalizer induces difficulty weighting")
    axes[1].legend(frameon=False)
    figure_path = figure_dir / "rlvr_estimator_tradeoffs.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(output_path)
    return output_path


if __name__ == "__main__":
    run()
