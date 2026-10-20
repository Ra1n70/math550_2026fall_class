"""Deterministic audit of PPO token ratios, signed clipping, and sequence ratios."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from math550.core import ppo_token_surrogate


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "output/reinforcement/analysis"
FIGURES = ROOT / "output/reinforcement/figures"


def _terms(ratios: torch.Tensor, advantage: float):
    return ppo_token_surrogate(
        ratios.log(),
        torch.zeros_like(ratios),
        torch.full_like(ratios, advantage),
        clip_epsilon=0.2,
    )


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    tokens = ["We", "compute", "48+24", "72"]
    token_ratios = torch.tensor([0.70, 0.95, 1.10, 1.35])
    positive = _terms(token_ratios, 1.5)
    exact_sequence_ratio = float(token_ratios.prod())
    geometric_mean_ratio = float(token_ratios.log().mean().exp())

    example_rows = []
    for index, token in enumerate(tokens):
        example_rows.append(
            {
                "position": index + 1,
                "token": token,
                "token_ratio": float(positive.ratio[index]),
                "unclipped_gradient_coefficient": float(1.5 * positive.ratio[index]),
                "ppo_gradient_coefficient": float(positive.gradient_coefficient[index]),
                "signed_clipping_active": bool(positive.clipping_active[index]),
            }
        )

    ratio_grid = torch.linspace(0.5, 1.5, 101)
    curve_rows = []
    for advantage in (1.0, -1.0):
        terms = _terms(ratio_grid, advantage)
        for ratio, surrogate, coefficient, active in zip(
            terms.ratio,
            terms.surrogate,
            terms.gradient_coefficient,
            terms.clipping_active,
            strict=True,
        ):
            curve_rows.append(
                {
                    "advantage": advantage,
                    "token_ratio": float(ratio),
                    "clipped_surrogate": float(surrogate),
                    "score_gradient_coefficient": float(coefficient),
                    "signed_clipping_active": bool(active),
                }
            )

    payload = {
        "scope": "deterministic estimator mechanics; not model-quality evidence",
        "clip_epsilon": 0.2,
        "sequence_advantage": 1.5,
        "token_example": example_rows,
        "exact_sequence_importance_ratio": exact_sequence_ratio,
        "geometric_mean_sequence_ratio": geometric_mean_ratio,
        "interpretation": (
            "PPO reuses tokens sampled from pi_old. Each live token gradient is scaled by "
            "A_t * pi_theta/pi_old; signed clipping sets that coefficient to zero only in "
            "the saturation direction. The product of token ratios is the exact sequence ratio."
        ),
    }
    (ANALYSIS / "ppo_token_reweighting.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (ANALYSIS / "ppo_token_reweighting.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = ["#64C7D0", "#3B82F6", "#17A673", "#E07A5F"]
    axes[0].bar(np.arange(len(tokens)), token_ratios.numpy(), color=colors)
    axes[0].axhspan(0.8, 1.2, color="#E8F4FF", alpha=0.8, label="PPO ratio window")
    axes[0].axhline(1.0, color="#1F2937", linewidth=1)
    axes[0].set_xticks(np.arange(len(tokens)), tokens)
    axes[0].set_ylim(0.5, 1.5)
    axes[0].set_ylabel(r"selected-token ratio $\pi_\theta/\pi_{old}$")
    axes[0].set_title("One response: every token has its own ratio")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(
        3,
        1.39,
        "signed clip active",
        ha="center",
        va="bottom",
        color="#B74435",
        fontsize=9,
    )

    for advantage, color, label in (
        (1.0, "#147D64", r"$\hat A=+1$"),
        (-1.0, "#E07A5F", r"$\hat A=-1$"),
    ):
        terms = _terms(ratio_grid, advantage)
        axes[1].plot(
            ratio_grid.numpy(),
            terms.gradient_coefficient.numpy(),
            color=color,
            linewidth=2.5,
            label=label,
        )
    axes[1].axvline(0.8, color="#94A3B8", linestyle="--", linewidth=1)
    axes[1].axvline(1.2, color="#94A3B8", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="#1F2937", linewidth=1)
    axes[1].set_xlabel(r"token ratio $r_t(\theta)$")
    axes[1].set_ylabel(r"coefficient on $\nabla\log\pi_\theta(y_t|s_t)$")
    axes[1].set_title("The advantage sign selects the saturation side")
    axes[1].legend(frameon=False)
    fig.suptitle("PPO is token-level importance reweighting with a pessimistic clip", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "ppo_token_reweighting.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
