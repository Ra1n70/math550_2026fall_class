"""Reasoning-specific policy objectives implemented directly in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from math550.core.alignment import (
    compute_group_advantages,
    masked_mean as _masked_mean,
    policy_gradient_loss,
    reference_kl_estimate,
)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(values, mask)


def response_token_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather shifted next-token log probabilities on generated response tokens."""

    if logits.ndim != 3 or input_ids.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("expected logits (batch,time,vocab) and two (batch,time) tensors")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != response_mask.shape:
        raise ValueError("logits, input_ids, and response_mask have incompatible shapes")
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    shifted_mask = response_mask[:, 1:].bool()
    log_probs = F.log_softmax(shifted_logits.float(), dim=-1)
    selected = log_probs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
    return selected, shifted_mask


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    group_size: int,
    baseline: str = "mean",
    normalizer: str = "std",
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Expose the baseline/normalizer choices behind named RLVR variants."""

    return compute_group_advantages(
        rewards,
        group_size=group_size,
        baseline=baseline,  # type: ignore[arg-type]
        normalizer=normalizer,  # type: ignore[arg-type]
        epsilon=epsilon,
    ).advantages


@dataclass(frozen=True)
class GRPOLoss:
    total: torch.Tensor
    policy: torch.Tensor
    reference_kl: torch.Tensor
    clip_fraction: torch.Tensor
    approximate_old_kl: torch.Tensor


def grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
    importance_method: str = "grpo",
    normalization: str = "sequence",
    normalization_constant: float | None = None,
) -> GRPOLoss:
    """Policy-gradient surrogate plus a frozen-reference KL penalty.

    The defaults implement sequence-normalized, token-clipped GRPO.  Passing
    other importance and denominator choices turns the same primitive into the
    on-policy variants and GSPO ablation taught in the module.
    """

    if not (
        new_log_probs.shape == old_log_probs.shape == reference_log_probs.shape == action_mask.shape
    ):
        raise ValueError("log-probability tensors and action_mask must match")
    if advantages.ndim != 1 or advantages.shape[0] != new_log_probs.shape[0]:
        raise ValueError("advantages must have one value per sampled completion")
    if not 0.0 < clip_epsilon < 1.0 or beta < 0:
        raise ValueError("invalid GRPO clipping or KL coefficient")
    objective = policy_gradient_loss(
        new_log_probs,
        advantages,
        action_mask,
        old_log_probs=old_log_probs,
        importance_method=importance_method,  # type: ignore[arg-type]
        clip_epsilon=clip_epsilon,
        normalization=normalization,  # type: ignore[arg-type]
        normalization_constant=normalization_constant,
    )
    policy = objective.loss
    reference_kl = reference_kl_estimate(
        new_log_probs,
        reference_log_probs,
        action_mask,
        normalization=normalization,  # type: ignore[arg-type]
        normalization_constant=normalization_constant,
    )
    total = policy + beta * reference_kl
    clip_fraction = objective.clip_fraction
    log_ratio = new_log_probs - old_log_probs
    approximate_old_kl = 0.5 * masked_mean(log_ratio.square(), action_mask)
    return GRPOLoss(total, policy, reference_kl, clip_fraction, approximate_old_kl)
