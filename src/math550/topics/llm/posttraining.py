"""Reward modeling and PPO-style RLHF objectives built from raw PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import GPT


def _last_valid_index(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (batch, time)")
    if torch.any(attention_mask.long().sum(dim=1) < 1):
        raise ValueError("each sequence must contain at least one valid token")
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device)
    positions = positions.expand(attention_mask.size(0), -1)
    return positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values


class RewardModel(nn.Module):
    """A GPT backbone plus a scalar score at the final non-padding token."""

    def __init__(self, backbone: GPT) -> None:
        super().__init__()
        self.backbone = backbone
        self.score_head = nn.Linear(backbone.config.n_embd, 1, bias=False)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids, return_hidden=True)
        if output.hidden_states is None:
            raise RuntimeError("backbone did not return hidden states")
        indices = _last_valid_index(attention_mask)
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        final_hidden = output.hidden_states[rows, indices]
        return self.score_head(final_hidden).squeeze(-1)


def preference_loss(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """Bradley-Terry negative log likelihood for pairwise preferences."""

    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError("chosen and rejected rewards must have matching shape")
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def preference_accuracy(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    return (chosen_rewards > rejected_rewards).float().mean()


class PolicyWithValueHead(nn.Module):
    """Shared GPT policy features with a per-token state-value head."""

    def __init__(self, policy: GPT) -> None:
        super().__init__()
        self.policy = policy
        self.value_head = nn.Linear(policy.config.n_embd, 1)
        nn.init.zeros_(self.value_head.bias)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.policy(input_ids, return_hidden=True)
        if output.hidden_states is None:
            raise RuntimeError("policy did not return hidden states")
        values = self.value_head(output.hidden_states).squeeze(-1)
        return output.logits, values


def token_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities for the observed next-token actions."""

    if logits.shape[:-1] != target_ids.shape:
        raise ValueError("logits time dimensions must match target_ids")
    return F.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have matching shape")
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def build_token_rewards(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    terminal_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    kl_coefficient: float,
) -> torch.Tensor:
    """Apply a sampled KL penalty per action and terminal reward at sequence end."""

    if policy_log_probs.shape != reference_log_probs.shape or policy_log_probs.shape != action_mask.shape:
        raise ValueError("log probabilities and action mask must match")
    if terminal_rewards.shape != (policy_log_probs.size(0),):
        raise ValueError("terminal_rewards must have shape (batch,)")
    rewards = -kl_coefficient * (policy_log_probs - reference_log_probs)
    rewards = rewards * action_mask.to(rewards.dtype)
    last_indices = _last_valid_index(action_mask)
    rows = torch.arange(rewards.size(0), device=rewards.device)
    rewards[rows, last_indices] += terminal_rewards
    return rewards


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked GAE and returns by a transparent reverse-time recursion."""

    if rewards.shape != values.shape or rewards.shape != mask.shape:
        raise ValueError("rewards, values, and mask must match")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(rewards.size(0), dtype=rewards.dtype, device=rewards.device)
    for time_index in range(rewards.size(1) - 1, -1, -1):
        valid = mask[:, time_index].to(rewards.dtype)
        if time_index + 1 < rewards.size(1):
            next_value = values[:, time_index + 1]
            next_valid = mask[:, time_index + 1].to(rewards.dtype)
        else:
            next_value = torch.zeros_like(running)
            next_valid = torch.zeros_like(running)
        delta = rewards[:, time_index] + gamma * next_value * next_valid - values[:, time_index]
        running = (delta + gamma * gae_lambda * running * next_valid) * valid
        advantages[:, time_index] = running
    returns = advantages + values
    return advantages, returns


@dataclass
class PPOLoss:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    entropy: torch.Tensor | None = None,
    clip_ratio: float = 0.2,
    value_clip: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> PPOLoss:
    """Clipped PPO policy/value objective on generated response tokens."""

    shapes = {
        tensor.shape
        for tensor in (new_log_probs, old_log_probs, advantages, new_values, old_values, returns, action_mask)
    }
    if len(shapes) != 1:
        raise ValueError("all PPO tensors must have matching shape")
    normalized_advantages = advantages
    selected = advantages[action_mask]
    if selected.numel() > 1:
        normalized_advantages = (advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8)
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * normalized_advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * normalized_advantages
    policy = -masked_mean(torch.minimum(unclipped, clipped), action_mask)

    clipped_values = old_values + (new_values - old_values).clamp(-value_clip, value_clip)
    value_error = torch.maximum((new_values - returns).square(), (clipped_values - returns).square())
    value = 0.5 * masked_mean(value_error, action_mask)
    entropy_term = torch.zeros((), device=new_log_probs.device)
    if entropy is not None:
        entropy_term = masked_mean(entropy, action_mask)
    total = policy + value_coefficient * value - entropy_coefficient * entropy_term
    approximate_kl = 0.5 * masked_mean(log_ratio.square(), action_mask)
    clip_fraction = masked_mean((torch.abs(ratio - 1.0) > clip_ratio).float(), action_mask)
    return PPOLoss(total, policy, value, entropy_term, approximate_kl, clip_fraction)
