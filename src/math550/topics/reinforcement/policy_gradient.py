"""Explicit policy-gradient estimators, PPO training, and GRPO objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from math550.core.alignment import (
    compute_group_advantages,
    policy_gradient_loss,
    ppo_token_surrogate,
    reference_kl_estimate,
)

from .data import MinAtarAdapter
from .devices import resolve_torch_device


def _as_float_tensor(values: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.float()
    return torch.as_tensor(values, dtype=torch.float32)


def discounted_returns(
    rewards: torch.Tensor | np.ndarray,
    terminated: torch.Tensor | np.ndarray,
    discount: float = 0.99,
    bootstrap_value: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Reward-to-go with terminal masks, used by the REINFORCE estimator."""

    rewards_tensor = _as_float_tensor(rewards)
    terminated_tensor = _as_float_tensor(terminated).to(rewards_tensor.device)
    if rewards_tensor.ndim != 1 or terminated_tensor.shape != rewards_tensor.shape:
        raise ValueError("rewards and terminated must be one-dimensional and aligned")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    running = torch.as_tensor(
        bootstrap_value, dtype=rewards_tensor.dtype, device=rewards_tensor.device
    )
    returns = torch.empty_like(rewards_tensor)
    for index in range(len(rewards_tensor) - 1, -1, -1):
        running = rewards_tensor[index] + discount * running * (
            1.0 - terminated_tensor[index]
        )
        returns[index] = running
    return returns


def reinforce_loss(
    action_log_probabilities: torch.Tensor,
    returns: torch.Tensor,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative likelihood-ratio objective for gradient descent."""

    if action_log_probabilities.shape != returns.shape:
        raise ValueError("log probabilities and returns must have matching shapes")
    weights = returns if baseline is None else returns - baseline
    if weights.shape != returns.shape:
        raise ValueError("baseline must match returns")
    return -(action_log_probabilities * weights.detach()).mean()


def generalized_advantage_estimate(
    rewards: torch.Tensor | np.ndarray,
    values: torch.Tensor | np.ndarray,
    terminated: torch.Tensor | np.ndarray,
    next_values: torch.Tensor | np.ndarray,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE and value targets for tensors shaped ``time x environments``."""

    rewards_tensor = _as_float_tensor(rewards)
    values_tensor = _as_float_tensor(values).to(rewards_tensor.device)
    terminated_tensor = _as_float_tensor(terminated).to(rewards_tensor.device)
    next_values_tensor = _as_float_tensor(next_values).to(rewards_tensor.device)
    squeeze = rewards_tensor.ndim == 1
    if squeeze:
        rewards_tensor = rewards_tensor[:, None]
        values_tensor = values_tensor[:, None]
        terminated_tensor = terminated_tensor[:, None]
        next_values_tensor = next_values_tensor.reshape(1)
    if rewards_tensor.ndim != 2:
        raise ValueError("rewards must have shape time or time x environments")
    if values_tensor.shape != rewards_tensor.shape:
        raise ValueError("values must match rewards")
    if terminated_tensor.shape != rewards_tensor.shape:
        raise ValueError("terminated must match rewards")
    if next_values_tensor.shape != rewards_tensor.shape[1:]:
        raise ValueError("next_values must have one value per environment")
    if not 0.0 <= discount <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("discount and gae_lambda must lie in [0, 1]")

    advantages = torch.zeros_like(rewards_tensor)
    running = torch.zeros_like(next_values_tensor)
    following_values = next_values_tensor
    for time_index in range(len(rewards_tensor) - 1, -1, -1):
        nonterminal = 1.0 - terminated_tensor[time_index]
        delta = (
            rewards_tensor[time_index]
            + discount * following_values * nonterminal
            - values_tensor[time_index]
        )
        running = delta + discount * gae_lambda * nonterminal * running
        advantages[time_index] = running
        following_values = values_tensor[time_index]
    value_targets = advantages + values_tensor
    if squeeze:
        return advantages[:, 0], value_targets[:, 0]
    return advantages, value_targets


def ppo_clipped_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    value_targets: torch.Tensor,
    entropy: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO's clipped policy surrogate plus clipped value and entropy terms."""

    expected = new_log_probabilities.shape
    tensors = (
        old_log_probabilities,
        advantages,
        new_values,
        old_values,
        value_targets,
        entropy,
    )
    if any(tensor.shape != expected for tensor in tensors):
        raise ValueError("all PPO tensors must have matching shapes")
    if clip_epsilon <= 0 or value_clip_epsilon <= 0:
        raise ValueError("clipping thresholds must be positive")
    token_terms = ppo_token_surrogate(
        new_log_probabilities,
        old_log_probabilities,
        advantages,
        clip_epsilon=clip_epsilon,
    )
    log_ratio = new_log_probabilities - old_log_probabilities.detach()
    ratio = token_terms.ratio
    policy_loss = -token_terms.surrogate.mean()

    clipped_values = old_values + torch.clamp(
        new_values - old_values,
        -value_clip_epsilon,
        value_clip_epsilon,
    )
    squared_error = (new_values - value_targets).square()
    clipped_squared_error = (clipped_values - value_targets).square()
    value_loss = 0.5 * torch.maximum(squared_error, clipped_squared_error).mean()
    entropy_mean = entropy.mean()
    total = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy_mean
    )
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
    return total, {
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy_mean,
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
        "clipping_active_fraction": token_terms.clipping_active.float().mean(),
        "mean_importance_ratio": ratio.mean(),
    }


def group_relative_advantages(
    group_rewards: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Standardize rewards inside each prompt group, as in outcome GRPO."""

    if group_rewards.ndim < 2 or group_rewards.shape[-1] < 2:
        raise ValueError("group_rewards must end in a group dimension of at least two")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return compute_group_advantages(
        group_rewards,
        normalizer="std",
        epsilon=epsilon,
    ).advantages


def grpo_clipped_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    reference_log_probabilities: torch.Tensor,
    group_rewards: torch.Tensor,
    mask: torch.Tensor | None = None,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Outcome-supervision GRPO objective for grouped variable-length samples."""

    if new_log_probabilities.ndim < 3:
        raise ValueError("log probabilities must have shape batch x group x tokens")
    if not (
        old_log_probabilities.shape
        == reference_log_probabilities.shape
        == new_log_probabilities.shape
    ):
        raise ValueError("new, old, and reference log probabilities must align")
    if group_rewards.shape != new_log_probabilities.shape[:-1]:
        raise ValueError("group_rewards must have shape batch x group")
    if clip_epsilon <= 0 or kl_coefficient < 0:
        raise ValueError("invalid GRPO coefficients")
    valid = torch.ones_like(new_log_probabilities, dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != new_log_probabilities.shape:
        raise ValueError("mask must match token log probabilities")

    advantages = group_relative_advantages(group_rewards).detach()
    flat_shape = (-1, new_log_probabilities.shape[-1])
    objective = policy_gradient_loss(
        new_log_probabilities.reshape(flat_shape),
        advantages.reshape(-1),
        valid.reshape(flat_shape),
        old_log_probs=old_log_probabilities.reshape(flat_shape),
        importance_method="grpo",
        clip_epsilon=clip_epsilon,
        normalization="sequence",
    )
    kl_mean = reference_kl_estimate(
        new_log_probabilities.reshape(flat_shape),
        reference_log_probabilities.reshape(flat_shape),
        valid.reshape(flat_shape),
        normalization="sequence",
    )
    policy_loss = objective.loss
    total = policy_loss + kl_coefficient * kl_mean
    clip_fraction = objective.clip_fraction
    return total, {
        "policy": policy_loss,
        "kl": kl_mean,
        "clip_fraction": clip_fraction,
        "group_advantage_mean": advantages.mean(),
    }


class MinAtarActorCritic(nn.Module):
    """Compact shared visual trunk with categorical policy and scalar critic."""

    def __init__(self, input_channels: int = 4, action_count: int = 3) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_count)
        self.value_head = nn.Linear(128, 1)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 4:
            raise ValueError("observations must have shape batch x channels x 10 x 10")
        features = self.trunk(observations)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 200_000
    num_environments: int = 8
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 2.5e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    target_kl: float = 0.03
    max_episode_steps: int = 10_000
    seed: int = 550
    device: str = "auto"

    def validate(self) -> None:
        positive_integers = (
            self.total_steps,
            self.num_environments,
            self.rollout_steps,
            self.update_epochs,
            self.minibatch_size,
            self.max_episode_steps,
        )
        if min(positive_integers) < 1:
            raise ValueError("PPO integer budgets must be positive")
        if self.total_steps % self.num_environments:
            raise ValueError("total_steps must be divisible by num_environments")
        if self.minibatch_size > self.num_environments * self.rollout_steps:
            raise ValueError("minibatch_size cannot exceed one rollout batch")
        if not 0.0 <= self.discount <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("discount and gae_lambda must lie in [0, 1]")
        positive = (
            self.clip_epsilon,
            self.value_clip_epsilon,
            self.max_gradient_norm,
            self.target_kl,
            self.learning_rate,
        )
        if min(positive) <= 0:
            raise ValueError("PPO clipping, optimization, and KL thresholds must be positive")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("PPO loss coefficients must be nonnegative")


@dataclass
class PPOTrainingResult:
    network: MinAtarActorCritic
    episode_returns: list[float]
    episode_lengths: list[int]
    policy_losses: list[float]
    value_losses: list[float]
    entropies: list[float]
    approximate_kls: list[float]
    clip_fractions: list[float]
    explained_variances: list[float]
    steps: int
    device: str


def _observations_to_tensor(
    observations: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    if tensor.ndim != 4:
        raise ValueError("MinAtar observations must be batch x height x width x channels")
    return tensor.permute(0, 3, 1, 2).contiguous()


def train_ppo(
    environments: Sequence[MinAtarAdapter],
    config: PPOConfig,
) -> PPOTrainingResult:
    """Train PPO from explicitly collected MinAtar trajectories."""

    config.validate()
    if len(environments) != config.num_environments:
        raise ValueError("environment count must match config.num_environments")
    if not environments:
        raise ValueError("at least one environment is required")
    first = environments[0]
    if any(
        environment.observation_shape != first.observation_shape
        or environment.action_count != first.action_count
        for environment in environments
    ):
        raise ValueError("all PPO environments must share shapes and actions")

    np_rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_torch_device(config.device)
    network = MinAtarActorCritic(
        input_channels=first.observation_shape[-1],
        action_count=first.action_count,
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, eps=1e-5)
    observations = np.stack([environment.reset() for environment in environments])
    running_returns = np.zeros(config.num_environments, dtype=np.float64)
    running_lengths = np.zeros(config.num_environments, dtype=np.int64)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []
    clip_fractions: list[float] = []
    explained_variances: list[float] = []
    steps_collected = 0

    while steps_collected < config.total_steps:
        remaining_per_environment = (
            config.total_steps - steps_collected
        ) // config.num_environments
        rollout_length = min(config.rollout_steps, remaining_per_environment)
        observation_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        reward_rows: list[np.ndarray] = []
        terminated_rows: list[np.ndarray] = []
        log_probability_rows: list[np.ndarray] = []
        value_rows: list[np.ndarray] = []

        for _ in range(rollout_length):
            observation_rows.append(observations.copy())
            with torch.inference_mode():
                logits, values = network(_observations_to_tensor(observations, device))
                distribution = Categorical(logits=logits)
                actions = distribution.sample()
                log_probabilities = distribution.log_prob(actions)
            actions_numpy = actions.cpu().numpy()
            rewards = np.zeros(config.num_environments, dtype=np.float32)
            terminated = np.zeros(config.num_environments, dtype=np.float32)
            next_observations = []
            for index, environment in enumerate(environments):
                next_observation, reward, environment_terminated = environment.step(
                    int(actions_numpy[index])
                )
                running_returns[index] += reward
                running_lengths[index] += 1
                time_limit = running_lengths[index] >= config.max_episode_steps
                done = bool(environment_terminated or time_limit)
                rewards[index] = reward
                if time_limit and not environment_terminated:
                    # Bootstrap the artificial cap, then cut the GAE trace before reset.
                    with torch.inference_mode():
                        _, timeout_value = network(
                            _observations_to_tensor(next_observation[None], device)
                        )
                    rewards[index] += config.discount * float(timeout_value.item())
                terminated[index] = float(done)
                if done:
                    episode_returns.append(float(running_returns[index]))
                    episode_lengths.append(int(running_lengths[index]))
                    running_returns[index] = 0.0
                    running_lengths[index] = 0
                    next_observation = environment.reset()
                next_observations.append(next_observation)
            action_rows.append(actions_numpy)
            reward_rows.append(rewards)
            terminated_rows.append(terminated)
            log_probability_rows.append(log_probabilities.cpu().numpy())
            value_rows.append(values.cpu().numpy())
            observations = np.stack(next_observations)
            steps_collected += config.num_environments

        with torch.inference_mode():
            _, bootstrap_values = network(_observations_to_tensor(observations, device))
        rewards_tensor = torch.as_tensor(np.asarray(reward_rows), dtype=torch.float32)
        values_tensor = torch.as_tensor(np.asarray(value_rows), dtype=torch.float32)
        terminated_tensor = torch.as_tensor(
            np.asarray(terminated_rows), dtype=torch.float32
        )
        advantages, value_targets = generalized_advantage_estimate(
            rewards_tensor,
            values_tensor,
            terminated_tensor,
            bootstrap_values.cpu(),
            discount=config.discount,
            gae_lambda=config.gae_lambda,
        )
        target_variance = torch.var(value_targets)
        explained = 1.0 - torch.var(value_targets - values_tensor) / target_variance.clamp_min(
            1e-8
        )
        explained_variances.append(float(explained))

        batch_observations = np.asarray(observation_rows).reshape(
            -1, *first.observation_shape
        )
        batch_actions = torch.as_tensor(
            np.asarray(action_rows).reshape(-1), dtype=torch.long, device=device
        )
        batch_old_log_probabilities = torch.as_tensor(
            np.asarray(log_probability_rows).reshape(-1),
            dtype=torch.float32,
            device=device,
        )
        batch_old_values = values_tensor.reshape(-1).to(device)
        batch_advantages = advantages.reshape(-1).to(device)
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (
            batch_advantages.std(unbiased=False) + 1e-8
        )
        batch_value_targets = value_targets.reshape(-1).to(device)
        batch_size = len(batch_actions)
        stop_for_kl = False
        for _ in range(config.update_epochs):
            order = np_rng.permutation(batch_size)
            for start in range(0, batch_size, config.minibatch_size):
                indices = torch.as_tensor(
                    order[start : start + config.minibatch_size],
                    dtype=torch.long,
                    device=device,
                )
                observations_tensor = _observations_to_tensor(
                    batch_observations[indices.cpu().numpy()], device
                )
                logits, new_values = network(observations_tensor)
                distribution = Categorical(logits=logits)
                new_log_probabilities = distribution.log_prob(batch_actions[indices])
                entropy = distribution.entropy()
                total_loss, metrics = ppo_clipped_loss(
                    new_log_probabilities,
                    batch_old_log_probabilities[indices],
                    batch_advantages[indices],
                    new_values,
                    batch_old_values[indices],
                    batch_value_targets[indices],
                    entropy,
                    clip_epsilon=config.clip_epsilon,
                    value_clip_epsilon=config.value_clip_epsilon,
                    value_coefficient=config.value_coefficient,
                    entropy_coefficient=config.entropy_coefficient,
                )
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), config.max_gradient_norm)
                optimizer.step()
                policy_losses.append(float(metrics["policy"].detach()))
                value_losses.append(float(metrics["value"].detach()))
                entropies.append(float(metrics["entropy"].detach()))
                approximate_kls.append(float(metrics["approximate_kl"].detach()))
                clip_fractions.append(float(metrics["clip_fraction"].detach()))
                if approximate_kls[-1] > config.target_kl:
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break

    return PPOTrainingResult(
        network=network,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        policy_losses=policy_losses,
        value_losses=value_losses,
        entropies=entropies,
        approximate_kls=approximate_kls,
        clip_fractions=clip_fractions,
        explained_variances=explained_variances,
        steps=steps_collected,
        device=device.type,
    )


def evaluate_ppo(
    environment: MinAtarAdapter,
    network: MinAtarActorCritic,
    episodes: int = 20,
    max_episode_steps: int = 10_000,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a deterministic categorical policy on disjoint episodes."""

    if episodes < 1 or max_episode_steps < 1:
        raise ValueError("evaluation budgets must be positive")
    resolved = resolve_torch_device(device)
    network = network.to(resolved)
    network.eval()
    returns = np.zeros(episodes, dtype=np.float64)
    lengths = np.zeros(episodes, dtype=np.int64)
    for episode in range(episodes):
        observation = environment.reset()
        for step in range(max_episode_steps):
            with torch.inference_mode():
                logits, _ = network(
                    _observations_to_tensor(observation[None], resolved)
                )
            if not isinstance(logits, torch.Tensor) or logits.shape != (
                1, environment.action_count
            ):
                raise ValueError(
                    "PPO evaluation output shape must be "
                    f"(1, {environment.action_count}), matching the environment's action count"
                )
            action = int(torch.argmax(logits, dim=-1).item())
            observation, reward, terminated = environment.step(action)
            returns[episode] += reward
            lengths[episode] = step + 1
            if terminated:
                break
    network.train()
    return returns, lengths
