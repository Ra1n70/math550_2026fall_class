"""Deep Q-network and explicit replay-learning loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..devices import resolve_torch_device
from ..replay import ReplayBuffer, TransitionBatch


class DiscreteEnvironment(Protocol):
    observation_shape: tuple[int, ...]
    action_count: int

    def reset(self) -> np.ndarray: ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool]: ...


class MinAtarQNetwork(nn.Module):
    """Compact convolutional action-value network for 10 x 10 MinAtar states."""

    def __init__(self, input_channels: int, action_count: int) -> None:
        super().__init__()
        if input_channels < 1 or action_count < 2:
            raise ValueError("invalid channel or action count")
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError("observations must have NCHW shape")
        return self.head(self.features(observations))


@dataclass(frozen=True)
class DQNConfig:
    total_steps: int = 500_000
    replay_capacity: int = 100_000
    warmup_steps: int = 5_000
    batch_size: int = 64
    discount: float = 0.99
    learning_rate: float = 2.5e-4
    target_update_steps: int = 1_000
    train_frequency: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    max_episode_steps: int = 10_000
    double_dqn: bool = True
    seed: int = 550
    device: str = "auto"

    def validate(self) -> None:
        positive = (
            self.total_steps,
            self.replay_capacity,
            self.warmup_steps,
            self.batch_size,
            self.target_update_steps,
            self.train_frequency,
            self.epsilon_decay_steps,
            self.max_episode_steps,
        )
        if min(positive) < 1:
            raise ValueError("step, buffer, and batch settings must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if not 0 <= self.discount <= 1 or self.learning_rate <= 0:
            raise ValueError("invalid discount or learning rate")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon schedule must decrease within [0, 1]")
        resolve_torch_device(self.device)


@dataclass
class DQNTrainingResult:
    network: MinAtarQNetwork
    episode_returns: list[float]
    episode_lengths: list[int]
    losses: list[float]
    mean_q_values: list[float]
    device: str
    replay_size: int
    replay_bytes: int


def epsilon_at_step(config: DQNConfig, step: int) -> float:
    fraction = min(max(step, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def _batch_to_device(
    batch: TransitionBatch, device: torch.device
) -> tuple[torch.Tensor, ...]:
    observations = torch.as_tensor(
        batch.observations, dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
    next_observations = torch.as_tensor(
        batch.next_observations, dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)
    terminated = torch.as_tensor(
        batch.terminated, dtype=torch.float32, device=device
    )
    return observations, actions, rewards, next_observations, terminated


def dqn_loss(
    online_network: nn.Module,
    target_network: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    discount: float,
    double_dqn: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the one-step DQN objective; terminal targets do not bootstrap."""

    observations, actions, rewards, next_observations, terminated = tensors
    predictions = online_network(observations).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        if double_dqn:
            next_actions = online_network(next_observations).argmax(dim=1)
            next_values = target_network(next_observations).gather(
                1, next_actions[:, None]
            ).squeeze(1)
        else:
            next_values = target_network(next_observations).max(dim=1).values
        targets = rewards + discount * (1.0 - terminated) * next_values
    loss = F.smooth_l1_loss(predictions, targets)
    return loss, predictions.detach(), targets.detach()


def train_dqn(
    environment: DiscreteEnvironment, config: DQNConfig
) -> DQNTrainingResult:
    """Interact, replay, bootstrap, and synchronize a target network explicitly."""

    config.validate()
    device = resolve_torch_device(config.device)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    input_channels = environment.observation_shape[-1]
    online = MinAtarQNetwork(input_channels, environment.action_count).to(device)
    target = MinAtarQNetwork(input_channels, environment.action_count).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(
        config.replay_capacity, environment.observation_shape, seed=config.seed + 1
    )

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    losses: list[float] = []
    mean_q_values: list[float] = []
    observation = environment.reset()
    episode_return = 0.0
    episode_length = 0
    for step in range(1, config.total_steps + 1):
        epsilon = epsilon_at_step(config, step - 1)
        if rng.random() < epsilon:
            action = int(rng.integers(environment.action_count))
        else:
            online.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(
                    observation[None], dtype=torch.float32, device=device
                ).permute(0, 3, 1, 2)
                action = int(online(tensor).argmax(dim=1).item())
        next_observation, reward, terminated = environment.step(action)
        episode_length += 1
        time_limit = episode_length >= config.max_episode_steps
        replay.add(observation, action, reward, next_observation, terminated)
        observation = next_observation
        episode_return += reward

        ready = (
            step >= config.warmup_steps
            and len(replay) >= config.batch_size
            and step % config.train_frequency == 0
        )
        if ready:
            online.train()
            tensors = _batch_to_device(replay.sample(config.batch_size), device)
            loss, q_values, _ = dqn_loss(
                online, target, tensors, config.discount, config.double_dqn
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mean_q_values.append(float(q_values.mean().cpu()))

        if step % config.target_update_steps == 0:
            target.load_state_dict(online.state_dict())

        if terminated or time_limit:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            observation = environment.reset()
            episode_return = 0.0
            episode_length = 0

    return DQNTrainingResult(
        network=online,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        losses=losses,
        mean_q_values=mean_q_values,
        device=str(device),
        replay_size=len(replay),
        replay_bytes=replay.bytes_allocated,
    )


def evaluate_dqn(
    environment: DiscreteEnvironment,
    network: nn.Module,
    episodes: int = 20,
    max_episode_steps: int = 10_000,
    device: str = "auto",
    epsilon: float = 0.001,
    seed: int = 550,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate without learning and return episode scores and lengths."""

    if episodes < 1 or max_episode_steps < 1:
        raise ValueError("episodes and max_episode_steps must be positive")
    resolved = resolve_torch_device(device)
    rng = np.random.default_rng(seed)
    network = network.to(resolved)
    network.eval()
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    for episode in range(episodes):
        observation = environment.reset()
        for step in range(max_episode_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(environment.action_count))
            else:
                with torch.no_grad():
                    tensor = torch.as_tensor(
                        observation[None], dtype=torch.float32, device=resolved
                    ).permute(0, 3, 1, 2)
                    q_values = network(tensor)
                    if not isinstance(q_values, torch.Tensor) or q_values.shape != (
                        1, environment.action_count
                    ):
                        raise ValueError(
                            "DQN evaluation output shape must be "
                            f"(1, {environment.action_count}), matching the environment's action count"
                        )
                    action = int(q_values.argmax(dim=1).item())
            observation, reward, terminated = environment.step(action)
            returns[episode] += reward
            lengths[episode] = step + 1
            if terminated:
                break
    return returns, lengths
