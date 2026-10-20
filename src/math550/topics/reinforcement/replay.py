"""An explicit memory-efficient replay buffer for value-based RL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray


class ReplayBuffer:
    """Fixed-capacity ring buffer storing binary states as bytes."""

    def __init__(
        self, capacity: int, observation_shape: tuple[int, ...], seed: int = 550
    ) -> None:
        if capacity < 1 or not observation_shape or min(observation_shape) < 1:
            raise ValueError("capacity and observation dimensions must be positive")
        self.capacity = int(capacity)
        self.observation_shape = tuple(int(v) for v in observation_shape)
        self.observations = np.empty(
            (self.capacity, *self.observation_shape), dtype=np.uint8
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.terminated = np.empty(self.capacity, dtype=np.bool_)
        self._next = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def bytes_allocated(self) -> int:
        arrays = (
            self.observations,
            self.next_observations,
            self.actions,
            self.rewards,
            self.terminated,
        )
        return int(sum(array.nbytes for array in arrays))

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        observation = np.asarray(observation, dtype=np.uint8)
        next_observation = np.asarray(next_observation, dtype=np.uint8)
        if observation.shape != self.observation_shape:
            raise ValueError("observation has the wrong shape")
        if next_observation.shape != self.observation_shape:
            raise ValueError("next_observation has the wrong shape")
        index = self._next
        self.observations[index] = observation
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_observations[index] = next_observation
        self.terminated[index] = bool(terminated)
        self._next = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size < 1 or batch_size > self._size:
            raise ValueError("batch_size must be positive and no larger than the buffer")
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return TransitionBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            terminated=self.terminated[indices],
        )
