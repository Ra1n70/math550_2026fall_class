"""Public environment adapters and precise dataset provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Raw MinAtar IDs remain unchanged for games that need all six controls.
MINATAR_ACTION_NAMES = ("no-op", "left", "up", "right", "down", "fire")
BREAKOUT_ACTION_NAMES = ("left", "right", "no-op")


def minatar_action_ids(game: str) -> tuple[int, ...]:
    """Map course action IDs to the underlying MinAtar controls."""

    return (1, 3, 0) if game == "breakout" else tuple(range(6))


def minatar_action_names(game: str) -> tuple[str, ...]:
    """Names in the same order used by the environment and network outputs."""

    return tuple(MINATAR_ACTION_NAMES[index] for index in minatar_action_ids(game))


@dataclass(frozen=True)
class EnvironmentMetadata:
    name: str
    version: str
    source_url: str
    paper_url: str
    license: str
    observation: str
    actions: str


@dataclass(frozen=True)
class OfflineDatasetMetadata:
    dataset_id: str
    source_url: str
    download_command: str
    steps: int
    episodes: int
    observation_shape: tuple[int, int, int]
    action_count: int
    collection_policy: str
    license_note: str


MINATAR_BREAKOUT = EnvironmentMetadata(
    name="MinAtar Breakout",
    version="1.0.15",
    source_url="https://github.com/kenjyoung/MinAtar",
    paper_url="https://arxiv.org/abs/1903.03176",
    license="GPL-3.0",
    observation="10 x 10 x 4 binary object channels",
    actions="3 actions: 0=left, 1=right, 2=no-op (MinAtar IDs 1, 3, 0)",
)

MINARI_BREAKOUT_EXPERT = OfflineDatasetMetadata(
    dataset_id="atari/breakout/expert-v0",
    source_url="https://minari.farama.org/main/datasets/atari/breakout/expert-v0/",
    download_command=(
        "uv run --extra rl-data minari download atari/breakout/expert-v0"
    ),
    steps=23_075,
    episodes=10,
    observation_shape=(210, 160, 3),
    action_count=4,
    collection_policy="CleanRL PPO-Impala expert policy",
    license_note=(
        "Minari is MIT-licensed; the dataset page and Arcade Learning "
        "Environment terms govern the observations and environment assets."
    ),
)


class MinAtarAdapter:
    """MinAtar with three Breakout controls; other games retain all six.

    Breakout action IDs are 0=left, 1=right, 2=no-op. No-op holds the
    paddle in place while the ball and the rest of the game advance.
    """

    def __init__(
        self,
        game: str = "breakout",
        seed: int = 550,
        sticky_action_probability: float = 0.1,
        difficulty_ramping: bool = True,
    ) -> None:
        try:
            from minatar import Environment
        except ImportError as error:  # pragma: no cover - installation guidance
            raise ImportError(
                "MinAtar is required; run `uv sync --locked --extra rl`"
            ) from error
        self.game = str(game)
        self.seed = int(seed)
        self._environment: Any = Environment(
            self.game,
            sticky_action_prob=float(sticky_action_probability),
            difficulty_ramping=bool(difficulty_ramping),
        )
        self._environment.seed(self.seed)
        self.raw_action_ids = minatar_action_ids(self.game)
        self.action_names = minatar_action_names(self.game)
        self.action_count = len(self.raw_action_ids)
        self.observation_shape = tuple(int(v) for v in self._environment.state_shape())

    def reset(self) -> np.ndarray:
        self._environment.reset()
        return np.asarray(self._environment.state(), dtype=np.uint8)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if not isinstance(action, (int, np.integer)) or action not in range(self.action_count):
            raise ValueError(f"action must lie in [0, {self.action_count})")
        reward, terminated = self._environment.act(self.raw_action_ids[int(action)])
        observation = np.asarray(self._environment.state(), dtype=np.uint8)
        return observation, float(reward), bool(terminated)

    @property
    def applied_action(self) -> int:
        """Last action actually executed, including MinAtar's sticky actions."""

        return self.raw_action_ids.index(int(self._environment.last_action))


def collect_random_transitions(
    environment: MinAtarAdapter, steps: int, seed: int = 550
) -> dict[str, np.ndarray]:
    """Collect a small reproducible replay dataset for buffer exercises."""

    if steps < 1:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    shape = environment.observation_shape
    observations = np.empty((steps, *shape), dtype=np.uint8)
    next_observations = np.empty_like(observations)
    actions = np.empty(steps, dtype=np.int64)
    rewards = np.empty(steps, dtype=np.float32)
    terminated = np.empty(steps, dtype=np.bool_)
    observation = environment.reset()
    for index in range(steps):
        action = int(rng.integers(environment.action_count))
        next_observation, reward, done = environment.step(action)
        observations[index] = observation
        next_observations[index] = next_observation
        actions[index] = action
        rewards[index] = reward
        terminated[index] = done
        observation = environment.reset() if done else next_observation
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminated": terminated,
    }
