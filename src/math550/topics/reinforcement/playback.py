"""Advance a learned MinAtar policy one step at a time for interactive playback."""

from __future__ import annotations

from torch import nn

from .data import MinAtarAdapter
from .devices import resolve_torch_device
from .visualization import (
    MINATAR_CHANNEL_COLORS,
    MINATAR_CHANNEL_NAMES,
    PolicyAlgorithm,
    minatar_policy_decision,
)


class MinAtarPolicyPlayer:
    """A paused episode: each call to ``step`` executes exactly one learned move.

    Network outputs always describe the displayed, current observation. A step
    records the selected and executed actions separately because the environment
    can repeat the previous action with its default 0.1 sticky probability.
    """

    def __init__(
        self,
        network: nn.Module,
        algorithm: PolicyAlgorithm = "dqn",
        game: str = "breakout",
        seed: int = 10_550,
        max_episode_steps: int = 1_000,
        device: str = "auto",
    ) -> None:
        if algorithm not in ("dqn", "ppo"):
            raise ValueError("algorithm must be 'dqn' or 'ppo'")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if game not in MINATAR_CHANNEL_NAMES:
            raise ValueError(f"unsupported MinAtar game: {game}")
        self.algorithm = algorithm
        self.game = game
        self.device = resolve_torch_device(device)
        self.network = network.to(self.device).eval()
        self.max_episode_steps = max_episode_steps
        self.reset(seed)

    def reset(self, seed: int) -> dict[str, object]:
        """Recreate the seeded environment, including sticky-action history."""

        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer between 0 and 4294967295")
        environment = MinAtarAdapter(game=self.game, seed=seed)
        observation = environment.reset()
        decision = minatar_policy_decision(
            self.network, observation, self.algorithm, self.device, environment.action_count
        )
        self.seed = seed
        self.environment = environment
        self.observation = observation
        self.step_count = 0
        self.reward = 0.0
        self.total_return = 0.0
        self.terminated = False
        self.truncated = False
        self.last_action = None
        self.applied_action = None
        self._decision = decision
        return self.snapshot()

    def step(self) -> dict[str, object]:
        """Apply the cached decision; retain the actual final board at game over."""

        if self.terminated or self.truncated:
            return self.snapshot()
        self.last_action = self._decision[0]
        self.observation, self.reward, self.terminated = self.environment.step(
            self.last_action
        )
        self.applied_action = self.environment.applied_action
        self.step_count += 1
        self.total_return += self.reward
        self.truncated = (
            self.step_count >= self.max_episode_steps and not self.terminated
        )
        self._decision = (
            (None, None, None)
            if self.terminated or self.truncated
            else minatar_policy_decision(
                self.network, self.observation, self.algorithm, self.device,
                self.environment.action_count,
            )
        )
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        """Return strict JSON-compatible state without advancing the episode."""

        next_action, statistics, state_value = self._decision
        return {
            "game": self.game,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "device": str(self.device),
            "step": self.step_count,
            "reward": self.reward,
            "total_return": self.total_return,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.terminated or self.truncated,
            "observation": self.observation.tolist(),
            "action_names": list(self.environment.action_names),
            "raw_action_ids": list(self.environment.raw_action_ids),
            "action_space_note": getattr(self.network, "action_space_note", None),
            "channel_names": list(MINATAR_CHANNEL_NAMES[self.game]),
            "channel_colors": MINATAR_CHANNEL_COLORS[
                : self.observation.shape[-1]
            ].astype(int).tolist(),
            "next_action": next_action,
            "action_statistics": None if statistics is None else statistics.tolist(),
            "state_value": state_value,
            "last_action": self.last_action,
            "applied_action": self.applied_action,
        }
