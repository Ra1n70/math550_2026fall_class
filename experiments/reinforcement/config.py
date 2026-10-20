"""Immutable configuration for the reinforcement-learning module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from math550.topics.reinforcement import DQNConfig, PPOConfig, SUPPORTED_TORCH_DEVICES


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: Path = Path("output")
    seed: int = 550
    device: str = "auto"
    game: str = "breakout"
    quick: bool = False
    dqn_steps: int = 500_000
    dqn_evaluation_episodes: int = 30
    ppo_steps: int = 200_000
    ppo_evaluation_episodes: int = 30
    alpha_zero_iterations: int = 12
    self_play_games_per_iteration: int = 30
    mcts_simulations: int = 64
    alpha_zero_epochs: int = 5
    alpha_zero_batch_size: int = 128
    alpha_zero_arena_games: int = 20

    @property
    def analysis_dir(self) -> Path:
        return self.output_root / "reinforcement" / "analysis"

    @property
    def figure_dir(self) -> Path:
        return self.output_root / "reinforcement" / "figures"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_root / "reinforcement" / "checkpoints"

    @property
    def pdf_dir(self) -> Path:
        return self.output_root / "pdf"

    @property
    def effective_dqn_steps(self) -> int:
        return 5_000 if self.quick else self.dqn_steps

    @property
    def effective_evaluation_episodes(self) -> int:
        return 6 if self.quick else self.dqn_evaluation_episodes

    @property
    def effective_ppo_steps(self) -> int:
        return 8_192 if self.quick else self.ppo_steps

    @property
    def effective_ppo_evaluation_episodes(self) -> int:
        return 6 if self.quick else self.ppo_evaluation_episodes

    @property
    def effective_alpha_zero_iterations(self) -> int:
        return 1 if self.quick else self.alpha_zero_iterations

    @property
    def effective_self_play_games(self) -> int:
        return 3 if self.quick else self.self_play_games_per_iteration

    @property
    def effective_mcts_simulations(self) -> int:
        return 8 if self.quick else self.mcts_simulations

    @property
    def effective_alpha_zero_epochs(self) -> int:
        return 2 if self.quick else self.alpha_zero_epochs

    @property
    def effective_arena_games(self) -> int:
        return 4 if self.quick else self.alpha_zero_arena_games

    def dqn_config(self) -> DQNConfig:
        steps = self.effective_dqn_steps
        return DQNConfig(
            total_steps=steps,
            replay_capacity=20_000 if self.quick else 100_000,
            warmup_steps=500 if self.quick else 5_000,
            batch_size=32 if self.quick else 64,
            target_update_steps=500 if self.quick else 1_000,
            epsilon_decay_steps=4_000 if self.quick else 100_000,
            max_episode_steps=2_000 if self.quick else 10_000,
            seed=self.seed,
            device=self.device,
        )

    def ppo_config(self) -> PPOConfig:
        return PPOConfig(
            total_steps=self.effective_ppo_steps,
            num_environments=8,
            rollout_steps=128,
            update_epochs=2 if self.quick else 4,
            minibatch_size=256,
            max_episode_steps=2_000 if self.quick else 10_000,
            seed=self.seed,
            device=self.device,
        )

    def validate(self) -> None:
        if self.device not in SUPPORTED_TORCH_DEVICES:
            raise ValueError(f"device must be one of {SUPPORTED_TORCH_DEVICES}")
        positive = (
            self.dqn_steps,
            self.dqn_evaluation_episodes,
            self.ppo_steps,
            self.ppo_evaluation_episodes,
            self.alpha_zero_iterations,
            self.self_play_games_per_iteration,
            self.mcts_simulations,
            self.alpha_zero_epochs,
            self.alpha_zero_batch_size,
            self.alpha_zero_arena_games,
        )
        if min(positive) < 1:
            raise ValueError("experiment budgets must be positive")
        self.dqn_config().validate()
        self.ppo_config().validate()
