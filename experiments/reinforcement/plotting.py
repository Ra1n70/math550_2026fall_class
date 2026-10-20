"""Plots built only from completed reinforcement-learning result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if len(values) < width:
        return values
    return np.convolve(values, np.ones(width) / width, mode="valid")


def plot_dqn_training(
    returns: np.ndarray,
    losses: np.ndarray,
    random_mean: float,
    evaluation_mean: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    width = min(20, max(1, len(returns) // 5))
    smoothed = _moving_average(returns, width)
    axes[0].plot(np.arange(len(smoothed)) + width - 1, smoothed, color="#1d4ed8")
    axes[0].axhline(random_mean, color="#64748b", linestyle="--", label="random eval")
    axes[0].axhline(evaluation_mean, color="#dc2626", linestyle=":", label="trained eval")
    axes[0].set(title="MinAtar Breakout", xlabel="completed episode", ylabel="return")
    axes[0].legend(frameon=False)
    if len(losses):
        axes[1].plot(_moving_average(losses, min(200, max(1, len(losses) // 20))))
    axes[1].set(title="Replay objective", xlabel="gradient update", ylabel="Huber loss")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / "dqn_training.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_ppo_training(
    returns: np.ndarray,
    policy_losses: np.ndarray,
    approximate_kls: np.ndarray,
    random_mean: float,
    evaluation_mean: float,
    output_dir: Path,
) -> Path:
    """Plot episodic progress and the two most useful PPO update diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    width = min(20, max(1, len(returns) // 5))
    smoothed = _moving_average(returns, width)
    axes[0].plot(
        np.arange(len(smoothed)) + width - 1,
        smoothed,
        color="#0f766e",
        label="training return",
    )
    axes[0].axhline(random_mean, color="#64748b", linestyle="--", label="random eval")
    axes[0].axhline(evaluation_mean, color="#dc2626", linestyle=":", label="PPO eval")
    axes[0].set(title="PPO on MinAtar Breakout", xlabel="completed episode", ylabel="return")
    axes[0].legend(frameon=False)
    loss_width = min(100, max(1, max(len(policy_losses), len(approximate_kls)) // 10))
    if len(policy_losses):
        axes[1].plot(_moving_average(policy_losses, loss_width), label="policy loss")
    if len(approximate_kls):
        axes[1].plot(_moving_average(approximate_kls, loss_width), label="approx. KL")
    axes[1].set(title="Trust-region diagnostics", xlabel="minibatch update", ylabel="value")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / "policy_gradient_training.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_self_play(
    iterations: np.ndarray,
    arena_scores: np.ndarray,
    policy_losses: np.ndarray,
    value_losses: np.ndarray,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(iterations, arena_scores, marker="o", color="#7c3aed")
    axes[0].axhline(0.5, color="#64748b", linestyle="--")
    axes[0].set(
        title="Search agent vs random",
        xlabel="self-play iteration",
        ylabel="win + 0.5 draw rate",
        ylim=(-0.03, 1.03),
    )
    axes[1].plot(iterations, policy_losses, marker="o", label="policy")
    axes[1].plot(iterations, value_losses, marker="s", label="value")
    axes[1].set(title="Self-play target fit", xlabel="iteration", ylabel="loss")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / "alpha_zero_self_play.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
