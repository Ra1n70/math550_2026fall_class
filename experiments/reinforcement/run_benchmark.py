"""Run tabular, DQN, PPO/GRPO-objective, and AlphaZero self-play tracks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from math550.topics.reinforcement import (
    MINARI_BREAKOUT_EXPERT,
    MINATAR_BREAKOUT,
    ConnectFourPolicyValueNetwork,
    GridWorld,
    MinAtarAdapter,
    augment_reflections,
    evaluate_against_random,
    evaluate_dqn,
    evaluate_ppo,
    greedy_rollout,
    grpo_clipped_loss,
    group_relative_advantages,
    q_learning,
    self_play_game,
    train_dqn,
    train_ppo,
    train_policy_value_network,
)

from .config import ExperimentConfig
from .plotting import plot_dqn_training, plot_ppo_training, plot_self_play


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_random(
    game: str, seed: int, episodes: int, max_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    environment = MinAtarAdapter(game=game, seed=seed)
    rng = np.random.default_rng(seed)
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    for episode in range(episodes):
        environment.reset()
        for step in range(max_steps):
            _, reward, terminated = environment.step(
                int(rng.integers(environment.action_count))
            )
            returns[episode] += reward
            lengths[episode] = step + 1
            if terminated:
                break
    return returns, lengths


def _run_gridworld(config: ExperimentConfig) -> dict[str, object]:
    environment = GridWorld()
    q_values, returns = q_learning(environment, seed=config.seed)
    path = greedy_rollout(environment, q_values)
    result = {
        "episodes": len(returns),
        "last_100_mean_return": float(returns[-100:].mean()),
        "greedy_path_length": len(path) - 1,
        "greedy_reaches_goal": path[-1] == environment.goal,
        "greedy_path": [list(state) for state in path],
    }
    np.save(config.analysis_dir / "gridworld_q_values.npy", q_values)
    return result


def _run_dqn(config: ExperimentConfig) -> dict[str, object]:
    dqn_config = config.dqn_config()
    environment = MinAtarAdapter(game=config.game, seed=config.seed)
    started = time.perf_counter()
    training = train_dqn(environment, dqn_config)
    elapsed = time.perf_counter() - started
    checkpoint = config.checkpoint_dir / "minatar_breakout_double_dqn.pt"
    torch.save(training.network.state_dict(), checkpoint)

    evaluation_environment = MinAtarAdapter(game=config.game, seed=config.seed + 10_000)
    evaluation_returns, evaluation_lengths = evaluate_dqn(
        evaluation_environment,
        training.network,
        episodes=config.effective_evaluation_episodes,
        max_episode_steps=dqn_config.max_episode_steps,
        device=config.device,
        seed=config.seed + 10_000,
    )
    random_returns, random_lengths = _evaluate_random(
        config.game,
        config.seed + 20_000,
        config.effective_evaluation_episodes,
        dqn_config.max_episode_steps,
    )
    episode_rows = [
        {"episode": index, "return": score, "length": length}
        for index, (score, length) in enumerate(
            zip(training.episode_returns, training.episode_lengths, strict=True), start=1
        )
    ]
    if episode_rows:
        _write_rows(config.analysis_dir / "dqn_training_episodes.csv", episode_rows)
    _write_rows(
        config.analysis_dir / "dqn_evaluation.csv",
        [
            {
                "policy": policy,
                "episode": episode,
                "return": score,
                "length": length,
            }
            for policy, scores, lengths in (
                ("trained_double_dqn", evaluation_returns, evaluation_lengths),
                ("uniform_random", random_returns, random_lengths),
            )
            for episode, (score, length) in enumerate(
                zip(scores, lengths, strict=True), start=1
            )
        ],
    )
    if training.losses:
        _write_rows(
            config.analysis_dir / "dqn_updates.csv",
            [
                {"update": index, "huber_loss": loss, "mean_chosen_q": mean_q}
                for index, (loss, mean_q) in enumerate(
                    zip(training.losses, training.mean_q_values, strict=True), start=1
                )
            ],
        )
    plot_dqn_training(
        np.asarray(training.episode_returns),
        np.asarray(training.losses),
        float(random_returns.mean()),
        float(evaluation_returns.mean()),
        config.figure_dir,
    )
    return {
        "steps": dqn_config.total_steps,
        "action_count": environment.action_count,
        "action_names": list(environment.action_names),
        "raw_minatar_action_ids": list(environment.raw_action_ids),
        "gradient_updates": len(training.losses),
        "completed_training_episodes": len(training.episode_returns),
        "training_seconds": elapsed,
        "steps_per_second": dqn_config.total_steps / elapsed,
        "device": training.device,
        "replay_size": training.replay_size,
        "replay_allocated_mb": training.replay_bytes / 1e6,
        "trained_evaluation_mean": float(evaluation_returns.mean()),
        "trained_evaluation_standard_error": float(
            evaluation_returns.std(ddof=1) / np.sqrt(len(evaluation_returns))
        ),
        "random_evaluation_mean": float(random_returns.mean()),
        "random_evaluation_standard_error": float(
            random_returns.std(ddof=1) / np.sqrt(len(random_returns))
        ),
        "checkpoint": str(checkpoint),
        "interpretation": (
            "Quick mode is a pipeline smoke test, not a converged Atari claim. "
            "Use the full 500,000-step configuration and multiple seeds for a course result."
            if config.quick
            else "This is a single-seed teaching run; publishable claims require multiple seeds."
        ),
    }


def _run_policy_gradient(config: ExperimentConfig) -> dict[str, object]:
    ppo_config = config.ppo_config()
    environments = [
        MinAtarAdapter(game=config.game, seed=config.seed + index)
        for index in range(ppo_config.num_environments)
    ]
    started = time.perf_counter()
    training = train_ppo(environments, ppo_config)
    elapsed = time.perf_counter() - started
    checkpoint = config.checkpoint_dir / "minatar_breakout_ppo.pt"
    torch.save(training.network.state_dict(), checkpoint)

    evaluation_returns, evaluation_lengths = evaluate_ppo(
        MinAtarAdapter(game=config.game, seed=config.seed + 30_000),
        training.network,
        episodes=config.effective_ppo_evaluation_episodes,
        max_episode_steps=ppo_config.max_episode_steps,
        device=config.device,
    )
    random_returns, random_lengths = _evaluate_random(
        config.game,
        config.seed + 40_000,
        config.effective_ppo_evaluation_episodes,
        ppo_config.max_episode_steps,
    )
    if training.episode_returns:
        _write_rows(
            config.analysis_dir / "ppo_training_episodes.csv",
            [
                {"episode": index, "return": score, "length": length}
                for index, (score, length) in enumerate(
                    zip(training.episode_returns, training.episode_lengths, strict=True),
                    start=1,
                )
            ],
        )
    _write_rows(
        config.analysis_dir / "ppo_updates.csv",
        [
            {
                "update": index,
                "policy_loss": policy,
                "value_loss": value,
                "entropy": entropy,
                "approximate_kl": kl,
                "clip_fraction": clipped,
            }
            for index, (policy, value, entropy, kl, clipped) in enumerate(
                zip(
                    training.policy_losses,
                    training.value_losses,
                    training.entropies,
                    training.approximate_kls,
                    training.clip_fractions,
                    strict=True,
                ),
                start=1,
            )
        ],
    )
    _write_rows(
        config.analysis_dir / "ppo_evaluation.csv",
        [
            {
                "policy": policy,
                "episode": episode,
                "return": score,
                "length": length,
            }
            for policy, scores, lengths in (
                ("trained_ppo_argmax", evaluation_returns, evaluation_lengths),
                ("uniform_random", random_returns, random_lengths),
            )
            for episode, (score, length) in enumerate(
                zip(scores, lengths, strict=True), start=1
            )
        ],
    )
    plot_ppo_training(
        np.asarray(training.episode_returns),
        np.asarray(training.policy_losses),
        np.asarray(training.approximate_kls),
        float(random_returns.mean()),
        float(evaluation_returns.mean()),
        config.figure_dir,
    )

    # A deterministic, inspectable GRPO calculation complements the environment PPO run.
    group_rewards = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 4.0, 4.0, 4.0]])
    group_advantages = group_relative_advantages(group_rewards)
    new_log_probabilities = torch.zeros(2, 4, 3)
    grpo_total, grpo_metrics = grpo_clipped_loss(
        new_log_probabilities,
        torch.full_like(new_log_probabilities, -0.1),
        torch.full_like(new_log_probabilities, -0.2),
        group_rewards,
    )
    return {
        "algorithm": "PPO for MinAtar; outcome-supervision GRPO objective demo",
        "steps": ppo_config.total_steps,
        "action_count": environments[0].action_count,
        "action_names": list(environments[0].action_names),
        "raw_minatar_action_ids": list(environments[0].raw_action_ids),
        "gradient_updates": len(training.policy_losses),
        "completed_training_episodes": len(training.episode_returns),
        "training_seconds": elapsed,
        "steps_per_second": ppo_config.total_steps / elapsed,
        "device": training.device,
        "trained_evaluation_mean": float(evaluation_returns.mean()),
        "trained_evaluation_standard_error": float(
            evaluation_returns.std(ddof=1) / np.sqrt(len(evaluation_returns))
        ),
        "random_evaluation_mean": float(random_returns.mean()),
        "random_evaluation_standard_error": float(
            random_returns.std(ddof=1) / np.sqrt(len(random_returns))
        ),
        "last_policy_loss": float(training.policy_losses[-1]),
        "last_value_loss": float(training.value_losses[-1]),
        "last_entropy": float(training.entropies[-1]),
        "last_approximate_kl": float(training.approximate_kls[-1]),
        "last_clip_fraction": float(training.clip_fractions[-1]),
        "last_explained_variance": float(training.explained_variances[-1]),
        "grpo_demo": {
            "group_rewards": group_rewards.tolist(),
            "group_advantages": group_advantages.tolist(),
            "total_loss": float(grpo_total),
            "positive_kl_estimate": float(grpo_metrics["kl"]),
        },
        "checkpoint": str(checkpoint),
        "interpretation": (
            "Quick PPO is a pipeline and diagnostics smoke test, not a convergence claim; "
            "the GRPO calculation tests the grouped objective rather than training an LLM."
            if config.quick
            else "This is a single-seed teaching run; stable comparisons require multiple seeds."
        ),
    }


def _run_alpha_zero(config: ExperimentConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    network = ConnectFourPolicyValueNetwork()
    replay: list[object] = []
    rows: list[dict[str, object]] = []
    initial = evaluate_against_random(
        network,
        games=config.effective_arena_games,
        simulations=config.effective_mcts_simulations,
        device=config.device,
        seed=config.seed,
    )
    rows.append(
        {
            "iteration": 0,
            "self_play_games": 0,
            "training_examples": 0,
            "mean_game_length": np.nan,
            "first_player_win_rate": np.nan,
            "policy_loss": np.nan,
            "value_loss": np.nan,
            "arena_wins": initial["wins"],
            "arena_draws": initial["draws"],
            "arena_losses": initial["losses"],
            "arena_score": initial["score"],
        }
    )
    started = time.perf_counter()
    for iteration in range(1, config.effective_alpha_zero_iterations + 1):
        games = []
        winners = []
        lengths = []
        for game in range(config.effective_self_play_games):
            examples, winner, length = self_play_game(
                network,
                simulations=config.effective_mcts_simulations,
                device=config.device,
                seed=config.seed + iteration * 1_000 + game,
            )
            games.extend(augment_reflections(examples))
            winners.append(winner)
            lengths.append(length)
        replay.extend(games)
        replay = replay[-50_000:]
        loss_history = train_policy_value_network(
            network,
            replay,
            epochs=config.effective_alpha_zero_epochs,
            batch_size=config.alpha_zero_batch_size,
            device=config.device,
            seed=config.seed + iteration,
        )
        arena = evaluate_against_random(
            network,
            games=config.effective_arena_games,
            simulations=config.effective_mcts_simulations,
            device=config.device,
            seed=config.seed + 50_000 + iteration,
        )
        rows.append(
            {
                "iteration": iteration,
                "self_play_games": config.effective_self_play_games,
                "training_examples": len(replay),
                "mean_game_length": float(np.mean(lengths)),
                "first_player_win_rate": float(np.mean(np.asarray(winners) == 1)),
                "policy_loss": loss_history["policy"][-1],
                "value_loss": loss_history["value"][-1],
                "arena_wins": arena["wins"],
                "arena_draws": arena["draws"],
                "arena_losses": arena["losses"],
                "arena_score": arena["score"],
            }
        )
        print(
            f"self-play iteration {iteration}: examples={len(replay)} "
            f"arena_score={arena['score']:.3f}"
        )
    elapsed = time.perf_counter() - started
    _write_rows(config.analysis_dir / "alpha_zero_iterations.csv", rows)
    checkpoint = config.checkpoint_dir / "connect_four_alpha_zero.pt"
    torch.save(network.state_dict(), checkpoint)
    plotted = rows[1:] if len(rows) > 1 else rows
    plot_self_play(
        np.asarray([row["iteration"] for row in plotted]),
        np.asarray([row["arena_score"] for row in plotted], dtype=float),
        np.asarray([row["policy_loss"] for row in plotted], dtype=float),
        np.asarray([row["value_loss"] for row in plotted], dtype=float),
        config.figure_dir,
    )
    return {
        "iterations": config.effective_alpha_zero_iterations,
        "games_per_iteration": config.effective_self_play_games,
        "simulations_per_move": config.effective_mcts_simulations,
        "training_seconds": elapsed,
        "initial_arena_score_vs_random": initial["score"],
        "final_arena_score_vs_random": rows[-1]["arena_score"],
        "final_training_examples_with_reflections": rows[-1]["training_examples"],
        "checkpoint": str(checkpoint),
        "interpretation": (
            "The quick arena has only four games and is a correctness smoke test; "
            "it is not evidence of monotone self-improvement."
            if config.quick
            else "Arena scores are diagnostic; stronger claims require more games and seeds."
        ),
    }


def run(config: ExperimentConfig, track: str = "all") -> Path:
    config.validate()
    for directory in (
        config.analysis_dir,
        config.figure_dir,
        config.checkpoint_dir,
        config.pdf_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "topic": "reinforcement",
        "quick": config.quick,
        "seed": config.seed,
        "requested_device": config.device,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "minatar": {
            "name": MINATAR_BREAKOUT.name,
            "version": MINATAR_BREAKOUT.version,
            "source_url": MINATAR_BREAKOUT.source_url,
            "paper_url": MINATAR_BREAKOUT.paper_url,
            "license": MINATAR_BREAKOUT.license,
            "observation": MINATAR_BREAKOUT.observation,
            "actions": MINATAR_BREAKOUT.actions,
        },
        "optional_full_atari_dataset": {
            "dataset_id": MINARI_BREAKOUT_EXPERT.dataset_id,
            "source_url": MINARI_BREAKOUT_EXPERT.source_url,
            "download_command": MINARI_BREAKOUT_EXPERT.download_command,
            "steps": MINARI_BREAKOUT_EXPERT.steps,
            "episodes": MINARI_BREAKOUT_EXPERT.episodes,
            "observation_shape": MINARI_BREAKOUT_EXPERT.observation_shape,
            "collection_policy": MINARI_BREAKOUT_EXPERT.collection_policy,
        },
        "gridworld": _run_gridworld(config),
    }
    if track in {"all", "dqn"}:
        print("training Double-DQN on MinAtar Breakout")
        results["dqn"] = _run_dqn(config)
    if track in {"all", "ppo"}:
        print("training PPO on MinAtar Breakout and checking the GRPO objective")
        results["policy_gradient"] = _run_policy_gradient(config)
    if track in {"all", "selfplay"}:
        print("training AlphaZero-style Connect Four by self-play")
        results["alpha_zero"] = _run_alpha_zero(config)
    path = config.analysis_dir / "run_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=550)
    parser.add_argument(
        "--track", choices=("all", "dqn", "ppo", "selfplay"), default="all"
    )
    args = parser.parse_args()
    run(
        ExperimentConfig(seed=args.seed, device=args.device, quick=args.quick),
        track=args.track,
    )


if __name__ == "__main__":
    main()
