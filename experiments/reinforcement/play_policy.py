"""Export a learned MinAtar episode, or watch it interactively with --ui."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
import hashlib
import json
from pathlib import Path
import warnings

import torch

from math550.topics.reinforcement import (
    MINATAR_BREAKOUT,
    MinAtarActorCritic,
    MinAtarAdapter,
    MinAtarPolicyRollout,
    MinAtarQNetwork,
    resolve_torch_device,
    rollout_minatar_policy,
    save_minatar_policy_animation,
    save_minatar_policy_contact_sheet,
)


class _LegacyBreakoutPolicy(torch.nn.Module):
    """Expose three physical choices from an unchanged six-action checkpoint."""

    def __init__(self, network: torch.nn.Module, algorithm: str) -> None:
        super().__init__()
        self.legacy_network = network
        self.algorithm = algorithm
        aggregation = (
            "no-op uses the maximum Q-value of old no-op/up/down/fire outputs."
            if algorithm == "dqn"
            else "no-op sums the old no-op/up/down/fire probabilities; this can "
            "change the greedy action."
        )
        self.action_space_note = (
            "Legacy six-action Breakout checkpoint adapted to left/right/no-op: "
            + aggregation
            + " The checkpoint file is unchanged; retrain for a native three-action model."
        )

    def forward(self, observations: torch.Tensor):
        output = self.legacy_network(observations)
        if self.algorithm == "dqn":
            scores = output
            hold = scores[:, [0, 2, 4, 5]].amax(dim=1)
            return torch.stack((scores[:, 1], scores[:, 3], hold), dim=1)
        logits, value = output
        # Softmax(logsumexp(group)) preserves the group's total probability.
        hold = torch.logsumexp(logits[:, [0, 2, 4, 5]], dim=1)
        return torch.stack((logits[:, 1], logits[:, 3], hold), dim=1), value


def _default_checkpoint(algorithm: str) -> Path:
    filename = (
        "minatar_breakout_double_dqn.pt"
        if algorithm == "dqn"
        else "minatar_breakout_ppo.pt"
    )
    return Path("output/reinforcement/checkpoints") / filename


def _load_network(
    algorithm: str,
    checkpoint: Path,
    input_channels: int,
    action_count: int,
    *,
    game: str = "breakout",
) -> torch.nn.Module:
    """Load native models or explicitly adapt the older six-action Breakout head."""

    if algorithm not in {"dqn", "ppo"}:
        raise ValueError("algorithm must be dqn or ppo")
    if game == "breakout" and action_count != 3:
        raise ValueError("Breakout playback requires three actions: left, right, no-op")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. Run `make quick-reinforcement` first."
        )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    head_key = "head.3.weight" if algorithm == "dqn" else "policy_head.weight"
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"checkpoint {checkpoint} must contain a model state_dict")
    head = state_dict.get(head_key)
    if not isinstance(head, torch.Tensor) or head.ndim != 2:
        raise ValueError(
            f"checkpoint {checkpoint} has no valid {algorithm.upper()} action head "
            f"({head_key}); select a matching algorithm and checkpoint"
        )
    saved_action_count = head.shape[0]
    legacy_breakout = game == "breakout" and saved_action_count == 6
    if saved_action_count != action_count and not legacy_breakout:
        raise ValueError(
            f"checkpoint {checkpoint} has {saved_action_count} actions, but "
            f"{game} requires {action_count}"
        )
    if algorithm == "dqn":
        network: torch.nn.Module = MinAtarQNetwork(input_channels, saved_action_count)
    else:
        network = MinAtarActorCritic(input_channels, saved_action_count)
    try:
        network.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(
            f"checkpoint {checkpoint} does not match the {algorithm.upper()} "
            f"network for {game}: {error}"
        ) from error
    if legacy_breakout:
        network = _LegacyBreakoutPolicy(network, algorithm)
        warnings.warn(network.action_space_note, RuntimeWarning, stacklevel=2)
    return network


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_trace(path: Path, rollout: MinAtarPolicyRollout) -> None:
    action_statistic_label = "q" if rollout.algorithm == "dqn" else "probability"
    fields = [
        "step",
        "action_id",
        "action_name",
        "reward",
        "cumulative_return",
        "state_value",
    ] + [
        f"{action_statistic_label}_{name.replace('-', '_').replace(' ', '_')}"
        for name in rollout.action_names
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(rollout.episode_length):
            row: dict[str, object] = {
                "step": index + 1,
                "action_id": int(rollout.actions[index]),
                "action_name": rollout.action_names[int(rollout.actions[index])],
                "reward": float(rollout.rewards[index]),
                "cumulative_return": float(rollout.cumulative_rewards[index]),
                "state_value": (
                    ""
                    if rollout.algorithm == "dqn"
                    else float(rollout.state_values[index])
                ),
            }
            row.update(
                {
                    field: float(value)
                    for field, value in zip(
                        fields[6:], rollout.action_statistics[index], strict=True
                    )
                }
            )
            writer.writerow(row)


def build_demo(args: argparse.Namespace) -> dict[str, object]:
    """Load a checkpoint, play one deterministic episode, and export its trace."""

    checkpoint = args.checkpoint or _default_checkpoint(args.algorithm)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = MinAtarAdapter(game=args.game, seed=args.seed)
    network = _load_network(
        args.algorithm,
        checkpoint,
        input_channels=environment.observation_shape[-1],
        action_count=environment.action_count,
        game=args.game,
    )
    device = resolve_torch_device(args.device)
    rollout = rollout_minatar_policy(
        environment,
        network,
        algorithm=args.algorithm,
        max_episode_steps=args.max_episode_steps,
        device=str(device),
    )
    stem = args.output_name or f"minatar_{args.game}_{args.algorithm}_learned_policy"
    gif_path = save_minatar_policy_animation(
        rollout,
        output_dir / f"{stem}.gif",
        fps=args.fps,
        max_rendered_frames=args.max_rendered_frames,
    )
    contact_sheet_path = save_minatar_policy_contact_sheet(
        rollout,
        output_dir / f"{stem}_contact_sheet.png",
        frame_count=args.contact_sheet_frames,
    )
    trace_path = output_dir / f"{stem}_trace.csv"
    _write_trace(trace_path, rollout)
    summary = {
        "algorithm": args.algorithm,
        "network": type(network).__name__,
        "game": args.game,
        "action_names": list(rollout.action_names),
        "raw_action_ids": list(environment.raw_action_ids),
        "action_space_note": getattr(network, "action_space_note", None),
        "environment_version": MINATAR_BREAKOUT.version,
        "environment_source": MINATAR_BREAKOUT.source_url,
        "environment_license": MINATAR_BREAKOUT.license,
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "greedy_action_selection": True,
        "sticky_action_probability": 0.1,
        "episode_return": rollout.episode_return,
        "episode_length": rollout.episode_length,
        "terminated": rollout.terminated,
        "gif": str(gif_path.resolve()),
        "contact_sheet": str(contact_sheet_path.resolve()),
        "trace": str(trace_path.resolve()),
        "interpretation": (
            "This visualizes one fixed-seed episode from the saved learned policy. "
            "It is a behavior trace, not an estimate of expected performance."
        ),
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path.resolve())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default="dqn")
    parser.add_argument("--game", default="breakout")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/reinforcement/demos"),
    )
    parser.add_argument("--output-name")
    parser.add_argument("--seed", type=int, default=10_550)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-rendered-frames", type=int, default=300)
    parser.add_argument("--contact-sheet-frames", type=int, default=12)
    parser.add_argument("--ui", action="store_true", help="Open the live browser player instead of exporting images")
    parser.add_argument("--port", type=int, default=8765, help="Preferred --ui port; automatically use a free port if busy")
    parser.add_argument("--no-browser", action="store_true", help="With --ui, print the URL without opening it")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ui:
        from .play_ui import serve_policy_ui

        serve_policy_ui(
            args.algorithm, args.checkpoint, args.game, args.seed, args.device,
            args.max_episode_steps, args.port, not args.no_browser,
        )
    else:
        print(json.dumps(build_demo(args), indent=2))
