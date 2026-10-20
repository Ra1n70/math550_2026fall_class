"""Record and visualize deterministic policies on MinAtar environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn

from .data import MINATAR_ACTION_NAMES, MinAtarAdapter, minatar_action_names
from .devices import resolve_torch_device


PolicyAlgorithm = Literal["dqn", "ppo"]

MINATAR_CHANNEL_NAMES: dict[str, tuple[str, ...]] = {
    "asterix": ("player", "enemy", "trail", "gold"),
    "breakout": ("paddle", "ball", "trail", "brick"),
    "freeway": ("chicken", "car", "speed 1", "speed 2", "speed 3", "speed 4", "speed 5"),
    "seaquest": (
        "sub front",
        "sub back",
        "friendly shot",
        "trail",
        "enemy shot",
        "fish",
        "enemy sub",
        "oxygen",
        "diver gauge",
        "diver",
    ),
    "space_invaders": (
        "cannon",
        "alien",
        "alien left",
        "alien right",
        "friendly shot",
        "enemy shot",
    ),
}

MINATAR_CHANNEL_COLORS = np.asarray(
    [
        (46, 196, 182),
        (255, 209, 102),
        (247, 140, 107),
        (239, 71, 111),
        (118, 200, 147),
        (17, 138, 178),
        (131, 56, 236),
        (255, 159, 28),
        (141, 153, 174),
        (244, 241, 222),
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class MinAtarPolicyRollout:
    """One greedy episode plus the action statistics that explain each move."""

    game: str
    algorithm: PolicyAlgorithm
    seed: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    cumulative_rewards: np.ndarray
    action_statistics: np.ndarray
    state_values: np.ndarray
    terminated: bool

    def __post_init__(self) -> None:
        steps = len(self.actions)
        if steps < 1:
            raise ValueError("a policy rollout must contain at least one step")
        if self.observations.ndim != 4 or len(self.observations) != steps:
            raise ValueError("observations must have shape steps x height x width x channels")
        expected_vectors = (self.rewards, self.cumulative_rewards, self.state_values)
        if any(vector.shape != (steps,) for vector in expected_vectors):
            raise ValueError("reward and value traces must contain one entry per step")
        if self.action_statistics.ndim != 2 or len(self.action_statistics) != steps:
            raise ValueError("action statistics must have shape steps x actions")
        if self.action_statistics.shape[1] != len(self.action_names):
            raise ValueError("action statistics do not match the game's action set")
        if np.any((self.actions < 0) | (self.actions >= len(self.action_names))):
            raise ValueError("rollout actions are outside the game's action set")
        if not np.isfinite(self.action_statistics).all():
            raise ValueError("action statistics must be finite")
        if self.algorithm == "ppo" and not np.isfinite(self.state_values).all():
            raise ValueError("PPO state values must be finite")
        if self.algorithm == "dqn" and not np.isnan(self.state_values).all():
            raise ValueError("DQN rollouts do not have critic state values")

    @property
    def action_names(self) -> tuple[str, ...]:
        # Historical six-output recordings retain their original labels.
        if self.action_statistics.shape[1] == 6:
            return MINATAR_ACTION_NAMES
        return minatar_action_names(self.game)

    @property
    def episode_return(self) -> float:
        return float(self.cumulative_rewards[-1])

    @property
    def episode_length(self) -> int:
        return len(self.actions)


def _observation_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        observation[None], dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)


def minatar_policy_decision(
    network: nn.Module,
    observation: np.ndarray,
    algorithm: PolicyAlgorithm,
    device: torch.device,
    action_count: int | None = None,
) -> tuple[int, np.ndarray, float | None]:
    """Choose a greedy action and expose the current state's network outputs."""

    with torch.inference_mode():
        output = network(_observation_tensor(observation, device))
        value = None
        if algorithm == "dqn":
            if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
                raise ValueError("a MinAtar DQN must return 1 x action_count Q-values")
            statistics = output[0]
        elif algorithm == "ppo":
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError("a PPO policy must return (policy logits, values)")
            logits, values = output
            if logits.ndim != 2 or logits.shape[0] != 1 or values.shape != (1,):
                raise ValueError("invalid MinAtar PPO network output shapes")
            statistics = torch.softmax(logits[0], dim=-1)
            value = float(values[0].cpu())
            if not np.isfinite(value):
                raise ValueError("the policy returned a non-finite critic value")
        else:
            raise ValueError("algorithm must be 'dqn' or 'ppo'")
        if statistics.numel() < 1 or (action_count is not None and statistics.numel() != action_count):
            raise ValueError("network output count does not match the environment's action count")
        statistics_array = statistics.cpu().numpy().astype(np.float32)
    if not np.isfinite(statistics_array).all():
        raise ValueError("the policy returned non-finite action statistics")
    return int(statistics_array.argmax()), statistics_array, value


def rollout_minatar_policy(
    environment: MinAtarAdapter,
    network: nn.Module,
    algorithm: PolicyAlgorithm,
    max_episode_steps: int = 1_000,
    device: str = "auto",
) -> MinAtarPolicyRollout:
    """Run one greedy episode while retaining the policy's decision trace."""

    if algorithm not in ("dqn", "ppo"):
        raise ValueError("algorithm must be 'dqn' or 'ppo'")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    resolved = resolve_torch_device(device)
    was_training = network.training
    network = network.to(resolved)
    network.eval()

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    cumulative_rewards: list[float] = []
    action_statistics: list[np.ndarray] = []
    state_values: list[float] = []
    observation = environment.reset()
    cumulative_reward = 0.0
    terminated = False

    try:
        for _ in range(max_episode_steps):
            action, statistics, value = minatar_policy_decision(
                network, observation, algorithm, resolved, environment.action_count
            )
            if action >= environment.action_count:
                raise ValueError("network selected an action outside the environment")
            next_observation, reward, terminated = environment.step(action)
            cumulative_reward += reward

            observations.append(np.asarray(observation, dtype=np.uint8).copy())
            actions.append(action)
            rewards.append(reward)
            cumulative_rewards.append(cumulative_reward)
            action_statistics.append(statistics)
            state_values.append(float("nan") if value is None else value)
            observation = next_observation
            if terminated:
                break
    finally:
        network.train(was_training)

    return MinAtarPolicyRollout(
        game=environment.game,
        algorithm=algorithm,
        seed=environment.seed,
        observations=np.stack(observations),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        cumulative_rewards=np.asarray(cumulative_rewards, dtype=np.float32),
        action_statistics=np.stack(action_statistics),
        state_values=np.asarray(state_values, dtype=np.float32),
        terminated=terminated,
    )


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        import matplotlib

        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
        return ImageFont.truetype(path, size=size)
    except (ImportError, OSError):  # pragma: no cover - minimal installations
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            return ImageFont.load_default()


def _state_image(observation: np.ndarray, cell_size: int) -> Image.Image:
    if observation.ndim != 3 or observation.shape[2] < 1:
        raise ValueError("a MinAtar state must have height x width x channels")
    height, width, channels = observation.shape
    if channels > len(MINATAR_CHANNEL_COLORS):
        raise ValueError("the renderer supports at most ten object channels")
    rgb = np.full((height, width, 3), (7, 17, 31), dtype=np.float32)
    active_count = observation.sum(axis=2, keepdims=True)
    colored_sum = np.einsum(
        "hwc,cd->hwd", observation.astype(np.float32), MINATAR_CHANNEL_COLORS[:channels]
    )
    occupied = active_count[..., 0] > 0
    rgb[occupied] = colored_sum[occupied] / active_count[occupied]
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((width * cell_size, height * cell_size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    grid_color = (35, 55, 75)
    for x in range(0, width * cell_size + 1, cell_size):
        draw.line((x, 0, x, height * cell_size), fill=grid_color, width=1)
    for y in range(0, height * cell_size + 1, cell_size):
        draw.line((0, y, width * cell_size, y), fill=grid_color, width=1)
    return image


def _relative_preferences(statistics: np.ndarray, algorithm: PolicyAlgorithm) -> np.ndarray:
    if algorithm == "ppo":
        total = statistics.sum()
        return statistics / total if total > 0 else np.full_like(statistics, 1 / len(statistics))
    shifted = statistics - statistics.max()
    exponentiated = np.exp(np.clip(shifted, -30.0, 0.0))
    return exponentiated / exponentiated.sum()


def render_minatar_policy_frame(
    rollout: MinAtarPolicyRollout,
    step_index: int,
    cell_size: int = 38,
) -> Image.Image:
    """Render one annotated decision frame from a recorded policy episode."""

    if step_index not in range(rollout.episode_length):
        raise IndexError("step_index is outside the rollout")
    board = _state_image(rollout.observations[step_index], cell_size)
    canvas = Image.new("RGB", (760, 520), color=(248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25, bold=True)
    heading_font = _font(17, bold=True)
    body_font = _font(15)
    small_font = _font(12)
    accent = (37, 99, 235)
    selected_color = (15, 118, 110)
    muted = (71, 85, 105)

    algorithm_name = "Double-DQN" if rollout.algorithm == "dqn" else "PPO"
    draw.text(
        (28, 20),
        f"MinAtar {rollout.game.title()} — learned {algorithm_name} policy",
        font=title_font,
        fill=(15, 23, 42),
    )
    canvas.paste(board, (28, 68))

    panel_x = 438
    action = int(rollout.actions[step_index])
    reward = float(rollout.rewards[step_index])
    cumulative = float(rollout.cumulative_rewards[step_index])
    action_name = rollout.action_names[action]
    draw.text((panel_x, 72), f"Step {step_index + 1} / {rollout.episode_length}", font=heading_font, fill=muted)
    draw.text((panel_x, 104), f"Action: {action_name}", font=heading_font, fill=selected_color)
    draw.text((panel_x, 136), f"Reward: {reward:+.0f}", font=body_font, fill=(15, 23, 42))
    draw.text((panel_x + 135, 136), f"Return: {cumulative:.0f}", font=body_font, fill=(15, 23, 42))
    if rollout.algorithm == "ppo":
        value = float(rollout.state_values[step_index])
        draw.text((panel_x, 162), f"Critic V(s): {value:+.2f}", font=body_font, fill=(15, 23, 42))

    label = "Action probabilities" if rollout.algorithm == "ppo" else "Q-values and relative preference"
    draw.text((panel_x, 198), label, font=heading_font, fill=(15, 23, 42))
    statistics = rollout.action_statistics[step_index]
    preferences = _relative_preferences(statistics, rollout.algorithm)
    for index, (name, raw, preference) in enumerate(
        zip(rollout.action_names, statistics, preferences, strict=True)
    ):
        y = 232 + index * 39
        color = selected_color if index == action else accent
        if index == action:
            draw.rounded_rectangle((panel_x - 5, y - 4, 724, y + 28), radius=6, fill=(220, 252, 231))
        draw.text((panel_x, y), name, font=body_font, fill=(15, 23, 42))
        draw.rounded_rectangle((512, y + 3, 644, y + 18), radius=5, fill=(226, 232, 240))
        draw.rounded_rectangle(
            (512, y + 3, 512 + max(2, int(132 * float(preference))), y + 18),
            radius=5,
            fill=color,
        )
        value_text = f"{raw:.1%}" if rollout.algorithm == "ppo" else f"{raw:+.2f}"
        draw.text((653, y), value_text, font=small_font, fill=(15, 23, 42))

    channel_names = MINATAR_CHANNEL_NAMES.get(
        rollout.game,
        tuple(f"channel {index}" for index in range(rollout.observations.shape[-1])),
    )
    legend_y = 470
    legend_x = 28
    for index, name in enumerate(channel_names):
        if legend_x > 350 and index < len(channel_names) - 1:
            break
        color = tuple(int(value) for value in MINATAR_CHANNEL_COLORS[index])
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 13, legend_y + 13), radius=2, fill=color)
        draw.text((legend_x + 18, legend_y - 2), name, font=small_font, fill=muted)
        legend_x += 22 + int(draw.textlength(name, font=small_font))
    footer = "Greedy action; sticky-action probability = 0.1"
    if step_index == rollout.episode_length - 1:
        ending = "terminated" if rollout.terminated else "time limit"
        footer = f"Episode {ending} · total return {rollout.episode_return:.0f}"
    draw.text((438, 493), footer, font=small_font, fill=muted)
    return canvas


def save_minatar_policy_animation(
    rollout: MinAtarPolicyRollout,
    output_path: Path | str,
    fps: int = 8,
    max_rendered_frames: int = 300,
) -> Path:
    """Export an annotated GIF, decimating only unusually long episodes."""

    if fps < 1 or max_rendered_frames < 2:
        raise ValueError("fps and max_rendered_frames must be positive")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(np.ceil(rollout.episode_length / max_rendered_frames)))
    indices = list(range(0, rollout.episode_length, stride))
    if indices[-1] != rollout.episode_length - 1:
        indices.append(rollout.episode_length - 1)
    frames = [render_minatar_policy_frame(rollout, index) for index in indices]
    duration = int(round(1_000 / fps))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[duration] * (len(frames) - 1) + [duration * 5],
        loop=0,
        optimize=False,
        disposal=2,
    )
    return path


def save_minatar_policy_contact_sheet(
    rollout: MinAtarPolicyRollout,
    output_path: Path | str,
    frame_count: int = 12,
) -> Path:
    """Save an inspectable PNG summary sampled evenly across an episode."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(frame_count, rollout.episode_length)
    indices = np.linspace(0, rollout.episode_length - 1, count, dtype=int)
    frames = [render_minatar_policy_frame(rollout, int(index)) for index in indices]
    thumbnail_size = (380, 260)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    sheet = Image.new("RGB", (columns * thumbnail_size[0], rows * thumbnail_size[1]), (226, 232, 240))
    for index, frame in enumerate(frames):
        thumbnail = frame.resize(thumbnail_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumbnail_size[0]
        y = (index // columns) * thumbnail_size[1]
        sheet.paste(thumbnail, (x, y))
    sheet.save(path)
    return path
