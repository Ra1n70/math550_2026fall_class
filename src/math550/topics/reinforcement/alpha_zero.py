"""A compact, complete AlphaZero-style policy/value and MCTS implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .connect_four import COLUMNS, ConnectFourState
from .devices import resolve_torch_device


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.norm1(self.conv1(inputs)))
        return F.relu(inputs + self.norm2(self.conv2(hidden)))


class ConnectFourPolicyValueNetwork(nn.Module):
    """Shared residual trunk with explicit policy and value heads."""

    def __init__(self, channels: int = 48, residual_blocks: int = 3) -> None:
        super().__init__()
        if channels < 4 or residual_blocks < 1:
            raise ValueError("channels and residual_blocks are too small")
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, COLUMNS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(self.stem(observations))
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(1)
        return logits, values


def alpha_zero_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy cross-entropy plus squared value error."""

    policy_loss = -(target_policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_values)
    return policy_loss + value_loss, policy_loss, value_loss


@dataclass
class SearchNode:
    prior: float
    to_play: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    def expand(self, state: ConnectFourState, policy: np.ndarray) -> None:
        for action in state.legal_actions():
            self.children[int(action)] = SearchNode(
                prior=float(policy[int(action)]), to_play=-state.current_player
            )


def _network_prediction(
    network: nn.Module, state: ConnectFourState, device: torch.device
) -> tuple[np.ndarray, float]:
    network.eval()
    with torch.no_grad():
        observation = torch.as_tensor(
            state.canonical_observation()[None], dtype=torch.float32, device=device
        )
        logits, value = network(observation)
        policy = torch.softmax(logits, dim=1)[0].cpu().numpy()
    legal = state.legal_actions()
    mask = np.zeros(COLUMNS, dtype=float)
    mask[legal] = 1.0
    policy = policy * mask
    total = policy.sum()
    if total <= 0:
        policy = mask / mask.sum()
    else:
        policy /= total
    return policy, float(value.item())


def _select_child(node: SearchNode, exploration_constant: float) -> tuple[int, SearchNode]:
    def score(child: SearchNode) -> float:
        prior_score = (
            exploration_constant
            * child.prior
            * math.sqrt(node.visit_count + 1)
            / (child.visit_count + 1)
        )
        return -child.value + prior_score

    return max(node.children.items(), key=lambda item: (score(item[1]), -item[0]))


def _backpropagate(
    search_path: list[SearchNode], value: float, value_player: int
) -> None:
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == value_player else -value
        node.visit_count += 1


def run_mcts(
    network: nn.Module,
    state: ConnectFourState,
    simulations: int = 80,
    exploration_constant: float = 1.5,
    device: str | torch.device = "auto",
    add_root_noise: bool = False,
    dirichlet_alpha: float = 0.30,
    noise_fraction: float = 0.25,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, SearchNode]:
    """Run PUCT search and return the improved visit-count policy."""

    if state.is_terminal():
        raise ValueError("MCTS requires a non-terminal state")
    if simulations < 1 or exploration_constant <= 0:
        raise ValueError("simulations and exploration_constant must be positive")
    resolved = device if isinstance(device, torch.device) else resolve_torch_device(device)
    network.to(resolved)
    rng = np.random.default_rng() if rng is None else rng
    root = SearchNode(prior=1.0, to_play=state.current_player)
    policy, _ = _network_prediction(network, state, resolved)
    root.expand(state, policy)
    if add_root_noise:
        actions = list(root.children)
        noise = rng.dirichlet(np.full(len(actions), dirichlet_alpha))
        for action, amount in zip(actions, noise, strict=True):
            child = root.children[action]
            child.prior = (1.0 - noise_fraction) * child.prior + noise_fraction * amount

    for _ in range(simulations):
        node = root
        simulation_state = state
        search_path = [node]
        while node.expanded:
            action, node = _select_child(node, exploration_constant)
            simulation_state = simulation_state.play(action)
            search_path.append(node)
            if simulation_state.is_terminal():
                break
        if simulation_state.is_terminal():
            value = simulation_state.outcome(simulation_state.current_player)
        else:
            leaf_policy, value = _network_prediction(network, simulation_state, resolved)
            node.expand(simulation_state, leaf_policy)
        _backpropagate(search_path, value, simulation_state.current_player)

    visit_policy = np.zeros(COLUMNS, dtype=np.float32)
    for action, child in root.children.items():
        visit_policy[action] = child.visit_count
    visit_policy /= visit_policy.sum()
    return visit_policy, root


@dataclass(frozen=True)
class SelfPlayExample:
    observation: np.ndarray
    policy: np.ndarray
    value: float


def self_play_game(
    network: nn.Module,
    simulations: int = 80,
    temperature_moves: int = 12,
    device: str = "auto",
    seed: int = 550,
    add_root_noise: bool = True,
) -> tuple[list[SelfPlayExample], int, int]:
    """Generate one game of search-improved targets from the current network."""

    rng = np.random.default_rng(seed)
    state = ConnectFourState.new()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move = 0
    while not state.is_terminal():
        policy, _ = run_mcts(
            network,
            state,
            simulations=simulations,
            device=device,
            add_root_noise=add_root_noise,
            rng=rng,
        )
        history.append((state.canonical_observation(), policy.copy(), state.current_player))
        if move < temperature_moves:
            action = int(rng.choice(COLUMNS, p=policy))
        else:
            action = int(np.argmax(policy))
        state = state.play(action)
        move += 1
    winner = state.winner()
    examples = [
        SelfPlayExample(
            observation=observation,
            policy=policy,
            value=0.0 if winner == 0 else (1.0 if player == winner else -1.0),
        )
        for observation, policy, player in history
    ]
    return examples, winner, move


def augment_reflections(examples: list[SelfPlayExample]) -> list[SelfPlayExample]:
    """Add the exact left-right symmetry of Connect Four."""

    augmented: list[SelfPlayExample] = []
    for example in examples:
        augmented.append(example)
        augmented.append(
            SelfPlayExample(
                observation=np.flip(example.observation, axis=2).copy(),
                policy=example.policy[::-1].copy(),
                value=example.value,
            )
        )
    return augmented


def train_policy_value_network(
    network: nn.Module,
    examples: list[SelfPlayExample],
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    seed: int = 550,
) -> dict[str, list[float]]:
    """Fit policy and value targets with an explicit PyTorch minibatch loop."""

    if not examples or epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("examples and optimization settings must be positive")
    resolved = resolve_torch_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network.to(resolved)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    observations = np.stack([example.observation for example in examples])
    policies = np.stack([example.policy for example in examples])
    values = np.asarray([example.value for example in examples], dtype=np.float32)
    history = {"total": [], "policy": [], "value": []}
    for _ in range(epochs):
        ordering = rng.permutation(len(examples))
        totals = np.zeros(3)
        seen = 0
        network.train()
        for start in range(0, len(examples), batch_size):
            indices = ordering[start : start + batch_size]
            batch_observations = torch.as_tensor(
                observations[indices], dtype=torch.float32, device=resolved
            )
            batch_policies = torch.as_tensor(
                policies[indices], dtype=torch.float32, device=resolved
            )
            batch_values = torch.as_tensor(
                values[indices], dtype=torch.float32, device=resolved
            )
            logits, predictions = network(batch_observations)
            total, policy_loss, value_loss = alpha_zero_loss(
                logits, predictions, batch_policies, batch_values
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            count = len(indices)
            totals += count * np.array(
                [float(total.detach().cpu()), float(policy_loss.detach().cpu()),
                 float(value_loss.detach().cpu())]
            )
            seen += count
        for key, value in zip(history, totals / seen, strict=True):
            history[key].append(float(value))
    return history


def evaluate_against_random(
    network: nn.Module,
    games: int = 20,
    simulations: int = 40,
    device: str = "auto",
    seed: int = 550,
) -> dict[str, float]:
    """Alternate colors and score search play against a uniform legal policy."""

    if games < 2 or simulations < 1:
        raise ValueError("use at least two games and one simulation")
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    for game in range(games):
        network_player = 1 if game % 2 == 0 else -1
        state = ConnectFourState.new()
        while not state.is_terminal():
            if state.current_player == network_player:
                policy, _ = run_mcts(
                    network,
                    state,
                    simulations=simulations,
                    device=device,
                    add_root_noise=False,
                    rng=rng,
                )
                action = int(np.argmax(policy))
            else:
                action = int(rng.choice(state.legal_actions()))
            state = state.play(action)
        winner = state.winner()
        if winner == network_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    return {
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "score": (wins + 0.5 * draws) / games,
    }
