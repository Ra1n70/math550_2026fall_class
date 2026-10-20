"""A tiny transparent Markov decision process and tabular Q-learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class GridWorld:
    """Deterministic grid used to introduce returns and Bellman updates."""

    rows: int = 5
    columns: int = 5
    start: tuple[int, int] = (4, 0)
    goal: tuple[int, int] = (0, 4)
    pit: tuple[int, int] = (1, 4)
    walls: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (3, 3))
    step_reward: float = -0.04

    def step(
        self, state: tuple[int, int], action: int
    ) -> tuple[tuple[int, int], float, bool]:
        if action not in range(len(ACTIONS)):
            raise ValueError("action must be an integer from 0 through 3")
        if self.is_terminal(state):
            raise ValueError("cannot step from a terminal state")
        row_delta, column_delta = ACTIONS[action]
        candidate = (state[0] + row_delta, state[1] + column_delta)
        outside = not (
            0 <= candidate[0] < self.rows and 0 <= candidate[1] < self.columns
        )
        next_state = state if outside or candidate in self.walls else candidate
        if next_state == self.goal:
            return next_state, 1.0, True
        if next_state == self.pit:
            return next_state, -1.0, True
        return next_state, self.step_reward, False

    def is_terminal(self, state: tuple[int, int]) -> bool:
        return state in {self.goal, self.pit}


def epsilon_greedy_action(
    q_values: np.ndarray,
    state: tuple[int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Sample an action while breaking greedy ties uniformly."""

    if rng.random() < epsilon:
        return int(rng.integers(len(ACTIONS)))
    values = q_values[state]
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def q_learning(
    environment: GridWorld,
    episodes: int = 1_500,
    learning_rate: float = 0.20,
    discount: float = 0.97,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    max_steps: int = 100,
    seed: int = 550,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn action values with the one-step off-policy Bellman update."""

    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    if not 0 < learning_rate <= 1 or not 0 <= discount <= 1:
        raise ValueError("invalid learning rate or discount")
    rng = np.random.default_rng(seed)
    q_values = np.zeros((environment.rows, environment.columns, len(ACTIONS)))
    returns = np.zeros(episodes)
    for episode in range(episodes):
        fraction = episode / max(episodes - 1, 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        state = environment.start
        for _ in range(max_steps):
            action = epsilon_greedy_action(q_values, state, epsilon, rng)
            next_state, reward, terminated = environment.step(state, action)
            bootstrap = 0.0 if terminated else float(q_values[next_state].max())
            target = reward + discount * bootstrap
            q_values[state][action] += learning_rate * (
                target - q_values[state][action]
            )
            returns[episode] += reward
            state = next_state
            if terminated:
                break
    return q_values, returns


def greedy_rollout(
    environment: GridWorld, q_values: np.ndarray, max_steps: int = 100
) -> list[tuple[int, int]]:
    """Return the deterministic greedy path for inspection."""

    state = environment.start
    path = [state]
    for _ in range(max_steps):
        if environment.is_terminal(state):
            return path
        action = int(np.argmax(q_values[state]))
        state, _, _ = environment.step(state, action)
        path.append(state)
    return path
