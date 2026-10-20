"""Rule-complete Connect Four state used by the AlphaZero self-play lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ROWS = 6
COLUMNS = 7


@dataclass(frozen=True, eq=False)
class ConnectFourState:
    """Immutable board with values -1, 0, and +1 and a player to move."""

    board: np.ndarray
    current_player: int = 1

    def __post_init__(self) -> None:
        board = np.asarray(self.board, dtype=np.int8)
        if board.shape != (ROWS, COLUMNS):
            raise ValueError(f"board must have shape {(ROWS, COLUMNS)}")
        if not np.isin(board, (-1, 0, 1)).all():
            raise ValueError("board entries must be -1, 0, or 1")
        if self.current_player not in (-1, 1):
            raise ValueError("current_player must be -1 or 1")
        board = board.copy()
        board.flags.writeable = False
        object.__setattr__(self, "board", board)

    @classmethod
    def new(cls) -> "ConnectFourState":
        return cls(np.zeros((ROWS, COLUMNS), dtype=np.int8), current_player=1)

    def legal_actions(self) -> np.ndarray:
        if self.is_terminal():
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.board[0] == 0).astype(np.int64)

    def play(self, column: int) -> "ConnectFourState":
        column = int(column)
        if column not in range(COLUMNS):
            raise ValueError("column must lie between 0 and 6")
        if self.is_terminal():
            raise ValueError("cannot play from a terminal state")
        empty_rows = np.flatnonzero(self.board[:, column] == 0)
        if len(empty_rows) == 0:
            raise ValueError("selected column is full")
        board = self.board.copy()
        board[int(empty_rows[-1]), column] = self.current_player
        return ConnectFourState(board, current_player=-self.current_player)

    def winner(self) -> int:
        board = self.board
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for row in range(ROWS):
            for column in range(COLUMNS):
                player = int(board[row, column])
                if player == 0:
                    continue
                for row_step, column_step in directions:
                    end_row = row + 3 * row_step
                    end_column = column + 3 * column_step
                    if not (0 <= end_row < ROWS and 0 <= end_column < COLUMNS):
                        continue
                    if all(
                        board[row + k * row_step, column + k * column_step] == player
                        for k in range(1, 4)
                    ):
                        return player
        return 0

    def is_terminal(self) -> bool:
        return self.winner() != 0 or not np.any(self.board == 0)

    def outcome(self, player: int) -> float:
        """Return a terminal result from ``player``'s perspective."""

        if not self.is_terminal():
            raise ValueError("outcome is defined only for terminal states")
        winner = self.winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == int(player) else -1.0

    def canonical_observation(self) -> np.ndarray:
        """Encode current pieces, opponent pieces, and player identity."""

        current = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        player_plane = np.full(
            (ROWS, COLUMNS), 1.0 if self.current_player == 1 else 0.0,
            dtype=np.float32,
        )
        return np.stack([current, opponent, player_plane], axis=0)

    def mirrored(self) -> "ConnectFourState":
        return ConnectFourState(np.fliplr(self.board), self.current_player)

    def key(self) -> bytes:
        return self.board.tobytes() + bytes([self.current_player + 1])
