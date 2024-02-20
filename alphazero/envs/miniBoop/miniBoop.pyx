# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True
from typing import List, Tuple, Any

from Game import GameState
from .miniBoopLogic import Board

import numpy as np

NUM_PLAYERS = 2
MAX_TURNS = 100
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 4 if MULTI_PLANE_OBSERVATION else 1

class Game(GameState):
    def __init__(self):
        super().__init__(self._get_board())

    @staticmethod
    def _get_board():
        return Board()

    def set_board(self, newBoardPieces, newPlayer):
        self._board.pieces = newBoardPieces
        self._player = newPlayer

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + bytes([self.turns]) + bytes([self._player]))

    def __eq__(self, other: 'Game') -> bool:
        return self._board.pieces == other._board.pieces and self._player == other._player and self.turns == other.turns

    def clone(self) -> 'Game':
        game = Game()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
        game._player = self._player
        game._turns = self.turns
        game.last_action = self.last_action
        return game

    @staticmethod
    def max_turns() -> int:
        return MAX_TURNS

    @staticmethod
    def has_draw() -> bool:
        return False

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size() -> int:
        return 36

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return NUM_CHANNELS, 6, 6

    def valid_moves(self):
        return np.asarray(self._board.get_valid_moves())

    def play_action(self, action: int) -> None:
        super().play_action(action)
        self._board.add_kitten(action, (1, -1)[self.player])
        self._update_turn()

    def win_state(self) -> np.ndarray:
        result = [False] * 2
        game_over, player = self._board.get_win_state()

        if game_over:
            index = -1
            if player == 1:
                index = 0
            elif player == -1:
                index = 1
            result[index] = True

        return np.array(result, dtype=np.uint8)

    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces)
            player1 = np.where(pieces == 1, 1, 0)
            player2 = np.where(pieces == -1, 1, 0)
            colour = np.full_like(pieces, self.player)
            turn = np.full_like(pieces, self.turns / MAX_TURNS, dtype=np.float32)
            return np.array([player1, player2, colour, turn], dtype=np.float32)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi: np.ndarray, winstate) -> List[Tuple[Any, int]]:
        # mirror, rotational
        assert (len(pi) == 36)

        pi_board = np.reshape(pi, (6, 6))
        result = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(np.asarray(self._board.pieces), i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)

                gs = self.clone()
                gs._board.pieces = new_b
                result.append((gs, new_pi.ravel(), winstate))

        return result


def display(board, action=None):
    if action:
        print(f'Action: {action}, Move: {action + 1}')
    print(" -----------------------")
    #print(' '.join(map(str, range(len(board[0])))))
    for row in board:
        for cell in row:
            if cell > 0:
                # Grey for positive numbers
                print("\033[90m{:>2}\033[0m".format(cell), end=" ")
            elif cell < 0:
                # Orange (or closest approximation) for negative numbers
                print("\033[91m{:>2}\033[0m".format(cell), end=" ")
            else:
                # Blue for zero
                print("\033[94m{:>2}\033[0m".format(cell), end=" ")
        print()  # Newline after each row
    print("Action: " + str(action))
    print(" -----------------------")
