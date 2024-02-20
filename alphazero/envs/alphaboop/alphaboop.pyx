# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True
from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.alphaboop.AlphaBoopLogic import Board

import numpy as np

NUM_PLAYERS = 2
MAX_TURNS = 100
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 7 if MULTI_PLANE_OBSERVATION else 1

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
        return self._board.pieces == other._board.pieces and self._player == other._player and self.turns == other.turns and self.turns == other.turns

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
        return 73

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return NUM_CHANNELS, 6, 6

    def valid_moves(self):
        return np.asarray(self._board.get_valid_moves((1, -1)[self._player]))

    def play_action(self, action: int) -> None:
        super().play_action(action)
        self._board.makeMove(action, (1, -1)[self._player])
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
            pieces = np.asarray(self._board.pieces[1:, :])
            player1 = np.where((pieces == 1) | (pieces == 2), pieces, 0)
            player2 = np.where((pieces == -1) | (pieces == -2), abs(pieces), 0)
            color = np.full_like(pieces, (1, -1)[self._player])
            if self._board.pieces[0][5] == 1:
                color = color * 2
            numOfGreyKittens = np.full_like(pieces, self._board.pieces[0][0])
            numOfOrangeKittens = np.full_like(pieces, self._board.pieces[0][1])
            numOfGreyCats = np.full_like(pieces, self._board.pieces[0][2])
            numOfOrangeCats = np.full_like(pieces, self._board.pieces[0][3])
            return np.array([player1, player2, color, numOfGreyKittens, numOfOrangeKittens, numOfGreyCats, numOfOrangeCats], dtype=np.float32)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi: np.ndarray, winstate) -> List[Tuple[Any, int]]:
        # mirror, rotational
        assert (len(pi) == 73)

        pi_kittens = np.reshape(pi[:36], (6, 6))
        pi_cats = np.reshape(pi[36:72], (6, 6))  # Adjusted to stop at 72, exclusive
        pass_turn = pi[72]  # This is the pass turn action
        result = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(np.asarray(self._board.pieces[1:, :]), i)  # Exclude the first row
                new_pi_kittens = np.rot90(pi_kittens, i)
                new_pi_cats = np.rot90(pi_cats, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi_kittens = np.fliplr(new_pi_kittens)
                    new_pi_cats = np.fliplr(new_pi_cats)

                gs = self.clone()
                gs._board.pieces = np.concatenate([self._board.pieces[:1, :], new_b], axis=0)  # Concatenate the first row back
                # Adjusted to include the pass turn action at the end
                result.append((gs, np.concatenate((new_pi_kittens.ravel(), new_pi_cats.ravel(), [pass_turn])), winstate))

        return result

def display(game, action=None):
    if action is not None:
        print(f'Action: {action}, Move: {action + 1}')
    print(" -----------------------")

    for i, row in enumerate(game._board.pieces):
        if i == 0:
            # Leave the first row as white (uncolored)
            print(' '.join("{:>2}".format(cell) for cell in row))
        else:
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
