# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True
# cython: profile=True

import numpy as np


cdef class Board():
    """
    MiniBoop Board.
    """

    cdef int height
    cdef int width
    cdef int length
    cdef int win_length
    cdef public int[:,:] pieces

    def __init__(self):
        """Set up initial board configuration."""
        self.pieces = np.zeros((6, 6), dtype=np.intc)
		
    def __getstate__(self):
        return self.height, self.width, self.win_length, np.asarray(self.pieces)

    def __setstate__(self, state):
        self.height, self.width, self.win_length, pieces = state
        self.pieces = np.asarray(pieces)
		
    def boop(self, int row, int col):
        cdef int dr, dc, new_row, new_col, new_new_row, new_new_col
        cdef list directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]  # 8 directions
        for direction in directions:
            dr, dc = direction
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 6 and 0 <= new_col < 6:  # check if in bounds
                if self.pieces[new_row, new_col] != 0:  # check if the spot is not empty
                    new_new_row, new_new_col = new_row + dr, new_col + dc
                    if 0 <= new_new_row < 6 and 0 <= new_new_col < 6:  # check if new spot is in bounds
                        if self.pieces[new_new_row, new_new_col] == 0:  # check if new spot is unoccupied
                            self.pieces[new_new_row, new_new_col] = self.pieces[new_row, new_col]  # move piece
                            self.pieces[new_row, new_col] = 0  # empty old spot
                    else:
                        # If the new spot is out of bounds, remove the piece and update the counts
                        self.pieces[new_row, new_col] = 0
						
    def add_kitten(self, int action, int player):
        """Place a piece on the board."""
        cdef Py_ssize_t row, col
        row = action // 6  # Integer division to get the row index
        col = action % 6  # Modulus to get the column index

        if self.pieces[row, col] == 0:
            self.pieces[row, col] = player
            self.boop(row, col)
        else:
            raise ValueError("Can't play action %s on board %s" % (action, self))

    def get_valid_moves(self):
        cdef Py_ssize_t r, c
        cdef int[:] valid = np.zeros(36, dtype=np.intc)
        for r in range(6):
            for c in range(6):
                if self.pieces[r,c] == 0:
                    valid[r*6 + c] = 1

        return valid

    def get_win_state(self):
        cdef int player
        cdef int total
        cdef int good
        cdef Py_ssize_t r, c, x
        for player in [1, -1]:
            # Check for 8 kittens on the board
            total = 0
            for r in range(6):
                for c in range(6):
                    if self.pieces[r, c] == player:
                        total += 1
            if total == 8:
                return (True, player)
            #check row wins
            for r in range(6):
                total = 0
                for c in range(6):
                    if self.pieces[r,c] == player:
                        total += 1
                    else:
                        total = 0
                    if total == 3:  # Check for three in a row
                        return (True, player)
            #check column wins
            for c in range(6):
                total = 0
                for r in range(6):
                    if self.pieces[r,c] == player:
                        total += 1
                    else:
                        total = 0
                    if total == 3:  # Check for three in a row
                        return (True, player)
            #check diagonal
            for r in range(6 - 3 + 1):
                for c in range(6 - 3 + 1):
                    good = True
                    for x in range(3):
                        if self.pieces[r+x,c+x] != player:
                            good = False
                            break
                    if good:
                        return (True, player)
                for c in range(3 - 1, 6):
                    good = True
                    for x in range(3):
                        if self.pieces[r+x,c-x] != player:
                            good = False
                            break
                    if good:
                        return (True, player)

        # Game is not ended yet.
        return (False, 0)

    def __str__(self):
        return str(np.asarray(self.pieces))
