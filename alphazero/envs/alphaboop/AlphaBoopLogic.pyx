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
    AlphaBoop Board.
    """

    cdef int height
    cdef int width
    cdef int length
    cdef int win_length
    cdef public int[:,:] pieces

    def __init__(self):
        """Set up initial board configuration."""
        self.pieces = np.zeros((7, 6), dtype=np.intc)
        self.pieces[0, 0] = 8  # grey kittens
        self.pieces[0, 1] = 8  # orange kittens
        self.pieces[0, 2] = 0  # grey cats
        self.pieces[0, 3] = 0  # orange cats

    def get_grey_kittens(self):
        return self.pieces[0, 0]

    def get_orange_kittens(self):
        return self.pieces[0, 1]

    def get_grey_cats(self):
        return self.pieces[0, 2]

    def get_orange_cats(self):
        return self.pieces[0, 3]

    def set_grey_kittens(self, int value):
        self.pieces[0, 0] = value

    def set_orange_kittens(self, int value):
        self.pieces[0, 1] = value

    def set_grey_cats(self, int value):
        self.pieces[0, 2] = value

    def set_orange_cats(self, int value):
        self.pieces[0, 3] = value

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
            new_row, new_col = row + dr + 1, col + dc  # add 1 to row to account for first row
            if 1 <= new_row < 7 and 0 <= new_col < 6:  # check if in bounds, first row excluded
                if self.pieces[new_row, new_col] != 0:  # check if the spot is not empty
                    # Do not move if the current piece is a kitten and the piece to be moved is a cat
                    if abs(self.pieces[row+1, col]) == 1 and abs(self.pieces[new_row, new_col]) == 2:
                        continue
                    new_new_row, new_new_col = new_row + dr, new_col + dc
                    if 1 <= new_new_row < 7 and 0 <= new_new_col < 6:  # check if new spot is in bounds, first row excluded
                        if self.pieces[new_new_row, new_new_col] == 0:  # check if new spot is unoccupied
                            self.pieces[new_new_row, new_new_col] = self.pieces[new_row, new_col]  # move piece
                            self.pieces[new_row, new_col] = 0  # empty old spot
                    else:
                        # If the new spot is out of bounds, remove the piece and update the counts
                        if self.pieces[new_row, new_col] == 1:  # Grey Kitten
                            self.set_grey_kittens(self.get_grey_kittens() + 1)
                        elif self.pieces[new_row, new_col] == -1:  # Orange Kitten
                            self.set_orange_kittens(self.get_orange_kittens() + 1)
                        elif self.pieces[new_row, new_col] == 2:  # Grey Cat
                            self.set_grey_cats(self.get_grey_cats() + 1)
                        elif self.pieces[new_row, new_col] == -2:  # Orange Cat
                            self.set_orange_cats(self.get_orange_cats() + 1)
                        self.pieces[new_row, new_col] = 0  # empty old spot

    def check_three_in_a_row(self, int player):
        cdef Py_ssize_t r, c, dr, dc, new_row, new_col
        cdef int total_cats, total_kittens
        cdef list directions = [(0, 1), (1, 0), (-1, 1), (1, 1)]  # 4 directions
        three_in_a_rows = []  # List to store the coordinates of each three-in-a-row

        for r in range(1, 7):  # Start from 1 to skip the first row
            for c in range(6):
                if self.pieces[r, c] == player or self.pieces[r, c] == 2 * player:  # If there is a piece of the player
                    for direction in directions:
                        dr, dc = direction
                        total_cats = 0
                        total_kittens = 0
                        temp_coordinates = []

                        for i in range(3):  # Check the next 2 pieces in the direction
                            new_row, new_col = r + i * dr, c + i * dc
                            if 1 <= new_row < 7 and 0 <= new_col < 6:  # Check if in bounds
                                temp_coordinates.append((new_row, new_col))
                                if self.pieces[new_row, new_col] == player:  # If there is a kitten of the player
                                    total_kittens += 1
                                elif self.pieces[new_row, new_col] == 2 * player:  # If there is a cat of the player
                                    total_cats += 1
                                else:  # If there is a piece of the other player or no piece
                                    break
                        else:  # If the loop didn't break (there are 3 pieces of the player in a row)
                            if (total_cats == 1 and total_kittens == 2) or (total_cats == 2 and total_kittens == 1) or (total_kittens == 3):  # If there is 1 cat and 2 kittens or 2 cats and 1 kitten or 3 kittens
                                three_in_a_rows.append(temp_coordinates)
        return three_in_a_rows

    def removeThreeInRow(self, three_in_a_rows, player):
        cdef Py_ssize_t new_row, new_col

        # Only execute the removal and count updating if exactly one three-in-a-row is found
        if len(three_in_a_rows) == 1:
            for coord in three_in_a_rows[0]:
                new_row, new_col = coord
                self.pieces[new_row, new_col] = 0
                if player == 1:
                    self.set_grey_cats(self.get_grey_cats() + 1)
                else:
                    self.set_orange_cats(self.get_orange_cats() + 1)

    def makeMove(self, int action, int player):
        """Place a piece on the board."""
        cdef Py_ssize_t row, col, r, c
        row = action // 6  # Integer division to get the row index
        col = action % 6  # Modulus to get the column index

        if action == 72:
            self.pieces[0, 5] = 0
            return

        three_in_a_rows = self.check_three_in_a_row(player)
        if len(three_in_a_rows) > 1 and action < 36:
            # Find and remove the specific sequence from the board
            for sequence in three_in_a_rows:
                midpoint = sequence[1]  # Assuming the sequence is always of length 3
                r, c = midpoint
                r = r - 1
                if row == r and col == c:
                    # Remove all pieces in this sequence from the board
                    for r, c in sequence:
                        self.pieces[r, c] = 0  # Removing the piece
                        if player == 1:
                            self.set_grey_cats(self.get_grey_cats() + 1)
                        else:
                            self.set_orange_cats(self.get_orange_cats() + 1)
                    break  # Stop searching once the sequence is found and removed

        elif action < 36:  # If action is between 0 and 35, place a kitten
            if self.pieces[row+1, col] == 0:  # Increment the row index to account for the first row
                self.pieces[row+1, col] = player
                if player == 1:
                    self.set_grey_kittens(self.get_grey_kittens() - 1)  # Use getters and setters to manipulate count
                else:
                    self.set_orange_kittens(self.get_orange_kittens() - 1)  # Use getters and setters to manipulate count
                self.boop(row, col)
                three_in_a_rows = self.check_three_in_a_row(player)
                self.removeThreeInRow(three_in_a_rows, player)
                if len(three_in_a_rows) > 1:
                    self.pieces[0, 5] = 1
                elif self.isAllEightFelinesOnBoard(player):
                    self.pieces[0, 5] = 1
            else:
                raise ValueError("Can't play action %s on board %s" % (action, self))
        else:  # If action is between 36 and 71, place a cat
            action -= 36  # Adjust action to map to the correct cell in the 6x6 grid
            row = action // 6  # Integer division to get the row index
            col = action % 6  # Modulus to get the column index
            if self.pieces[row+1, col] == 0:  # Increment the row index to account for the first row
                self.pieces[row+1, col] = 2 * player
                if player == 1:
                    self.set_grey_cats(self.get_grey_cats() - 1)  # Use getters and setters to manipulate count
                else:
                    self.set_orange_cats(self.get_orange_cats() - 1)  # Use getters and setters to manipulate count
                self.boop(row, col)
                three_in_a_rows = self.check_three_in_a_row(player)
                self.removeThreeInRow(three_in_a_rows, player)
                if len(three_in_a_rows) > 1:
                    self.pieces[0, 5] = 1
                elif self.isAllEightFelinesOnBoard(player):
                    self.pieces[0, 5] = 1
            elif abs(self.pieces[row+1, col]) == 1:
                self.pieces[row+1, col] = 0
                if player == 1:
                    self.set_grey_cats(self.get_grey_cats() + 1)  # Use getters and setters to manipulate count
                else:
                    self.set_orange_cats(self.get_orange_cats() + 1)  # Use getters and setters to manipulate count
            else:
                raise ValueError("Can't play action %s on board %s" % (action, self))

    def isAllEightFelinesOnBoard(self, player):
        cdef int feline_count = 0

        # Count the number of kittens on the board for the current player
        for r in range(1, 7):
            for c in range(6):
                if self.pieces[r,c] == player or self.pieces[r,c] == 2 * player:
                    feline_count += 1

        if feline_count == 8:
            return True
        else:
            return False

    def get_valid_moves(self, int player):
        cdef Py_ssize_t r, c
        cdef int[:] valid = np.zeros(73, dtype=np.intc)

        three_in_a_rows = self.check_three_in_a_row(player)

        if self.pieces[0, 5] == 1:
            valid[72] = 1
            return valid

        # If the player has 8 kittens on the board, only the locations of those kittens are marked as valid
        if self.isAllEightFelinesOnBoard(player):
            for r in range(1, 7):  # Start from the second row
                for c in range(6):
                    if self.pieces[r,c] == player:
                        valid[36 + (r-1)*6 + c] = 1

        if len(three_in_a_rows) > 1:
            for sequence in three_in_a_rows:
                # Calculate the midpoint of the sequence
                midpoint = sequence[1]
                r, c = midpoint
                valid[(r-1)*6 + c] = 1
        else:
            for r in range(1, 7):  # Start from the second row
                for c in range(6):
                    if self.pieces[r,c] == 0:
                        if (player == 1 and self.get_grey_kittens() > 0) or (player == -1 and self.get_orange_kittens() > 0):
                            valid[(r-1)*6 + c] = 1
                        if (player == 1 and self.get_grey_cats() > 0) or (player == -1 and self.get_orange_cats() > 0):
                            valid[36 + (r-1)*6 + c] = 1

        return valid

    def get_win_state(self):
        cdef int player
        cdef int total
        cdef int good
        cdef Py_ssize_t r, c, x
        for player in [1, -1]:
            # Check for 8 cats on the board
            total = 0
            for r in range(1, 7):  # Start from the second row
                for c in range(6):
                    if self.pieces[r, c] == 2 * player:
                        total += 1
            if total == 8:
                return (True, player)
            # Check row wins
            for r in range(1, 7):  # Start from the second row
                total = 0
                for c in range(6):
                    if self.pieces[r,c] == 2 * player:
                        total += 1
                    else:
                        total = 0
                    if total == 3:  # Check for three in a row
                        return (True, player)
            # Check column wins
            for c in range(6):
                total = 0
                for r in range(1, 7):  # Start from the second row
                    if self.pieces[r,c] == 2 * player:
                        total += 1
                    else:
                        total = 0
                    if total == 3:  # Check for three in a row
                        return (True, player)
            # Check diagonals
            for r in range(1, 7 - 3 + 1):  # Start from the second row
                for c in range(6 - 3 + 1):
                    good = True
                    for x in range(3):
                        if self.pieces[r+x,c+x] != 2 * player:
                            good = False
                            break
                    if good:
                        return (True, player)
                for c in range(3 - 1, 6):
                    good = True
                    for x in range(3):
                        if self.pieces[r+x,c-x] != 2 * player:
                            good = False
                            break
                    if good:
                        return (True, player)

        # Game is not ended yet.
        return (False, 0)

    def __str__(self):
        return str(np.asarray(self.pieces))
