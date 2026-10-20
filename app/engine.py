"""Connect-4 game engine with board state, rules, and tensor conversion."""

import copy
import numpy as np


ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


class ConnectFourGame:
    """Connect-4 game state and logic.

    Board layout: row 0 = top, row 5 = bottom.
    Values: 0 = empty, 1 = player 1, 2 = player 2.
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = PLAYER1
        self.move_count = 0
        self._last_winner = None

    def get_valid_moves(self):
        """Return list of columns that are not full."""
        return [c for c in range(COLS) if self.board[0][c] == EMPTY]

    def make_move(self, col):
        """Drop a piece into the given column.

        Returns the row where the piece landed, or -1 if the column is full.
        Toggles current_player after a successful move.
        """
        if col < 0 or col >= COLS or self.board[0][col] != EMPTY:
            return -1
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                self.move_count += 1
                self._last_winner = None
                self.current_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1
                return row
        return -1

    def undo_move(self, col):
        """Remove the topmost piece from the given column.

        Toggles current_player back. Used by minimax to avoid cloning.
        """
        for row in range(ROWS):
            if self.board[row][col] != EMPTY:
                self.board[row][col] = EMPTY
                self.move_count -= 1
                self._last_winner = None
                self.current_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1
                return
        raise ValueError(f"Cannot undo move on empty column {col}")

    def check_winner(self):
        """Return the winning player (1 or 2), or 0 if no winner yet."""
        if self._last_winner is not None:
            return self._last_winner

        b = self.board
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                val = b[r][c]
                if val != EMPTY and val == b[r][c+1] == b[r][c+2] == b[r][c+3]:
                    self._last_winner = val
                    return val
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                val = b[r][c]
                if val != EMPTY and val == b[r+1][c] == b[r+2][c] == b[r+3][c]:
                    self._last_winner = val
                    return val
        # Diagonal down-right
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                val = b[r][c]
                if val != EMPTY and val == b[r+1][c+1] == b[r+2][c+2] == b[r+3][c+3]:
                    self._last_winner = val
                    return val
        # Diagonal up-right
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                val = b[r][c]
                if val != EMPTY and val == b[r-1][c+1] == b[r-2][c+2] == b[r-3][c+3]:
                    self._last_winner = val
                    return val

        self._last_winner = 0
        return 0

    def is_full(self):
        """Return True if the board has no empty cells."""
        return self.move_count >= ROWS * COLS

    def is_terminal(self):
        """Return True if the game is over (winner or draw)."""
        return self.check_winner() != 0 or self.is_full()

    def to_tensor(self):
        """Convert board to a 3x6x7 one-hot tensor (numpy array).

        Channel 0: current player's pieces
        Channel 1: opponent's pieces
        Channel 2: empty cells

        This perspective-relative encoding ensures the model works
        equally well as either player.
        """
        opponent = PLAYER2 if self.current_player == PLAYER1 else PLAYER1
        tensor = np.zeros((3, ROWS, COLS), dtype=np.float32)
        tensor[0] = (self.board == self.current_player).astype(np.float32)
        tensor[1] = (self.board == opponent).astype(np.float32)
        tensor[2] = (self.board == EMPTY).astype(np.float32)
        return tensor

    def clone(self):
        """Return a deep copy of this game state."""
        new_game = ConnectFourGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        new_game._last_winner = self._last_winner
        return new_game

    def __str__(self):
        symbols = {EMPTY: '.', PLAYER1: 'X', PLAYER2: 'O'}
        rows = []
        for r in range(ROWS):
            rows.append(' '.join(symbols[self.board[r][c]] for c in range(COLS)))
        rows.append('0 1 2 3 4 5 6')
        return '\n'.join(rows)


if __name__ == '__main__':
    print("=== Connect-4 Engine Tests ===\n")
    errors = 0

    # Test 1: Horizontal win
    g = ConnectFourGame()
    for c in [0, 0, 1, 1, 2, 2, 3]:
        g.make_move(c)
    assert g.check_winner() == PLAYER1, "Horizontal win failed"
    print("[PASS] Horizontal win")

    # Test 2: Vertical win
    g = ConnectFourGame()
    for c in [0, 1, 0, 1, 0, 1, 0]:
        g.make_move(c)
    assert g.check_winner() == PLAYER1, "Vertical win failed"
    print("[PASS] Vertical win")

    # Test 3: Diagonal down-right win
    g = ConnectFourGame()
    moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]
    for c in moves:
        g.make_move(c)
    assert g.check_winner() == PLAYER1, "Diagonal down-right win failed"
    print("[PASS] Diagonal down-right win")

    # Test 4: Diagonal up-right win
    g = ConnectFourGame()
    moves = [3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0]
    for c in moves:
        g.make_move(c)
    assert g.check_winner() == PLAYER1, "Diagonal up-right win failed"
    print("[PASS] Diagonal up-right win")

    # Test 5: No winner
    g = ConnectFourGame()
    g.make_move(0)
    g.make_move(1)
    assert g.check_winner() == 0, "False winner detected"
    print("[PASS] No winner")

    # Test 6: Column overflow
    g = ConnectFourGame()
    for _ in range(ROWS):
        g.make_move(0)
    assert g.make_move(0) == -1, "Column overflow not detected"
    assert 0 not in g.get_valid_moves(), "Full column in valid moves"
    print("[PASS] Column overflow")

    # Test 7: Undo move symmetry
    g = ConnectFourGame()
    original_board = g.board.copy()
    for c in [3, 4, 3, 4, 3]:
        g.make_move(c)
    for c in [3, 4, 3, 4, 3]:
        g.undo_move(c)
    assert np.array_equal(g.board, original_board), "Undo move asymmetry"
    assert g.current_player == PLAYER1, "Player not restored after undo"
    print("[PASS] Undo move symmetry")

    # Test 8: Clone independence
    g = ConnectFourGame()
    g.make_move(3)
    clone = g.clone()
    clone.make_move(4)
    assert g.board[5][4] == EMPTY, "Clone modified original"
    print("[PASS] Clone independence")

    # Test 9: Tensor encoding
    g = ConnectFourGame()
    g.make_move(0)  # P1 at (5,0), now P2's turn
    t = g.to_tensor()
    # Current player is P2, so channel 0 should be P2's pieces (none yet)
    assert t[0][5][0] == 0.0, "Tensor channel 0 wrong (should be current player)"
    # Channel 1 should be opponent (P1) pieces
    assert t[1][5][0] == 1.0, "Tensor channel 1 wrong (should be opponent)"
    # Channel 2 should be empty
    assert t[2][5][0] == 0.0, "Tensor channel 2 wrong for occupied cell"
    assert t[2][0][0] == 1.0, "Tensor channel 2 wrong for empty cell"
    print("[PASS] Tensor encoding")

    # Test 10: Full board detection
    g = ConnectFourGame()
    # Fill board in a pattern that produces no winner
    pattern = [
        [1, 2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 2, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 2, 1],
        [2, 1, 2, 1, 2, 1, 2],
    ]
    g.board = np.array(pattern, dtype=np.int8)
    g.move_count = 42
    assert g.is_full(), "Full board not detected"
    assert g.check_winner() == 0, "False winner on draw board"
    assert g.is_terminal(), "Terminal not detected on full board"
    print("[PASS] Full board / draw detection")

    print(f"\n=== All tests passed ===")
