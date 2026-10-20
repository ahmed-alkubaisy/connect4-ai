"""Alpha-Beta Minimax agent with heuristic evaluation for Connect-4."""

import time
from app.agents.base import Agent
from app.engine import ROWS, COLS, EMPTY, PLAYER1, PLAYER2


# Center-first move ordering for better alpha-beta pruning
MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]

# Precompute all 69 possible 4-cell windows on a 6x7 board
WINDOWS = []
for r in range(ROWS):
    for c in range(COLS - 3):
        WINDOWS.append([(r, c), (r, c+1), (r, c+2), (r, c+3)])  # Horizontal
for r in range(ROWS - 3):
    for c in range(COLS):
        WINDOWS.append([(r, c), (r+1, c), (r+2, c), (r+3, c)])  # Vertical
for r in range(ROWS - 3):
    for c in range(COLS - 3):
        WINDOWS.append([(r, c), (r+1, c+1), (r+2, c+2), (r+3, c+3)])  # Diag ↘
for r in range(3, ROWS):
    for c in range(COLS - 3):
        WINDOWS.append([(r, c), (r-1, c+1), (r-2, c+2), (r-3, c+3)])  # Diag ↗


class MinimaxAgent(Agent):
    """Alpha-Beta Minimax agent with configurable search depth.

    Uses mutate-and-restore (make_move/undo_move) instead of cloning
    for 10-100x faster search. The caller must pass a cloned game
    object if thread safety is needed.
    """

    def __init__(self, player_id=PLAYER1, depth=6):
        self.player_id = player_id
        self.depth = depth

    @property
    def name(self):
        return "Minimax"

    def select_move(self, game):
        start = time.perf_counter()
        score, move = self._alpha_beta(
            game, self.depth, float('-inf'), float('inf'), True
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if move is None:
            valid = game.get_valid_moves()
            move = valid[0] if valid else 0

        return {
            'move': move,
            'score': score,
            'depth': self.depth,
            'compute_time_ms': elapsed_ms,
        }

    def _alpha_beta(self, game, depth, alpha, beta, maximizing):
        """Recursive alpha-beta search.

        maximizing=True means it's this agent's turn to move.
        Returns (score, best_move).
        """
        winner = game.check_winner()
        if winner == self.player_id:
            return 100_000 + depth, None  # Prefer faster wins
        if winner != 0:
            return -100_000 - depth, None  # Opponent won
        if game.is_full():
            return 0, None
        if depth == 0:
            return self._evaluate(game), None

        valid_moves = [c for c in MOVE_ORDER if c in game.get_valid_moves()]

        if maximizing:
            max_eval = float('-inf')
            best_move = valid_moves[0]
            for col in valid_moves:
                game.make_move(col)
                eval_score, _ = self._alpha_beta(game, depth - 1, alpha, beta, False)
                game.undo_move(col)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = col
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = valid_moves[0]
            for col in valid_moves:
                game.make_move(col)
                eval_score, _ = self._alpha_beta(game, depth - 1, alpha, beta, True)
                game.undo_move(col)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = col
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def _evaluate(self, game):
        """Heuristic evaluation from this agent's perspective.

        Scans all 69 four-cell windows and scores based on piece counts.
        Adds a center column bonus.
        """
        board = game.board
        opponent = PLAYER2 if self.player_id == PLAYER1 else PLAYER1
        score = 0

        # Center column bonus
        center_col = board[:, 3]
        score += int((center_col == self.player_id).sum()) * 6

        # Window-based scoring
        for window in WINDOWS:
            cells = [board[r][c] for r, c in window]
            own = cells.count(self.player_id)
            opp = cells.count(opponent)
            empty = cells.count(EMPTY)

            if own == 4:
                score += 100_000
            elif own == 3 and empty == 1:
                score += 50
            elif own == 2 and empty == 2:
                score += 10

            if opp == 4:
                score -= 100_000
            elif opp == 3 and empty == 1:
                score -= 50
            elif opp == 2 and empty == 2:
                score -= 10

        return score


if __name__ == '__main__':
    from app.engine import ConnectFourGame

    print("=== Minimax Agent Tests ===\n")

    # Test 1: Take the winning move
    g = ConnectFourGame()
    for c in [0, 6, 1, 6, 2]:  # P1 has 3 in a row at bottom: cols 0,1,2
        g.make_move(c)
    # P2's turn, but let's test P1 agent
    g.undo_move(2)  # Undo P1's last, now it's P1's turn with 2 in row
    g.make_move(2)  # P1 plays col 2 again
    # Actually let's do a cleaner test
    g = ConnectFourGame()
    for c in [0, 6, 1, 6, 2, 6]:  # P1: 0,1,2 at bottom. P2: 6,6,6
        g.make_move(c)
    # P1's turn, can win with col 3
    agent = MinimaxAgent(player_id=PLAYER1, depth=4)
    result = agent.select_move(g)
    assert result['move'] == 3, f"Should take winning move 3, got {result['move']}"
    print(f"[PASS] Takes winning move (col 3), score={result['score']}, time={result['compute_time_ms']:.1f}ms")

    # Test 2: Block opponent's winning move
    g = ConnectFourGame()
    # P1 plays scattered, P2 builds a horizontal threat at bottom: cols 0,1,2
    for c in [5, 0, 4, 1, 6, 2]:  # P1: 5,4,6 (scattered). P2: 0,1,2 (3-in-row)
        g.make_move(c)
    # P1's turn, must block col 3 to prevent P2's horizontal win
    agent = MinimaxAgent(player_id=PLAYER1, depth=4)
    result = agent.select_move(g)
    assert result['move'] == 3, f"Should block at col 3, got {result['move']}"
    print(f"[PASS] Blocks opponent (col 3), score={result['score']}, time={result['compute_time_ms']:.1f}ms")

    # Test 3: Performance — depth 6 from empty board
    g = ConnectFourGame()
    agent = MinimaxAgent(player_id=PLAYER1, depth=6)
    result = agent.select_move(g)
    assert result['compute_time_ms'] < 10_000, "Depth 6 too slow (>10s)"
    print(f"[PASS] Depth 6 empty board: move={result['move']}, time={result['compute_time_ms']:.1f}ms")

    print("\n=== All minimax tests passed ===")
