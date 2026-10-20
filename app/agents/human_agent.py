"""Human player agent — receives moves from Pygame mouse clicks."""

from app.agents.base import Agent


class HumanAgent(Agent):
    """Non-blocking human input agent.

    The UI sets pending moves via set_move(). select_move() returns
    None if no valid move is pending, allowing the game loop to
    continue rendering without blocking.
    """

    def __init__(self):
        self._pending_move = None

    @property
    def name(self):
        return "Human"

    def set_move(self, col):
        self._pending_move = col

    def select_move(self, game):
        if self._pending_move is not None and self._pending_move in game.get_valid_moves():
            move = self._pending_move
            self._pending_move = None
            return {'move': move, 'compute_time_ms': 0.0}
        return None
