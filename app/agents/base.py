"""Abstract base class for Connect-4 agents."""

from abc import ABC, abstractmethod


class Agent(ABC):
    """Base interface for all Connect-4 playing agents.

    select_move returns a dict with at minimum:
        'move': int (column index 0-6)
        'compute_time_ms': float
    Subclasses may add agent-specific keys (e.g. 'score', 'probabilities').
    """

    @abstractmethod
    def select_move(self, game):
        pass

    @property
    @abstractmethod
    def name(self):
        pass
