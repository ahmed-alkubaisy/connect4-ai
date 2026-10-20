"""XGBoost-based Connect-4 agent.

Uses a trained XGBRegressor to evaluate board positions.
Move selection uses value-based 1-ply lookahead (same approach as ResNet).
"""

import os
import time
import random
import numpy as np

from app.agents.base import Agent
from app.engine import ROWS, COLS


class XGBoostAgent(Agent):
    """Connect-4 agent using a trained XGBoost model.

    For each valid column, plays the move, flattens the resulting
    board tensor to 126 features, predicts the value with XGBoost,
    and selects the column with the best evaluation.

    Falls back to center-biased random play if no model is loaded.
    """

    def __init__(self, player_id=1, model_path=None, finetuned=False):
        self.player_id = player_id
        self.model = None
        self.model_loaded = False
        self._finetuned = finetuned

        if model_path is None:
            filename = 'connect4_xgboost_finetuned4.json' if finetuned else 'connect4_xgboost.json'
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'models', filename
            )

        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            self.model_loaded = True
            print(f"XGBoost model loaded from {model_path}")
        except (FileNotFoundError, Exception) as e:
            self.model = None
            print(f"WARNING: Could not load XGBoost model ({e}). Using random fallback.")

    @property
    def name(self):
        return "XGBoost (RL)" if self._finetuned else "XGBoost"

    def select_move(self, game):
        start = time.perf_counter()

        if self.model is None:
            result = self._random_fallback(game)
        else:
            result = self._model_inference(game)

        result['compute_time_ms'] = (time.perf_counter() - start) * 1000
        return result

    def _model_inference(self, game):
        """Evaluate each valid column using XGBoost (1-ply lookahead)."""
        valid_moves = game.get_valid_moves()
        values = {}

        for col in valid_moves:
            game_copy = game.clone()
            game_copy.make_move(col)
            tensor = game_copy.to_tensor()  # (3, 6, 7)
            flat = tensor.reshape(1, -1)    # (1, 126)
            pred = self.model.predict(flat)[0]
            # Negate: tensor is from the next player's perspective
            values[col] = -float(np.clip(pred, -1.0, 1.0))

        # Softmax over values for probability display and move sampling
        cols = sorted(values.keys())
        vals = np.array([values[c] for c in cols])
        # Temperature-scaled softmax
        exp_vals = np.exp((vals - vals.max()) * 3.0)
        probs_arr = exp_vals / exp_vals.sum()

        probabilities = [0.0] * COLS
        for i, c in enumerate(cols):
            probabilities[c] = float(probs_arr[i])

        # Sample from distribution instead of argmax for varied play
        chosen_idx = np.random.choice(len(cols), p=probs_arr)
        chosen_col = cols[chosen_idx]

        return {
            'move': chosen_col,
            'probabilities': probabilities,
            'value': values[chosen_col],
        }

    def _random_fallback(self, game):
        """Center-biased random move when no model is loaded."""
        valid = game.get_valid_moves()
        weights = [1, 2, 3, 4, 3, 2, 1]
        valid_weights = [weights[c] for c in valid]
        total = sum(valid_weights)
        probs_valid = [w / total for w in valid_weights]

        col = random.choices(valid, weights=valid_weights, k=1)[0]

        probabilities = [0.0] * COLS
        for i, c in enumerate(valid):
            probabilities[c] = probs_valid[i]

        return {
            'move': col,
            'probabilities': probabilities,
            'value': 0.0,
        }
