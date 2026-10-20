"""ResNet-based Connect-4 agent with policy/value heads.

Contains the neural network architecture (used by both inference and
training notebook) and the inference agent with graceful fallback
when no trained model is available.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.agents.base import Agent
from app.engine import ROWS, COLS


class ResidualBlock(nn.Module):
    """Single residual block: Conv→BN→ReLU→Conv→BN + skip → ReLU."""

    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class Connect4ResNet(nn.Module):
    """ResNet for Connect-4 with dual heads (policy + value).

    Architecture:
        Input: 3x6x7 (one-hot encoded board)
        Initial Conv: 3 → num_filters, 3x3
        Residual Blocks: num_blocks × ResidualBlock
        Policy Head: 1x1 conv → flatten → FC → 7 outputs
        Value Head: 1x1 conv → flatten → FC → FC → 1 output (tanh)
    """

    def __init__(self, num_blocks=5, num_filters=128):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, num_filters, 3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head: predicts move probabilities for each column
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ROWS * COLS, COLS)

        # Value head: predicts board evaluation in [-1, 1]
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Shared trunk
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = self.blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # Raw logits (shape: [batch, 7])

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Range [-1, 1]

        return p, v


class ResNetAgent(Agent):
    """Connect-4 agent using a trained ResNet model.

    Move selection uses value-based 1-ply lookahead: for each valid
    column, play the move, evaluate the resulting position with the
    value head, then select the column with the best value. Softmax
    over the values provides per-column pseudo-probabilities for XAI.

    Falls back to center-biased random play if no model is loaded.
    """

    def __init__(self, player_id=1, model_path=None, finetuned=False):
        self.player_id = player_id
        self.model = None
        self.model_loaded = False
        self._finetuned = finetuned

        if model_path is None:
            filename = 'connect4_resnet_finetuned4.pt' if finetuned else 'connect4_resnet.pt'
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'models', filename
            )

        try:
            self.model = Connect4ResNet()
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            self.model.eval()
            self.model_loaded = True
            print(f"ResNet model loaded from {model_path}")
        except (FileNotFoundError, RuntimeError) as e:
            self.model = None
            print(f"WARNING: Could not load model ({e}). Using random fallback.")

    @property
    def name(self):
        return "ResNet (RL)" if self._finetuned else "ResNet"

    def select_move(self, game):
        start = time.perf_counter()

        if self.model is None:
            result = self._random_fallback(game)
        else:
            result = self._model_inference(game)

        result['compute_time_ms'] = (time.perf_counter() - start) * 1000
        return result

    def _model_inference(self, game):
        """Evaluate each valid column using the value head (1-ply lookahead)."""
        valid_moves = game.get_valid_moves()
        values = {}

        with torch.no_grad():
            for col in valid_moves:
                game_copy = game.clone()
                game_copy.make_move(col)
                tensor = torch.tensor(game_copy.to_tensor()).unsqueeze(0)
                _, value = self.model(tensor)
                # Negate because to_tensor encodes from the NEXT player's
                # perspective after make_move, but we want the current
                # player's evaluation
                values[col] = -value.item()

        # Softmax over values for probability display and move sampling
        cols = sorted(values.keys())
        vals = torch.tensor([values[c] for c in cols])
        probs_tensor = F.softmax(vals * 3.0, dim=0)  # Temperature-scaled

        probabilities = [0.0] * COLS
        for i, c in enumerate(cols):
            probabilities[c] = probs_tensor[i].item()

        # Sample from distribution instead of argmax for varied play
        probs_np = probs_tensor.numpy()
        chosen_idx = np.random.choice(len(cols), p=probs_np)
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


if __name__ == '__main__':
    from app.engine import ConnectFourGame

    print("=== ResNet Agent Tests ===\n")

    # Test 1: Fallback mode produces valid moves
    agent = ResNetAgent(player_id=1)
    assert not agent.model_loaded, "Model should not be loaded (no weights file)"

    g = ConnectFourGame()
    for _ in range(20):
        result = agent.select_move(g.clone())
        assert result['move'] in g.get_valid_moves(), "Invalid move from fallback"
        assert 'probabilities' in result, "Missing probabilities"
        assert len(result['probabilities']) == COLS, "Wrong probability length"
    print("[PASS] Fallback mode produces valid moves")

    # Test 2: Probabilities sum to ~1 for valid columns
    result = agent.select_move(g)
    prob_sum = sum(result['probabilities'])
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}"
    print("[PASS] Probabilities sum to 1.0")

    # Test 3: Model architecture forward pass
    model = Connect4ResNet()
    dummy = torch.randn(1, 3, 6, 7)
    policy, value = model(dummy)
    assert policy.shape == (1, 7), f"Policy shape wrong: {policy.shape}"
    assert value.shape == (1, 1), f"Value shape wrong: {value.shape}"
    assert -1.0 <= value.item() <= 1.0, f"Value out of range: {value.item()}"
    print("[PASS] Model forward pass shapes correct")

    print("\n=== All ResNet agent tests passed ===")
