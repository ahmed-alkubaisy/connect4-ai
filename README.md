# Connect-4: A Comparative Study of AI Approaches

**Author:** Ahmed Wajdi AlKubaisy<br>
**Supervisor:** Abdul Wahid<br>
**Programme:** BSc Artificial Intelligence and Computer Science, University of Birmingham<br>
**Year:** 2025-26

## Overview

A comparative study of **Deep Learning (ResNet)**, **Gradient Boosting (XGBoost)**, and **Heuristic Search (Alpha-Beta Minimax)** for playing Connect-4.

The project investigates whether preserving spatial structure through convolutional networks gives a measurable advantage over flat-feature models. Both ML models were trained on the same data, use the same move selection strategy, and were benchmarked under identical conditions. The only variable is how they represent the board.

Key findings: ResNet (35% win rate vs Minimax) outperforms XGBoost (15%) by more than 2x, with the gap attributable to translation invariance in convolutional architectures.

## Project Structure

```
connect4-ai/
├── app/                              # Pygame application
│   ├── main.py                       # Entry point — game loop, threading, state machine
│   ├── engine.py                     # Connect-4 rules, board state, tensor conversion
│   ├── renderer.py                   # Board rendering, animations, XAI display
│   ├── logger.py                     # Per-game CSV logging
│   ├── agents/
│   │   ├── base.py                   # Abstract Agent interface
│   │   ├── human_agent.py            # Non-blocking human input
│   │   ├── minimax_agent.py          # Alpha-beta pruning with heuristic evaluation
│   │   ├── resnet_agent.py           # ResNet model definition + inference
│   │   └── xgboost_agent.py          # XGBoost model loading + inference
│   ├── models/
│   │   ├── connect4_resnet.pt        # Base ResNet weights (supervised)
│   │   ├── connect4_resnet_finetuned4.pt  # RL fine-tuned ResNet weights
│   │   ├── connect4_xgboost.json     # Base XGBoost model
│   │   └── connect4_xgboost_finetuned4.json  # RL fine-tuned XGBoost model
│   └── ui/
│       ├── menu.py                   # Mode selection menu
│       └── xai_overlay.py            # Confidence bars and eval score overlays
├── notebooks/                        # Training & analysis (.ipynb)
│   ├── 01_Data_and_Training.ipynb    # UCI dataset → tensor → train ResNet
│   ├── 02_Comparative_Analysis.ipynb # Win rates, timing, heatmaps, game length
│   ├── 03_XGBoost_Training.ipynb     # Train XGBoost on flattened board features
│   └── 04_Reinforcement_Learning.ipynb  # RL fine-tuning via self-play vs Minimax
├── data/                             # Benchmark game logs (300 games, 6 matchups)
│   ├── ResNet_vs_XGBoost/            # 50 games
│   ├── ResNet_vs_Minimax/            # 50 games
│   ├── XGBoost_vs_Minimax/           # 50 games
│   ├── ResNet_RL_vs_XGBoost_RL/      # 50 games
│   ├── ResNet_RL_vs_Minimax/         # 50 games
│   └── XGBoost_RL_vs_Minimax/        # 50 games
├── requirements.txt
└── README.md
```

## How to Run

### Prerequisites
- Python 3.12+
- Set up a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Pre-trained Models

The `app/models/` directory already contains trained weights for all agents (both base and RL-finetuned). You can skip the training step and go straight to running the game.

### 1. Train the Models (Optional)

All notebooks can be run locally or on Google Colab. Colab is recommended for ResNet training (GPU speeds up training).

**ResNet** (Google Colab recommended for GPU):
1. Open `notebooks/01_Data_and_Training.ipynb`
2. Run all cells — fetches UCI dataset, trains ResNet, saves weights
3. Place `connect4_resnet.pt` in `app/models/`

**XGBoost**:
1. Open `notebooks/03_XGBoost_Training.ipynb`
2. Run all cells — trains XGBoost on flattened features
3. Place `connect4_xgboost.json` in `app/models/`

**RL Fine-Tuning** (optional):
1. Open `notebooks/04_Reinforcement_Learning.ipynb`
2. Run all cells — fine-tunes both models via self-play against Minimax
3. Produces `connect4_resnet_finetuned4.pt` and `connect4_xgboost_finetuned4.json`

### 2. Run the Game
```bash
python -m app.main
```

### Play Modes
- **Human vs AI** — Play against ResNet, XGBoost, or Minimax
- **AI vs AI** — Watch two agents play, with per-move CSV logging

## Agents

### ResNet
- 5-block residual CNN with dual heads (value + policy)
- Trained on 67,557 UCI Connect-4 positions (MSE loss, Adam, lr=0.001, 25 epochs)
- 3-channel input: current player, opponent, empty cells (6x7 grid)
- Value-based 1-ply lookahead with temperature-scaled softmax (T=3.0)
- Inference time: ~20ms per move

### XGBoost
- 500 gradient-boosted trees (max depth 8, lr=0.05, subsample 0.8)
- Same 3-channel input flattened to 126 scalar features
- Same 1-ply lookahead and softmax move selection as ResNet
- Inference time: ~7ms per move

### Minimax
- Alpha-beta pruning with centre-first move ordering [3,2,4,1,5,0,6]
- Window-based heuristic: +100,000 (four), +50 (three), +10 (two), +6 (centre)
- Configurable depth (4 / 6 / 8)
- Make/undo move optimisation (no board cloning)
- Inference time: ~50ms per move (depth 4)

## XAI Overlays
- **ML Agents (ResNet/XGBoost)**: Real-time bar chart showing per-column confidence (0-100%)
- **Minimax**: Evaluation score, search depth, and win/loss status

## Benchmark Results (300 games)

| Agent | Games | Wins | Win % |
|-------|-------|------|-------|
| Minimax | 200 | 198 | 99.0% |
| ResNet | 100 | 35 | 35.0% |
| ResNet (RL) | 100 | 37 | 37.0% |
| XGBoost | 100 | 15 | 15.0% |
| XGBoost (RL) | 100 | 14 | 14.0% |

## Dependencies

See `requirements.txt`. Key packages: pygame, numpy, torch, xgboost, scikit-learn, matplotlib, seaborn, pandas.

