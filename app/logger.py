"""CSV logger for AI vs AI benchmark mode.

Each game is saved to its own CSV file, organized into subfolders by matchup:
    data/ResNet_vs_XGBoost/game_001.csv
    data/ResNet_RL_vs_Minimax/game_001.csv
"""

import csv
import os
import re


class GameLogger:
    """Records move data during games. Each game gets its own CSV file.

    CSV fields: Turn_Number, Agent, Move, Compute_Time_ms,
    Score_or_Probability, Winner
    """

    def __init__(self, directory='data'):
        self.base_directory = directory
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.game_id = 0
        self.turn_number = 0
        self.filepath = None

    def _name_to_tag(self, name):
        """Convert agent name to a filename-safe tag."""
        return name.replace(' ', '_').replace('(', '').replace(')', '')

    def _count_existing(self, directory):
        """Find the highest existing game number in directory."""
        if not os.path.isdir(directory):
            return 0
        numbers = []
        for f in os.listdir(directory):
            match = re.search(r'_(\d{3})\.csv$', f)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers) if numbers else 0

    def new_game(self, p1_name='P1', p2_name='P2'):
        """Start a new game — creates a new CSV file in a matchup subfolder."""
        tag1 = self._name_to_tag(p1_name)
        tag2 = self._name_to_tag(p2_name)
        matchup_folder = f'{tag1}_vs_{tag2}'

        self.directory = os.path.join(self.base_directory, matchup_folder)
        os.makedirs(self.directory, exist_ok=True)

        self.game_id = self._count_existing(self.directory) + 1
        self.turn_number = 0
        self.filepath = os.path.join(
            self.directory, f'game_{self.game_id:03d}.csv'
        )
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Turn_Number', 'Agent', 'Move',
                'Compute_Time_ms', 'Score_or_Probability'
            ])

    def log_move(self, agent_name, result):
        """Log a single move to the current game's CSV."""
        if self.filepath is None:
            return
        self.turn_number += 1

        if 'score' in result:
            score_field = result['score']
        elif 'probabilities' in result:
            score_field = str(result['probabilities'])
        else:
            score_field = ''

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.turn_number,
                agent_name,
                result['move'],
                f"{result['compute_time_ms']:.2f}",
                score_field,
            ])
            f.flush()

    def log_game_result(self, winner_name):
        """Append a summary row marking the game winner."""
        if self.filepath is None:
            return
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                '', '', '', '', f'Winner: {winner_name}'
            ])
            f.flush()
