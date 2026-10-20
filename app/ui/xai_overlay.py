"""XAI visualization overlays for the Pygame UI.

Draws real-time bar charts (ResNet confidence) and score displays
(Minimax evaluation) in the bottom panel of the game window.
"""

import pygame


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
DARK_GRAY = (60, 60, 60)
BAR_COLOR = (70, 130, 230)
BAR_HIGHLIGHT = (100, 200, 255)
GREEN = (50, 200, 50)
RED = (220, 50, 50)
YELLOW = (255, 220, 50)


def draw_resnet_xai(surface, rect, probabilities, selected_col=None, title="ResNet Column Confidence"):
    """Draw per-column confidence bars.

    Args:
        surface: Pygame surface to draw on.
        rect: pygame.Rect defining the overlay area.
        probabilities: list of 7 floats (0-1) for each column.
        selected_col: int or None — the column the agent chose (highlighted).
    """
    font = pygame.font.SysFont('monospace', 14)
    title_font = pygame.font.SysFont('monospace', 16, bold=True)

    # Background
    pygame.draw.rect(surface, DARK_GRAY, rect)
    pygame.draw.rect(surface, GRAY, rect, 1)

    # Title
    title = title_font.render(title, True, WHITE)
    surface.blit(title, (rect.x + 10, rect.y + 5))

    # Bar chart area
    bar_area_top = rect.y + 28
    bar_area_height = rect.height - 50
    bar_width = (rect.width - 40) // 7
    bar_gap = 4

    max_prob = max(probabilities) if max(probabilities) > 0 else 1.0

    for i in range(7):
        prob = probabilities[i]
        bar_height = int((prob / max_prob) * bar_area_height) if max_prob > 0 else 0
        bar_x = rect.x + 20 + i * bar_width
        bar_y = bar_area_top + bar_area_height - bar_height

        color = BAR_HIGHLIGHT if i == selected_col else BAR_COLOR
        if prob == 0:
            color = GRAY

        pygame.draw.rect(surface, color,
                         (bar_x + bar_gap // 2, bar_y,
                          bar_width - bar_gap, bar_height))

        # Percentage label
        pct_text = f"{prob * 100:.0f}%"
        label = font.render(pct_text, True, WHITE)
        label_x = bar_x + (bar_width - label.get_width()) // 2
        surface.blit(label, (label_x, bar_y - 16))

        # Column number
        col_label = font.render(str(i), True, GRAY)
        col_x = bar_x + (bar_width - col_label.get_width()) // 2
        surface.blit(col_label, (col_x, bar_area_top + bar_area_height + 4))


def draw_minimax_xai(surface, rect, score, depth):
    """Draw minimax evaluation score and search depth.

    Args:
        surface: Pygame surface to draw on.
        rect: pygame.Rect defining the overlay area.
        score: int — heuristic evaluation score.
        depth: int — search depth used.
    """
    font = pygame.font.SysFont('monospace', 16, bold=True)
    detail_font = pygame.font.SysFont('monospace', 14)

    # Background
    pygame.draw.rect(surface, DARK_GRAY, rect)
    pygame.draw.rect(surface, GRAY, rect, 1)

    # Title
    title = font.render("Minimax Evaluation", True, WHITE)
    surface.blit(title, (rect.x + 10, rect.y + 5))

    # Score display with color
    if score > 1000:
        color = GREEN
        status = "Winning"
    elif score < -1000:
        color = RED
        status = "Losing"
    else:
        color = YELLOW
        status = "Even"

    score_text = font.render(f"Eval: {score:+,}", True, color)
    surface.blit(score_text, (rect.x + 10, rect.y + 35))

    depth_text = detail_font.render(f"Depth: {depth}", True, GRAY)
    surface.blit(depth_text, (rect.x + 10, rect.y + 60))

    status_text = detail_font.render(f"Status: {status}", True, color)
    surface.blit(status_text, (rect.x + 10, rect.y + 82))

    # Score bar visualization
    bar_x = rect.x + 200
    bar_y = rect.y + 40
    bar_width = rect.width - 220
    bar_height = 30

    pygame.draw.rect(surface, GRAY, (bar_x, bar_y, bar_width, bar_height), 1)

    # Clamp score to [-100000, 100000] for display
    clamped = max(-100000, min(100000, score))
    ratio = (clamped + 100000) / 200000  # 0 to 1
    fill_width = int(ratio * bar_width)

    bar_color = GREEN if ratio > 0.5 else RED
    pygame.draw.rect(surface, bar_color,
                     (bar_x, bar_y, fill_width, bar_height))

    # Center line
    center_x = bar_x + bar_width // 2
    pygame.draw.line(surface, WHITE,
                     (center_x, bar_y), (center_x, bar_y + bar_height), 2)


def draw_fallback_xai(surface, rect):
    """Draw fallback message when ResNet model is not loaded."""
    font = pygame.font.SysFont('monospace', 16, bold=True)

    pygame.draw.rect(surface, DARK_GRAY, rect)
    pygame.draw.rect(surface, GRAY, rect, 1)

    text = font.render("ResNet: RANDOM FALLBACK (model not trained)", True, YELLOW)
    surface.blit(text, (rect.x + 10, rect.y + rect.height // 2 - 10))
