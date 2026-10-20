"""Pygame renderer for the Connect-4 board and UI elements."""

import pygame
from app.engine import ROWS, COLS, EMPTY, PLAYER1, PLAYER2
from app.ui.xai_overlay import draw_resnet_xai, draw_minimax_xai, draw_fallback_xai


# Default / minimum window dimensions
DEFAULT_WIDTH = 850
DEFAULT_HEIGHT = 800
MIN_WIDTH = 700
MIN_HEIGHT = 650

# Colors
BG_COLOR = (18, 18, 28)
BOARD_COLOR = (25, 70, 170)
BOARD_BORDER = (40, 100, 210)
EMPTY_COLOR = (12, 12, 22)
PLAYER1_COLOR = (230, 55, 55)
PLAYER1_GLOW = (255, 100, 100)
PLAYER2_COLOR = (255, 210, 40)
PLAYER2_GLOW = (255, 240, 120)
WHITE = (255, 255, 255)
GRAY = (140, 140, 150)
LIGHT_GRAY = (200, 200, 210)
GREEN = (60, 210, 60)

BTN_BLUE = (40, 100, 200)
BTN_BLUE_HOVER = (60, 130, 240)
BTN_RED = (180, 50, 50)
BTN_RED_HOVER = (220, 70, 70)


class Renderer:
    """Handles all Pygame rendering for the Connect-4 game.

    All layout is computed dynamically from the current window size
    via recalc_layout(), so the UI scales when the window is resized.
    """

    def __init__(self, screen):
        self.screen = screen
        self.piece_colors = {PLAYER1: PLAYER1_COLOR, PLAYER2: PLAYER2_COLOR}
        self.piece_glows = {PLAYER1: PLAYER1_GLOW, PLAYER2: PLAYER2_GLOW}

        # Layout values (recomputed on resize)
        self.sw = 0
        self.sh = 0
        self.cell = 0
        self.bx = 0
        self.by = 0
        self.bw = 0
        self.bh = 0
        self.xai_y = 0
        self.xai_h = 0
        self.btn_play_again = pygame.Rect(0, 0, 0, 0)
        self.btn_main_menu = pygame.Rect(0, 0, 0, 0)

        # Fonts (recomputed on resize)
        self._build_fonts(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        self.recalc_layout()

    def _build_fonts(self, w, h):
        """Create fonts scaled to window size."""
        scale = min(w / DEFAULT_WIDTH, h / DEFAULT_HEIGHT)
        self.status_font = pygame.font.SysFont('arial', max(18, int(28 * scale)), bold=True)
        self.small_font = pygame.font.SysFont('arial', max(12, int(18 * scale)))
        self.col_font = pygame.font.SysFont('arial', max(11, int(16 * scale)), bold=True)
        self.btn_font = pygame.font.SysFont('arial', max(14, int(22 * scale)), bold=True)

    def recalc_layout(self):
        """Recompute all layout values from the current screen size."""
        self.sw, self.sh = self.screen.get_size()

        # Reserve space: top for status, bottom for XAI
        top_margin = int(self.sh * 0.12)
        xai_height = int(self.sh * 0.16)
        board_area_h = self.sh - top_margin - xai_height - 30
        board_area_w = self.sw - 40

        # Cell size: fit the board into available space
        cell_from_h = board_area_h // ROWS
        cell_from_w = board_area_w // COLS
        self.cell = max(40, min(cell_from_h, cell_from_w))

        self.bw = self.cell * COLS
        self.bh = self.cell * ROWS
        self.bx = (self.sw - self.bw) // 2
        self.by = top_margin
        self.xai_y = self.by + self.bh + 15
        self.xai_h = max(60, self.sh - self.xai_y - 10)

        # Game over buttons — centered over the board
        btn_w = max(140, int(self.sw * 0.2))
        btn_h = max(36, int(self.sh * 0.055))
        btn_gap = 20
        total_w = btn_w * 2 + btn_gap
        btn_y = self.by + self.bh // 2 + int(self.cell * 0.4)
        self.btn_play_again = pygame.Rect(
            (self.sw - total_w) // 2, btn_y, btn_w, btn_h
        )
        self.btn_main_menu = pygame.Rect(
            (self.sw - total_w) // 2 + btn_w + btn_gap, btn_y, btn_w, btn_h
        )

        self._build_fonts(self.sw, self.sh)

    # -- Convenience accessors for main.py animation code --
    @property
    def board_x(self):
        return self.bx

    @property
    def board_y(self):
        return self.by

    @property
    def cell_size(self):
        return self.cell

    def draw_board(self, game, hover_col=None, hover_player=None):
        """Draw the Connect-4 board with pieces."""
        self.screen.fill(BG_COLOR)
        cell = self.cell
        bx, by = self.bx, self.by
        radius = cell // 2 - 8

        # Column numbers
        for c in range(COLS):
            cx = bx + c * cell + cell // 2
            label = self.col_font.render(str(c + 1), True, GRAY)
            self.screen.blit(label, label.get_rect(centerx=cx, bottom=by - 4))

        # Hover piece
        if hover_col is not None and hover_player is not None:
            cx = bx + hover_col * cell + cell // 2
            cy = by - int(cell * 0.3)
            glow = self.piece_glows.get(hover_player, WHITE)
            color = self.piece_colors.get(hover_player, WHITE)
            pygame.draw.circle(self.screen, glow, (cx, cy), radius + 4, 3)
            pygame.draw.circle(self.screen, color, (cx, cy), radius)

        # Board background
        pygame.draw.rect(self.screen, BOARD_BORDER,
                         (bx - 4, by - 4, self.bw + 8, self.bh + 8),
                         border_radius=12)
        pygame.draw.rect(self.screen, BOARD_COLOR,
                         (bx, by, self.bw, self.bh), border_radius=10)

        # Cells
        for r in range(ROWS):
            for c in range(COLS):
                cx = bx + c * cell + cell // 2
                cy = by + r * cell + cell // 2
                val = game.board[r][c]

                # Shadow
                pygame.draw.circle(self.screen, (10, 10, 18),
                                   (cx + 2, cy + 2), radius)

                if val == EMPTY:
                    pygame.draw.circle(self.screen, EMPTY_COLOR, (cx, cy), radius)
                else:
                    color = self.piece_colors[val]
                    glow = self.piece_glows[val]
                    pygame.draw.circle(self.screen, glow, (cx, cy), radius + 2, 2)
                    pygame.draw.circle(self.screen, color, (cx, cy), radius)
                    hl_offset = max(4, radius // 5)
                    pygame.draw.circle(self.screen, glow,
                                       (cx - hl_offset, cy - hl_offset),
                                       max(3, radius // 5))

    def draw_status(self, text, color=WHITE):
        label = self.status_font.render(text, True, color)
        self.screen.blit(label, label.get_rect(centerx=self.sw // 2, y=6))

    def draw_turn_info(self, text):
        label = self.small_font.render(text, True, GRAY)
        self.screen.blit(label, label.get_rect(centerx=self.sw // 2, y=38))

    def draw_game_over_buttons(self, show_next=False):
        overlay = pygame.Surface(
            (self.btn_main_menu.right - self.btn_play_again.x + 40, self.btn_play_again.h + 40),
            pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (self.btn_play_again.x - 20, self.btn_play_again.y - 20))

        mouse_pos = pygame.mouse.get_pos()

        # Play Again / Next Game
        hover = self.btn_play_again.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, BTN_BLUE_HOVER if hover else BTN_BLUE,
                         self.btn_play_again, border_radius=8)
        pygame.draw.rect(self.screen, WHITE, self.btn_play_again, 2, border_radius=8)
        lbl = "Next Game" if show_next else "Play Again"
        text = self.btn_font.render(lbl, True, WHITE)
        self.screen.blit(text, text.get_rect(center=self.btn_play_again.center))

        # Main Menu
        hover = self.btn_main_menu.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, BTN_RED_HOVER if hover else BTN_RED,
                         self.btn_main_menu, border_radius=8)
        pygame.draw.rect(self.screen, WHITE, self.btn_main_menu, 2, border_radius=8)
        text = self.btn_font.render("Main Menu", True, WHITE)
        self.screen.blit(text, text.get_rect(center=self.btn_main_menu.center))

    def draw_xai(self, agent_type, result_data, model_loaded=True):
        xai_rect = pygame.Rect(10, self.xai_y, self.sw - 20, self.xai_h)
        if agent_type in ('resnet', 'xgboost'):
            if not model_loaded:
                draw_fallback_xai(self.screen, xai_rect)
            else:
                title = "ResNet Column Confidence" if agent_type == 'resnet' else "XGBoost Column Confidence"
                draw_resnet_xai(self.screen, xai_rect,
                                result_data.get('probabilities', [0.0] * 7),
                                result_data.get('move'),
                                title=title)
        elif agent_type == 'minimax':
            draw_minimax_xai(self.screen, xai_rect,
                             result_data.get('score', 0),
                             result_data.get('depth', 0))

    def draw_winning_line(self, game):
        board = game.board
        winner = game.check_winner()
        if winner == 0:
            return
        for r in range(ROWS):
            for c in range(COLS - 3):
                if board[r][c] == winner == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                    self._highlight_cells([(r, c+i) for i in range(4)])
                    return
        for r in range(ROWS - 3):
            for c in range(COLS):
                if board[r][c] == winner == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                    self._highlight_cells([(r+i, c) for i in range(4)])
                    return
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if board[r][c] == winner == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                    self._highlight_cells([(r+i, c+i) for i in range(4)])
                    return
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if board[r][c] == winner == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]:
                    self._highlight_cells([(r-i, c+i) for i in range(4)])
                    return

    def _highlight_cells(self, cells):
        cell = self.cell
        radius = cell // 2 - 4
        for r, c in cells:
            cx = self.bx + c * cell + cell // 2
            cy = self.by + r * cell + cell // 2
            pygame.draw.circle(self.screen, WHITE, (cx, cy), radius, 4)
            pygame.draw.circle(self.screen, GREEN, (cx, cy), radius + 2, 2)

    def get_column_from_mouse(self, x):
        if x < self.bx or x >= self.bx + self.bw:
            return None
        return (x - self.bx) // self.cell
