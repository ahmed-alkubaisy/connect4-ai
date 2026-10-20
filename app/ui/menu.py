"""Main menu for Connect-4 game mode selection."""

import pygame


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BG = (18, 18, 28)
GRAY = (140, 140, 150)
LIGHT_GRAY = (200, 200, 210)
DIM_GRAY = (80, 80, 100)
YELLOW = (255, 210, 40)
ACCENT = (100, 200, 255)
DIVIDER = (50, 50, 70)

BTN_IDLE = (35, 80, 170)
BTN_HOVER = (50, 105, 210)
BTN_BORDER = (60, 110, 220)
BTN_BORDER_HOVER = (90, 150, 255)
BADGE_BG = (20, 55, 130)

SEL_ON = YELLOW
SEL_OFF = (40, 40, 55)
SEL_BORDER_OFF = (70, 70, 90)

# Game modes
MODE_HUMAN_VS_RESNET = "human_vs_resnet"
MODE_HUMAN_VS_MINIMAX = "human_vs_minimax"
MODE_HUMAN_VS_XGBOOST = "human_vs_xgboost"
MODE_AI_VS_AI = "ai_vs_ai"

# AI matchup options (p1_type, p2_type, label)
MATCHUP_RN_XG = 0
MATCHUP_RN_MM = 1
MATCHUP_XG_MM = 2
_MATCHUP_DATA = [
    ('resnet', 'xgboost', 'RN v XG'),
    ('resnet', 'minimax', 'RN v MM'),
    ('xgboost', 'minimax', 'XG v MM'),
]

# Button data (mode, label, detail, icon)
_BUTTON_DATA = [
    (MODE_HUMAN_VS_RESNET, 'Human vs ResNet',
     'Deep learning CNN model', 'NN'),
    (MODE_HUMAN_VS_MINIMAX, 'Human vs Minimax',
     'Alpha-beta pruning search', 'AB'),
    (MODE_HUMAN_VS_XGBOOST, 'Human vs XGBoost',
     'Gradient boosting model', 'XG'),
    (MODE_AI_VS_AI, 'AI vs AI  (Benchmark)',
     'Automated with logging', 'VS'),
]


class Menu:
    """Main menu with mode selection buttons and settings. Scales with window size."""

    def __init__(self, screen_width, screen_height):
        self.depth_options = [4, 6, 8]
        self.selected_depth_idx = 1
        self.model_variants = ['Base', 'Fine-tuned (RL)']
        self.selected_variant_idx = 0
        self.selected_matchup_idx = 0
        self.buttons = []
        self.depth_rects = []
        self.variant_rects = []
        self.matchup_rects = []
        self.sw = screen_width
        self.sh = screen_height
        self._build_fonts()
        self._layout()

    def resize(self, screen_width, screen_height):
        self.sw = screen_width
        self.sh = screen_height
        self._build_fonts()
        self._layout()

    def _build_fonts(self):
        scale = min(self.sw / 850, self.sh / 800)
        self.title_font = pygame.font.SysFont('arial', max(28, int(48 * scale)), bold=True)
        self.subtitle_font = pygame.font.SysFont('arial', max(12, int(17 * scale)))
        self.button_font = pygame.font.SysFont('arial', max(14, int(20 * scale)), bold=True)
        self.detail_font = pygame.font.SysFont('arial', max(10, int(13 * scale)))
        self.label_font = pygame.font.SysFont('arial', max(11, int(14 * scale)), bold=True)
        self.selector_font = pygame.font.SysFont('arial', max(12, int(16 * scale)), bold=True)

    def _layout(self):
        sw, sh = self.sw, self.sh
        scale = min(sw / 850, sh / 800)

        # Mode buttons
        btn_width = max(300, int(400 * scale))
        btn_height = max(52, int(68 * scale))
        btn_x = (sw - btn_width) // 2
        btn_start_y = int(sh * 0.20)
        btn_gap = max(55, int(74 * scale))

        self.buttons = []
        for i, (mode, label, detail, icon) in enumerate(_BUTTON_DATA):
            self.buttons.append({
                'rect': pygame.Rect(btn_x, btn_start_y + btn_gap * i,
                                    btn_width, btn_height),
                'label': label,
                'detail': detail,
                'mode': mode,
                'icon': icon,
            })

        # Settings rows below buttons
        last_btn_bottom = self.buttons[-1]['rect'].bottom
        row_h = max(28, int(34 * scale))
        label_gap = max(22, int(28 * scale))  # space for label above each row
        row_gap = row_h + label_gap + int(12 * scale)

        # Row 1: Model variant
        row1_y = last_btn_bottom + int(28 * scale) + label_gap
        variant_btn_w = max(90, int(125 * scale))
        variant_gap = int(8 * scale)
        total_variant_w = variant_btn_w * 2 + variant_gap
        variant_start_x = (sw - total_variant_w) // 2

        self.variant_rects = []
        for i in range(len(self.model_variants)):
            self.variant_rects.append(
                pygame.Rect(variant_start_x + i * (variant_btn_w + variant_gap),
                            row1_y, variant_btn_w, row_h)
            )

        # Row 2: AI matchup
        row2_y = row1_y + row_gap
        matchup_btn_w = max(70, int(95 * scale))
        matchup_gap = int(8 * scale)
        total_matchup_w = matchup_btn_w * 3 + matchup_gap * 2
        matchup_start_x = (sw - total_matchup_w) // 2

        self.matchup_rects = []
        for i in range(len(_MATCHUP_DATA)):
            self.matchup_rects.append(
                pygame.Rect(matchup_start_x + i * (matchup_btn_w + matchup_gap),
                            row2_y, matchup_btn_w, row_h)
            )

        # Row 3: Minimax depth
        row3_y = row2_y + row_gap
        depth_btn_w = max(38, int(50 * scale))
        depth_gap = int(8 * scale)
        total_depth_w = depth_btn_w * 3 + depth_gap * 2
        depth_start_x = (sw - total_depth_w) // 2

        self.depth_rects = []
        for i in range(len(self.depth_options)):
            self.depth_rects.append(
                pygame.Rect(depth_start_x + i * (depth_btn_w + depth_gap),
                            row3_y, depth_btn_w, row_h)
            )

    @property
    def minimax_depth(self):
        return self.depth_options[self.selected_depth_idx]

    @property
    def use_finetuned(self):
        return self.selected_variant_idx == 1

    @property
    def ai_matchup(self):
        """Return (p1_type, p2_type) for the selected AI vs AI matchup."""
        return _MATCHUP_DATA[self.selected_matchup_idx][:2]

    def handle_click(self, pos):
        for btn in self.buttons:
            if btn['rect'].collidepoint(pos):
                return btn['mode']
        for i, rect in enumerate(self.variant_rects):
            if rect.collidepoint(pos):
                self.selected_variant_idx = i
        for i, rect in enumerate(self.matchup_rects):
            if rect.collidepoint(pos):
                self.selected_matchup_idx = i
        for i, rect in enumerate(self.depth_rects):
            if rect.collidepoint(pos):
                self.selected_depth_idx = i
        return None

    def draw(self, surface):
        sw, sh = self.sw, self.sh
        scale = min(sw / 850, sh / 800)
        surface.fill(DARK_BG)

        # Title
        title = self.title_font.render("Connect 4", True, YELLOW)
        surface.blit(title, title.get_rect(centerx=sw // 2, y=int(sh * 0.04)))

        # Subtitle
        sub = self.subtitle_font.render("ResNet  vs  XGBoost  vs  Minimax", True, ACCENT)
        surface.blit(sub, sub.get_rect(centerx=sw // 2, y=int(sh * 0.10)))

        sub2 = self.detail_font.render("Comparative Study", True, GRAY)
        surface.blit(sub2, sub2.get_rect(centerx=sw // 2, y=int(sh * 0.14)))

        # Divider
        div_y = int(sh * 0.17)
        div_w = min(300, int(sw * 0.4))
        pygame.draw.line(surface, DIVIDER,
                         (sw // 2 - div_w // 2, div_y),
                         (sw // 2 + div_w // 2, div_y), 1)

        # Mode buttons
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            rect = btn['rect']
            hovered = rect.collidepoint(mouse_pos)

            bg = BTN_HOVER if hovered else BTN_IDLE
            border = BTN_BORDER_HOVER if hovered else BTN_BORDER
            pygame.draw.rect(surface, bg, rect, border_radius=10)
            pygame.draw.rect(surface, border, rect, 2, border_radius=10)

            # Icon badge
            badge_size = rect.height - 22
            icon_rect = pygame.Rect(rect.x + 12, rect.y + 11,
                                    max(30, badge_size), badge_size)
            pygame.draw.rect(surface, BADGE_BG, icon_rect, border_radius=6)
            icon_text = self.label_font.render(btn['icon'], True, ACCENT)
            surface.blit(icon_text, icon_text.get_rect(center=icon_rect.center))

            # Label (upper line)
            text_x = rect.x + icon_rect.width + 24
            label = self.button_font.render(btn['label'], True, WHITE)
            surface.blit(label, (text_x, rect.y + rect.height // 2 - label.get_height() + 1))

            # Detail (lower line, inside button)
            detail = self.detail_font.render(btn['detail'], True, (160, 175, 210))
            surface.blit(detail, (text_x, rect.y + rect.height // 2 + 4))

        # Settings divider
        settings_div_y = self.variant_rects[0].top - int(20 * scale)
        div_w2 = min(400, int(sw * 0.5))
        pygame.draw.line(surface, DIVIDER,
                         (sw // 2 - div_w2 // 2, settings_div_y),
                         (sw // 2 + div_w2 // 2, settings_div_y), 1)

        # Row 1: Model variant
        self._draw_selector_row(surface, mouse_pos, "MODEL",
                                self.model_variants, self.variant_rects,
                                self.selected_variant_idx)

        # Row 2: AI matchup
        matchup_labels = [m[2] for m in _MATCHUP_DATA]
        self._draw_selector_row(surface, mouse_pos, "AI MATCHUP",
                                matchup_labels, self.matchup_rects,
                                self.selected_matchup_idx)

        # Row 3: Minimax depth
        depth_labels = [str(d) for d in self.depth_options]
        self._draw_selector_row(surface, mouse_pos, "MINIMAX DEPTH",
                                depth_labels, self.depth_rects,
                                self.selected_depth_idx)

        # Footer
        footer = self.detail_font.render("Press ESC to quit", True, (55, 55, 70))
        surface.blit(footer, footer.get_rect(centerx=sw // 2, bottom=sh - 12))

    def _draw_selector_row(self, surface, mouse_pos, title, labels, rects, selected_idx):
        """Draw a labelled row of selector buttons."""
        center_x = (rects[0].left + rects[-1].right) // 2
        lbl = self.label_font.render(title, True, DIM_GRAY)
        surface.blit(lbl, lbl.get_rect(centerx=center_x, bottom=rects[0].top - 10))

        for i, (label, rect) in enumerate(zip(labels, rects)):
            selected = (i == selected_idx)
            hovered = rect.collidepoint(mouse_pos)
            bg = SEL_ON if selected else (50, 50, 65) if hovered else SEL_OFF
            fg = BLACK if selected else WHITE
            border_c = SEL_ON if selected else SEL_BORDER_OFF
            pygame.draw.rect(surface, bg, rect, border_radius=8)
            pygame.draw.rect(surface, border_c, rect, 2, border_radius=8)
            text = self.selector_font.render(label, True, fg)
            surface.blit(text, text.get_rect(center=rect.center))
