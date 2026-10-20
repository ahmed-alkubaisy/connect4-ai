"""Main entry point for the Connect-4 Pygame application.

Supports three modes:
    1. Human vs ResNet
    2. Human vs Minimax
    3. AI vs AI (ResNet vs Minimax) — automated benchmark

Uses ThreadPoolExecutor for non-blocking AI computation.
Window is resizable — all layout scales dynamically.
"""

import sys
import queue
import time
from concurrent.futures import ThreadPoolExecutor

import pygame

from app.engine import ConnectFourGame, PLAYER1, PLAYER2
from app.renderer import Renderer, DEFAULT_WIDTH, DEFAULT_HEIGHT, MIN_WIDTH, MIN_HEIGHT
from app.agents.human_agent import HumanAgent
from app.agents.minimax_agent import MinimaxAgent
from app.agents.resnet_agent import ResNetAgent
from app.agents.xgboost_agent import XGBoostAgent
from app.logger import GameLogger
from app.ui.menu import (Menu, MODE_HUMAN_VS_RESNET,
                          MODE_HUMAN_VS_MINIMAX, MODE_HUMAN_VS_XGBOOST,
                          MODE_AI_VS_AI, MATCHUP_RN_XG)

# Game states
STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"

# Animation
DROP_FRAMES = 15
AI_VS_AI_DELAY_MS = 500
AI_VS_AI_NUM_GAMES = 50

FPS = 60


class App:
    """Main application class managing game state and rendering."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption("Connect 4 — ResNet vs Minimax")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.menu = Menu(DEFAULT_WIDTH, DEFAULT_HEIGHT)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.result_queue = queue.Queue()

        self.state = STATE_MENU
        self.current_mode = None
        self.game = None
        self.agents = {}
        self.agent_types = {}
        self.ai_thinking = False
        self.last_result = {}
        self.last_agent_type = None

        # Animation state
        self.animating = False
        self.anim_col = 0
        self.anim_target_row = 0
        self.anim_player = PLAYER1
        self.anim_frame = 0
        self.anim_pending_result = None

        # AI vs AI state
        self.ai_vs_ai = False
        self.ai_delay_timer = 0
        self.logger = None
        self.games_played = 0
        self.ai_wins = {PLAYER1: 0, PLAYER2: 0, 'draw': 0}

    def run(self):
        """Main application loop."""
        running = True
        while running:
            dt = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state != STATE_MENU:
                            self._go_to_menu()
                        else:
                            running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)

            if self.state == STATE_PLAYING and not self.animating:
                self._update_playing(dt)

            self._render()
            pygame.display.flip()

        self.executor.shutdown(wait=False)
        pygame.quit()

    def _handle_resize(self, w, h):
        """Handle window resize — recalculate all layout."""
        w = max(w, MIN_WIDTH)
        h = max(h, MIN_HEIGHT)
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.renderer.screen = self.screen
        self.renderer.recalc_layout()
        self.menu.resize(w, h)

    def _handle_click(self, pos):
        if self.state == STATE_MENU:
            mode = self.menu.handle_click(pos)
            if mode is not None:
                self._start_game(mode)

        elif self.state == STATE_PLAYING and not self.ai_thinking and not self.animating:
            current = self.game.current_player
            agent = self.agents[current]
            if isinstance(agent, HumanAgent):
                col = self.renderer.get_column_from_mouse(pos[0])
                if col is not None:
                    agent.set_move(col)

        elif self.state == STATE_GAME_OVER:
            if self.renderer.btn_play_again.collidepoint(pos):
                self._play_again()
            elif self.renderer.btn_main_menu.collidepoint(pos):
                self._go_to_menu()

    def _start_game(self, mode):
        self.current_mode = mode
        self.game = ConnectFourGame()
        self.state = STATE_PLAYING
        self.ai_thinking = False
        self.last_result = {}
        self.last_agent_type = None
        self.animating = False

        depth = self.menu.minimax_depth
        finetuned = self.menu.use_finetuned

        if mode == MODE_HUMAN_VS_RESNET:
            self.agents = {PLAYER1: HumanAgent(), PLAYER2: ResNetAgent(player_id=PLAYER2, finetuned=finetuned)}
            self.agent_types = {PLAYER1: 'human', PLAYER2: 'resnet'}
            self.ai_vs_ai = False
        elif mode == MODE_HUMAN_VS_MINIMAX:
            self.agents = {PLAYER1: HumanAgent(), PLAYER2: MinimaxAgent(player_id=PLAYER2, depth=depth)}
            self.agent_types = {PLAYER1: 'human', PLAYER2: 'minimax'}
            self.ai_vs_ai = False
        elif mode == MODE_HUMAN_VS_XGBOOST:
            self.agents = {PLAYER1: HumanAgent(), PLAYER2: XGBoostAgent(player_id=PLAYER2, finetuned=finetuned)}
            self.agent_types = {PLAYER1: 'human', PLAYER2: 'xgboost'}
            self.ai_vs_ai = False
        elif mode == MODE_AI_VS_AI:
            p1_type, p2_type = self.menu.ai_matchup
            self.agents = {
                PLAYER1: self._make_agent(p1_type, PLAYER1, depth, finetuned),
                PLAYER2: self._make_agent(p2_type, PLAYER2, depth, finetuned),
            }
            self.agent_types = {PLAYER1: p1_type, PLAYER2: p2_type}
            self.ai_vs_ai = True
            self.logger = GameLogger()
            self.games_played = 0
            self.ai_wins = {PLAYER1: 0, PLAYER2: 0, 'draw': 0}
            self.logger.new_game(self.agents[PLAYER1].name,
                                 self.agents[PLAYER2].name)
            self.ai_delay_timer = 0

    def _make_agent(self, agent_type, player_id, depth, finetuned):
        """Create an agent by type string."""
        if agent_type == 'resnet':
            return ResNetAgent(player_id=player_id, finetuned=finetuned)
        elif agent_type == 'xgboost':
            return XGBoostAgent(player_id=player_id, finetuned=finetuned)
        elif agent_type == 'minimax':
            return MinimaxAgent(player_id=player_id, depth=depth)

    def _play_again(self):
        if self.ai_vs_ai:
            if self.games_played < AI_VS_AI_NUM_GAMES:
                self._start_next_ai_game()
                return
            self._start_game(self.current_mode)
        else:
            self._start_game(self.current_mode)

    def _start_next_ai_game(self):
        self.game = ConnectFourGame()
        self.state = STATE_PLAYING
        self.ai_thinking = False
        self.last_result = {}
        self.last_agent_type = None
        self.animating = False
        self.ai_delay_timer = 0
        self.logger.new_game(self.agents[PLAYER1].name,
                             self.agents[PLAYER2].name)

    def _update_playing(self, dt):
        try:
            result = self.result_queue.get_nowait()
            self.ai_thinking = False
            self._start_drop_animation(result)
            return
        except queue.Empty:
            pass

        if self.ai_thinking:
            return

        current = self.game.current_player
        agent = self.agents[current]

        if isinstance(agent, HumanAgent):
            result = agent.select_move(self.game)
            if result is not None:
                self._start_drop_animation(result)
        else:
            if self.ai_vs_ai:
                self.ai_delay_timer += dt
                if self.ai_delay_timer < AI_VS_AI_DELAY_MS:
                    return
                self.ai_delay_timer = 0

            game_copy = self.game.clone()
            self.ai_thinking = True

            def compute():
                return agent.select_move(game_copy)

            future = self.executor.submit(compute)
            future.add_done_callback(
                lambda f: self.result_queue.put(f.result())
            )

    def _start_drop_animation(self, result):
        col = result['move']
        for row in range(5, -1, -1):
            if self.game.board[row][col] == 0:
                self.anim_target_row = row
                break

        self.anim_col = col
        self.anim_player = self.game.current_player
        self.anim_frame = 0
        self.anim_pending_result = result
        self.animating = True

        self.last_agent_type = self.agent_types[self.game.current_player]
        self.last_result = result

    def _finish_animation(self):
        result = self.anim_pending_result
        col = result['move']
        current_player = self.game.current_player

        if self.ai_vs_ai and self.logger:
            self.logger.log_move(self.agents[current_player].name, result)

        self.game.make_move(col)
        self.animating = False
        self.anim_pending_result = None

        winner = self.game.check_winner()
        if winner != 0 or self.game.is_full():
            self.state = STATE_GAME_OVER
            if self.ai_vs_ai:
                self.games_played += 1
                if winner != 0:
                    self.ai_wins[winner] += 1
                    winner_name = self.agents[winner].name
                else:
                    self.ai_wins['draw'] += 1
                    winner_name = "Draw"
                if self.logger:
                    self.logger.log_game_result(winner_name)

    def _go_to_menu(self):
        self.state = STATE_MENU
        self.game = None
        self.current_mode = None
        self.ai_thinking = False
        self.animating = False
        self.ai_vs_ai = False

    def _render(self):
        if self.state == STATE_MENU:
            self.menu.draw(self.screen)
            return

        # Use dynamic layout from renderer
        bx = self.renderer.board_x
        by = self.renderer.board_y
        cell = self.renderer.cell_size

        hover_col = None
        hover_player = None
        if (self.state == STATE_PLAYING and not self.ai_thinking
                and not self.animating):
            current = self.game.current_player
            if isinstance(self.agents[current], HumanAgent):
                mx, _ = pygame.mouse.get_pos()
                hover_col = self.renderer.get_column_from_mouse(mx)
                hover_player = current

        self.renderer.draw_board(self.game, hover_col, hover_player)

        # Drop animation
        if self.animating:
            progress = self.anim_frame / DROP_FRAMES
            current_row = progress * self.anim_target_row
            cx = int(bx + self.anim_col * cell + cell // 2)
            cy = int(by + current_row * cell + cell // 2)
            color = self.renderer.piece_colors[self.anim_player]
            pygame.draw.circle(self.screen, color, (cx, cy), cell // 2 - 8)

            self.anim_frame += 1
            if self.anim_frame >= DROP_FRAMES:
                self._finish_animation()

        # Status
        if self.state == STATE_PLAYING:
            if self.ai_thinking:
                self.renderer.draw_status("AI Thinking...", (100, 200, 255))
                agent = self.agents[self.game.current_player]
                self.renderer.draw_turn_info(f"{agent.name} is computing...")
            else:
                current = self.game.current_player
                agent = self.agents[current]
                self.renderer.draw_status(f"{agent.name}'s Turn",
                                          self.renderer.piece_colors[current])

        elif self.state == STATE_GAME_OVER:
            winner = self.game.check_winner()
            self.renderer.draw_winning_line(self.game)
            if winner != 0:
                agent = self.agents[winner]
                self.renderer.draw_status(f"{agent.name} Wins!",
                                          self.renderer.piece_colors[winner])
            else:
                self.renderer.draw_status("Draw!", (200, 200, 200))


            show_next = self.ai_vs_ai and self.games_played < AI_VS_AI_NUM_GAMES
            self.renderer.draw_game_over_buttons(show_next=show_next)

        # XAI overlay
        if self.last_agent_type and self.last_result:
            model_loaded = True
            if self.last_agent_type == 'resnet':
                resnet_agents = [a for a in self.agents.values()
                                 if isinstance(a, ResNetAgent)]
                if resnet_agents:
                    model_loaded = resnet_agents[0].model_loaded
            elif self.last_agent_type == 'xgboost':
                xgb_agents = [a for a in self.agents.values()
                              if isinstance(a, XGBoostAgent)]
                if xgb_agents:
                    model_loaded = xgb_agents[0].model_loaded
            self.renderer.draw_xai(self.last_agent_type, self.last_result,
                                   model_loaded)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
