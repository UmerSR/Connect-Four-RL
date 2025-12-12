from pathlib import Path

import pygame

from envs.connect_four_env import ConnectFourEnv


# Colors (RGB)
COLOR_BG = (26, 28, 34)
COLOR_PANEL = (38, 40, 48)
COLOR_GRID = (36, 74, 145)
COLOR_EMPTY = (16, 18, 25)
COLOR_P1 = (219, 84, 97)   # Player 1 discs
COLOR_P2 = (245, 196, 79)  # Player 2 discs
COLOR_TEXT = (232, 236, 245)
COLOR_BUTTON = (88, 100, 140)
COLOR_BUTTON_ACTIVE = (108, 164, 255)
COLOR_BUTTON_BORDER = (64, 72, 96)

SQUARE_SIZE = 80
PADDING_TOP = 110  # Space for text/status
MENU_HEIGHT = 190  # Space for opponent select menu and spacing
BUTTONS_PER_ROW = 5

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PPO_DIR = ARTIFACTS_DIR / "ppo"
DQN_DIR = ARTIFACTS_DIR / "dqn"
REINFORCE_MANUAL_DIR = ARTIFACTS_DIR / "reinforce_manual"
REINFORCE_TIANSHOU_DIR = ARTIFACTS_DIR / "reinforce_tianshou"


def draw_board(screen, env, font, status_lines, board_left):
    rows, cols = env.rows, env.cols
    board_top = MENU_HEIGHT + PADDING_TOP

    # Draw status text
    status_rect = pygame.Rect(0, MENU_HEIGHT, screen.get_width(), PADDING_TOP)
    pygame.draw.rect(screen, COLOR_PANEL, status_rect)
    y_offset = MENU_HEIGHT + 12
    for line in status_lines:
        text_surface = font.render(line, True, COLOR_TEXT)
        screen.blit(text_surface, (20, y_offset))
        y_offset += font.get_height() + 4

    # Draw grid + discs
    for c in range(cols):
        for r in range(rows):
            cell_x = board_left + c * SQUARE_SIZE
            cell_y = board_top + r * SQUARE_SIZE
            pygame.draw.rect(screen, COLOR_GRID, (cell_x, cell_y, SQUARE_SIZE, SQUARE_SIZE))

            center_x = cell_x + SQUARE_SIZE // 2
            center_y = cell_y + SQUARE_SIZE // 2
            piece = env.board[r, c]
            if piece == 0:
                color = COLOR_EMPTY
            elif piece == 1:
                color = COLOR_P1
            else:
                color = COLOR_P2
            pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 2 - 5)

    pygame.display.flip()


def main():
    pygame.init()
    pygame.display.set_caption("Connect Four")
    env = ConnectFourEnv()
    env.reset()

    cols = env.cols
    rows = env.rows

    board_left = (cols * SQUARE_SIZE) // 10  # placeholder to init
    width = cols * SQUARE_SIZE + 140  # add margin for breathing room
    height = rows * SQUARE_SIZE + PADDING_TOP + MENU_HEIGHT + 80
    screen = pygame.display.set_mode((width, height))

    board_left = (width - cols * SQUARE_SIZE) // 2

    font = pygame.font.SysFont("Arial", 22, bold=False)
    menu_font = pygame.font.SysFont("Arial", 18)

    running = True
    game_over = False
    last_info = {}
    last_message = ""
    clock = pygame.time.Clock()
    opponent = None
    opponent_name = "Human vs Human"
    opponent_kind = None
    ai_player = None  # 1 or 2 for which side AI controls
    human_player = 1  # default human side

    def render_status_lines():
        nonlocal game_over, last_info, last_message
        prefix = f"Mode: {opponent_name} | You are Player {human_player}"
        if game_over:
            if "winner" in last_info:
                winner = last_info["winner"]
                body = f"Game over: Player {winner + 1} wins! Press R to restart."
            elif "draw" in last_info:
                body = "Game over: Draw. Press R to restart."
            elif "illegal_move" in last_info:
                body = "Illegal move! Game over. Press R to restart."
            else:
                body = "Game over. Press R to restart."
        else:
            body = f"Player {env.current_player + 1}'s turn"
            if last_message:
                body = f"{body} | {last_message}"
        return [prefix, body]

    def draw_menu():
        menu_surface = pygame.Surface((width, MENU_HEIGHT))
        menu_surface.fill(COLOR_PANEL)
        ppo_model = PPO_DIR / "ppo.pth"
        ppo_pool_model = PPO_DIR / "ppo_pool.pth"
        ppo_dense_model = PPO_DIR / "ppo_dense.pth"
        dqn_model = DQN_DIR / "dqn_connect4.pth"  # torch-only fallback; zip works if SB3 is installed
        reinforce_manual_model = REINFORCE_MANUAL_DIR / "reinforce_connect4.pth"
        reinforce_ts_model = REINFORCE_TIANSHOU_DIR / "reinforce_connect4.pth"
        options = [
            ("Human", None, None, None),
            ("Random", "random", 2, None),
            ("Heuristic", "heuristic", 2, None),
            ("PPO", "ppo", 2, ppo_model),
            ("PPO Pool", "ppo_pool", 2, ppo_pool_model),
            ("PPO Dense", "ppo_dense", 2, ppo_dense_model),
            ("DQN", "dqn", 2, dqn_model),
            ("REINFORCE", "reinforce_manual", 2, reinforce_manual_model),
            ("REINFORCE TS", "reinforce_tianshou", 2, reinforce_ts_model),
        ]
        btns = []
        padding = 14
        btn_height = 46
        row_count = (len(options) + BUTTONS_PER_ROW - 1) // BUTTONS_PER_ROW
        btn_width = (width - padding * (BUTTONS_PER_ROW + 1)) // BUTTONS_PER_ROW
        for idx, (label, kind, ai_side, model_path) in enumerate(options):
            row = idx // BUTTONS_PER_ROW
            col = idx % BUTTONS_PER_ROW
            x = padding + col * (btn_width + padding)
            y = 12 + row * (btn_height + 10)
            rect = pygame.Rect(x, y, btn_width, btn_height)
            btns.append((rect, label, kind, ai_side, model_path))
            is_active = opponent_kind == kind
            fill = COLOR_BUTTON_ACTIVE if is_active else COLOR_BUTTON
            pygame.draw.rect(menu_surface, fill, rect, border_radius=10)
            pygame.draw.rect(menu_surface, COLOR_BUTTON_BORDER, rect, width=2, border_radius=10)
            text = menu_font.render(label, True, COLOR_TEXT)
            menu_surface.blit(
                text,
                (
                    rect.x + (rect.width - text.get_width()) // 2,
                    rect.y + (rect.height - text.get_height()) // 2,
                ),
            )
        # Human side buttons
        side_btns = []
        side_opts = [("You: Player 1 (Red)", 1), ("You: Player 2 (Yellow)", 2)]
        side_width = (width - padding * (len(side_opts) + 1)) // len(side_opts)
        side_height = 50
        # place side buttons below the last row of mode buttons
        y2 = 12 + row_count * (btn_height + 10) + 12
        for i, (label, side) in enumerate(side_opts):
            x = padding + i * (side_width + padding)
            rect = pygame.Rect(x, y2, side_width, side_height)
            side_btns.append((rect, label, side))
            is_active = human_player == side
            fill = COLOR_BUTTON_ACTIVE if is_active else COLOR_BUTTON
            pygame.draw.rect(menu_surface, fill, rect, border_radius=10)
            pygame.draw.rect(menu_surface, COLOR_BUTTON_BORDER, rect, width=2, border_radius=10)
            text = menu_font.render(label, True, COLOR_TEXT)
            menu_surface.blit(
                text,
                (
                    rect.x + (rect.width - text.get_width()) // 2,
                    rect.y + (rect.height - text.get_height()) // 2,
                ),
            )
        return menu_surface, btns, side_btns

    def reset_game(selected_opponent, ai_side, opponent_label, selected_kind=None):
        nonlocal opponent, opponent_name, opponent_kind, ai_player, game_over, last_info, last_message
        opponent = selected_opponent
        opponent_name = opponent_label
        ai_player = ai_side
        opponent_kind = selected_kind
        if opponent is not None:
            ai_player = 1 if human_player == 2 else 2
        env.reset()
        game_over = False
        last_info = {}
        last_message = f"You are Player {human_player}"
        draw_board(screen, env, font, render_status_lines(), board_left)

    def set_human_side(side: int):
        nonlocal human_player, ai_player, opponent, opponent_name, last_message
        human_player = side
        if opponent is not None:
            ai_player = 1 if human_player == 2 else 2
        else:
            ai_player = None
            opponent_name = "Human vs Human"
        last_message = f"You are Player {human_player}"

    while running:
        clock.tick(30)

        screen.fill(COLOR_BG)
        menu_surface, menu_buttons, side_buttons = draw_menu()
        screen.blit(menu_surface, (0, 0))
        draw_board(screen, env, font, render_status_lines(), board_left)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Restart game
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset_game(opponent, ai_player, opponent_name, opponent_kind)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Menu selection
                if my < MENU_HEIGHT:
                    for rect, label, kind, ai_side, model_path in menu_buttons:
                        if rect.collidepoint(mx, my):
                            from agents.opponents import get_opponent
                            selected = None
                            try:
                                if kind is None:
                                    selected = None
                                else:
                                    selected = get_opponent(kind, model_path=model_path)
                                reset_game(selected, ai_side, label, kind)
                            except Exception as e:
                                print(f"Failed to load opponent {label}: {e}")
                    for rect, label, side in side_buttons:
                        if rect.collidepoint(mx, my):
                            set_human_side(side)
                            reset_game(opponent, ai_player, opponent_name, opponent_kind)
                    continue

                # Human move
                board_top = MENU_HEIGHT + PADDING_TOP
                if not game_over and my >= board_top:
                    if board_left <= mx < board_left + cols * SQUARE_SIZE:
                        col = (mx - board_left) // SQUARE_SIZE
                        human_turn = ai_player is None or (env.current_player + 1) != ai_player
                        if human_turn:
                            legal = env._legal_moves()
                            if col not in legal:
                                last_message = "Column is full. Choose another."
                            else:
                                _, reward, terminated, truncated, info = env.step(col)
                                game_over = terminated or truncated
                                last_info = info
                                last_message = ""

        # AI move if applicable
        if not game_over and opponent is not None:
            ai_turn = ai_player is not None and (env.current_player + 1) == ai_player
            if ai_turn:
                legal = env._legal_moves()
                if not legal:
                    game_over = True
                    last_info = {"illegal_move": True}
                else:
                    col = opponent.select_action(env)
                    if col not in legal:
                        # Fallback: pick a legal move
                        col = legal[0]
                    _, reward, terminated, truncated, info = env.step(col)
                    game_over = terminated or truncated
                    last_info = info
                    last_message = ""

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
