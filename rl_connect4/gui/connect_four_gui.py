from pathlib import Path

import pygame
import pygame.gfxdraw

from envs.connect_four_env import ConnectFourEnv


# Style constants (visual only)
COLOR_BG = (11, 18, 32)          # #0b1220
COLOR_PANEL = (15, 23, 42)       # #0f172a
COLOR_CARD = (17, 28, 50)        # #111c32
COLOR_GRID = (36, 74, 145)
COLOR_EMPTY = (16, 18, 25)
COLOR_P1 = (219, 84, 97)   # Player 1 discs
COLOR_P2 = (245, 196, 79)  # Player 2 discs
COLOR_TEXT = (229, 231, 235)     # #e5e7eb
COLOR_MUTED = (148, 163, 184)
COLOR_BUTTON = (48, 58, 80)
COLOR_BUTTON_ACTIVE = (59, 130, 246)  # #3b82f6
COLOR_BUTTON_BORDER = (71, 85, 105)
COLOR_OUTLINE = (35, 47, 70)

LEFT_W = 360
PAD = 16
HEADER_H = 56
STATUS_H = 72
BOTTOM_MARGIN = 0  # reserved if needed
BUTTONS_PER_ROW = 3
MIN_CELL = 80
MIN_W = LEFT_W + 7 * MIN_CELL + 3 * PAD
MIN_H = HEADER_H + STATUS_H + 6 * MIN_CELL + 3 * PAD

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PPO_DIR = ARTIFACTS_DIR / "ppo"
DQN_DIR = ARTIFACTS_DIR / "dqn"
REINFORCE_MANUAL_DIR = ARTIFACTS_DIR / "reinforce_manual"
REINFORCE_TIANSHOU_DIR = ARTIFACTS_DIR / "reinforce_tianshou"


def compute_layout(screen_w, screen_h, rows, cols):
    left_rect = pygame.Rect(0, 0, LEFT_W, screen_h)
    right_rect = pygame.Rect(LEFT_W, 0, screen_w - LEFT_W, screen_h)
    board_area_w = right_rect.width - 2 * PAD
    board_area_h = screen_h - (HEADER_H + STATUS_H + 3 * PAD)
    cell_size = min(board_area_w / cols, board_area_h / rows)
    grid_w = cell_size * cols
    grid_h = cell_size * rows
    board_left = LEFT_W + PAD + (board_area_w - grid_w) / 2
    board_top = HEADER_H + STATUS_H + 2 * PAD + (board_area_h - grid_h) / 2
    board_rect = pygame.Rect(board_left, board_top, grid_w, grid_h)
    status_rect = pygame.Rect(LEFT_W + PAD, HEADER_H + PAD, right_rect.width - 2 * PAD, STATUS_H)
    header_rect = pygame.Rect(LEFT_W, 0, right_rect.width, HEADER_H)
    return {
        "left_rect": left_rect,
        "right_rect": right_rect,
        "board_rect": board_rect,
        "status_rect": status_rect,
        "header_rect": header_rect,
        "cell_size": cell_size,
    }


def draw_board(screen, env, font, status_lines, layout):
    rows, cols = env.rows, env.cols
    board_rect = layout["board_rect"]
    status_rect = layout["status_rect"]
    header_rect = layout["header_rect"]
    cell_size = layout["cell_size"]

    # Header and status
    pygame.draw.rect(screen, COLOR_PANEL, header_rect)
    pygame.draw.rect(screen, COLOR_CARD, status_rect, border_radius=10)
    pygame.draw.rect(screen, COLOR_OUTLINE, status_rect, width=1, border_radius=10)
    title_surface = font.render("Connect Four RL Playground", True, COLOR_TEXT)
    screen.blit(title_surface, (header_rect.x + 10, header_rect.y + 10))
    y_offset = status_rect.y + 12
    for line in status_lines:
        text_surface = font.render(line, True, COLOR_TEXT)
        screen.blit(text_surface, (status_rect.x + 10, y_offset))
        y_offset += font.get_height() + 4

    # Board background card
    board_card = board_rect.inflate(PAD, PAD)
    pygame.draw.rect(screen, COLOR_CARD, board_card, border_radius=14)
    pygame.draw.rect(screen, COLOR_OUTLINE, board_card, width=1, border_radius=14)

    # Draw grid + discs
    for c in range(cols):
        for r in range(rows):
            cell_x = board_rect.x + c * cell_size
            cell_y = board_rect.y + r * cell_size
            pygame.draw.rect(screen, COLOR_GRID, (cell_x, cell_y, cell_size, cell_size))

            center_x = int(cell_x + cell_size // 2)
            center_y = int(cell_y + cell_size // 2)
            piece = env.board[r, c]
            if piece == 0:
                color = COLOR_EMPTY
            elif piece == 1:
                color = COLOR_P1
            else:
                color = COLOR_P2
            radius = int(cell_size // 2 - 5)
            pygame.gfxdraw.aacircle(screen, center_x, center_y, radius, color)
            pygame.gfxdraw.filled_circle(screen, center_x, center_y, radius, color)
            highlight = tuple(min(255, int(c * 1.1)) for c in color)
            pygame.gfxdraw.filled_circle(screen, center_x, center_y - radius // 3, max(1, radius // 4), highlight)

    pygame.display.flip()


def main():
    pygame.init()
    pygame.display.set_caption("Connect Four")
    env = ConnectFourEnv()
    env.reset()

    cols = env.cols
    rows = env.rows

    width = max(MIN_W, cols * MIN_CELL + LEFT_W + 2 * PAD)
    height = max(MIN_H, rows * MIN_CELL + HEADER_H + STATUS_H + 3 * PAD)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

    font = pygame.font.SysFont("Arial", 22, bold=False)
    menu_font = pygame.font.SysFont("Arial", 17)

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
    online_mode = False  # toggle for guided opponents

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

    def draw_menu(panel_w, panel_h):
        menu_surface = pygame.Surface((panel_w, panel_h))
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
        btn_width = (panel_w - padding * (BUTTONS_PER_ROW + 1)) // BUTTONS_PER_ROW
        mouse_pos = pygame.mouse.get_pos()

        def draw_card(y_start, height, title_text):
            card_rect = pygame.Rect(padding // 2, y_start, panel_w - padding, height)
            pygame.draw.rect(menu_surface, COLOR_CARD, card_rect, border_radius=12)
            pygame.draw.rect(menu_surface, COLOR_OUTLINE, card_rect, width=1, border_radius=12)
            title = menu_font.render(title_text, True, COLOR_MUTED)
            menu_surface.blit(title, (card_rect.x + padding // 2, card_rect.y + 8))
            return card_rect

        def draw_button(rect, label, active):
            hovered = rect.collidepoint((mouse_pos[0], mouse_pos[1]))
            fill = COLOR_BUTTON_ACTIVE if active else (COLOR_BUTTON if not hovered else COLOR_BUTTON_BORDER)
            pygame.draw.rect(menu_surface, fill, rect, border_radius=12)
            pygame.draw.rect(menu_surface, COLOR_BUTTON_BORDER, rect, width=1, border_radius=12)
            text = menu_font.render(label, True, COLOR_TEXT)
            menu_surface.blit(
                text,
                (
                    rect.x + (rect.width - text.get_width()) // 2,
                    rect.y + (rect.height - text.get_height()) // 2,
                ),
            )

        # Card 1: Agent selection
        card_y = padding
        card_h = row_count * (btn_height + 10) + padding * 2 + 10
        card_rect_agents = draw_card(card_y, card_h, "Agent")
        btn_y_offset = card_rect_agents.y + 30
        for idx, (label, kind, ai_side, model_path) in enumerate(options):
            row = idx // BUTTONS_PER_ROW
            col = idx % BUTTONS_PER_ROW
            x = padding + col * (btn_width + padding)
            y = btn_y_offset + row * (btn_height + 10)
            rect = pygame.Rect(x, y, btn_width, btn_height)
            btns.append((rect, label, kind, ai_side, model_path))
            is_active = opponent_kind == kind
            draw_button(rect, label, is_active)

        # Card 2: Player side selection
        side_btns = []
        side_opts = [("You: Player 1 (Red)", 1), ("You: Player 2 (Yellow)", 2)]
        side_width = (panel_w - padding * (len(side_opts) + 1)) // len(side_opts)
        side_height = 50
        card_y2 = card_rect_agents.bottom + padding
        card_h2 = side_height + padding * 2 + 8
        card_rect_you = draw_card(card_y2, card_h2, "You")
        y2 = card_rect_you.y + 30
        for i, (label, side) in enumerate(side_opts):
            x = padding + i * (side_width + padding)
            rect = pygame.Rect(x, y2, side_width, side_height)
            side_btns.append((rect, label, side))
            is_active = human_player == side
            draw_button(rect, label, is_active)

        # Card 3: Mode toggle
        card_y3 = card_rect_you.bottom + padding
        toggle_height = 50
        card_h3 = toggle_height + padding * 2 + 8
        card_rect_mode = draw_card(card_y3, card_h3, "Mode")
        toggle_width = panel_w - 2 * padding
        toggle_x = padding
        toggle_y = card_rect_mode.y + 30
        toggle_rect = pygame.Rect(toggle_x, toggle_y, toggle_width, toggle_height)
        fill = COLOR_BUTTON_ACTIVE if online_mode else COLOR_BUTTON
        pygame.draw.rect(menu_surface, fill, toggle_rect, border_radius=12)
        pygame.draw.rect(menu_surface, COLOR_BUTTON_BORDER, toggle_rect, width=1, border_radius=12)
        toggle_label = "Online" if online_mode else "Offline"
        text = menu_font.render(f"Mode: {toggle_label}", True, COLOR_TEXT)
        menu_surface.blit(
            text,
            (
                toggle_rect.x + (toggle_rect.width - text.get_width()) // 2,
                toggle_rect.y + (toggle_rect.height - text.get_height()) // 2,
            ),
        )

        return menu_surface, btns, side_btns, toggle_rect

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
        w, h = screen.get_size()
        layout_local = compute_layout(w, h, rows, cols)
        draw_board(screen, env, font, render_status_lines(), layout_local)

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

        w, h = screen.get_size()
        layout = compute_layout(w, h, rows, cols)
        left_rect = layout["left_rect"]
        board_rect = layout["board_rect"]
        cell_size = layout["cell_size"]

        screen.fill(COLOR_BG)
        menu_surface, menu_buttons, side_buttons, toggle_rect = draw_menu(left_rect.width, h)
        screen.blit(menu_surface, (0, 0))
        draw_board(screen, env, font, render_status_lines(), layout)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Restart game
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset_game(opponent, ai_player, opponent_name, opponent_kind)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Menu selection
                if left_rect.collidepoint(mx, my):
                    for rect, label, kind, ai_side, model_path in menu_buttons:
                        if rect.collidepoint(mx, my):
                            from agents.opponents import get_opponent
                            selected = None
                            try:
                                if kind is None:
                                    selected = None
                                else:
                                    op_kind = f"online_{kind}" if online_mode else kind
                                    selected = get_opponent(op_kind, model_path=model_path)
                                reset_game(selected, ai_side, label, kind)
                            except Exception as e:
                                print(f"Failed to load opponent {label}: {e}")
                    for rect, label, side in side_buttons:
                        if rect.collidepoint(mx, my):
                            set_human_side(side)
                            reset_game(opponent, ai_player, opponent_name, opponent_kind)
                    if toggle_rect.collidepoint(mx, my):
                        online_mode = not online_mode
                    continue

                # Human move
                if not game_over and board_rect.collidepoint(mx, my):
                    col = int((mx - board_rect.x) // cell_size)
                    if col < 0 or col >= cols:
                        continue
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
