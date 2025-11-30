import pygame

from envs.connect_four_env import ConnectFourEnv


# Colors (RGB)
COLOR_BG = (30, 30, 30)
COLOR_GRID = (0, 0, 160)
COLOR_EMPTY = (10, 10, 10)
COLOR_P1 = (220, 60, 60)   # Player 1 discs
COLOR_P2 = (250, 210, 50)  # Player 2 discs
COLOR_TEXT = (230, 230, 230)

SQUARE_SIZE = 90
PADDING_TOP = 100  # Space for text/status
MENU_HEIGHT = 220  # Space for opponent select menu and spacing


def draw_board(screen, env, font, status_text):
    rows, cols = env.rows, env.cols
    board_top = MENU_HEIGHT + PADDING_TOP

    # Draw status text
    text_surface = font.render(status_text, True, COLOR_TEXT)
    screen.blit(text_surface, (10, MENU_HEIGHT + 10))

    # Draw grid + discs
    for c in range(cols):
        for r in range(rows):
            # Cell background (board)
            cell_x = c * SQUARE_SIZE
            cell_y = board_top + r * SQUARE_SIZE
            pygame.draw.rect(
                screen,
                COLOR_GRID,
                (cell_x, cell_y, SQUARE_SIZE, SQUARE_SIZE),
            )

            # Disc
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

    width = cols * SQUARE_SIZE + 60  # add margin for breathing room
    height = rows * SQUARE_SIZE + PADDING_TOP + MENU_HEIGHT + 40
    screen = pygame.display.set_mode((width, height))

    font = pygame.font.SysFont("Arial", 24)
    menu_font = pygame.font.SysFont("Arial", 24)

    running = True
    game_over = False
    last_info = {}
    last_message = ""
    clock = pygame.time.Clock()
    opponent = None
    opponent_name = "Human vs Human"
    ai_player = None  # 1 or 2 for which side AI controls
    human_player = 1  # default human side

    def status_line():
        nonlocal game_over, last_info, last_message
        prefix = f"Mode: {opponent_name} | You are Player {human_player}"
        if game_over:
            if "winner" in last_info:
                winner = last_info["winner"]
                return f"{prefix} | Game over: Player {winner + 1} wins! Press R to restart."
            elif "draw" in last_info:
                return f"{prefix} | Game over: Draw. Press R to restart."
            elif "illegal_move" in last_info:
                return f"{prefix} | Illegal move! Game over. Press R to restart."
            else:
                return f"{prefix} | Game over. Press R to restart."
        else:
            turn_text = f"{prefix} | Player {env.current_player + 1}'s turn"
            if last_message:
                return f"{turn_text} | {last_message}"
            return turn_text

    def draw_menu():
        menu_surface = pygame.Surface((width, MENU_HEIGHT))
        menu_surface.fill((40, 40, 40))
        options = [
            ("Human", None, None, None),
            ("Random", "random", 2, None),
            ("Heuristic", "heuristic", 2, None),
            ("PPO", "ppo", 2, "PPO/ppo.pth"),
            ("PPO_pool", "ppo_pool", 2, "PPO/ppo_pool.pth"),
            ("PPO_dense", "ppo_dense", 2, "PPO/ppo_dense.pth"),
        ]
        btns = []
        padding = 25
        btn_width = (width - padding * (len(options) + 1)) // len(options)
        btn_height = 70
        y = 20
        for i, (label, kind, ai_side, model_path) in enumerate(options):
            x = padding + i * (btn_width + padding)
            rect = pygame.Rect(x, y, btn_width, btn_height)
            btns.append((rect, label, kind, ai_side, model_path))
            pygame.draw.rect(menu_surface, (70, 70, 120), rect, border_radius=8)
            text = menu_font.render(label, True, COLOR_TEXT)
            menu_surface.blit(text, (rect.x + (rect.width - text.get_width()) // 2,
                                     rect.y + (rect.height - text.get_height()) // 2))
        # Human side buttons
        side_btns = []
        side_opts = [("You: Player 1 (Red)", 1), ("You: Player 2 (Yellow)", 2)]
        side_width = (width - padding * (len(side_opts) + 1)) // len(side_opts)
        side_height = 60
        y2 = y + btn_height + 40
        for i, (label, side) in enumerate(side_opts):
            x = padding + i * (side_width + padding)
            rect = pygame.Rect(x, y2, side_width, side_height)
            side_btns.append((rect, label, side))
            pygame.draw.rect(menu_surface, (90, 90, 140), rect, border_radius=8)
            text = menu_font.render(label, True, COLOR_TEXT)
            menu_surface.blit(text, (rect.x + (rect.width - text.get_width()) // 2,
                                     rect.y + (rect.height - text.get_height()) // 2))
        return menu_surface, btns, side_btns

    menu_surface, menu_buttons, side_buttons = draw_menu()

    def reset_game(selected_opponent, ai_side, opponent_label):
        nonlocal opponent, opponent_name, ai_player, game_over, last_info, last_message
        opponent = selected_opponent
        opponent_name = opponent_label
        ai_player = ai_side
        if opponent is not None:
            ai_player = 1 if human_player == 2 else 2
        env.reset()
        game_over = False
        last_info = {}
        last_message = f"You are Player {human_player}"
        draw_board(screen, env, font, status_line())

    def set_human_side(side: int):
        nonlocal human_player, ai_player, opponent, opponent_name, last_message
        human_player = side
        if opponent is not None:
            ai_player = 1 if human_player == 2 else 2
        else:
            ai_player = None
            opponent_name = "Human vs Human"
        last_message = f"You are Player {human_player}"

    draw_board(screen, env, font, status_line())

    while running:
        clock.tick(30)

        screen.fill(COLOR_BG)
        screen.blit(menu_surface, (0, 0))
        draw_board(screen, env, font, status_line())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Restart game
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset_game(opponent, ai_player, opponent_name)

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
                                reset_game(selected, ai_side, label)
                            except Exception as e:
                                print(f"Failed to load opponent {label}: {e}")
                    for rect, label, side in side_buttons:
                        if rect.collidepoint(mx, my):
                            set_human_side(side)
                            reset_game(opponent, ai_player, opponent_name)
                    continue

                # Human move
                if not game_over and my >= MENU_HEIGHT:
                    col = mx // SQUARE_SIZE
                    if 0 <= col < cols:
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
