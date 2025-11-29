import pygame

from envs.connect_four_env import ConnectFourEnv


# Colors (RGB)
COLOR_BG = (30, 30, 30)
COLOR_GRID = (0, 0, 160)
COLOR_EMPTY = (10, 10, 10)
COLOR_P1 = (220, 60, 60)   # Player 1 discs
COLOR_P2 = (250, 210, 50)  # Player 2 discs
COLOR_TEXT = (230, 230, 230)

SQUARE_SIZE = 80
PADDING_TOP = 80  # Space for text/status


def draw_board(screen, env, font, status_text):
    rows, cols = env.rows, env.cols
    screen.fill(COLOR_BG)

    # Draw status text
    text_surface = font.render(status_text, True, COLOR_TEXT)
    screen.blit(text_surface, (10, 10))

    # Draw grid + discs
    board_top = PADDING_TOP
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
    pygame.display.set_caption("Connect Four (Human vs Human)")
    env = ConnectFourEnv()
    env.reset()

    cols = env.cols
    rows = env.rows

    width = cols * SQUARE_SIZE
    height = rows * SQUARE_SIZE + PADDING_TOP
    screen = pygame.display.set_mode((width, height))

    font = pygame.font.SysFont("Arial", 24)

    running = True
    game_over = False
    last_info = {}
    clock = pygame.time.Clock()

    def status_line():
        nonlocal game_over, last_info
        if game_over:
            if "winner" in last_info:
                winner = last_info["winner"]
                return f"Game over: Player {winner + 1} wins! Press R to restart."
            elif "draw" in last_info:
                return "Game over: Draw. Press R to restart."
            elif "illegal_move" in last_info:
                return "Illegal move! Game over. Press R to restart."
            else:
                return "Game over. Press R to restart."
        else:
            # current_player has already been flipped after last move,
            # so this is the player who is about to play.
            return f"Player {env.current_player + 1}'s turn"

    draw_board(screen, env, font, status_line())

    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Restart game
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                game_over = False
                last_info = {}
                draw_board(screen, env, font, status_line())

            # Mouse click -> choose column (for human players)
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                x, y = event.pos
                if y >= PADDING_TOP:
                    col = x // SQUARE_SIZE
                    if 0 <= col < cols:
                        _, reward, terminated, truncated, info = env.step(col)
                        game_over = terminated or truncated
                        last_info = info
                        draw_board(screen, env, font, status_line())

    pygame.quit()


if __name__ == "__main__":
    main()
