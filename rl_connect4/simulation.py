import random
from pathlib import Path

import pygame
import pygame.gfxdraw

from envs.connect_four_env import ConnectFourEnv
from agents.opponents import get_opponent

# Visual constants (aligned with main GUI style)
COLOR_BG = (11, 18, 32)          # background
COLOR_PANEL = (15, 23, 42)       # panels
COLOR_CARD = (17, 28, 50)        # cards
COLOR_GRID = (36, 74, 145)
COLOR_EMPTY = (16, 18, 25)
COLOR_P1 = (219, 84, 97)
COLOR_P2 = (245, 196, 79)
COLOR_TEXT = (229, 231, 235)
COLOR_MUTED = (148, 163, 184)
COLOR_BUTTON = (48, 58, 80)
COLOR_BUTTON_ACTIVE = (59, 130, 246)
COLOR_BUTTON_BORDER = (71, 85, 105)
COLOR_OUTLINE = (35, 47, 70)

LEFT_W = 360
PAD = 16
HEADER_H = 56
STATUS_H = 110
BUTTONS_PER_ROW = 2
MIN_CELL = 80
MIN_W = LEFT_W + 7 * MIN_CELL + 4 * PAD
MIN_H = HEADER_H + STATUS_H + 6 * MIN_CELL + 4 * PAD

ROOT_DIR = Path(__file__).resolve().parent.parent
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


def load_agent(kind, online_mode=False):
    model_map = {
        "ppo": PPO_DIR / "ppo.pth",
        "ppo_pool": PPO_DIR / "ppo_pool.pth",
        "ppo_dense": PPO_DIR / "ppo_dense.pth",
        "dqn": DQN_DIR / "dqn_connect4.pth",
        "reinforce_manual": REINFORCE_MANUAL_DIR / "reinforce_connect4.pth",
        "reinforce_tianshou": REINFORCE_TIANSHOU_DIR / "reinforce_connect4.pth",
    }
    if kind is None or kind == "human":
        return None
    op_kind = f"guided_{kind}" if online_mode else kind
    model_path = model_map.get(kind)
    return get_opponent("online_" + op_kind, model_path=model_path)


def select_action(agent_obj, kind, env):
    legal = env._legal_moves()
    if not legal:
        return None
    if agent_obj is None or kind in (None, "human"):
        return random.choice(legal)
    try:
        action = agent_obj.select_action(env)
    except Exception:
        action = random.choice(legal)
    if action not in legal:
        action = random.choice(legal)
    return action


def draw_button(surface, rect, label, font, active=False):
    pygame.draw.rect(surface, COLOR_BUTTON_ACTIVE if active else COLOR_BUTTON, rect, border_radius=12)
    pygame.draw.rect(surface, COLOR_BUTTON_BORDER, rect, width=1, border_radius=12)
    text = font.render(label, True, COLOR_TEXT)
    surface.blit(text, (rect.x + (rect.width - text.get_width()) // 2, rect.y + (rect.height - text.get_height()) // 2))


def draw_board(screen, env, font, status_lines, layout, anim=None):
    rows, cols = env.rows, env.cols
    board_rect = layout["board_rect"]
    status_rect = layout["status_rect"]
    header_rect = layout["header_rect"]
    cell_size = layout["cell_size"]

    pygame.draw.rect(screen, COLOR_PANEL, header_rect)
    pygame.draw.rect(screen, COLOR_CARD, status_rect, border_radius=10)
    pygame.draw.rect(screen, COLOR_OUTLINE, status_rect, width=1, border_radius=10)
    title_surface = font.render("Connect Four RL Playground — Agent vs Agent", True, COLOR_TEXT)
    screen.blit(title_surface, (header_rect.x + 10, header_rect.y + 10))

    y_offset = status_rect.y + 12
    for line in status_lines:
        text_surface = font.render(line, True, COLOR_TEXT)
        screen.blit(text_surface, (status_rect.x + 10, y_offset))
        y_offset += font.get_height() + 4

    board_card = board_rect.inflate(PAD, PAD)
    pygame.draw.rect(screen, COLOR_CARD, board_card, border_radius=14)
    pygame.draw.rect(screen, COLOR_OUTLINE, board_card, width=1, border_radius=14)

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

    # Falling disc animation overlay
    if anim:
        col = anim["col"]
        y_pos = anim["y"]
        piece = anim["piece"]
        color = COLOR_P1 if piece == 1 else COLOR_P2
        center_x = int(board_rect.x + col * cell_size + cell_size // 2)
        radius = int(cell_size // 2 - 5)
        pygame.gfxdraw.aacircle(screen, center_x, int(y_pos), radius, color)
        pygame.gfxdraw.filled_circle(screen, center_x, int(y_pos), radius, color)
        highlight = tuple(min(255, int(c * 1.1)) for c in color)
        pygame.gfxdraw.filled_circle(screen, center_x, int(y_pos) - radius // 3, max(1, radius // 4), highlight)

    pygame.display.flip()


def main():
    pygame.init()
    pygame.display.set_caption("Connect Four — Agent vs Agent")
    env = ConnectFourEnv()
    env.reset()
    cols, rows = env.cols, env.rows

    width = max(MIN_W, cols * MIN_CELL + LEFT_W + 2 * PAD)
    height = max(MIN_H, rows * MIN_CELL + HEADER_H + STATUS_H + 3 * PAD)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

    font = pygame.font.SysFont("Arial", 20, bold=False)
    menu_font = pygame.font.SysFont("Arial", 16)

    clock = pygame.time.Clock()
    running = False
    paused = False
    game_over = False
    move_count = 0
    last_action = None
    status_message = ""
    online_mode = False  # keep consistent with existing guided_ logic

    agent_options = [
        ("Random", "random"),
        ("Heuristic", "heuristic"),
        ("PPO", "ppo"),
        ("PPO Pool", "ppo_pool"),
        ("PPO Dense", "ppo_dense"),
        ("DQN", "dqn"),
        ("REINFORCE", "reinforce_manual"),
        ("REINFORCE TS", "reinforce_tianshou"),
    ]
    agent_a_kind = "random"
    agent_b_kind = "heuristic"
    agent_a = load_agent(agent_a_kind, online_mode)
    agent_b = load_agent(agent_b_kind, online_mode)

    speed_map = {"Slow": 6, "Normal": 12, "Fast": 20}
    speed_label = "Normal"
    anim_speed = speed_map[speed_label]
    anim_state = None

    def reset_env():
        nonlocal running, paused, game_over, move_count, last_action, status_message, anim_state
        env.reset()
        running = False
        paused = False
        game_over = False
        move_count = 0
        last_action = None
        status_message = "Ready"
        anim_state = None

    reset_env()

    def status_lines():
        turn = env.current_player + 1
        line1 = f"Mode: Agent vs Agent | Turn: Player {turn}"
        line2 = f"Move: {move_count} | Last action: {last_action if last_action is not None else '-'} | Speed: {speed_label}"
        if game_over and "winner" in last_info:
            line2 = f"Winner: Player {last_info['winner'] + 1}"
        elif game_over and "draw" in last_info:
            line2 = "Result: Draw"
        return [line1, line2, status_message]

    last_info = {}

    def start_match():
        nonlocal running, paused, status_message
        running = True
        paused = False
        status_message = "Running"

    def toggle_pause():
        nonlocal paused, status_message
        paused = not paused
        status_message = "Paused" if paused else "Running"

    def step_once():
        nonlocal running, paused
        if not game_over:
            running = False
            paused = False
            run_agent_step()

    def run_agent_step():
        nonlocal anim_state, last_action, move_count, game_over, last_info, status_message
        if game_over or anim_state:
            return
        legal = env._legal_moves()
        if not legal:
            game_over = True
            last_info = {"draw": True}
            status_message = "No legal moves"
            return
        current_piece = env.current_player + 1
        agent_obj = agent_a if current_piece == 1 else agent_b
        kind = agent_a_kind if current_piece == 1 else agent_b_kind
        action = select_action(agent_obj, kind, env)
        if action is None:
            status_message = "No action selected"
            game_over = True
            return
        # find target row for animation
        target_row = None
        for r in reversed(range(rows)):
            if env.board[r, action] == 0:
                target_row = r
                break
        if target_row is None:
            # fall back to first legal
            action = random.choice(legal)
            for r in reversed(range(rows)):
                if env.board[r, action] == 0:
                    target_row = r
                    break
        if target_row is None:
            status_message = "No space to drop"
            game_over = True
            return
        board_rect = compute_layout(*screen.get_size(), rows, cols)["board_rect"]
        cell_size = compute_layout(*screen.get_size(), rows, cols)["cell_size"]
        start_y = board_rect.y - cell_size
        target_y = board_rect.y + target_row * cell_size + cell_size / 2
        anim_state = {
            "col": action,
            "piece": current_piece,
            "y": start_y,
            "target_y": target_y,
            "action": action,
        }
        last_action = action
        move_count += 1

    running = False
    paused = False

    def maybe_load_agents():
        nonlocal agent_a, agent_b
        agent_a = load_agent(agent_a_kind, online_mode)
        agent_b = load_agent(agent_b_kind, online_mode)

    maybe_load_agents()

    running_loop = True
    while running_loop:
        clock.tick(60)
        w, h = screen.get_size()
        layout = compute_layout(w, h, rows, cols)
        left_rect = layout["left_rect"]

        screen.fill(COLOR_BG)

        # Left panel
        menu_surface = pygame.Surface((left_rect.width, h))
        menu_surface.fill(COLOR_PANEL)
        padding = 10
        gap = 8
        y_cursor = padding

        def draw_card(title, height):
            nonlocal y_cursor
            card_rect = pygame.Rect(padding // 2, y_cursor, left_rect.width - padding, height)
            pygame.draw.rect(menu_surface, COLOR_CARD, card_rect, border_radius=12)
            pygame.draw.rect(menu_surface, COLOR_OUTLINE, card_rect, width=1, border_radius=12)
            title_surf = menu_font.render(title, True, COLOR_MUTED)
            menu_surface.blit(title_surf, (card_rect.x + padding // 2, card_rect.y + 8))
            y_cursor += height + padding
            return card_rect

        # Agent A selection
        card_height_a = ((len(agent_options) + BUTTONS_PER_ROW - 1) // BUTTONS_PER_ROW) * (42 + gap) + padding + 20
        card_a = draw_card("Agent A", card_height_a)
        btn_width = (left_rect.width - padding * (BUTTONS_PER_ROW + 1)) // BUTTONS_PER_ROW
        btn_height = 42
        mouse_pos = pygame.mouse.get_pos()
        btns_a = []
        start_y = card_a.y + 30
        for idx, (label, kind) in enumerate(agent_options):
            row = idx // BUTTONS_PER_ROW
            col = idx % BUTTONS_PER_ROW
            x = padding + col * (btn_width + padding)
            y = start_y + row * (btn_height + gap)
            rect = pygame.Rect(x, y, btn_width, btn_height)
            btns_a.append((rect, kind))
            active = agent_a_kind == kind
            hovered = rect.collidepoint(mouse_pos)
            fill = COLOR_BUTTON_ACTIVE if active else (COLOR_BUTTON if not hovered else COLOR_BUTTON_BORDER)
            pygame.draw.rect(menu_surface, fill, rect, border_radius=12)
            pygame.draw.rect(menu_surface, COLOR_BUTTON_BORDER, rect, width=1, border_radius=12)
            text = menu_font.render(label, True, COLOR_TEXT)
            menu_surface.blit(text, (rect.x + (rect.width - text.get_width()) // 2, rect.y + (rect.height - text.get_height()) // 2))

        # Agent B selection
        card_height_b = card_height_a
        card_b = draw_card("Agent B", card_height_b)
        btns_b = []
        start_y_b = card_b.y + 30
        for idx, (label, kind) in enumerate(agent_options):
            row = idx // BUTTONS_PER_ROW
            col = idx % BUTTONS_PER_ROW
            x = padding + col * (btn_width + padding)
            y = start_y_b + row * (btn_height + gap)
            rect = pygame.Rect(x, y, btn_width, btn_height)
            btns_b.append((rect, kind))
            active = agent_b_kind == kind
            hovered = rect.collidepoint(mouse_pos)
            fill = COLOR_BUTTON_ACTIVE if active else (COLOR_BUTTON if not hovered else COLOR_BUTTON_BORDER)
            pygame.draw.rect(menu_surface, fill, rect, border_radius=12)
            pygame.draw.rect(menu_surface, COLOR_BUTTON_BORDER, rect, width=1, border_radius=12)
            text = menu_font.render(label, True, COLOR_TEXT)
            menu_surface.blit(text, (rect.x + (rect.width - text.get_width()) // 2, rect.y + (rect.height - text.get_height()) // 2))

        # Controls card
        control_height = 190
        card_ctrl = draw_card("Controls", control_height)
        ctrl_y = card_ctrl.y + 30
        ctrl_padding = padding
        btn_w_full = card_ctrl.width - 2 * ctrl_padding
        start_rect = pygame.Rect(card_ctrl.x + ctrl_padding, ctrl_y, btn_w_full, 40)
        pause_rect = pygame.Rect(card_ctrl.x + ctrl_padding, ctrl_y + 46, btn_w_full, 40)
        reset_rect = pygame.Rect(card_ctrl.x + ctrl_padding, ctrl_y + 92, btn_w_full, 40)
        draw_button(menu_surface, start_rect, "Start", menu_font, active=False)
        draw_button(menu_surface, pause_rect, "Pause / Resume", menu_font, active=False)
        draw_button(menu_surface, reset_rect, "Reset", menu_font, active=False)

        speed_y = ctrl_y + 140
        speed_rects = []
        speed_labels = ["Slow", "Normal", "Fast"]
        btn_sw = (card_ctrl.width - ctrl_padding * 2 - (len(speed_labels) - 1) * 8) // len(speed_labels)
        for i, lbl in enumerate(speed_labels):
            x = card_ctrl.x + ctrl_padding + i * (btn_sw + 8)
            rect = pygame.Rect(x, speed_y, btn_sw, 34)
            speed_rects.append((rect, lbl))
            draw_button(menu_surface, rect, lbl, menu_font, active=(speed_label == lbl))

        # Status card
        status_height = 140
        card_status = draw_card("Status", status_height)
        status_txt = [
            f"A: {agent_a_kind}",
            f"B: {agent_b_kind}",
            f"State: {'Running' if running and not paused else ('Paused' if paused else 'Idle')}",
        ]
        sy = card_status.y + 30
        for line in status_txt:
            t = menu_font.render(line, True, COLOR_TEXT)
            menu_surface.blit(t, (card_status.x + ctrl_padding, sy))
            sy += menu_font.get_height() + 6

        screen.blit(menu_surface, (0, 0))

        # Board + status right side
        draw_board(screen, env, font, status_lines(), layout, anim_state)

        # Handle animation progress
        if anim_state:
            anim_state["y"] += anim_speed
            if anim_state["y"] >= anim_state["target_y"]:
                anim_state["y"] = anim_state["target_y"]
                # apply env step once
                _, _, terminated, truncated, info = env.step(anim_state["action"])
                last_info = info
                game_over = terminated or truncated
                anim_state = None
                if game_over:
                    running = False
                    status_message = "Finished"

        # Auto-play loop
        if running and not paused and not game_over and not anim_state:
            run_agent_step()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_loop = False
            if event.type == pygame.VIDEORESIZE:
                new_w = max(event.w, MIN_W)
                new_h = max(event.h, MIN_H)
                screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if left_rect.collidepoint(mx, my):
                    # translate to menu_surface coords (0,0)
                    local_x, local_y = mx, my
                    # Agent A
                    for rect, kind in btns_a:
                        r = rect.move(0, 0)
                        if r.collidepoint(local_x, local_y):
                            agent_a_kind = kind
                            maybe_load_agents()
                    # Agent B
                    for rect, kind in btns_b:
                        r = rect.move(0, 0)
                        if r.collidepoint(local_x, local_y):
                            agent_b_kind = kind
                            maybe_load_agents()
                    # Controls
                    if start_rect.collidepoint(local_x, local_y):
                        start_match()
                    if pause_rect.collidepoint(local_x, local_y):
                        toggle_pause()
                    if reset_rect.collidepoint(local_x, local_y):
                        reset_env()
                    for rect, lbl in speed_rects:
                        if rect.collidepoint(local_x, local_y):
                            speed_label = lbl
                            anim_speed = speed_map[speed_label]
                    # Step if right-click on status? Add explicit: shift-click on start => step
                else:
                    # optional: step on right panel right-click
                    if event.button == 3 and not game_over:
                        step_once()

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
