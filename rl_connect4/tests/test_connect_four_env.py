import numpy as np
import pytest

from envs.connect_four_env import ConnectFourEnv


def test_reset_shapes_and_mask():
    env = ConnectFourEnv()
    obs, info = env.reset()
    assert info == {}
    assert obs["observation"].shape == (env.rows, env.cols, 2)
    assert obs["observation"].dtype == np.int8
    assert obs["action_mask"].tolist() == [1] * env.cols


def test_illegal_move_penalty_and_termination():
    env = ConnectFourEnv()
    env.reset()
    col = 0
    for _ in range(env.rows):
        env.step(col)

    _, reward, terminated, truncated, info = env.step(col)
    assert terminated is True
    assert truncated is False
    assert info.get("illegal_move") is True
    assert reward == env.illegal_move_penalty


def test_horizontal_win_detection():
    env = ConnectFourEnv()
    env.reset()
    moves = [0, 6, 1, 6, 2, 6, 3]
    for move in moves[:-1]:
        env.step(move)
    _, reward, terminated, truncated, info = env.step(moves[-1])
    assert terminated is True
    assert truncated is False
    assert info.get("winner") == 0
    assert reward == env.win_reward


def test_vertical_win_detection():
    env = ConnectFourEnv()
    env.reset()
    moves = [0, 1, 0, 1, 0, 1, 0]
    for move in moves[:-1]:
        env.step(move)
    _, reward, terminated, truncated, info = env.step(moves[-1])
    assert terminated is True
    assert truncated is False
    assert info.get("winner") == 0
    assert reward == env.win_reward


def test_diagonal_win_detection():
    env = ConnectFourEnv()
    env.reset()
    moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]
    for move in moves[:-1]:
        env.step(move)
    _, reward, terminated, truncated, info = env.step(moves[-1])
    assert terminated is True
    assert truncated is False
    assert info.get("winner") == 0
    assert reward == env.win_reward


def test_action_mask_updates_when_column_full():
    env = ConnectFourEnv()
    env.reset()
    last_obs = None
    for _ in range(env.rows):
        last_obs, _, _, _, _ = env.step(0)
    assert last_obs is not None
    assert last_obs["action_mask"][0] == 0
