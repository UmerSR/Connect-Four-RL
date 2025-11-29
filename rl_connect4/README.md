# RL Connect Four

Minimal self-play friendly Connect Four environment, GUI, and PPO scaffold.

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python rl_connect4/main.py  # launch human vs human GUI
```

## Environment (`envs/connect_four_env.py`)

- Gymnasium-style single-agent API with action mask for legal columns.
- Configurable `rows`, `cols`, `connect_n`, rewards, and `render_mode="ansi"`.
- Rewards: win = `win_reward` (default 1.0), illegal = `illegal_move_penalty` (default -1.0), draw = `draw_reward` (default 0.0).

## Agents and self-play

- `agents/simple_agents.py` provides `RandomAgent` and a small heuristic (win/block/center).
- Run quick evaluation:

```bash
python -m training.self_play --episodes 50 --heuristic
```

## PPO scaffold

- `training/ppo.py` contains a lightweight manual PPO loop with a ConvNet actor-critic.
- Kick off a small self-play run (CPU by default):

```bash
python -m training.ppo
```

## Tests

- Pytest coverage for core environment rules:

```bash
python -m pytest rl_connect4/tests
```
