# Connect Four RL Experiments

End-to-end Connect Four self-play experiments using the custom `rl_connect4` environment, with PPO, DQN, and REINFORCE agents, comparison notebooks, and a playable GUI.

## Repository Layout
- `rl_connect4/` — environment, agents (including guided/online modes), GUI (`python rl_connect4/main.py`), and utilities.
- `notebooks/`
  - `ppo/` — PPO training variants (`ppo.ipynb`, `ppo_dense.ipynb`, `ppo_pool.ipynb`).
  - `dqn/` — DQN notebooks (`dqn.ipynb` clean run, `dqn_run_outputs.ipynb` executed snapshot).
  - `reinforce/` — REINFORCE notebooks (manual and Tianshou variants).
  - `analysis/` — head-to-head and meta-analysis notebooks.
- `artifacts/`
  - `ppo/` — PPO weights (`ppo.pth`, `ppo_dense.pth`, `ppo_pool.pth`).
  - `dqn/` — DQN checkpoints (`dqn_connect4_selfplay.zip`, `dqn_connect4.pth`) and outputs.
  - `reinforce_manual/`, `reinforce_tianshou/` — REINFORCE weights.
- `docs/` — project documents and checkpoint reports.

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r rl_connect4/requirements.txt
```

## Running
- **GUI / play**: `python rl_connect4/main.py` (Select opponent agent and turn, online/offline toggles guided opponents).
- **Agent-vs-Agent simulator**: `python rl_connect4/simulation.py` (animates matches between two selected agents with start/pause/reset and speed control).
- **Training scripts**: see `rl_connect4/training/` (e.g., `python -m training.ppo`).
- **Notebooks**: open under `notebooks/` in Jupyter/Colab; each handles its own setup.

## Models & Reports
- Trained weights live under `artifacts/` as above.
- Documents and reports are in `docs/`.
