# Connect Four RL Experiments

End-to-end Connect Four self-play experiments built around the `rl_connect4` environment, with PPO and DQN training notebooks plus saved checkpoints.

## Repository Layout
- `rl_connect4/` – environment, agents, training scripts, and tests.
- `notebooks/`
  - `ppo/` – PPO training variants (`ppo.ipynb`, `ppo_dense.ipynb`, `ppo_pool.ipynb`).
  - `dqn/` – DQN notebooks (`dqn.ipynb` clean run, `dqn_run_outputs.ipynb` with executed cells/outputs).
- `artifacts/`
  - `ppo/` – saved PPO weights (`ppo.pth`, `ppo_dense.pth`, `ppo_pool.pth`).
  - `dqn/` – saved DQN checkpoints (`dqn_connect4_selfplay.zip`, `dqn_connect4.pth`) and run outputs (`outputs.zip`).
- `docs/` – project documents and checkpoint report.
- `.venv/` – local virtual environment (ignored).

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r rl_connect4/requirements.txt
```

## Running
- **Environment & CLI/GUI**: `python rl_connect4/main.py`
- **Training scripts**: `python -m training.ppo` (see `rl_connect4/training/` for options)
- **Notebooks**: open files under `notebooks/` in Jupyter/Colab; each notebook handles its own repo checkout and dependencies.

## Models & Reports
- Trained weights live in `artifacts/ppo` and `artifacts/dqn`.
- Documents (plan/report) are in `docs/`.
