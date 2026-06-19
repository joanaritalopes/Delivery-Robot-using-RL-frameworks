# Delivery Robot using RL Frameworks

A reinforcement learning project that trains a simplified delivery robot to navigate warehouse grid environments. Two RL algorithms are implemented: **DQN** (Deep Q-Network) and **PPO** (Proximal Policy Optimization).

---

## Setup

**Requirements:** Python 3.10+, PyTorch

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── train.py              # Main training script
├── sweep.py              # Hyperparameter sweep (OAT)
├── agents/
│   ├── dqn_agent.py      # DQN implementation
│   ├── ppo_agent.py      # PPO implementation
│   ├── random_agent.py   # Random baseline agent
│   └── null_agent.py     # No-op baseline agent
├── world/
│   ├── environment.py    # RL environment
│   ├── grid.py           # Grid representation
│   ├── grid_creator.py   # Web-based grid editor
│   └── path_visualizer.py
├── grid_configs/         # Predefined .npy grid files
└── results/              # Training logs (CSV) and plots (auto-generated)
```

---

## Training an Agent

```bash
python3 train.py <GRID> --agent <dqn|ppo> [options]
```

The `GRID` argument is required and must be a path to a `.npy` grid file. You can pass **multiple grids** to train sequentially.

### Quick start examples

```bash
# Train DQN on the flying V warehouse grid (no GUI, 500k steps)
python3 train.py grid_configs/flying_v.npy --agent dqn --no_gui --iter 500000

# Train PPO on the fishbone grid
python3 train.py grid_configs/fishbone.npy --agent ppo --no_gui --iter 500000

# Train DQN on multiple grids back-to-back
python3 train.py grid_configs/fishbone.npy grid_configs/flying_v.npy --agent dqn --no_gui --iter 500000
```

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `GRID` | *(required)* | Path(s) to `.npy` grid file(s) |
| `--agent` | `dqn` | Agent type: `dqn` or `ppo` |
| `--no_gui` | off | Disable rendering for faster training |
| `--iter` | `200000` | Total environment steps to train for |
| `--sigma` | `0.1` | Environment stochasticity (0 = deterministic) |
| `--fps` | `30` | Render FPS (ignored if `--no_gui`) |
| `--random_seed` | `0` | Random seed |
| `--start_pos` | auto | Agent start position as `row,col` (e.g. `2,3`) |
| `--eval_episodes` | `20` | Episodes per evaluation run |
| `--eval_every` | `10` | Evaluate every N training episodes |
| `--converge_patience` | `5` | Early-stop after N consecutive evals at threshold |
| `--converge_threshold` | `0.95` | Success rate threshold for early stopping |
| `--lr` | `0.001` | Learning rate |
| `--gamma` | `0.99` | Discount factor |

**DQN-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | `64` | Replay buffer batch size |
| `--epsilon_decay` | `0.9995` | Multiplicative epsilon decay per update |
| `--min_epsilon` | `0.01` | Minimum epsilon value |
| `--min_buffer` | `1000` | Minimum buffer size before training starts |
| `--max_buffer` | `50000` | Maximum replay buffer size |
| `--target_update` | `200` | Steps between target network updates |

**PPO-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--clip_eps` | `0.2` | Clipping epsilon for surrogate loss |
| `--rollout_size` | `256` | Steps collected per rollout before updating |
| `--ppo_epochs` | `10` | Update epochs per rollout |
| `--gae_lambda` | `0.95` | Lambda for Generalized Advantage Estimation |
| `--entropy_coef` | `0.01` | Entropy bonus coefficient |
| `--value_coef` | `0.5` | Value loss coefficient |

### Custom hyperparameter examples

```bash
# DQN with custom hyperparameters
python3 train.py grid_configs/warehouse_small.npy --agent dqn --no_gui --iter 500000 --lr 0.0001 --gamma 0.99 --batch_size 128 --epsilon_decay 0.9999 --target_update 500

# PPO with custom hyperparameters
python3 train.py grid_configs/warehouse_small.npy --agent ppo --no_gui --iter 500000 --lr 0.0003 --gamma 0.99 --clip_eps 0.2 --rollout_size 512 --ppo_epochs 10
```

### Output

Training logs are saved automatically to `results/` with timestamps:
- `results/<timestamp>_<agent>_<grid>_train.csv` — per-episode metrics
- `results/<timestamp>_<agent>_<grid>_eval.csv` — periodic evaluation metrics

---

## Available Grid Configs

Pre-built grids are in `grid_configs/`:

| File | Description |
|---|---|
| `example_grid.npy` | Small example grid for quick tests |
| `small_grid.npy` | Small grid |
| `large_grid.npy` | Large grid |
| `super_hard.npy` | Challenging environment |
| `A1_grid.npy` | Assignment 1 grid |
| `warehouse_small.npy` | Small warehouse layout |
| `warehouse_large.npy` | Large warehouse layout |
| `fishbone.npy` | Fishbone aisle pattern |
| `flying_v.npy` | Flying-V aisle pattern |
| `half_aisles.npy` | Half-aisle layout |


---

## Hyperparameter Sweep

Script for one-at-a-time (OAT) hyperparameter search across DQN and PPO.

### `sweep.py` — Standard sweep and the one we choose to use at the end given that is more efficient

```bash
# Full sweep: both agents, all 3 grids (fishbone, flying_v, half_aisles)
python3 sweep.py

# Sweep only DQN
python3 sweep.py --agent dqn

# Sweep specific grids
python3 sweep.py --agent dqn --grids fishbone flying_v

# Resume a crashed sweep
python3 sweep.py --resume

# Run only the finalize step (re-run top-2 configs with 3 seeds)
python3 sweep.py --finalize

# Save results to a custom directory
python3 sweep.py --results_dir my_sweep_output
```

| Argument | Default | Description |
|---|---|---|
| `--agent` | `both` | Agent(s) to sweep: `dqn`, `ppo`, or `both` |
| `--grids` | all three | Grids to sweep: `fishbone`, `flying_v`, `half_aisles` |
| `--results_dir` | `sweep_results` | Output directory for sweep results |
| `--resume` | off | Skip already-completed configs |
| `--finalize` | off | Skip sweep; only re-run top-2 configs with multiple seeds |

### Sweep output

```
sweep_results/
├── sweep_summary.csv         # One row per config: metrics + hyperparams
├── configs.json              # Full config dict for each config_id
├── <config_id>_train.csv     # Per-episode training log
├── <config_id>_eval.csv      # Evaluation log
└── final_runs/
    ├── final_summary.csv     # Results for top-2 configs × 3 seeds
    └── best_commands.json    # Best CLI commands to reproduce top runs
```

---

## Final Best models CLI comand with best hyperparameters

