"""
One-at-a-time (OAT) hyperparameter sweep for DQN and PPO.

Early stopping: train.py is called with --converge_patience and
   --converge_threshold. If eval_success_rate ≥ threshold for that many
   consecutive eval checks, training stops before exhausting --iter

Seeds = 3

Primary metric: eval_success_rate (delivery task completion rate),
   as recommended by the professor. eval_mean_reward used as tiebreaker.

Sweepable environment/training parameters:
   - max_steps_per_episode: caps each episode so early training doesn't
     spend all budget on one random walk
   - iter (total step budget): test if more steps help
   - eval_every: how often to checkpoint (affects early stopping granularity)
   - sigma: environment stochasticity
   All hyperparameters (lr, gamma, batch_size, etc.) are also swept.

Usage
─────
  # Full sweep (both agents, 3 grids, 1 seed):
  python3 sweep.py

  # One agent only:
  python3 sweep.py --agent dqn
  python3 sweep.py --agent ppo

  # Dry run (print commands, don't execute):
  python3 sweep.py --dry_run

  # Resume if interrupted:
  python3 sweep.py --resume

  # After sweep: re-run top-2 per (agent, grid) with 3 seeds:
  python3 sweep.py --finalize

Output
──────
  sweep_results/
    sweep_summary.csv        one row per config, best eval metrics
    configs.json             config dict per config_id (needed for finalize)
    commands_run.txt         full audit log of all commands executed
    <config_id>_eval.csv    eval CSV from train.py (moved here)
    <config_id>_train.csv   train CSV from train.py (moved here)
    final_runs/
      final_summary.csv      best 2 configs x 3 seeds x 6 combos = 36 rows
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


GRIDS = [
    "grid_configs/fishbone.npy",
    "grid_configs/flying_v.npy",
    "grid_configs/half_aisles.npy",
]

SWEEP_SEEDS = [0, 1, 2]          # 1 seed during OAT sweep (seed=0)
FINAL_SEEDS = [0, 1, 2]          # 3 seeds for final top-2 validation

DEFAULTS = {
    # ── Environment / training ──────────────────────────────────────────────
    "sigma":                 0.1,    # env stochasticity
    "iter":                  500000, # total step budget
    "max_steps_per_episode": 2000,   # cap per episode (prevents runaway episodes)
    "eval_every":            20,     # evaluation checkpoint frequency (in episodes)
    "eval_episodes":         10,     # episodes per evaluation checkpoint
    "converge_patience":     5,      # consecutive evals at threshold to stop early
    "converge_threshold":    0.95,   # success rate that counts as "converged"
    # ── Shared hyperparameters ───────────────────────────────────────────────
    "lr":                    0.001,
    "gamma":                 0.99,
    # ── DQN-specific ────────────────────────────────────────────────────────
    "batch_size":            64,
    "epsilon_decay":         0.9995,
    "min_epsilon":           0.01,
    "min_buffer":            1000,
    "max_buffer":            50000,
    "target_update":         200,
    # ── PPO-specific ────────────────────────────────────────────────────────
    "clip_eps":              0.2,
    "rollout_size":          256,
    "ppo_epochs":            10,
    "gae_lambda":            0.95,
    "entropy_coef":          0.01,
    "value_coef":            0.5,
}


SHARED_SWEEP = {
    "lr":    [1e-4, 5e-4, 1e-3],
    "gamma": [0.95, 0.97, 0.99, 0.999],
    # ── Environment parameters ──────────────────────────────────────────────
    "sigma":                 [0.0, 0.05, 0.1, 0.2],
    "iter":                  [200000, 500000, 1000000],
    "max_steps_per_episode": [500, 1000, 2000, 5000],
    "eval_every":            [10, 20, 50],
}


DQN_SWEEP = {
    "epsilon_decay": [0.999, 0.9995, 0.9998, 0.99995],  # default: 0.9995
    "batch_size":    [32, 64, 128, 256],                 # default: 64
    "target_update": [100, 200, 500, 1000],              # default: 200
    "max_buffer":    [10000, 50000, 100000, 200000],     # default: 50000
}

# PPO-only parameters
PPO_SWEEP = {
    "clip_eps":     [0.1, 0.2, 0.3, 0.4],               # default: 0.2
    "rollout_size": [128, 256, 512, 1024],               # default: 256
    "ppo_epochs":   [5, 10, 20, 40],                     # default: 10
    "gae_lambda":   [0.9, 0.95, 0.98, 1.0],             # default: 0.95
    "entropy_coef": [0.0, 0.01, 0.05, 0.1],             # default: 0.01
    "value_coef":   [0.25, 0.5, 1.0],                   # default: 0.5
}


def generate_configs(agents: list[str]) -> list[dict]:
    """
    Generate all OAT configs.
    For each (agent, grid, param, value) combination, one config dict is
    produced with all parameters at default except the swept one.
    One baseline config (all defaults) is also generated per (agent, grid).
    """
    configs = []

    for agent in agents:
        # Determine which param sweeps apply
        param_sweeps = dict(SHARED_SWEEP)
        if agent == "dqn":
            param_sweeps.update(DQN_SWEEP)
        else:
            param_sweeps.update(PPO_SWEEP)

        for grid in GRIDS:
            baseline = {
                "agent":       agent,
                "grid":        grid,
                "swept_param": "baseline",
                "swept_value": "default",
                "random_seed": SWEEP_SEEDS[0],   # single seed for sweep
                **DEFAULTS,
            }
            configs.append(baseline)

            for param, values in param_sweeps.items():
                for value in values:
                    # Skip if this equals the default (baseline covers it)
                    if value == DEFAULTS.get(param):
                        continue
                    cfg = {
                        "agent":       agent,
                        "grid":        grid,
                        "swept_param": param,
                        "swept_value": value,
                        "random_seed": SWEEP_SEEDS[0],
                        **DEFAULTS,
                        param: value,   # override the swept param
                    }
                    configs.append(cfg)

    return configs


def config_to_cmd(cfg: dict) -> list[str]:
    """
    Build the train.py CLI command for this config.
    Early stopping flags:
      --converge_patience  N  stop if eval_success_rate ≥ threshold for N evals
      --converge_threshold T  success rate threshold for early stopping
    """
    return [
        sys.executable, "train.py",
        cfg["grid"],
        "--agent",                   cfg["agent"],
        "--no_gui",
        "--iter",                    str(cfg["iter"]),
        "--sigma",                   str(cfg["sigma"]),
        "--random_seed",             str(cfg["random_seed"]),
        "--eval_every",              str(cfg["eval_every"]),
        "--eval_episodes",           str(cfg["eval_episodes"]),
        "--max_steps_per_episode",   str(cfg["max_steps_per_episode"]),
        "--converge_patience",       str(cfg["converge_patience"]),
        "--converge_threshold",      str(cfg["converge_threshold"]),
        # Shared hyperparams
        "--lr",                      str(cfg["lr"]),
        "--gamma",                   str(cfg["gamma"]),
        # DQN hyperparams (ignored by PPO in train.py)
        "--batch_size",              str(cfg["batch_size"]),
        "--epsilon_decay",           str(cfg["epsilon_decay"]),
        "--min_epsilon",             str(cfg["min_epsilon"]),
        "--min_buffer",              str(cfg["min_buffer"]),
        "--max_buffer",              str(cfg["max_buffer"]),
        "--target_update",           str(cfg["target_update"]),
        # PPO hyperparams (ignored by DQN in train.py)
        "--clip_eps",                str(cfg["clip_eps"]),
        "--rollout_size",            str(cfg["rollout_size"]),
        "--ppo_epochs",              str(cfg["ppo_epochs"]),
        "--gae_lambda",              str(cfg["gae_lambda"]),
        "--entropy_coef",            str(cfg["entropy_coef"]),
        "--value_coef",              str(cfg["value_coef"]),
    ]


def extract_best_eval(eval_csv: Path) -> dict:
    """
    Read an eval CSV and return the row with the highest eval_success_rate.
    Tiebreaker: eval_mean_reward (higher = better), then latest episode.
    """
    if not eval_csv.exists():
        return {}
    rows = []
    with open(eval_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "episode":           int(row["episode"]),
                    "steps_so_far":      int(row["steps_so_far"]),
                    "eval_mean_reward":  float(row["eval_mean_reward"]),
                    "eval_std_reward":   float(row["eval_std_reward"]),
                    "eval_success_rate": float(row["eval_success_rate"]),
                    "eval_mean_length":  float(row["eval_mean_length"]),
                })
            except (ValueError, KeyError):
                continue
    if not rows:
        return {}
    best = max(
        rows,
        key=lambda r: (r["eval_success_rate"], r["eval_mean_reward"], r["episode"])
    )
    return best


def run_config(cfg: dict, config_id: str, out_dir: Path,
               cmd_log: Path, dry_run: bool = False) -> dict:
    """
    Execute train.py for one config. Move the resulting CSVs into out_dir
    with config_id prefix. Return the best eval metrics from the eval CSV.
    """
    cmd = config_to_cmd(cfg)
    cmd_str = " ".join(str(c) for c in cmd)

    # Audit log
    with open(cmd_log, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] [{config_id}] {cmd_str}\n")

    print(f"\n{'─'*72}")
    print(f"  Config:  {config_id}")
    print(f"  Agent:   {cfg['agent']}  |  Grid: {Path(cfg['grid']).stem}")
    print(f"  Sweep:   {cfg['swept_param']} = {cfg['swept_value']}")
    print(f"  Seed:    {cfg['random_seed']}")
    print(f"  CMD:     {cmd_str}")
    print(f"{'─'*72}")

    if dry_run:
        print("  [DRY RUN — not executing]")
        return {}

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    before = set(results_dir.glob("*.csv"))

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"warning: train.py exited with code {result.returncode}")

    # Identify CSVs created by this run and move them to out_dir
    after = set(results_dir.glob("*.csv"))
    new_files = after - before

    eval_dest  = out_dir / f"{config_id}_eval.csv"
    train_dest = out_dir / f"{config_id}_train.csv"

    for fp in new_files:
        if fp.name.endswith("_eval.csv"):
            fp.rename(eval_dest)
        elif fp.name.endswith("_train.csv"):
            fp.rename(train_dest)

    return extract_best_eval(eval_dest)


SUMMARY_FIELDS = [
    "config_id", "agent", "grid", "swept_param", "swept_value", "random_seed",
    # Environment / training
    "sigma", "iter", "max_steps_per_episode", "eval_every",
    "converge_patience", "converge_threshold",
    # Shared
    "lr", "gamma",
    # DQN
    "batch_size", "epsilon_decay", "min_buffer", "max_buffer", "target_update",
    # PPO
    "clip_eps", "rollout_size", "ppo_epochs", "gae_lambda",
    "entropy_coef", "value_coef",
    # Best eval results (primary: eval_success_rate)
    "best_eval_episode", "best_eval_steps",
    "eval_success_rate", "eval_mean_reward", "eval_std_reward", "eval_mean_length",
]


def init_summary(path: Path):
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()


def append_summary(path: Path, cfg: dict, config_id: str, best: dict):
    row = {
        "config_id":          config_id,
        "agent":              cfg["agent"],
        "grid":               Path(cfg["grid"]).stem,
        "swept_param":        cfg["swept_param"],
        "swept_value":        cfg["swept_value"],
        "random_seed":        cfg["random_seed"],
        "sigma":              cfg["sigma"],
        "iter":               cfg["iter"],
        "max_steps_per_episode": cfg["max_steps_per_episode"],
        "eval_every":         cfg["eval_every"],
        "converge_patience":  cfg["converge_patience"],
        "converge_threshold": cfg["converge_threshold"],
        "lr":                 cfg["lr"],
        "gamma":              cfg["gamma"],
        "batch_size":         cfg["batch_size"],
        "epsilon_decay":      cfg["epsilon_decay"],
        "min_buffer":         cfg["min_buffer"],
        "max_buffer":         cfg["max_buffer"],
        "target_update":      cfg["target_update"],
        "clip_eps":           cfg["clip_eps"],
        "rollout_size":       cfg["rollout_size"],
        "ppo_epochs":         cfg["ppo_epochs"],
        "gae_lambda":         cfg["gae_lambda"],
        "entropy_coef":       cfg["entropy_coef"],
        "value_coef":         cfg["value_coef"],
        "best_eval_episode":  best.get("episode", ""),
        "best_eval_steps":    best.get("steps_so_far", ""),
        "eval_success_rate":  best.get("eval_success_rate", ""),
        "eval_mean_reward":   best.get("eval_mean_reward", ""),
        "eval_std_reward":    best.get("eval_std_reward", ""),
        "eval_mean_length":   best.get("eval_mean_length", ""),
    }
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow(row)


def finalize(results_dir: Path, dry_run: bool):
    """
    Read sweep_summary.csv. For each (agent, grid) group, pick top 2
    configs ranked by eval_success_rate DESC.
    Re-run each winning config with all 3 seeds and save to final_runs.

    Returns 2-best-models-per-agent-per-grid table.
    """
    summary_path = results_dir / "sweep_summary.csv"
    configs_path = results_dir / "configs.json"

    if not summary_path.exists():
        print(f"[ERROR] {summary_path} not found. Run the sweep first.")
        sys.exit(1)
    if not configs_path.exists():
        print(f"[ERROR] {configs_path} not found. Run the sweep first.")
        sys.exit(1)

    # Load sweep results
    rows = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["eval_success_rate"] = float(row["eval_success_rate"] or -1)
                row["eval_mean_reward"]  = float(row["eval_mean_reward"] or -9999)
            except ValueError:
                row["eval_success_rate"] = -1
                row["eval_mean_reward"]  = -9999
            rows.append(row)

    with open(configs_path) as f:
        all_configs: dict = json.load(f)

    final_dir = results_dir / "final_runs"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_summary_path = final_dir / "final_summary.csv"
    final_cmd_log      = final_dir / "commands_run.txt"

    final_fields = SUMMARY_FIELDS + ["rank"]
    with open(final_summary_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=final_fields).writeheader()

    print(f"\n{'='*72}")
    print(f"Top-2 configs x {len(FINAL_SEEDS)} seeds "
          f"per (agent, grid)")
    print(f"Seeds: {FINAL_SEEDS}")
    print(f"Primary metric: eval_success_rate")
    print(f"{'='*72}")

    # Group by (agent, grid)
    groups: dict = defaultdict(list)
    for row in rows:
        groups[(row["agent"], row["grid"])].append(row)

    for (agent, grid), group_rows in sorted(groups.items()):
        sorted_rows = sorted(
            group_rows,
            key=lambda r: (r["eval_success_rate"], r["eval_mean_reward"]),
            reverse=True,
        )
        top2 = sorted_rows[:2]

        print(f"\n[{agent.upper()} | {grid}] Top-2:")
        for rank, row in enumerate(top2, 1):
            print(f"  #{rank}: {row['config_id']} | "
                  f"success={row['eval_success_rate']:.3f} | "
                  f"reward={row['eval_mean_reward']:.2f} | "
                  f"swept: {row['swept_param']}={row['swept_value']}")

        for rank, row in enumerate(top2, 1):
            cfg_id = row["config_id"]
            if cfg_id not in all_configs:
                print(f"  [WARNING] {cfg_id} not in configs.json, skipping")
                continue
            base_cfg = all_configs[cfg_id]

            for seed in FINAL_SEEDS:
                run_cfg = {**base_cfg, "random_seed": seed}
                final_id = f"final_{agent}_{grid}_rank{rank}_seed{seed}"
                best = run_config(run_cfg, final_id, final_dir,
                                  final_cmd_log, dry_run=dry_run)

                out_row = {
                    "config_id":          final_id,
                    "agent":              run_cfg["agent"],
                    "grid":               grid,
                    "swept_param":        run_cfg.get("swept_param", ""),
                    "swept_value":        run_cfg.get("swept_value", ""),
                    "random_seed":        seed,
                    "sigma":              run_cfg["sigma"],
                    "iter":               run_cfg["iter"],
                    "max_steps_per_episode": run_cfg["max_steps_per_episode"],
                    "eval_every":         run_cfg["eval_every"],
                    "converge_patience":  run_cfg["converge_patience"],
                    "converge_threshold": run_cfg["converge_threshold"],
                    "lr":                 run_cfg["lr"],
                    "gamma":              run_cfg["gamma"],
                    "batch_size":         run_cfg["batch_size"],
                    "epsilon_decay":      run_cfg["epsilon_decay"],
                    "min_buffer":         run_cfg["min_buffer"],
                    "max_buffer":         run_cfg["max_buffer"],
                    "target_update":      run_cfg["target_update"],
                    "clip_eps":           run_cfg["clip_eps"],
                    "rollout_size":       run_cfg["rollout_size"],
                    "ppo_epochs":         run_cfg["ppo_epochs"],
                    "gae_lambda":         run_cfg["gae_lambda"],
                    "entropy_coef":       run_cfg["entropy_coef"],
                    "value_coef":         run_cfg["value_coef"],
                    "best_eval_episode":  best.get("episode", ""),
                    "best_eval_steps":    best.get("steps_so_far", ""),
                    "eval_success_rate":  best.get("eval_success_rate", ""),
                    "eval_mean_reward":   best.get("eval_mean_reward", ""),
                    "eval_std_reward":    best.get("eval_std_reward", ""),
                    "eval_mean_length":   best.get("eval_mean_length", ""),
                    "rank":               rank,
                }
                with open(final_summary_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=final_fields).writerow(out_row)

    print(f"\nFinal results: {final_summary_path}")
    print(f"\nTo compute mean ± std per (agent, grid, rank) across seeds:")
    print(f"  python3 -c \"")
    print(f"    import pandas as pd")
    print(f"    df = pd.read_csv('{final_summary_path}')")
    print(f"    g = df.groupby(['agent','grid','rank'])['eval_success_rate']")
    print(f"    print(g.agg(['mean','std']))\"")


def parse_args():
    p = argparse.ArgumentParser(
        description="OAT hyperparameter sweep for DQN and PPO")
    p.add_argument("--agent", choices=["dqn", "ppo", "both"], default="both",
                   help="Which agent(s) to sweep.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--finalize", action="store_true",
                   help="Skip sweep; run top-2 configs with all seeds.")
    p.add_argument("--results_dir", type=Path, default=Path("sweep_results"),
                   help="Output directory. Default: sweep_results/")
    p.add_argument("--resume", action="store_true",
                   help="Skip config_ids already in sweep_summary.csv.")
    return p.parse_args()


def make_config_id(cfg: dict, existing: dict) -> str:
    """Build a deterministic, collision-safe config ID."""
    grid_stem  = Path(cfg["grid"]).stem
    val_str    = str(cfg["swept_value"]).replace(".", "p").replace("-", "m")
    base_id    = f"{cfg['agent']}_{grid_stem}_{cfg['swept_param']}_{val_str}"
    config_id  = base_id
    collision  = 0
    while config_id in existing and existing[config_id] != cfg:
        collision += 1
        config_id = f"{base_id}_{collision}"
    return config_id


def main():
    args = parse_args()
    out_dir = args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        finalize(out_dir, dry_run=args.dry_run)
        return

    agents  = ["dqn", "ppo"] if args.agent == "both" else [args.agent]
    configs = generate_configs(agents)

    summary_path = out_dir / "sweep_summary.csv"
    cmd_log_path = out_dir / "commands_run.txt"
    configs_path = out_dir / "configs.json"

    # Load already-completed IDs if resuming
    done_ids: set = set()
    if args.resume and summary_path.exists():
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["config_id"])
        print(f"[RESUME] Skipping {len(done_ids)} already-completed configs.")
    else:
        init_summary(summary_path)

    # Load or init config store
    existing_cfgs: dict = {}
    if configs_path.exists():
        with open(configs_path) as f:
            existing_cfgs = json.load(f)

    total = len(configs)
    print(f"\nSweep: {total} configs | Agents: {agents} | Grids: {len(GRIDS)}")
    print(f"Output: {out_dir}/")
    print(f"\nEarly stopping: enabled (patience={DEFAULTS['converge_patience']}, "
          f"threshold={DEFAULTS['converge_threshold']})")
    print(f"Seeds per config: 1 (seed=0). Final runs: {len(FINAL_SEEDS)} seeds.")
    print(f"\nPackage location is not in the state vector.")
    print(f"State = [gps_x, gps_y] — 2 floats (normalized row, col).")
    print(f"The agent has no direct access to package coordinates.\n")

    for i, cfg in enumerate(configs, 1):
        config_id = make_config_id(cfg, existing_cfgs)

        print(f"[{i:>4}/{total}] {config_id}", end="", flush=True)

        if config_id in done_ids:
            print("  [SKIP — already done]")
            continue
        print()

        # Save to config store before running (so finalize can find it later)
        existing_cfgs[config_id] = cfg
        with open(configs_path, "w") as f:
            json.dump(existing_cfgs, f, indent=2)

        best = run_config(cfg, config_id, out_dir, cmd_log_path,
                          dry_run=args.dry_run)
        append_summary(summary_path, cfg, config_id, best)

    print(f"\n{'='*72}")
    print(f"Sweep done. {total - len(done_ids)} configs run.")
    print(f"Results: {summary_path}")
    print(f"\nNext step:")
    print(f"  python3 sweep.py --finalize --results_dir {out_dir}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
