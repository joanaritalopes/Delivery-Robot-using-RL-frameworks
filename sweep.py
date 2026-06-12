"""
One-at-a-time (OAT) hyperparameter sweep for DQN and PPO.

Runs all configs with seed=0. After sweep, picks top-2 per (agent, grid)
by eval_success_rate and re-runs each with seeds 0, 1, 2.

At the end, saves the best CLI commands to final_runs/best_commands.json
for the final runs.

Usage:
  python3 sweep.py                 # full sweep
  python3 sweep.py --agent dqn
  python3 sweep.py --agent dqn --grids fishbone --results_dir sweep_test
  python3 sweep.py --resume        # if the sweep crashes or stop halfway through
  python3 sweep.py --finalize      # run only the finalize step (re-running top-2 configs with 3 seeds and saving results)
"""

import argparse
import csv
import json
import subprocess
import sys
import pandas as pd
from collections import defaultdict
from pathlib import Path


GRIDS = [
    "grid_configs/fishbone.npy",
    "grid_configs/flying_v.npy",
    "grid_configs/half_aisles.npy",
]

SWEEP_SEED  = 0
FINAL_SEEDS = [0, 1, 2]

DEFAULTS = {
    "sigma":              0.1,
    "iter":               500000,
    "eval_every":         20,
    "eval_episodes":      10,
    "converge_patience":  5,
    "converge_threshold": 0.95,
    "lr":                 0.001,
    "gamma":              0.99,
    "batch_size":         64,
    "epsilon_decay":      0.9995,
    "min_epsilon":        0.01,
    "min_buffer":         1000,
    "max_buffer":         50000,
    "target_update":      200,
    "clip_eps":           0.2,
    "rollout_size":       256,
    "ppo_epochs":         10,
    "gae_lambda":         0.95,
    "entropy_coef":       0.01,
    "value_coef":         0.5,
}

SHARED_SWEEP = {
    "lr":         [1e-4, 3e-4, 1e-3],
    "gamma":      [0.95, 0.999],
    "sigma":      [0.0, 0.1, 0.2],
    "iter":       [200000, 500000, 1000000],
}

DQN_SWEEP = {
    "epsilon_decay": [0.99997, 0.99998],
    "batch_size":    [64, 128],
    "target_update": [200, 1000],
    "max_buffer":    [50000, 100000],
}

PPO_SWEEP = {
    "clip_eps":     [0.1, 0.2],
    "rollout_size": [256, 512],
    "gae_lambda":   [0.9, 0.95],
}

SUMMARY_FIELDS = [
    "config_id", "agent", "grid", "swept_param", "swept_value",
    "eval_success_rate", "eval_mean_reward","eval_mean_length",
    "best_eval_steps",
]


def generate_configs(agents, grids=None):
    if grids is None:
        grids = GRIDS
    configs = []
    for agent in agents:
        param_sweeps = dict(SHARED_SWEEP)
        param_sweeps.update(DQN_SWEEP if agent == "dqn" else PPO_SWEEP)
        for grid in grids:
            configs.append({
                "agent": agent, "grid": grid,
                "swept_param": "baseline", "swept_value": "default",
                "random_seed": SWEEP_SEED, **DEFAULTS,
            })
            for param, values in param_sweeps.items():
                for value in values:
                    if value == DEFAULTS.get(param):
                        continue
                    configs.append({
                        "agent": agent, "grid": grid,
                        "swept_param": param, "swept_value": value,
                        "random_seed": SWEEP_SEED, **DEFAULTS, param: value,
                    })
    return configs


def config_to_cmd(cfg):
    return [
        sys.executable, "train.py", cfg["grid"],
        "--agent",              cfg["agent"],
        "--no_gui",
        "--iter",               str(cfg["iter"]),
        "--sigma",              str(cfg["sigma"]),
        "--random_seed",        str(cfg["random_seed"]),
        "--eval_every",         str(cfg["eval_every"]),
        "--eval_episodes",      str(cfg["eval_episodes"]),
        "--converge_patience",  str(cfg["converge_patience"]),
        "--converge_threshold", str(cfg["converge_threshold"]),
        "--lr",                 str(cfg["lr"]),
        "--gamma",              str(cfg["gamma"]),
        "--batch_size",         str(cfg["batch_size"]),
        "--epsilon_decay",      str(cfg["epsilon_decay"]),
        "--min_epsilon",        str(cfg["min_epsilon"]),
        "--min_buffer",         str(cfg["min_buffer"]),
        "--max_buffer",         str(cfg["max_buffer"]),
        "--target_update",      str(cfg["target_update"]),
        "--clip_eps",           str(cfg["clip_eps"]),
        "--rollout_size",       str(cfg["rollout_size"]),
        "--ppo_epochs",         str(cfg["ppo_epochs"]),
        "--gae_lambda",         str(cfg["gae_lambda"]),
        "--entropy_coef",       str(cfg["entropy_coef"]),
        "--value_coef",         str(cfg["value_coef"]),
    ]


def extract_best_eval(eval_csv):
    if not eval_csv.exists():
        return {}
    rows = []
    with open(eval_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "eval_success_rate": float(row["eval_success_rate"]),
                    "eval_mean_reward":  float(row["eval_mean_reward"]),
                    "eval_mean_length":  float(row["eval_mean_length"]),
                    "best_eval_steps":   int(row["steps_so_far"]),
                })
            except (ValueError, KeyError):
                continue
    if not rows:
        return {}
    return max(rows, key=lambda r: (r["eval_success_rate"],
            r["eval_mean_reward"],
            -r["eval_mean_length"],
            -r["best_eval_steps"],))


def run_config(cfg, config_id, out_dir):
    cmd = config_to_cmd(cfg)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    before = set(results_dir.glob("*.csv"))

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [WARNING] {config_id} exited with code {result.returncode}")

    after = set(results_dir.glob("*.csv"))
    eval_dest  = out_dir / f"{config_id}_eval.csv"
    train_dest = out_dir / f"{config_id}_train.csv"
    for fp in after - before:
        if fp.name.endswith("_eval.csv"):
            fp.rename(eval_dest)
        elif fp.name.endswith("_train.csv"):
            fp.rename(train_dest)

    return extract_best_eval(eval_dest)


def make_config_id(cfg, existing):
    grid_stem = Path(cfg["grid"]).stem
    val_str   = str(cfg["swept_value"]).replace(".", "p").replace("-", "m")
    base_id   = f"{cfg['agent']}_{grid_stem}_{cfg['swept_param']}_{val_str}"
    config_id = base_id
    collision = 0
    while config_id in existing and existing[config_id] != cfg:
        collision += 1
        config_id = f"{base_id}_{collision}"
    return config_id


def cfg_to_cli(cfg, seed):
    cmd = config_to_cmd({**cfg, "random_seed": seed})
    return " ".join(str(c) for c in cmd)


def finalize(results_dir):
    summary_path = results_dir / "sweep_summary.csv"
    configs_path = results_dir / "configs.json"

    if not summary_path.exists():
        print(f"[ERROR] {summary_path} not found. Run the sweep first.")
        sys.exit(1)
    if not configs_path.exists():
        print(f"[ERROR] {configs_path} not found. Run the sweep first.")
        sys.exit(1)

    rows = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["eval_success_rate"] = float(row["eval_success_rate"] or -1)
                row["eval_mean_reward"]  = float(row["eval_mean_reward"] or -9999)
                row["eval_mean_length"]  = float(row["eval_mean_length"] or 999999)
                row["best_eval_steps"]   = float(row["best_eval_steps"] or 999999)


            except ValueError:
                row["eval_success_rate"] = -1
                row["eval_mean_reward"]  = -9999
                row["eval_mean_length"]  = 999999
                row["best_eval_steps"]   = 999999
            rows.append(row)

    with open(configs_path) as f:
        all_configs = json.load(f)

    final_dir = results_dir / "final_runs"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_summary_path = final_dir / "final_summary.csv"

    final_fields = ["config_id", "agent", "grid", "best", "seed",
                    "eval_success_rate", "eval_mean_reward", "eval_mean_length", "best_eval_steps"]
    with open(final_summary_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=final_fields).writeheader()

    groups = defaultdict(list)
    for row in rows:
        groups[(row["agent"], row["grid"])].append(row)

    best_configs = {}

    for (agent, grid), group_rows in sorted(groups.items()):
        top2 = sorted(group_rows,
                      key=lambda r: (r["eval_success_rate"], r["eval_mean_reward"],-r["eval_mean_length"],
        -r["best_eval_steps"]),
                      reverse=True)[:2]
        best_configs[(agent, grid)] = top2

        for rank, row in enumerate(top2, 1):
            cfg_id = row["config_id"]
            if cfg_id not in all_configs:
                print(f"  [WARNING] {cfg_id} not in configs.json, skipping")
                continue
            base_cfg = all_configs[cfg_id]
            for seed in FINAL_SEEDS:
                run_cfg  = {**base_cfg, "random_seed": seed}
                final_id = f"final_{agent}_{grid}_best{rank}_seed{seed}"
                print(f"  Running {final_id}")
                metrics  = run_config(run_cfg, final_id, final_dir)
                with open(final_summary_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=final_fields).writerow({
                        "config_id":         final_id,
                        "agent":             agent,
                        "grid":              grid,
                        "best":              rank,
                        "seed":              seed,
                        "eval_success_rate": metrics.get("eval_success_rate", ""),
                        "eval_mean_reward":  metrics.get("eval_mean_reward", ""),
                        "eval_mean_length":  metrics.get("eval_mean_length", ""),  
                        "best_eval_steps":   metrics.get("best_eval_steps", "")
                    })

    df = pd.read_csv(final_summary_path)
    summary = (
        df.groupby(["agent", "grid", "best"])["eval_success_rate"]
        .agg(["mean", "std"])
        .round(4)
    )
    print(f"\neval_success_rate — mean ± std across {len(FINAL_SEEDS)} seeds:\n")
    print(summary.to_string())

    best_commands = {}
    for (agent, grid), top2 in best_configs.items():
        key = f"{agent}_{Path(grid).stem}"
        best_commands[key] = {}
        for rank, row in enumerate(top2, 1):
            cfg_id = row["config_id"]
            if cfg_id not in all_configs:
                continue
            best_commands[key][f"best{rank}"] = cfg_to_cli(all_configs[cfg_id], seed=0)

    with open(final_dir / "best_commands.json", "w") as f:
        json.dump(best_commands, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="OAT hyperparameter sweep for DQN and PPO.")
    p.add_argument("--agent", choices=["dqn", "ppo", "both"], default="both")
    p.add_argument("--grids", nargs="+",
                   choices=["fishbone", "flying_v", "half_aisles"],
                   default=["fishbone", "flying_v", "half_aisles"],
                   help="Which grid(s) to sweep. Default: all three.")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--results_dir", type=Path, default=Path("sweep_results"))
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        finalize(out_dir)
        return

    agents  = ["dqn", "ppo"] if args.agent == "both" else [args.agent]
    grids   = [f"grid_configs/{g}.npy" for g in args.grids]
    configs = generate_configs(agents, grids)

    summary_path = out_dir / "sweep_summary.csv"
    configs_path = out_dir / "configs.json"

    done_ids = set()
    if args.resume and summary_path.exists():
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["config_id"])
        print(f"Resuming — skipping {len(done_ids)} completed configs.")
    else:
        with open(summary_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()

    existing_cfgs = {}
    if configs_path.exists():
        with open(configs_path) as f:
            existing_cfgs = json.load(f)

    total = len(configs)
    print(f"Sweeping: {total} configs | agents: {agents} | grids: {args.grids}\n")

    for i, cfg in enumerate(configs, 1):
        config_id = make_config_id(cfg, existing_cfgs)

        if config_id in done_ids:
            continue

        print(f"[{i}/{total}] {cfg['agent']} | {Path(cfg['grid']).stem} | {cfg['swept_param']}={cfg['swept_value']}")

        existing_cfgs[config_id] = cfg
        with open(configs_path, "w") as f:
            json.dump(existing_cfgs, f, indent=2)

        metrics = run_config(cfg, config_id, out_dir)

        with open(summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow({
                "config_id":         config_id,
                "agent":             cfg["agent"],
                "grid":              Path(cfg["grid"]).stem,
                "swept_param":       cfg["swept_param"],
                "swept_value":       cfg["swept_value"],
                "eval_success_rate": metrics.get("eval_success_rate", ""),
                "eval_mean_reward":  metrics.get("eval_mean_reward", ""),
                "eval_mean_length":  metrics.get("eval_mean_length", ""),
                "best_eval_steps":   metrics.get("best_eval_steps", ""),
            })

    print("\nSweep done. Running finalize...\n")
    finalize(out_dir)


if __name__ == "__main__":
    main()
