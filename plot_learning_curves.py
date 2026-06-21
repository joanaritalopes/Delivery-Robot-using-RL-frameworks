

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


AGENT_STYLE = {
    "dqn": dict(color="#E76F51", label="DQN",  lw=2.2, zorder=3),
    "ppo": dict(color="#264653", label="PPO",  lw=2.2, zorder=3),
}
SHADE_ALPHA = 0.18

GRID_TITLES = {
    "fishbone":    "Fishbone",
    "flying_v":    "Flying-V",
    "half_aisles": "Half-Aisles",
}

METRIC_CONFIG = {
    "eval_mean_reward":  dict(ylabel="Mean Episode Reward",    title="Mean Reward"),
    "eval_success_rate": dict(ylabel="Success Rate",           title="Success Rate",
                              ylim=(0.0, 1.05), pct=True),
    "eval_mean_length":  dict(ylabel="Mean Episode Length (steps)", title="Episode Length"),
}



def parse_args():
    p = argparse.ArgumentParser(description="Plot DQN vs PPO learning curves.")
    p.add_argument("--csv",    type=Path, default=None,
                   help="Path to all_eval_curves.csv (auto-detected if omitted).")
    p.add_argument("--out",    type=Path, default=Path("learning_curve_plots"),
                   help="Output directory for PNG files (default: learning_curve_plots/).")
    p.add_argument("--smooth", type=int,  default=3,
                   help="Rolling-window size in evaluation checkpoints (default: 3).")
    p.add_argument("--dpi",    type=int,  default=150,
                   help="PNG DPI (default: 150).")
    return p.parse_args()



SEARCH_PATHS = [
    Path("report_outputs/tables/all_eval_curves.csv"),
    Path("Delivery-Robot-using-RL-frameworks/report_outputs/tables/all_eval_curves.csv"),
    Path("../report_outputs/tables/all_eval_curves.csv"),
]

def find_csv(hint: Path | None) -> Path:
    if hint is not None:
        if hint.exists():
            return hint
        sys.exit(f"   CSV not found: {hint}")
    for candidate in SEARCH_PATHS:
        if candidate.exists():
            print(f"   Found data at: {candidate}")
            return candidate
    sys.exit(
        "   Could not find all_eval_curves.csv.\n"
        "        Run analysis_report.py first, or pass --csv <path>."
    )


def load_and_aggregate(csv_path: Path, smooth: int) -> dict:
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = {"agent", "grid", "step_for_plot", "source_file",
                "eval_mean_reward", "eval_success_rate", "eval_mean_length"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"   CSV is missing columns: {missing}")

    result = {}
    for grid in df["grid"].unique():
        result[grid] = {}
        g = df[df["grid"] == grid]

        for agent in ("dqn", "ppo"):
            a = g[g["agent"] == agent]
            if a.empty:
                continue

            
            step_min = a["step_for_plot"].min()
            step_max = a["step_for_plot"].max()
            n_bins   = 50
            bins     = np.linspace(step_min, step_max, n_bins + 1)
            bin_mids = 0.5 * (bins[:-1] + bins[1:])

            runs = a["source_file"].unique()

            series_dict = {m: [] for m in METRIC_CONFIG}

            for run in runs:
                r = a[a["source_file"] == run].sort_values("step_for_plot")
                run_steps  = r["step_for_plot"].values

                for metric in METRIC_CONFIG:
                    run_vals = r[metric].values
                    
                    binned = np.full(n_bins, np.nan)
                    for i in range(n_bins):
                        mask = (run_steps >= bins[i]) & (run_steps < bins[i + 1])
                        if mask.any():
                            binned[i] = run_vals[mask].mean()
                    series_dict[metric].append(binned)

            result[grid][agent] = {}
            for metric in METRIC_CONFIG:
                mat = np.array(series_dict[metric])   
                
                mean = np.nanmean(mat, axis=0)
                std  = np.nanstd(mat,  axis=0)

                
                if smooth > 1:
                    def _roll(arr):
                        s = pd.Series(arr)
                        return s.rolling(smooth, min_periods=1, center=True).mean().values
                    mean = _roll(mean)
                    std  = _roll(std)

                
                valid = ~np.isnan(np.nanmean(mat, axis=0))
                result[grid][agent][metric] = (
                    bin_mids[valid],
                    mean[valid],
                    std[valid],
                )

    return result



def plot_grid(grid: str, data: dict, out_dir: Path, dpi: int):
    """One figure per grid, 3 subplots side by side."""
    metrics  = list(METRIC_CONFIG.keys())
    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5),
                             constrained_layout=True)

    grid_title = GRID_TITLES.get(grid, grid.replace("_", " ").title())
    fig.suptitle(f"DQN vs PPO — {grid_title} grid",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax, metric in zip(axes, metrics):
        cfg  = METRIC_CONFIG[metric]
        any_data = False

        for agent, style in AGENT_STYLE.items():
            if agent not in data.get(grid, {}):
                continue
            if metric not in data[grid][agent]:
                continue

            steps, mean, std = data[grid][agent][metric]
            if len(steps) == 0:
                continue

            any_data = True
            ax.plot(steps, mean,
                    color=style["color"],
                    label=style["label"],
                    lw=style["lw"],
                    zorder=style["zorder"])
            ax.fill_between(steps,
                            mean - std, mean + std,
                            color=style["color"],
                            alpha=SHADE_ALPHA,
                            zorder=style["zorder"] - 1)

        
        ax.set_xlabel("Environment Steps", fontsize=11)
        ax.set_ylabel(cfg["ylabel"], fontsize=11)
        ax.set_title(cfg["title"], fontsize=12, fontweight="semibold", pad=8)

        if "ylim" in cfg:
            ax.set_ylim(cfg["ylim"])
        if cfg.get("pct"):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}")
        )

        if any_data:
            ax.legend(framealpha=0.85, fontsize=10, loc="best")

    out_path = out_dir / f"learning_curves_{grid}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_combined(data: dict, out_dir: Path, dpi: int):
    
    grids   = [g for g in ("fishbone", "flying_v", "half_aisles") if g in data]
    metrics = list(METRIC_CONFIG.keys())
    n_rows  = len(grids)
    n_cols  = len(metrics)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows),
                             constrained_layout=True)

    fig.suptitle("DQN vs PPO — All Grids", fontsize=16,
                 fontweight="bold", y=1.01)

    if n_rows == 1:
        axes = axes[np.newaxis, :]  

    for r, grid in enumerate(grids):
        grid_title = GRID_TITLES.get(grid, grid.replace("_", " ").title())
        for c, metric in enumerate(metrics):
            ax  = axes[r, c]
            cfg = METRIC_CONFIG[metric]

            for agent, style in AGENT_STYLE.items():
                if agent not in data.get(grid, {}):
                    continue
                if metric not in data[grid][agent]:
                    continue

                steps, mean, std = data[grid][agent][metric]
                if len(steps) == 0:
                    continue

                ax.plot(steps, mean,
                        color=style["color"],
                        label=style["label"],
                        lw=2.0, zorder=3)
                ax.fill_between(steps,
                                mean - std, mean + std,
                                color=style["color"],
                                alpha=SHADE_ALPHA, zorder=2)

            if r == 0:
                ax.set_title(cfg["title"], fontsize=12,
                             fontweight="semibold", pad=6)
            if c == 0:
                ax.set_ylabel(f"{grid_title}\n{cfg['ylabel']}", fontsize=10)
            else:
                ax.set_ylabel(cfg["ylabel"], fontsize=10)

            ax.set_xlabel("Steps", fontsize=10)

            if "ylim" in cfg:
                ax.set_ylim(cfg["ylim"])
            if cfg.get("pct"):
                ax.yaxis.set_major_formatter(
                    mticker.PercentFormatter(xmax=1.0))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"
                )
            )

            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=9, loc="best", framealpha=0.85)

    out_path = out_dir / "learning_curves_all_grids.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")



def main():
    args    = parse_args()
    csv_fp  = find_csv(args.csv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Smooth window : {args.smooth} checkpoints")
    print(f"   Output dir    : {out_dir.resolve()}")

    print("   Loading & aggregating data …")
    data = load_and_aggregate(csv_fp, smooth=args.smooth)

    grids_found = sorted(data.keys())
    if not grids_found:
        sys.exit("   No valid grid data found in the CSV.")
    print(f"   Grids found   : {grids_found}")

    
    for grid in grids_found:
        plot_grid(grid, data, out_dir, args.dpi)

    
    plot_combined(data, out_dir, args.dpi)

    print(" All plots saved.")


if __name__ == "__main__":
    main()
