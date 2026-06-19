import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_step_col(df):
    for col in ["steps_so_far", "step", "steps", "total_steps", "training_steps", "iteration", "iter"]:
        if col in df.columns:
            return col
    raise ValueError(f"No step column found. Columns: {list(df.columns)}")


def find_metric_col(df, metric):
    candidates = {
        "success": ["eval_success_rate", "success_rate", "success"],
        "reward": ["eval_mean_reward", "mean_reward", "reward"],
        "length": ["eval_mean_length", "mean_length", "episode_length"],
    }

    for col in candidates[metric]:
        if col in df.columns:
            return col

    raise ValueError(f"No {metric} column found. Columns: {list(df.columns)}")


def find_eval_file(results_dir, config_id):
    candidates = list(results_dir.rglob(f"{config_id}_eval.csv"))
    if candidates:
        return candidates[0]

    candidates = list(results_dir.rglob(f"*{config_id}*eval.csv"))
    if candidates:
        return candidates[0]

    return None


def select_best_configs(sweep_df):
    best = (
        sweep_df.sort_values(
            by=[
                "agent",
                "grid",
                "eval_success_rate",
                "eval_mean_reward",
                "eval_mean_length",
                "best_eval_steps",
            ],
            ascending=[True, True, False, False, True, True],
        )
        .groupby(["agent", "grid"])
        .head(1)
        .reset_index(drop=True)
    )

    return best


def load_curves(results_dir, best_df):
    curves = []

    for _, row in best_df.iterrows():
        config_id = row["config_id"]
        agent = row["agent"]
        grid = row["grid"]

        eval_path = find_eval_file(results_dir, config_id)

        if eval_path is None:
            print(f"Missing eval file for {config_id}")
            continue

        df = pd.read_csv(eval_path)
        step_col = find_step_col(df)

        success_col = find_metric_col(df, "success")
        reward_col = find_metric_col(df, "reward")
        length_col = find_metric_col(df, "length")

        out = pd.DataFrame({
            "agent": agent,
            "grid": grid,
            "config_id": config_id,
            "step": df[step_col],
            "success": df[success_col],
            "reward": df[reward_col],
            "length": df[length_col],
        })

        curves.append(out)

    if not curves:
        raise RuntimeError("No convergence curves could be loaded.")

    return pd.concat(curves, ignore_index=True)


def plot_metric(curves, metric, ylabel, output_path):
    grids = ["fishbone", "flying_v", "half_aisles"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=(metric == "success"))

    for ax, grid in zip(axes, grids):
        sub = curves[curves["grid"] == grid]

        for agent, group in sub.groupby("agent"):
            group = group.sort_values("step")
            ax.plot(
                group["step"],
                group[metric],
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=agent.upper(),
            )

        ax.set_title(grid.replace("_", "-"))
        ax.set_xlabel("Training steps")
        ax.grid(alpha=0.3)

        if metric == "success":
            ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="final_exp2")
    parser.add_argument("--output_dir", default="report_outputs/convergence")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = results_dir / "sweep_summary.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(f"Missing {sweep_path}")

    sweep_df = pd.read_csv(sweep_path)
    best_df = select_best_configs(sweep_df)
    best_df.to_csv(output_dir / "best_configs_used_for_convergence.csv", index=False)

    curves = load_curves(results_dir, best_df)
    curves.to_csv(output_dir / "convergence_curves_data.csv", index=False)

    plot_metric(
        curves,
        "success",
        "Evaluation success rate",
        output_dir / "convergence_success_rate.png",
    )

    plot_metric(
        curves,
        "reward",
        "Evaluation mean reward",
        output_dir / "convergence_mean_reward.png",
    )

    plot_metric(
        curves,
        "length",
        "Evaluation mean episode length",
        output_dir / "convergence_episode_length.png",
    )

    print("Convergence plots created:")
    print(output_dir / "convergence_success_rate.png")
    print(output_dir / "convergence_mean_reward.png")
    print(output_dir / "convergence_episode_length.png")


if __name__ == "__main__":
    main()