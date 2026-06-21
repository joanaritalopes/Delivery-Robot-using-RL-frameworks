import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_dirs(output_dir):
    plots = output_dir / "plots"
    tables = output_dir / "tables"
    notes = output_dir / "notes"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)
    return plots, tables, notes


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def add_label(df):
    df = df.copy()
    df["label"] = df["agent"].str.upper() + " / " + df["grid"]
    return df


def select_best_sweep(sweep_df):
    return (
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


def final_seed_summary(final_df):
    return (
        final_df.groupby(["agent", "grid"])
        .agg(
            success_mean=("eval_success_rate", "mean"),
            success_std=("eval_success_rate", "std"),
            reward_mean=("eval_mean_reward", "mean"),
            reward_std=("eval_mean_reward", "std"),
            length_mean=("eval_mean_length", "mean"),
            length_std=("eval_mean_length", "std"),
            steps_mean=("best_eval_steps", "mean"),
            steps_std=("best_eval_steps", "std"),
        )
        .reset_index()
        .fillna(0)
    )


def bar_plot(df, x, y, title, ylabel, output_path, yerr=None, ylim=None, colors=None):
    plt.figure(figsize=(9, 4.5))
    plt.bar(df[x], df[y], yerr=yerr, capsize=4 if yerr is not None else 0, color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def grouped_bar(summary, metric, std_metric, title, ylabel, output_path, ylim=None, bounded=False):
    df = add_label(summary)
    df = df.sort_values(["grid", "agent"]).reset_index(drop=True)

    colors = df["agent"].map({"dqn": "#1f77b4", "ppo": "#ff7f0e"}).values

    if std_metric in df.columns:
        if bounded:
            lower = np.minimum(df[std_metric].values, df[metric].values)
            upper = np.minimum(df[std_metric].values, 1 - df[metric].values)
            yerr = np.vstack([lower, upper])
        else:
            yerr = df[std_metric].values
    else:
        yerr = None

    bar_plot(df, "label", metric, title, ylabel, output_path, yerr=yerr, ylim=ylim, colors=colors)


def save_latex_table(df, path, caption, label):
    latex = df.to_latex(index=False, escape=False, caption=caption, label=label)
    path.write_text(latex)


def plot_final_results(final_summary, plots_dir):
    grouped_bar(
        final_summary,
        "success_mean",
        "success_std",
        "Final success rate across seeds",
        "Success rate",
        plots_dir / "final_success_rate_mean_std.png",
        ylim=(0, 1.05),
        bounded=True,
    )

    grouped_bar(
        final_summary,
        "reward_mean",
        "reward_std",
        "Final mean reward across seeds",
        "Mean reward",
        plots_dir / "final_mean_reward_mean_std.png",
    )

    grouped_bar(
        final_summary,
        "length_mean",
        "length_std",
        "Final episode length across seeds",
        "Mean episode length",
        plots_dir / "final_episode_length_mean_std.png",
    )

    grouped_bar(
        final_summary,
        "steps_mean",
        "steps_std",
        "Final convergence speed across seeds",
        "Best evaluation step",
        plots_dir / "final_convergence_speed_mean_std.png",
    )


def plot_best_sweep(best_sweep, plots_dir):
    best_sweep = add_label(best_sweep)

    bar_plot(
        best_sweep,
        "label",
        "eval_success_rate",
        "Best success rate by agent and grid",
        "Success rate",
        plots_dir / "best_success_rate.png",
        ylim=(0, 1.05),
    )

    bar_plot(
        best_sweep,
        "label",
        "eval_mean_reward",
        "Best mean reward by agent and grid",
        "Mean reward",
        plots_dir / "best_mean_reward.png",
    )

    bar_plot(
        best_sweep,
        "label",
        "eval_mean_length",
        "Best episode length by agent and grid",
        "Mean episode length",
        plots_dir / "best_episode_length.png",
    )

    bar_plot(
        best_sweep,
        "label",
        "best_eval_steps",
        "Best convergence speed by agent and grid",
        "Best evaluation step",
        plots_dir / "best_convergence_speed.png",
    )
def plot_combined_sensitivity_by_agent(sweep_df, plots_dir):

    for agent in sweep_df["agent"].unique():

        agent_df = sweep_df[sweep_df["agent"] == agent].copy()
        agent_df["param_label"] = (
                agent_df["swept_param"].astype(str)
                + "="
                + agent_df["swept_value"].astype(str)
        )

        param_labels = agent_df["param_label"].drop_duplicates().tolist()
        grids = agent_df["grid"].unique()

        n_groups = len(param_labels)
        n_bars = len(grids)
        bar_width = 0.5 / n_bars
        x = np.arange(n_groups)

        plt.figure(figsize=(max(12, n_groups * 0.6), 5))

        for i, grid in enumerate(grids):

            grid_df = agent_df[agent_df["grid"] == grid].copy()
            grid_df = grid_df.set_index("param_label").reindex(param_labels)

            # So you can actually see the 0 values
            plot_values  = grid_df["eval_success_rate"].clip(lower=0.01)

            offset = (i - (n_bars - 1) / 2) * bar_width
            plt.bar(
                x + offset,
                plot_values,
                width=bar_width,
                label=grid.replace("_", "-")
            )

        plt.xticks(
            x,
            param_labels,
            rotation=45,
            ha="right"
        )

        plt.ylabel("Evaluation Success Rate")
        plt.xlabel("Hyperparameter Variation")
        plt.title(
            f"{agent.upper()} Hyperparameter Sensitivity Across Warehouse Layouts"
        )

        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3, axis="y")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            plots_dir / f"combined_sensitivity_{agent}.png",
            dpi=250
        )

        plt.close()

def plot_hyperparameter_impact(sweep_df, plots_dir):
    """
    Creates compact ablation-style plots showing which swept parameter affected performance most.
    Impact is measured as the range of success rates caused by changing each hyperparameter.
    """

    rows = []

    for (agent, grid, param), group in sweep_df.groupby(["agent", "grid", "swept_param"]):
        if param == "baseline":
            continue

        success_range = group["eval_success_rate"].max() - group["eval_success_rate"].min()
        reward_range = group["eval_mean_reward"].max() - group["eval_mean_reward"].min()
        length_range = group["eval_mean_length"].max() - group["eval_mean_length"].min()

        rows.append(
            {
                "agent": agent,
                "grid": grid,
                "swept_param": param,
                "success_impact": success_range,
                "reward_impact": reward_range,
                "length_impact": length_range,
            }
        )

    impact_df = pd.DataFrame(rows)

    if impact_df.empty:
        return impact_df

    impact_df.to_csv(plots_dir.parent / "tables" / "hyperparameter_impact.csv", index=False)

    for (agent, grid), group in impact_df.groupby(["agent", "grid"]):
        group = group.sort_values("success_impact", ascending=False)

        plt.figure(figsize=(7.5, 3.8))
        plt.bar(group["swept_param"], group["success_impact"])
        plt.title(f"Hyperparameter impact on success: {agent.upper()} / {grid}")
        plt.ylabel("Success-rate range")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"hyperparameter_impact_{agent}_{grid}.png", dpi=250)
        plt.close()

    return impact_df

def plot_success_rate_heatmap(final_summary_df, plots_dir):
    import numpy as np
    import matplotlib.pyplot as plt

    agents = ["dqn", "ppo"]
    grids = ["fishbone", "flying_v", "half_aisles"]

    heatmap = np.zeros((len(grids), len(agents)))

    for i, grid in enumerate(grids):
        for j, agent in enumerate(agents):
            row = final_summary_df[
                (final_summary_df["agent"] == agent)
                & (final_summary_df["grid"] == grid)
            ]
            heatmap[i, j] = row["success_mean"].values[0]

    plt.figure(figsize=(7, 4.5))
    im = plt.imshow(
    heatmap,
    cmap="Greens",
    vmin=0,
    vmax=1,
    aspect="auto"

)

    plt.colorbar(im)
    plt.xticks(range(len(agents)), ["DQN", "PPO"])
    plt.yticks(range(len(grids)), ["fishbone", "flying_v", "half_aisles"])

    for i in range(len(grids)):
        for j in range(len(agents)):
            plt.text(
                j,
                i,
                f"{heatmap[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    plt.title("Success rate heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "success_rate_heatmap.png", dpi=250)
    plt.close()

def plot_hyperparameter_sensitivity(sweep_df, plots_dir):
    """
    Creates one plot per agent-grid pair showing success rate for every tested configuration.
    This is useful for ablation/hyperparameter sensitivity discussion.
    """

    for (agent, grid), group in sweep_df.groupby(["agent", "grid"]):
        group = group.copy()
        group["param_label"] = group["swept_param"].astype(str) + "=" + group["swept_value"].astype(str)

        group = group.sort_values(
            by=["eval_success_rate", "eval_mean_reward", "eval_mean_length"],
            ascending=[False, False, True],
        )

        plt.figure(figsize=(9, 4.5))
        plt.bar(group["param_label"], group["eval_success_rate"])
        plt.title(f"Hyperparameter sensitivity: {agent.upper()} / {grid}")
        plt.ylabel("Evaluation success rate")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"sensitivity_{agent}_{grid}.png", dpi=250)
        plt.close()


def find_step_column(df):
    candidates = [
        "steps_so_far",
        "step",
        "steps",
        "total_steps",
        "training_steps",
        "iteration",
        "iter",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def parse_agent_grid_from_filename(path):
    name = path.stem

    if name.endswith("_eval"):
        name = name[:-5]
    elif name.endswith("_train"):
        name = name[:-6]

    parts = [p for p in name.split("_") if p != ""]

    known_agents = {
        "dqn": ("dqn", 1),
        "ppo": ("ppo", 1),
    }

    agent = "unknown"
    rest = parts

    for i, part in enumerate(parts):
        if part == "dueling" and i + 1 < len(parts) and parts[i + 1] == "dqn":
            agent = "dueling_dqn"
            rest = parts[i + 2:]
            break
        if part in known_agents:
            agent_name, skip = known_agents[part]
            agent = agent_name
            rest = parts[i + skip:]
            break

    known_grids = ["fishbone", "flying_v", "half_aisles"]

    grid = "unknown"
    for g in known_grids:
        g_parts = g.split("_")
        if rest[: len(g_parts)] == g_parts:
            grid = g
            break

    return agent, grid


def plot_convergence_curves(results_dir, plots_dir, smoothing=10):
    results_dir = Path("results")

    eval_files = list(results_dir.glob("*_eval.csv"))
    train_files = list(results_dir.glob("*_train.csv"))

    if not eval_files and not train_files:
        print("No *_eval.csv or *_train.csv files found for convergence plots.")
        return

    agent_colors = {"dqn": "#1f77b4", "ppo": "#ff7f0e"}

    def load_files(file_list, kind):
        rows = []
        for path in file_list:
            df = pd.read_csv(path)
            agent, grid = parse_agent_grid_from_filename(path)
            df = df.copy()
            df["agent"] = df.get("agent", agent)
            df["grid"] = df.get("grid", grid)
            df["source_file"] = path.name
            df["kind"] = kind
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    eval_df = load_files(eval_files, "eval")
    train_df = load_files(train_files, "train")

    if not eval_df.empty:
        eval_df.to_csv(plots_dir.parent / "tables" / "all_eval_curves.csv", index=False)
    if not train_df.empty:
        train_df.to_csv(plots_dir.parent / "tables" / "all_train_curves.csv", index=False)

    grids = set(eval_df["grid"].unique() if not eval_df.empty else []) | \
            set(train_df["grid"].unique() if not train_df.empty else [])

    metrics = [
        ("eval_success_rate", "success", "Success rate", "convergence_success", (-0.05, 1.05)),
        ("eval_mean_reward", "ep_reward", "Mean reward", "convergence_reward", None),
        ("eval_mean_length", "ep_length", "Mean episode length", "convergence_length", None),
    ]

    for grid in grids:
        if grid == "unknown":
            continue

        eval_g = eval_df[eval_df["grid"] == grid] if not eval_df.empty else pd.DataFrame()
        train_g = train_df[train_df["grid"] == grid] if not train_df.empty else pd.DataFrame()

        for eval_col, train_col, ylabel, fname, ylim in metrics:

            has_eval = not eval_g.empty and eval_col in eval_g.columns
            has_train = not train_g.empty and train_col in train_g.columns

            if not has_eval and not has_train:
                continue

            plt.figure(figsize=(8, 4))

            agents = set(eval_g["agent"].unique() if has_eval else []) | \
                     set(train_g["agent"].unique() if has_train else [])

            for agent in agents:
                color = agent_colors.get(agent, None)

                if has_train:
                    t = (
                        train_g[train_g["agent"] == agent]
                        .groupby("episode")[train_col]
                        .mean()
                        .reset_index()
                        .sort_values("episode")
                    )
                    if not t.empty:
                        t[train_col] = t[train_col].rolling(smoothing, min_periods=1).mean()
                        plt.plot(
                            t["episode"], t[train_col],
                            linestyle="-", linewidth=1, alpha=0.5,
                            color=color, label=f"{agent.upper()} (train)"
                        )

                if has_eval:
                    cols_to_agg = [eval_col]
                    if eval_col == "eval_mean_reward" and "eval_std_reward" in eval_g.columns:
                        cols_to_agg.append("eval_std_reward")

                    e = (
                        eval_g[eval_g["agent"] == agent]
                        .groupby("episode")[cols_to_agg]
                        .mean()
                        .reset_index()
                        .sort_values("episode")
                    )
                    if not e.empty:
                        e[eval_col] = e[eval_col].rolling(smoothing, min_periods=1).mean()

                        if "eval_std_reward" in e.columns:
                            e["eval_std_reward"] = e["eval_std_reward"].rolling(smoothing, min_periods=1).mean()
                            plt.fill_between(
                                e["episode"],
                                e[eval_col] - e["eval_std_reward"],
                                e[eval_col] + e["eval_std_reward"],
                                color=color, alpha=0.15
                            )

                        plt.plot(
                            e["episode"], e[eval_col],
                            linestyle="-", linewidth=1,
                            color=color, label=f"{agent.upper()} (eval)"
                        )

            plt.title(f"{ylabel} per episode on {grid}")
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{fname}_{grid}.png", dpi=250)
            plt.close()


def write_notes(best_sweep, final_summary, impact_df, notes_dir):
    lines = []

    lines.append("EMPIRICAL RESULTS SUMMARY\n")
    lines.append("=========================\n\n")

    lines.append("1. Best sweep configurations\n")
    for _, r in best_sweep.iterrows():
        lines.append(
            f"- {r['agent'].upper()} on {r['grid']}: "
            f"{r['config_id']} | "
            f"success={r['eval_success_rate']:.3f}, "
            f"reward={r['eval_mean_reward']:.2f}, "
            f"length={r['eval_mean_length']:.2f}, "
            f"best_steps={int(r['best_eval_steps'])}\n"
        )

    lines.append("\n2. Final multi-seed results\n")
    for _, r in final_summary.iterrows():
        lines.append(
            f"- {r['agent'].upper()} on {r['grid']}: "
            f"success={r['success_mean']:.3f} ± {r['success_std']:.3f}, "
            f"reward={r['reward_mean']:.2f} ± {r['reward_std']:.2f}, "
            f"length={r['length_mean']:.2f} ± {r['length_std']:.2f}, "
            f"steps={r['steps_mean']:.0f} ± {r['steps_std']:.0f}\n"
        )

    if impact_df is not None and not impact_df.empty:
        lines.append("\n3. Most influential hyperparameters by success-rate range\n")
        for (agent, grid), group in impact_df.groupby(["agent", "grid"]):
            best = group.sort_values("success_impact", ascending=False).iloc[0]
            lines.append(
                f"- {agent.upper()} on {grid}: {best['swept_param']} "
                f"(success impact={best['success_impact']:.3f})\n"
            )

    lines.append("\n4. Interpretation guide\n")
    lines.append("- Higher success rate is better.\n")
    lines.append("- Higher mean reward is better because rewards are mostly negative penalties.\n")
    lines.append("- Lower episode length indicates more efficient navigation.\n")
    lines.append("- Lower best evaluation steps indicates faster convergence.\n")
    lines.append("- Episode length 5000 indicates timeout and unsuccessful termination.\n")

    (notes_dir / "empirical_results_notes.txt").write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="final_exp")
    parser.add_argument("--output_dir", default="report_outputs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    plots_dir, tables_dir, notes_dir = make_dirs(output_dir)

    sweep_df = read_csv(results_dir / "sweep_summary.csv")
    final_df = read_csv(results_dir / "final_runs" / "final_summary.csv")

    best_sweep = select_best_sweep(sweep_df)
    best_sweep.to_csv(tables_dir / "best_sweep_configs.csv", index=False)

    final_summary = final_seed_summary(final_df)
    plot_success_rate_heatmap(final_summary, plots_dir)
    final_summary.to_csv(tables_dir / "final_mean_std.csv", index=False)

    save_latex_table(
        final_summary,
        tables_dir / "final_mean_std.tex",
        "Final multi-seed performance summary.",
        "tab:final_mean_std",
    )

    save_latex_table(
        best_sweep,
        tables_dir / "best_sweep_configs.tex",
        "Best hyperparameter configuration per agent and grid.",
        "tab:best_sweep_configs",
    )

    plot_best_sweep(best_sweep, plots_dir)
    plot_final_results(final_summary, plots_dir)
    

    plot_convergence_curves(results_dir, plots_dir)
    

    plot_hyperparameter_sensitivity(sweep_df, plots_dir)

    plot_combined_sensitivity_by_agent(
        sweep_df,
        plots_dir
    )

    impact_df = plot_hyperparameter_impact(
        sweep_df,
        plots_dir
    )

    write_notes(best_sweep, final_summary, impact_df, notes_dir)

    print("Analysis completed.")
    print(f"Plots saved to: {plots_dir}")
    print(f"Tables saved to: {tables_dir}")
    print(f"Notes saved to: {notes_dir}")


if __name__ == "__main__":
    main()