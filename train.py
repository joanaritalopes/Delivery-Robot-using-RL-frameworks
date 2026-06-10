"""
Train RL Agent.

Usage example:
    # train on 1 grid
    python3 train.py grid_configs/example_grid.npy --agent dqn --no_gui --iter 100000
    # train on multiple grids
    python3 train.py grid_configs/fishbone.npy grid_configs/flying_v.npy grid_configs/warehouse_small.npy --agent dqn --no_gui
"""

import csv
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
from pathlib import Path
from tqdm import trange

from world import Environment
from agents.dqn_agent import DQN
from agents.ppo_agent import PPO


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo"],
                   help="Which agent to train: dqn, ppo, or random.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--iter", type=int, default=100000,
                   help="Total number of steps to train for (across all episodes).")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as row,col (e.g. 2,3).")
    p.add_argument("--eval_every", type=int, default=10,
                   help="Run evaluation every N episodes during training.")
    p.add_argument("--eval_episodes", type=int, default=30,
                   help="Number of episodes to run during each evaluation.")
    return p.parse_args()


def build_agent(agent_name: str):
    """Instantiate the chosen agent with default hyperparameters based on the CLI argument"""
    match agent_name:
        case "dqn":
            return DQN(input_dim=2, output_dim=4)
        case "ppo":
            return PPO(state_dim=2, action_dim=4)
        case _:
            raise ValueError(f"Unknown agent: {agent_name}")


def agent_update(agent, state, next_state, reward, action, terminated):
    """
    Handles the different update signatures of DQN and PPO.
 
    DQN: update(next_state, reward, action, terminated)
    PPO: update(state, reward, action)
    """
    if isinstance(agent, DQN):
        agent.update(next_state, reward, action, terminated)
    elif isinstance(agent, PPO):
        agent.update(state, reward, action)


# During evaluation, want the agent to act greedily, using its best
# known policy with no exploration noise. This is critical because
# evaluation measures true learned performance, not exploratory behavior.
# For DQN: we temporarily set epsilon to 0 (no random actions) and
#   restore it afterward so training exploration is not disrupted.
# For PPO: we set agent.eval = True so take_action() uses argmax over
#   the logits instead of sampling from the distribution.
# We always restore the original mode after evaluation finishes.

def set_eval_mode(agent, eval_mode: bool, saved_epsilon: float = 0.0):
    """
    Puts the agent in greedy / eval mode (no exploration).
 
    DQN: sets epsilon to 0 when eval_mode=True, restores saved_epsilon when eval_mode=False
    PPO: sets agent.eval flag so take_action uses argmax
    """
    if isinstance(agent, DQN):
        if eval_mode:
            current_ep = agent.ep
            agent.ep = 0.0
            return current_ep
        else:
            agent.ep = saved_epsilon
    elif isinstance(agent, PPO):
        agent.eval = eval_mode

    return 0.0


def init_csv(log_path: Path, fieldnames: list[str]):
    """Creates a CSV file with headers"""
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def append_csv(log_path: Path, row: dict):
    """Appends one row to an existing CSV"""
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)




# Evaluation is deliberately kept separate from training for two reasons:
# 1. The agent does NOT learn during evaluation — no update() is called.
#    This gives a clean measurement of what the policy has learned so far,
#    unaffected by exploration noise or ongoing weight updates.
# 2. We use the same grid and same random seed so results are comparable
#    across different evaluation checkpoints and across agents.

def evaluate(agent, grid_fp: Path, eval_episodes: int,
             sigma: float, start_pos, random_seed: int) -> dict:
    """
    Runs the agent greedily for eval_episodes episodes on the same grid.
    No learning happens here.
 
    Returns a dict with mean reward, std reward, and success rate.
    """
    # Create a separate environment instance for evaluation so it does
    # not interfere with the training environment's internal state.
    env = Environment(grid_fp, no_gui=True, sigma=sigma,
                      agent_start_pos=start_pos,
                      target_fps=-1, random_seed=random_seed)
 
    set_eval_mode(agent, True)
 
    episode_rewards = []
    episode_lengths = []
    successes = 0
 
    for _ in range(eval_episodes):
        state = env.reset()
        ep_reward = 0
        ep_steps = 0
        terminated = False
 
        # Run one full episode greedily — no update() calls
        while not terminated:
            action = agent.take_action(state)
            state, reward, terminated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1
 
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
 
        # Count episode as success only if agent reached terminal state
        if terminated:
            successes += 1
 
    # Restore training mode so the agent continues learning after eval
    set_eval_mode(agent, False)
 
    return {
        "eval_mean_reward":  round(float(np.mean(episode_rewards)), 4),
        "eval_std_reward":   round(float(np.std(episode_rewards)), 4),
        "eval_success_rate": round(successes / eval_episodes, 4),
        "eval_mean_length":  round(float(np.mean(episode_lengths)), 2),
    }



# The training loop follows a fixed total step budget (--iter)
# spread across as many episodes as needed. The environment resets
# automatically whenever a terminal state is reached or the step budget
# runs out mid-episode.
# This design is more flexible than a fixed number of episodes because
# episode length varies, especially early in training when the agent
# wanders randomly and rarely reaches the goal.
# The loop structure is:
#   outer while: runs until total step budget is exhausted
#   inner while: runs one episode until terminated or budget exhausted
# After each episode we log training metrics to CSV and print a summary.
# Every eval_every episodes we pause and run a full greedy evaluation,
# logging the results to a separate CSV for plotting.
 
def train(agent, grid_fp: Path, total_steps: int, no_gui: bool,
          sigma: float, fps: int, start_pos, random_seed: int,
          eval_every: int, eval_episodes: int,
          train_log: Path, eval_log: Path):
    """
    Main training loop.
 
    Runs until total_steps environment steps have been taken,
    across as many episodes as needed. Resets the environment
    automatically when a terminal state is reached.
 
    Logs per-episode metrics to train_log CSV.
    Logs evaluation metrics to eval_log CSV every eval_every episodes.
    """
    env = Environment(grid_fp, no_gui=no_gui, sigma=sigma,
                      target_fps=fps, agent_start_pos=start_pos,
                      random_seed=random_seed)
 
    # Initialize CSV files with appropriate column headers before training
    # so they exist even if training is interrupted early
    init_csv(train_log, ["episode", "steps_so_far", "ep_reward",
                         "ep_length", "success", "epsilon"])
    init_csv(eval_log,  ["episode", "steps_so_far", "eval_mean_reward",
                         "eval_std_reward", "eval_success_rate",
                         "eval_mean_length"])
 
    step_count = 0
    episode = 0
 
    print(f"\nTraining on {grid_fp.name} for {total_steps} total steps...\n")
 
    # Outer loop: keep running episodes until total step budget is used up
    while step_count < total_steps:
        state = env.reset()
        ep_reward = 0
        ep_steps = 0
        terminated = False
        episode += 1
 
        # Inner loop: run one episode step by step
        while not terminated:
            # Agent selects action based on current state
            action = agent.take_action(state)
 
            # Environment executes the action and returns the outcome.
            # info["actual_action"] may differ from action when sigma > 0
            # (stochastic environment) — we pass the actual action to
            # the agent so it learns from what really happened.
            next_state, reward, terminated, info = env.step(action)
 
            # Update the agent using the observed transition.
            # agent_update() handles the DQN vs PPO signature difference.
            agent_update(agent, state, next_state, reward,
                         info["actual_action"], terminated)
 
            # Advance to the next state
            state = next_state
            ep_reward += reward
            ep_steps += 1
            step_count += 1
 
            # Exit inner loop early if total budget is exhausted mid-episode
            if step_count >= total_steps:
                break
 
        # Epsilon is only meaningful for DQN (controls exploration rate).
        # For PPO we log "N/A" since exploration is handled via stochastic
        # policy sampling rather than an explicit epsilon parameter.
        epsilon = round(agent.ep, 5) if isinstance(agent, DQN) else "N/A"
 
        # Append this episode's metrics to the training CSV
        append_csv(train_log, {
            "episode":      episode,
            "steps_so_far": step_count,
            "ep_reward":    round(ep_reward, 4),
            "ep_length":    ep_steps,
            "success":      int(terminated),
            "epsilon":      epsilon,
        })
 
        print(f"Ep {episode:>5} | Steps {step_count:>7}/{total_steps} | "
              f"Reward {ep_reward:>8.2f} | Length {ep_steps:>5} | "
              f"Success {int(terminated)} | ε {epsilon}")
 
        # Run periodic evaluation every eval_every episodes.
        # This gives us a learning curve of true (greedy) performance
        # over training time, which is the main plot in the report.
        if episode % eval_every == 0:
            eval_metrics = evaluate(agent, grid_fp, eval_episodes,
                                    sigma, start_pos, random_seed)
            append_csv(eval_log, {
                "episode":      episode,
                "steps_so_far": step_count,
                **eval_metrics,
            })
            print(f"  → EVAL | Mean R {eval_metrics['eval_mean_reward']} | "
                  f"Std {eval_metrics['eval_std_reward']} | "
                  f"Success {eval_metrics['eval_success_rate']}")
 
    print(f"\nTraining complete. {step_count} steps over {episode} episodes.")
 
 

# After training is complete we run Environment.evaluate_agent() which
# produces a path visualization image saved to the results/ directory.
# This image shows the frequency with which the agent visited each cell,
# giving a qualitative view of the learned navigation behavior —
# useful for the situational analysis section of the report.
# No learning happens here — the agent acts purely from its learned policy.
 
def final_evaluate(agent, grid_fp: Path, total_steps: int,
                   sigma: float, start_pos, random_seed: int):
    """Runs Environment.evaluate_agent for a final path visualization."""
    print("\nRunning final evaluation (path visualization)...")
    Environment.evaluate_agent(
        grid_fp, agent, total_steps,
        sigma=sigma,
        agent_start_pos=start_pos,
        random_seed=random_seed,
        show_images=False,
    )


def main():
    args = parse_args()
 
    # Parse optional start position from "row,col" string to tuple
    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(",")
        start_pos = (int(parts[0]), int(parts[1]))
 
    # Timestamp ensures each run produces uniquely named log files,
    # so results from different seeds or hyperparameter configs are
    # never accidentally overwritten
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    log_dir = Path("results")
    log_dir.mkdir(parents=True, exist_ok=True)
 
    for grid_fp in args.GRID:
        print(f"\n{'='*60}")
        print(f"Grid: {grid_fp.name} | Agent: {args.agent} | "
              f"Steps: {args.iter} | Seed: {args.random_seed}")
        print(f"{'='*60}")
 
        agent = build_agent(args.agent)
 
        # Build log file paths: one training CSV and one eval CSV per run
        stem = f"{timestamp}_{args.agent}_{grid_fp.stem}"
        train_log = log_dir / f"{stem}_train.csv"
        eval_log  = log_dir / f"{stem}_eval.csv"
 
        train(
            agent=agent,
            grid_fp=grid_fp,
            total_steps=args.iter,
            no_gui=args.no_gui,
            sigma=args.sigma,
            fps=args.fps,
            start_pos=start_pos,
            random_seed=args.random_seed,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            train_log=train_log,
            eval_log=eval_log,
        )
 
        final_evaluate(agent, grid_fp, args.iter,
                       args.sigma, start_pos, args.random_seed)
 
        print(f"\nLogs saved to:")
        print(f"  {train_log}")
        print(f"  {eval_log}")
 
 
if __name__ == "__main__":
    main()





























# def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
#          sigma: float, random_seed: int, start_pos: tuple[int, int] | None):
#     """Main loop of the program."""

#     for grid in grid_paths:
        
#         # Set up the environment
#         env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
#                           agent_start_pos=start_pos,
#                           random_seed=random_seed)
        
#         # Initialize agent
#         agent = DQN()
#         # agent = PPO()
#         # agent = RandomAgent()
        
#         # Always reset the environment to initial state
#         state = env.reset()
#         initial_pos = env.agent_pos
#         for _ in trange(iters):
            
#             # Agent takes an action based on the latest observation and info.
#             action = agent.take_action(state)

#             # The action is performed in the environment
#             next_state, reward, terminated, info = env.step(action)
#             agent.update(next_state, reward, info["actual_action"])
#             state = next_state
            
#             # If the final state is reached, stop.
#             if terminated:
#                 break

#         # Evaluate the agent
#         Environment.evaluate_agent(grid, agent, iters, sigma,
#                                    agent_start_pos=initial_pos,
#                                    random_seed=random_seed)


# if __name__ == '__main__':
#     args = parse_args()
#     start_pos = None
#     if args.start_pos is not None:
#         parts = args.start_pos.split(',')
#         start_pos = (int(parts[0]), int(parts[1]))
#     main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma,
#          args.random_seed, start_pos)
