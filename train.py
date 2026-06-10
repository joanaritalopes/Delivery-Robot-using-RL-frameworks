"""
Train RL Agent.

Usage example:
    # train on 1 grid
    python3 train.py grid_configs/example_grid.npy --agent dqn --no_gui --iter 100000

    # train DQN on one grid with default hyperparameters
    python3 train.py grid_configs/warehouse_small.npy --agent dqn --no_gui --iter 500000
 
    # train PPO on one grid with default hyperparameters
    python3 train.py grid_configs/warehouse_small.npy --agent ppo --no_gui --iter 500000
 
    # train DQN with custom hyperparameters (for tuning experiments)
    python3 train.py grid_configs/warehouse_small.npy --agent dqn --no_gui --iter 500000 --lr 0.0001 --gamma 0.99 --batch_size 128 --epsilon_decay 0.9999 --target_update 500
 
    # train PPO with custom hyperparameters
    python3 train.py grid_configs/warehouse_small.npy --agent ppo --no_gui --iter 500000 --lr 0.0003 --gamma 0.99 --clip_eps 0.2 --rollout_size 512 --ppo_epochs 10
"""

import csv
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
 
from world import Environment
from agents.dqn_agent import DQN
from agents.ppo_agent import PPO


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
 
    # ── Environment arguments ──
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo"],
                   help="Which agent to train - dqn or ppo.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--iter", type=int, default=200000,
                   help="Total number of steps to train for (across all episodes).")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as row,col (e.g. 2,3).")
    p.add_argument("--eval_episodes", type=int, default=20,
                   help="Number of episodes to run during each evaluation.")
    p.add_argument("--eval_every", type=int, default=10,
                   help="Run evaluation every N episodes during training.")


    # ── Shared hyperparameters (both agents) ──
    p.add_argument("--lr", type=float, default=0.001,
                   help="Learning rate for the optimizer.")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor gamma.")

    # ── DQN-specific hyperparameters ──
    p.add_argument("--batch_size", type=int, default=64,
                   help="[DQN] Batch size for replay buffer sampling.")
    p.add_argument("--epsilon_decay", type=float, default=0.9995,
                   help="[DQN] Multiplicative epsilon decay per update.")
    p.add_argument("--min_epsilon", type=float, default=0.01,
                   help="[DQN] Minimum epsilon value.")
    p.add_argument("--min_buffer", type=int, default=1000,
                   help="[DQN] Minimum replay buffer size before training starts.")
    p.add_argument("--max_buffer", type=int, default=50000,
                   help="[DQN] Maximum replay buffer size.")
    p.add_argument("--target_update", type=int, default=200,
                   help="[DQN] Steps between target network hard updates.")

    # ── PPO-specific hyperparameters ──
    p.add_argument("--clip_eps", type=float, default=0.2,
                   help="[PPO] Clipping epsilon for the surrogate loss.")
    p.add_argument("--rollout_size", type=int, default=256,
                   help="[PPO] Number of steps to collect per rollout before updating.")
    p.add_argument("--ppo_epochs", type=int, default=10,
                   help="[PPO] Number of update epochs per rollout.")
    p.add_argument("--gae_lambda", type=float, default=0.95,
                   help="[PPO] Lambda for Generalized Advantage Estimation.")
    p.add_argument("--entropy_coef", type=float, default=0.01,
                   help="[PPO] Entropy bonus coefficient to encourage exploration.")
    p.add_argument("--value_coef", type=float, default=0.5,
                   help="[PPO] Value loss coefficient.")

    return p.parse_args()


def build_agent(agent_name: str, args):
    """Instantiate the chosen agent with hyperparameters from CLI args"""
    match agent_name:
        case "dqn":
            return DQN(
                input_dim=2,
                output_dim=4,
                lr=args.lr,
                df=args.gamma,
                batch_size=args.batch_size,
                ep_decay=args.epsilon_decay,
                min_ep=args.min_epsilon,
                min_buffer=args.min_buffer,
                max_buffer=args.max_buffer,
                target_update_interval=args.target_update,
                seed=args.random_seed,
            )
        case "ppo":
            return PPO(
                state_dim=2,
                action_dim=4,
                lr=args.lr,
                gamma=args.gamma,
                clip_eps=args.clip_eps,
                rollout_size=args.rollout_size,
                update_epochs=args.ppo_epochs,
                gae_lambda=args.gae_lambda,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
            )
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

    Returns the epsilon value active before switching (DQN only),
    so the caller can restore it after evaluation.
    """
    if isinstance(agent, DQN):
        if eval_mode:
            current_ep = agent.ep
            agent.ep = 0.0
            return current_ep       # caller must store and pass back
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
             sigma: float, start_pos, random_seed: int,
             max_steps: int = 5000) -> dict:
    """
    Runs the agent greedily for eval_episodes episodes on the same grid.
    There is no learning here.
    """
    # Create a separate environment instance for evaluation so it does
    # not interfere with the training environment's internal state.
    env = Environment(grid_fp, no_gui=True, sigma=sigma,
                      agent_start_pos=start_pos,
                      target_fps=-1, random_seed=random_seed)

    saved_epsilon = set_eval_mode(agent, eval_mode=True)

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for _ in range(eval_episodes):
        state = env.reset()
        ep_reward = 0
        ep_steps = 0
        terminated = False

        while not terminated and ep_steps < max_steps:
            action = agent.take_action(state)
            state, reward, terminated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        if terminated:
            successes += 1

    set_eval_mode(agent, eval_mode=False, saved_epsilon=saved_epsilon)

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

    Runs until total_steps environment steps have been taken across as
    many episodes as needed. Logs per-episode metrics to train_log CSV
    and evaluation metrics to eval_log CSV every eval_every episodes.
    """
    env = Environment(grid_fp, no_gui=no_gui, sigma=sigma,
                      target_fps=fps, agent_start_pos=start_pos,
                      random_seed=random_seed)


    init_csv(train_log, ["episode", "steps_so_far", "ep_reward",
                         "ep_length", "success_rate", "epsilon"])
    init_csv(eval_log,  ["episode", "steps_so_far", "eval_mean_reward",
                         "eval_std_reward", "eval_success_rate",
                         "eval_mean_length"])

    step_count = 0
    episode = 0

    print(f"\nTraining on {grid_fp.name} for {total_steps} total steps\n")

    # keep running episodes until total step budget is used up
    while step_count < total_steps:
        state = env.reset()
        ep_reward = 0
        ep_steps = 0
        terminated = False
        episode += 1

        # run one episode step by step
        while not terminated:
            action = agent.take_action(state)
            next_state, reward, terminated, info = env.step(action)

            agent_update(agent, state, next_state, reward,
                         info["actual_action"], terminated)

            state = next_state
            ep_reward += reward
            ep_steps += 1
            step_count += 1

            # Exit inner loop early if total budget is exhausted mid-episode
            if step_count >= total_steps:
                break

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
    """Runs Environment.evaluate_agent for a final path visualization"""
    print("\nRunning final evaluation ")
    Environment.evaluate_agent(
        grid_fp, agent, total_steps,
        sigma=sigma,
        agent_start_pos=start_pos,
        random_seed=random_seed,
        show_images=False,
    )


def main():
    """Main loop of the program"""

    args = parse_args()

    # Parse optional start position from "row,col" string to tuple
    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(",")
        start_pos = (int(parts[0]), int(parts[1]))

    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    log_dir = Path("results")
    log_dir.mkdir(parents=True, exist_ok=True)

    for grid_fp in args.GRID:
        print(f"\n{'='*60}")
        print(f"Grid: {grid_fp.name} | Agent: {args.agent} | "
              f"Steps: {args.iter} | Seed: {args.random_seed}")
        print(f"{'='*60}")

        # Initialize agent
        agent = build_agent(args.agent, args)

        # One training CSV and one evaluation CSV per run
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

        # Evaluate the agent
        final_evaluate(agent, grid_fp, args.iter,
                       args.sigma, start_pos, args.random_seed)

        print(f"\nLogs saved to:")
        print(f"{train_log}")
        print(f"{eval_log}")


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
