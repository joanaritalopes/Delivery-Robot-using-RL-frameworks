"""Deep Q Network (DQN) Agent"""
from collections import namedtuple
from agents import BaseAgent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



class NeuralNetworkDQN(nn.Module):
    """
    The DQN neural network.
    Input: state
    Output: Q-value
    """
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim: Input dimension (positional info).
            output_dim: Output dimension (possible actions).
        """
        super(NeuralNetworkDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)



class DQN(BaseAgent):
    """
    Deep Q Network (DQN) agent.
    """
    def __init__(
        self,
        lr: float = 0.001,                  # 0.0001, 0.001
        df: float = 0.99,                   # 0.99
        ep: float = 1.0,                    # 1.0
        ep_decay: float = 0.9995,           # 0.995, 0.9995, 0.99995
        min_ep: float = 0.01,               # 0.01, 0.05
        batch_size: int = 64,               # 32, 64, 128
        min_buffer: int = 1000,             # 1 000, 5 000
        max_buffer: int = 5000,             # 10 000, 100 000
        target_update_interval: int = 200,  # 100, 200, 1000
        seed: int | None = None,
        input_dim: int = 2,
        output_dim: int = 4,
    ):
        """
        Args:
            lr: Learning rate (optimizer).
            df: Discount factor gamma.
            ep: Epsilon for epsilon-greedy exploration.
            ep_decay: Multiplicative epsilon decay per update.
            min_ep: Minimum epsilon.
            batch_size: Batch size.
            min_buffer: Minimum replay buffer size.
            max_buffer: Maximum replay buffer size.
            target_update_interval: Target network update interval duration.
            seed: Random seed for reproducibility.
            input_dim: Input dimension (positional info).
            output_dim: Output dimension (possible actions).
        """
        # Hyperparameters
        self.df = df
        self.ep = ep
        self.ep_decay = ep_decay
        self.min_ep = min_ep
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.max_buffer = max_buffer
        self.target_update_interval = target_update_interval

        # Fixed parameters
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Internal parameters
        self.last_state: torch.tensor | None = None
        self.last_action: int | None = None
        self.update_counter = 0
        self.Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "terminated"])
        self.replay_buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_network = NeuralNetworkDQN(input_dim, output_dim).to(self.device)
        self.target_network = NeuralNetworkDQN(input_dim, output_dim).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
        for param in self.target_network.parameters():
            param.requires_grad = False

    def update(
        self,
        state: np.ndarray,
        reward: float,
        action: int | None = None,
        terminated: bool = False,
    ):
        """
        Build experiences, update the replay buffer, update the online and target networks.

        Args:
            state: New state after taking the previous action.
            reward: Reward received.
            action: Kept for compatibility with BaseAgent, but this implementation
                uses self.last_action because that is the action chosen by the agent.
            terminated: Whether the new state is terminal.
        """
        # Prevent the first step from crashing
        if self.last_state is None or self.last_action is None:
            return

        # Build experiences (state, action, Reward(next_state), next_state, terminated)
        next_x, next_y = state
        next_state = torch.tensor([next_x, next_y], dtype=torch.float32)
        experience = self.Experience(self.last_state, self.last_action, reward, next_state, terminated)

        # Update replay buffer
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer.pop(0)

        # Choose random experience from buffer
        if len(self.replay_buffer) > self.min_buffer:
            # Get mini batch
            indices = self.rng.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
            batch = self.Experience(*zip(*[self.replay_buffer[i] for i in indices]))

            # Get mini batch variables
            states = torch.stack(batch.state).to(self.device)
            actions = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
            next_states = torch.stack(batch.next_state).to(self.device)
            terminations = torch.tensor(batch.terminated, dtype=torch.bool).to(self.device)

            # Calculate MSE-loss
            q_values = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.target_network(next_states).max(dim=1).values
            y = rewards + self.df*(torch.logical_not(terminations))*next_q_values
            mse_loss = self.loss_fn(q_values, y)

            # Backward pass (update online network)
            self.optimizer.zero_grad()
            mse_loss.backward()
            self.optimizer.step()

            # Scheduled target network update
            if self.update_counter >= self.target_update_interval:
                self.target_network.load_state_dict(self.online_network.state_dict())
                for param in self.target_network.parameters():
                    param.requires_grad = False
                self.update_counter = 0
            else:
                self.update_counter += 1

            # Decay exploration after learning
            self.ep = max(self.min_ep, self.ep*self.ep_decay)

    def take_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy.

        Actions:
            0: down
            1: up
            2: left
            3: right
        """
        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float32)

        # Take random or best action
        if self.rng.random() < self.ep:
            action = int(self.rng.integers(0, 4))
        else:
            with torch.no_grad():
                action = int((self.online_network(state)).argmax())

        # Update variables
        self.last_state = state
        self.last_action = action
        return action



def compute_ep_decay(min_ep, target_step):
    """
    Helper function to compute a good epsilon decay.
    """
    print((min_ep) ** (1 / target_step))

compute_ep_decay(0.01, 8000)