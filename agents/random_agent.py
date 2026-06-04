"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
from agents import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    """Agent that performs a random action every time. """
    def update(self, state: np.ndarray, reward: float, action):
        pass

    def take_action(self, state: np.ndarray) -> int:
        return randint(0, 3)