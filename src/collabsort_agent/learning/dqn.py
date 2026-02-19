"""
Deep Q-Learning algorithm.
"""

import random

from gym_collabsort.config import Action
from gymnasium import Env

from collabsort_agent.learning.learning import Config as LearningConfig
from collabsort_agent.learning.learning import LearningAlgorithm


class DQN(LearningAlgorithm):
    """Deep Q-Learning algorithm"""

    def __init__(self, config: LearningConfig) -> None:
        self.config = config

    def select_action(self, env: Env) -> Action:
        # Exploration
        if random.random() < self.config.epsilon:
            return env.action_space.sample()
        else:
            # TODO implement exploitation
            return Action.NONE
