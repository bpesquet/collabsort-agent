"""
Deep Q-Learning algorithm.
"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning.learning import Config as LearningConfig
from collabsort_agent.learning.learning import LearningAlgorithm


class QNetwork(nn.Module):
    """Neural network for Q-value estimation over all actions"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Tuple[int, int] = (100, 100),
    ) -> None:
        super().__init__()

        # Create network layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features=prev_size, out_features=hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(in_features=prev_size, out_features=output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.net(x)


class DQN(LearningAlgorithm):
    """Deep Q-Learning algorithm"""

    def __init__(self, config: LearningConfig, state_size: int, n_actions: int) -> None:
        self.config = config
        self.n_actions = n_actions

        self.model = QNetwork(input_size=state_size, output_size=n_actions)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.config.lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state: np.ndarray) -> int:
        # With probability epsilon: explore (random action)
        if np.random.random() < self.config.epsilon:
            return random.randint(0, self.n_actions - 1)

        # With probability (1-epsilon): exploit (best known action)
        # TODO Use Q-network
        return 0
