"""
Common definitions for learning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


@dataclass
class Config:
    """Learning configuration"""

    # Learning algorithm to use
    algorithm: Literal["dqn"] = "dqn"

    # Exponential decay parameter for epsilon-greedy algorithms: epsilon = beta^t
    beta: float = 0.99

    # Discount factor for Temporal-Difference algorithms
    gamma: float = 0.99

    # Learning rate for gradient descent
    lr: float = 1e-3

    # Batch size
    batch_size: int = 32

    # Size of the DQN replay buffer
    replay_buffer_size: int = 10000


class LearningAlgorithm(ABC):
    """Abstract base class for learning algorithms"""

    def __init__(self, logger: SummaryWriter) -> None:
        # TensorBoard writer for logging
        self.logger = logger

    @abstractmethod
    def choose_action(self, state: np.ndarray) -> int:
        """Select an action to perform"""

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Store a state transition for later learning"""

    @abstractmethod
    def learn(self) -> None:
        """Perform a learning update"""
