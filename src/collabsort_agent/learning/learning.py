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

    # Discount factor for Temporal-Difference algorithms
    gamma: float = 0.99

    # Learning rate for gradient descent
    lr: float = 1e-3

    # Batch size for sampling from replay buffer
    batch_size: int = 64

    # Size of the DQN replay buffer
    replay_buffer_size: int = 10000

    # Starting exploration probability
    epsilon_start: float = 1

    # Minimum exploration probability at the end of decay
    epsilon_min: float = 0.05

    # Exploration probability decay algorithm
    exploration_decay: Literal["lin", "exp"] = "lin"

    # Percentage of training time during which exploration probability is decayed
    exploration_decay_span: float = 0.5

    # Interval in learning steps to copy online weights to target network.
    target_network_sync_freq: int = 500


class LearningAlgorithm(ABC):
    """Abstract base class for learning algorithms"""

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def choose_action(self, state: np.ndarray, training_step: int | None) -> int:
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
        """Update model parameters"""

    @abstractmethod
    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

    @abstractmethod
    def save(self, dir: str) -> None:
        """Save the learning component"""

    @abstractmethod
    def load(self, dir: str) -> None:
        """Load a previously saved learning component"""
