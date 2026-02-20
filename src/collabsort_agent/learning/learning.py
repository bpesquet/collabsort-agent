"""
Common definitions for learning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Config:
    """Learning configuration"""

    # Learning algorithm to use
    algorithm: Literal["dqn"] = "dqn"

    # Exploration/exploitation threshold for epsilon-greedy algorithms
    epsilon: float = 0.1

    # Learning rate for gradient descent
    lr: float = 1e-3


class LearningAlgorithm(ABC):
    """Abstract base class for learning algorithms"""

    @abstractmethod
    def choose_action(self, state: np.ndarray) -> int:
        """Select an action to perform"""
