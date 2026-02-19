"""
Common definitions for learning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from gym_collabsort.config import Action
from gymnasium import Env


@dataclass
class Config:
    """Learning configuration"""

    # Learning algorithm to use
    algorithm: Literal["dqn"] = "dqn"

    # Exploration/exploitation threshold for epsilon-greedy algorithms
    epsilon: float = 0.1


class LearningAlgorithm(ABC):
    """Abstract base class for learning algorithms"""

    @abstractmethod
    def select_action(self, env: Env) -> Action:
        """Select an action to perform"""

        pass
