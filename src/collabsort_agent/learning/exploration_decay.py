"""
Exploration/exploitation ratio management.
"""

from abc import ABC, abstractmethod

import numpy as np

from collabsort_agent.learning import Config as LearningConfig


class ExplorationDecay(ABC):
    """Abstract base class for exploration decay algorithms"""

    def __init__(self, config: LearningConfig, total_steps: int) -> None:
        self.config = config

        # Number of steps during which exploration probability is decayed
        self.decay_steps: int = int(total_steps * self.config.exploration_decay_span)

    def get_epsilon(self, training_step: int) -> float:
        """Return the exploration probability epsilon"""

        epsilon_decayed = self._decay_epsilon(training_step=training_step)

        return max(epsilon_decayed, self.config.epsilon_min)

    @abstractmethod
    def _decay_epsilon(self, training_step: int) -> float:
        """Compute the decayed value of exploration probability"""


class LinearExplorationDecay(ExplorationDecay):
    """Linear exploration decay"""

    def __init__(self, config: LearningConfig, total_steps: int) -> None:
        super().__init__(config=config, total_steps=total_steps)

    def _decay_epsilon(self, training_step: int) -> float:
        decay_slope = (
            self.config.epsilon_min - self.config.epsilon_start
        ) / self.decay_steps

        # Decay epsilon linearly: ε = ε_min + t * (ε_min - ε_start) / decay_steps
        return self.config.epsilon_start + decay_slope * training_step


class ExponentialExplorationDecay(ExplorationDecay):
    """Exponential exploration decay"""

    def __init__(self, config: LearningConfig, total_steps: int) -> None:
        super().__init__(config=config, total_steps=total_steps)

    def _decay_epsilon(self, training_step: int) -> float:
        # Maximal difference between current and minimum values of epsilon for stopping decay
        epsilon_delta = 0.01

        # Compute the decay rate λ to reach (epsilon_min + epsilon_delta) at decay_steps
        decay_rate: float = (
            -np.log(
                epsilon_delta / (self.config.epsilon_start - self.config.epsilon_min)
            )
            / self.decay_steps
        )

        # Decay epsilon exponentially: ε = ε_min + (ε_start - ε_min) * exp(-λt)
        return self.config.epsilon_min + (
            self.config.epsilon_start - self.config.epsilon_min
        ) * np.exp(-decay_rate * training_step)
