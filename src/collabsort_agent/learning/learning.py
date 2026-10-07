"""
Common definitions for learning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import Literal

import numpy as np
from torch.utils.tensorboard import SummaryWriter


@dataclass
class LearningConfig:
    """Learning configuration"""

    # Learning algorithm to use
    algorithm: Literal[
        "ql",
        "dqn",
        "dueling_dqn",
        "ddqn",
        "dd_dqn",
        "per",
        "n_step",
    ] = "ql"

    # Discount factor for Temporal-Difference algorithms
    gamma: float = 0.99

    # Learning rate for gradient descent
    lr: float = 1e-3

    # Initial value for variable learning rate (adjusted via metacognition)
    alpha_start: float = 0.1

    # Minimum variable learning rate
    alpha_min: float = 0.01

    # Maximum variable learning rate
    alpha_max: float = 0.5

    # Batch size for sampling from replay buffer
    batch_size: int = 256

    # Size of the DQN replay buffer
    replay_buffer_size: int = 100000

    # Number of steps for n-step returns (1 = standard DQN)
    n_step: int = 3

    # Interval in learning steps to copy online weights to target network.
    target_network_sync_freq: int = 500

    # Initial Q-Value
    q_start: float = 0

    # Number of training episodes
    n_episodes: int = 300

    # Maximal number of steps in an episode
    n_steps_episode: int = 1000


class ActionValueEstimator(ABC):
    """Base class for action value estimators."""

    def __init__(
        self,
        config: LearningConfig,
        n_actions: int,
    ) -> None:
        self.config = config
        self.n_actions = n_actions

        # Recorded loss values (used for logging).
        # Algorithm-specific (e.g. |TD-error| for Q-Learning, batch MSE for DQN):
        # only comparable within an algorithm family.
        self.losses: list[float] = []

        # Per-transition metrics computed identically for all estimators
        # on the transitions actually experienced (used for logging).
        # See record_transition_metrics().
        self.abs_td_errors: list[float] = []
        self.q_taken_values: list[float] = []
        self.q_max_values: list[float] = []

        # Signed TD-error (reward-prediction error, Eq 1) from the most
        # recent update_action_values() call. Used by outcome-based
        # confidence calibration (extension 5). Estimators that don't
        # naturally expose a single per-transition TD-error (e.g. batched
        # off-policy algorithms) may leave this at its default of 0.0.
        self.last_td_error: float = 0.0

    @abstractmethod
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """Return the action values for all actions"""

    @abstractmethod
    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        """Update action values after an action was taken"""

    def record_transition_metrics(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """
        Record estimator-agnostic metrics for an experienced transition.

        Must be called before update_action_values(), so that metrics reflect
        the estimates the agent actually acted upon. Uses the one-step greedy
        TD-error δ = r + γ · max_a' Q(s', a') − Q(s, a) computed with
        get_action_values() for every estimator, regardless of the target
        each algorithm actually learns from (target network, n-step returns...).
        This makes the values comparable across learning algorithms.
        """

        q_values = self.get_action_values(state)
        q_current = float(q_values[action])
        q_target = (
            reward
            if done
            else reward
            + self.config.gamma * float(self.get_action_values(next_state).max())
        )

        self.abs_td_errors.append(abs(q_target - q_current))
        self.q_taken_values.append(q_current)
        self.q_max_values.append(float(q_values.max()))

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        # Lists may be empty, e.g. before a replay buffer holds a full batch
        for tag, values in (
            ("learning/loss", self.losses),
            ("learning/abs_td_error", self.abs_td_errors),
            ("learning/q_taken", self.q_taken_values),
            ("learning/q_max", self.q_max_values),
        ):
            if values:
                logger.add_scalar(
                    tag=tag, scalar_value=mean(values), global_step=episode
                )

            # Reset episode data
            values.clear()

    @abstractmethod
    def save_state(self, dir: str) -> None:
        """Save the estimator state to disk"""

    @abstractmethod
    def load_state(self, dir: str) -> None:
        """Load a previously saved estimator state from disk"""

    @property
    def state_filename(self) -> str:
        """Return the file name for saving/loading estimator state"""

        return "estimator.pth"
