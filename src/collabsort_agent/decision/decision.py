"""
Common definitions for decision-making algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.learning import ActionValueEstimator


@dataclass
class DecisionConfig:
    """Decision configuration"""

    # Deision algorithm to use
    algorithm: Literal["eps", "ard"] = "eps"

    # ---------- Exploration decay ----------
    # Used by both algorithms: epsilon-greedy decays its exploration probability,
    # ARD increases its advantage weight w_d from 0 (random choices) to its final value.

    # Starting exploration probability
    epsilon_start: float = 1

    # Minimum exploration probability at the end of decay
    epsilon_min: float = 0.05

    # Exploration probability decay algorithm
    exploration_decay: Literal["lin", "exp"] = "lin"

    # Percentage of training time during which exploration probability is decayed
    decay_span: float = 0.5

    # ---------- Advantage Racing Diffusion ----------

    # Method for running the accumulator race:
    # - "wald": exact sampling of first-passage times (fast, no evidence history).
    # - "euler": step-by-step Euler-Maruyama simulation (slower, records evidence history for plotting).
    accumulation: Literal["wald", "euler"] = "wald"

    # Rule for ending evidence accumulation and choosing an action
    decision_rule: Literal["win-all"] = "win-all"

    # Initial value for decision threshold (adjusted via metacognition)
    theta_start: float = 1.0

    # Minimum decision threshold
    theta_min: float = 0.2

    # Maximum decision threshold
    theta_max: float = 3.0

    # Weight of the advantage (Q_i - Q_j) term, reached at the end of exploration decay
    w_d: float = 2.0

    # Exponent k of the advantage weight schedule: w_d(t) = w_d * progress(t)^k,
    # where progress(t) in [0, 1] follows the exploration decay.
    # Choice greediness rises quickly with w_d, so k > 1 keeps exploration going longer
    # (k = 3 roughly matches the greediness of epsilon-greedy along its schedule).
    w_d_schedule_power: float = 3.0

    # Weight of the sum (Q_i + Q_j) term.
    # Disabled by default (limited RL-lARD variant of Miletic2021): the sum term assumes
    # positive, bounded Q-values, whereas this environment yields mostly negative ones.
    w_s: float = 0.0

    # Urgency / baseline drift added to every accumulator.
    # Must be large enough (relative to normalized advantages) for decisions to terminate.
    V_0: float = 1.5

    # If enabled, divide Q-values by a running estimate of their spread across actions
    # before computing drift rates, so that drifts do not depend on the learned Q-value scale
    normalize_q_values: bool = True

    # Decay factor of the exponential moving average used to estimate the Q-value spread
    q_scale_decay: float = 0.99

    # Mean of accumulation noise
    noise_mean: float = 0.0

    # Standard deviation of accumulation noise (denoted s in Miletic2021 paper)
    noise_std: float = 0.03

    # Maximal decision time, in accumulation steps of duration dt
    max_steps: int = 100

    # Euler-Maruyama timestep
    dt: float = 0.01


class Deliberator(ABC):
    """Base class for decision-making algorithms."""

    def __init__(
        self,
        config: DecisionConfig,
        estimator: ActionValueEstimator,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.estimator = estimator
        self.rng = rng

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray,
        training_step: int | None,
        deterministic: bool = False,
    ) -> int:
        """
        Choose the action to perform.

        An undefined training_step is used for non-training mode (demo),
        in which the loaded exploration probability is not decayed.

        deterministic, when True, disables exploration and always returns
        the greedy action. Used for evaluation.
        """

    def update_calibration(self, td_error: float) -> None:
        """
        Update outcome-based confidence calibration if this deliberator supports metacognitive calibration.
        No-op by default.
        """
        return

    @abstractmethod
    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

    @abstractmethod
    def save_state(self, dir: str) -> None:
        """Save the deliberator state to disk"""

    @abstractmethod
    def load_state(self, dir: str) -> None:
        """Load a previously saved deliberator state from disk"""

    @property
    def state_filename(self) -> str:
        """Return the file name for saving/loading deliberator state"""

        return "deliberator.pth"
