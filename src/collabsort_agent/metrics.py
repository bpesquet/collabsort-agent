"""
Metrics-related definitions.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter


def get_action_name(action: int) -> str:
    """Return the name associated to an action value"""

    if action == Action.NONE.value:
        return "none"
    elif action == Action.UP.value:
        return "up"
    elif action == Action.DOWN.value:
        return "down"
    elif action == Action.PICK.value:
        return "pick"
    else:
        raise ValueError(f"Unrecognized action {action}")


@dataclass
class ArmMetrics:
    """Episode metrics for an arm (agent or robot)"""

    # Arm name (used for logging)
    name: Literal["agent", "robot"]

    # Episodic return (cumulated reward)
    reward: float = 0.0

    # Number of placed (collected) objects
    n_collected_objects: int = 0

    # List of actions taken
    actions: list[int] = field(default_factory=list[int])

    def log(self, logger: SummaryWriter, episode: int) -> None:
        """Log arm metrics at end of episode"""

        logger.add_scalar(
            tag=f"{self.name}/episodic_return",
            scalar_value=(self.reward),
            global_step=episode,
        )

        logger.add_scalar(
            tag=f"{self.name}/n_collected_objects",
            scalar_value=(self.n_collected_objects),
            global_step=episode,
        )

        if len(self.actions) > 0:
            # Log action frequency
            n_actions = len(Action)
            action_counts = np.bincount(self.actions, minlength=n_actions)
            action_freqs = action_counts / action_counts.sum()
            for action in range(n_actions):
                action_name = get_action_name(action)
                logger.add_scalar(
                    tag=f"{self.name}/action_{action_name}",
                    scalar_value=action_freqs[action],
                    global_step=episode,
                )


@dataclass
class EpisodeMetrics:
    """Episode metrics"""

    # Agent metrics
    agent: ArmMetrics = field(default_factory=lambda: ArmMetrics(name="agent"))

    # Robot metrics
    robot: ArmMetrics = field(default_factory=lambda: ArmMetrics(name="robot"))

    # Episode time step (= number of time steps since beginning of episode)
    step: int = 0

    # Maximum possible reward (total value of all objects)
    maximum_reward: float = 0.0

    # Number or missed objects (fallen from treadmills)
    n_missed_objects: int = 0

    # Number of collisions
    n_collisions: int = 0

    # Number of steps per second
    sps: float = 0.0

    def log(
        self,
        logger: SummaryWriter,
        episode: int,
    ) -> None:
        """Log metrics at end of episode"""

        # Log and and robot metrics
        self.agent.log(logger=logger, episode=episode)
        self.robot.log(logger=logger, episode=episode)

        # Log collaboration and technical metrics
        logger.add_scalar(
            tag="n_missed_objects",
            scalar_value=self.n_missed_objects,
            global_step=episode,
        )
        logger.add_scalar(
            tag="n_collisions",
            scalar_value=self.n_collisions,
            global_step=episode,
        )
        logger.add_scalar(
            tag="steps_per_seconds",
            scalar_value=self.sps,
            global_step=episode,
        )
