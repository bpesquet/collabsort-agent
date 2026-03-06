"""
Agent definitions.
"""

import numpy as np
from gym_collabsort.config import Action
from torch.utils.tensorboard.writer import SummaryWriter

from collabsort_agent.learning import LearningAlgorithm
from collabsort_agent.memory import Memory
from collabsort_agent.perception import Perceiver


class Agent:
    """An agent"""

    def __init__(
        self, perceiver: Perceiver, memory: Memory, learner: LearningAlgorithm
    ) -> None:
        self.perceiver = perceiver
        self.memory = memory
        self.learner = learner

        # Init internal data
        self.cureent_sensory_state: np.ndarray | None = None
        self.current_action: Action | None = None

    def act(self, obs: dict, training_step: int) -> Action:
        """Select an action"""

        self.cureent_sensory_state = self.perceiver.get_sensory_state(obs=obs)
        extended_state = self.memory.get_extended_state(
            sensory_state=self.cureent_sensory_state
        )

        self.current_action = Action(
            self.learner.choose_action(
                state=extended_state, training_step=training_step
            )
        )
        return self.current_action

    def update(self, next_obs: dict, reward: float, done: bool) -> None:
        """Update agent after an action"""

        if self.cureent_sensory_state is None or not self.current_action:
            raise Exception("Trying to update agent with non-existent state")

        # Store state transition
        next_sensory_state = self.perceiver.get_sensory_state(obs=next_obs)
        self.learner.store_transition(
            state=self.cureent_sensory_state,
            action=self.current_action.value,
            reward=reward,
            next_state=next_sensory_state,
            done=done,
        )

        # Perform learning
        self.learner.learn()

    def log(self, logger: SummaryWriter, episode: int) -> None:
        """Log agent information for an episode"""

        self.learner.log(logger=logger, episode=episode)
