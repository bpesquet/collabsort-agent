"""
Agent definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import DecisionConfig, Deliberator
from collabsort_agent.learning import LearningConfig
from collabsort_agent.memory import Memory, MemoryConfig
from collabsort_agent.metacognition import MetaConfig
from collabsort_agent.perception import Perceiver, PerceptionConfig


@dataclass
class AgentConfig:
    """Agent configuration"""

    # Perception configuration
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    # Memory configuration
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Decision configuration
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    # Learning configuration
    learning: LearningConfig = field(default_factory=LearningConfig)

    # Metacognition configuration
    meta: MetaConfig = field(default_factory=MetaConfig)


class Agent:
    """An agent interacting with its environment."""

    def __init__(
        self, perceiver: Perceiver, memory: Memory, deliberator: Deliberator
    ) -> None:
        self.perceiver = perceiver
        self.memory = memory
        self.deliberator = deliberator

        # Current extended state (sensory + memory)
        self.current_extended_state: np.ndarray | None = None

        # Newest action chosen by the agent
        self.current_action: Action | None = None

    def act(
        self,
        obs: dict,
        training_step: int | None,
    ) -> Action:
        """Select an action"""

        sensory_state = self.perceiver.get_sensory_state(obs=obs)
        extended_state = self.memory.get_extended_state(sensory_state=sensory_state)

        self.current_extended_state = extended_state

        # Choose the next action to perform
        self.current_action = Action(
            self.deliberator.choose_action(
                state=extended_state,
                training_step=training_step,
            )
        )
        return self.current_action

    def update(self, next_obs: dict, reward: float, done: bool) -> None:
        """Update agent after an action"""

        if self.current_extended_state is None or not self.current_action:
            raise ValueError("Trying to update agent with non-existent state")

        # Compute next extended state (sensory + memory) for the transition
        next_sensory_state = self.perceiver.get_sensory_state(obs=next_obs)
        next_extended_state = self.memory.get_extended_state(
            sensory_state=next_sensory_state
        )

        # Update action values
        self.deliberator.estimator.update_action_values(
            state=self.current_extended_state,
            action=self.current_action.value,
            reward=reward,
            next_state=next_extended_state,
            done=done,
        )

    def log_episode(self, logger: SummaryWriter | None, episode: int) -> None:
        """Log agent information after an episode"""

        if logger is not None:
            self.deliberator.log_episode(logger=logger, episode=episode)
            self.deliberator.estimator.log_episode(logger=logger, episode=episode)

    def serialize(self, dir: str) -> None:
        """Save the agent state to disk"""

        self.deliberator.save_state(dir=dir)
        self.deliberator.estimator.save_state(dir=dir)

    def deserialize(self, dir: str) -> None:
        """Load the agent state from disk"""

        self.deliberator.estimator.load_state(dir=dir)
        self.deliberator.load_state(dir=dir)
