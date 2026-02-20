"""
Agent definitions.
"""

from gym_collabsort.config import Action

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

    def act(self, obs: dict) -> Action:
        """Select an action"""

        sensory_state = self.perceiver.get_sensory_state(obs=obs)
        extended_state = self.memory.get_extended_state(sensory_state=sensory_state)

        return Action(value=self.learner.choose_action(state=extended_state))
