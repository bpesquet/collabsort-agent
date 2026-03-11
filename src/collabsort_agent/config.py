"""
Configuration definitions.
"""

import pickle
from dataclasses import dataclass

from gym_collabsort.config import Action
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.agent import Agent
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN
from collabsort_agent.learning.exploration_decay import (
    ExponentialExplorationDecay,
    LinearExplorationDecay,
)
from collabsort_agent.memory import Config as MemoryConfig
from collabsort_agent.memory import Memory
from collabsort_agent.perception import Config as PerceptionConfig
from collabsort_agent.perception import Perceiver


@dataclass
class Config:
    """Training configuration"""

    # Environment configuration
    env: EnvConfig

    # Perception configuration
    perception: PerceptionConfig

    # Memory configuration
    memory: MemoryConfig

    # Learning configuration
    learning: LearningConfig

    # Number of training episodes
    n_episodes: int = 300

    # Maximal number of steps in an episode
    n_steps_episode: int = 1000

    # Log training events
    log_events: bool = True

    # Save state at end of training
    save_state: bool = True


def save_cfg(config: Config, dir: str) -> None:
    """Save a configuration object to disk"""

    with open(file=f"{dir}/config.pkl", mode="wb") as file:
        pickle.dump(obj=config, file=file)


def load_cfg(dir: str) -> Config:
    """Load a configuration object from disk"""

    with open(file=f"{dir}/config.pkl", mode="rb") as file:
        return pickle.load(file=file)


def create_agent(config: Config, sample_obs: dict) -> Agent:
    """Create an agent with a specific configuration"""

    # Initialize perception
    perceiver = Perceiver(
        config=config.perception,
        treadmill_rows=[config.env.upper_treadmill_row, config.env.lower_treadmill_row],
    )
    sample_sensory_state = perceiver.get_sensory_state(obs=sample_obs)

    if config.memory.type == "none":
        memory = Memory()
    else:
        raise Exception(f"Unrecognized memory type: {config.memory.type}")

    sample_extended_state = memory.get_extended_state(
        sensory_state=sample_sensory_state
    )

    # Compute learning hyperparameters
    extended_state_size = len(sample_extended_state)
    n_actions = len(Action) + len(memory.get_actions())

    # Initialize learning
    if config.learning.algorithm == "dqn":
        # Total number of training steps
        total_steps: int = config.n_episodes * config.n_steps_episode

        # Initialize exploration probability decay algorithm
        if config.learning.exploration_decay == "lin":
            exploration_decay = LinearExplorationDecay(
                config=config.learning, total_steps=total_steps
            )
        elif config.learning.exploration_decay == "exp":
            exploration_decay = ExponentialExplorationDecay(
                config=config.learning, total_steps=total_steps
            )
        else:
            raise Exception(
                f"Unrecognized exploration decay: {config.learning.exploration_decay}"
            )

        learner = DQN(
            config=config.learning,
            exploration_decay=exploration_decay,
            state_size=extended_state_size,
            n_actions=n_actions,
        )
    else:
        raise Exception(f"Unrecognized learning algorithm: {config.learning.algorithm}")

    return Agent(perceiver=perceiver, memory=memory, learner=learner)
