"""
Configuration definitions.
"""

import pickle
from dataclasses import dataclass, field

from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.decision import DecisionConfig
from collabsort_agent.learning import LearningConfig
from collabsort_agent.memory import MemoryConfig
from collabsort_agent.metacognition import MetaConfig
from collabsort_agent.perception import PerceptionConfig


@dataclass
class Config:
    """Training configuration"""

    # Environment configuration
    env: EnvConfig = field(default_factory=EnvConfig)

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

    # Environment version
    env_id: str = "CollabSort-v1"

    # Number of training episodes
    n_episodes: int = 300

    # Maximal number of steps in an episode
    n_steps_episode: int = 1000

    # Log training events
    log_events: bool = True

    # Save state at end of training
    save_state: bool = True

    @property
    def total_steps(self) -> int:
        """Total number of training steps"""

        return self.n_steps_episode * self.n_episodes


def save_cfg(config: Config, dir: str) -> None:
    """Save a configuration object to disk"""

    with open(file=f"{dir}/config.pkl", mode="wb") as file:
        pickle.dump(obj=config, file=file)


def load_cfg(dir: str) -> Config:
    """Load a configuration object from disk"""

    with open(file=f"{dir}/config.pkl", mode="rb") as file:
        return pickle.load(file=file)
