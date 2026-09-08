"""
Configuration definitions.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.agent import AgentConfig

# File name used to (de)serialize a configuration object
SERIALIZATION_FILENAME: str = "config.pkl"


@dataclass
class Config:
    """Global configuration"""

    # Environment configuration
    env: EnvConfig = field(default_factory=EnvConfig)

    # Environment version
    env_id: str = "CollabSort-v1"

    # Agent configuration
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Number of training episodes
    n_episodes: int = 300

    # Maximal number of steps in an episode
    n_steps_episode: int = 1000

    # Directory used to load a previously saved configuration
    load_dir: str | None = None

    @property
    def total_steps(self) -> int:
        """Total number of training steps"""

        return self.n_steps_episode * self.n_episodes

    def serialize(self, dir: str) -> None:
        """Save a configuration object to disk"""

        # Save agent and environment configurations only (see below)
        with open(file=f"{dir}/agent_{SERIALIZATION_FILENAME}", mode="wb") as file:
            pickle.dump(obj=self.agent, file=file)
        with open(file=f"{dir}/env_{SERIALIZATION_FILENAME}", mode="wb") as file:
            pickle.dump(obj=self.env, file=file)

    def deserialize(self, dir: str) -> None:
        """Load a configuration object from disk"""

        # Load agent and environment configurations only.
        # This prevents overriding other configuration parameters when loading a previously saved run
        with open(file=f"{dir}/agent_{SERIALIZATION_FILENAME}", mode="rb") as file:
            self.agent = pickle.load(file=file)
        with open(file=f"{dir}/env_{SERIALIZATION_FILENAME}", mode="rb") as file:
            self.env = pickle.load(file=file)
