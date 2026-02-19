"""
Train an agent.
"""

from dataclasses import dataclass

import gym_collabsort
import gymnasium as gym
import tyro
from gym_collabsort.config import Config as EnvConfig
from gym_collabsort.envs.env import RenderMode

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning import LearningAlgorithm
from collabsort_agent.learning.dqn import DQN


@dataclass
class Config:
    """Training configuration"""

    # Environment configuration
    env: EnvConfig

    # Learning configuration
    learning: LearningConfig

    # Environment rendering mode
    render_mode: RenderMode = RenderMode.NONE


def train(config: Config) -> None:
    """Train an agent"""

    print(f"Using gym-collabsort {gym_collabsort.__version__}")

    env = gym.make("CollabSort-v0", render_mode=config.render_mode, config=config.env)

    if config.learning.algorithm == "dqn":
        learner: LearningAlgorithm = DQN(config=config.learning)

    env.reset()

    ep_over: bool = False
    while not ep_over:
        _, _, terminated, truncated, _ = env.step(action=learner.select_action(env=env))
        ep_over = terminated or truncated

    env.close()


if __name__ == "__main__":
    # Create training configuration from command line args
    config: Config = tyro.cli(Config)

    train(config=config)
