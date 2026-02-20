"""
Train an agent.
"""

from dataclasses import dataclass

import gymnasium as gym
import tyro
from gym_collabsort.config import Action
from gym_collabsort.config import Config as EnvConfig
from gym_collabsort.envs.env import RenderMode

from collabsort_agent.agent import Agent
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN
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

    # Environment rendering mode
    render_mode: RenderMode = RenderMode.NONE


def create_agent(config: Config, sample_obs: dict) -> Agent:
    """Create an agent with a specific configuration"""

    # Initialize perception
    perceiver = Perceiver(
        config=config.perception,
        treadmill_rows=[config.env.upper_treadmill_row, config.env.lower_treadmill_row],
    )
    sample_sensory_state = perceiver.get_sensory_state(obs=sample_obs)

    # Initialize memory
    if config.memory.type == "none":
        memory = Memory()
    sample_extended_state = memory.get_extended_state(
        sensory_state=sample_sensory_state
    )

    # Compute learning hyperparameters
    extended_state_size = len(sample_extended_state)
    n_actions = len(Action) + len(memory.get_actions())

    # Initialize learning
    if config.learning.algorithm == "dqn":
        learner = DQN(
            config=config.learning,
            state_size=extended_state_size,
            n_actions=n_actions,
        )

    return Agent(perceiver=perceiver, memory=memory, learner=learner)


def train(config: Config) -> None:
    """Train an agent"""

    # Initialize environment
    env = gym.make("CollabSort-v0", render_mode=config.render_mode, config=config.env)
    obs, _ = env.reset()

    # Create agent
    agent = create_agent(config=config, sample_obs=obs)

    # Training loop
    ep_over = False
    while not ep_over:
        # Agent chooses an action
        action: Action = agent.act(obs=obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, _ = env.step(action=action)

        # TODO Learn from this experience

        # Move to next state
        ep_over = terminated or truncated
        obs = next_obs

    env.close()


if __name__ == "__main__":  # pragma: no cover
    # Create training configuration from command line args
    config: Config = tyro.cli(Config)

    train(config=config)
