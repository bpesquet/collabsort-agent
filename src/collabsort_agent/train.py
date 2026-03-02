"""
Train an agent.
"""

import time
from dataclasses import dataclass

import gymnasium as gym
import tyro
from gym_collabsort.config import Action
from gym_collabsort.config import Config as EnvConfig
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

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

    # Number of training episodes
    n_episodes: int = 50

    # Maximal number of steps in an episode
    n_steps_episode: int = 1000


def create_agent(config: Config, sample_obs: dict, logger: SummaryWriter) -> Agent:
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
            logger=logger,
        )

    return Agent(perceiver=perceiver, memory=memory, learner=learner)


def train(config: Config) -> None:
    """Train an agent"""

    # Initialize logging
    run_name: str = f"train_{config.learning.algorithm}_{int(time.time())}"
    logger = SummaryWriter(f"runs/{run_name}")

    # Initialize environment
    env = gym.make("CollabSort-v0", config=config.env)

    # Create agent
    agent = create_agent(
        config=config, sample_obs=env.observation_space.sample(), logger=logger
    )

    for episode in trange(config.n_episodes, desc="Training progress"):
        # Reset environment and metrics for new episode
        obs, _ = env.reset()
        ep_reward: float = 0.0
        ep_collisions: int = 0
        ep_collected_objects: int = 0
        ep_step: int = 0
        ep_over: bool = False

        # Training loop
        while not ep_over:
            # Agent chooses an action
            action: Action = agent.act(obs=obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action=action)

            # TODO Learn from this experience

            # Move to next state
            obs = next_obs

            # Update episode metrics
            ep_reward += float(reward)
            ep_collisions += info["n_collisions"]
            ep_collected_objects += info["n_placed_objects"]
            ep_step += 1
            ep_over = terminated or truncated or ep_step >= config.n_steps_episode

        # Log episode metrics
        logger.add_scalar(
            tag="training/reward", scalar_value=ep_reward, global_step=episode
        )
        logger.add_scalar(
            tag="training/collisions", scalar_value=ep_collisions, global_step=episode
        )
        logger.add_scalar(
            tag="training/collected objects",
            scalar_value=ep_collected_objects,
            global_step=episode,
        )

    env.close()


if __name__ == "__main__":  # pragma: no cover
    # Create training configuration from command line args
    config: Config = tyro.cli(Config)

    train(config=config)
