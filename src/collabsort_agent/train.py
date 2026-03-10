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

    # Save the trained agent
    save_agent: bool = False


@dataclass
class EpisodeMetrics:
    """Episode metrics"""

    # Cumulated reward
    reward: float = 0

    # Number of collisions
    collisions: int = 0

    # Number of collected objects
    collected_objects: int = 0

    # Episode time step (= number of time steps since beginning of episode)
    step: int = 0

    # Number of steps per second
    sps: float = 0

    def log(
        self,
        logger: SummaryWriter | None,
        episode: int,
    ) -> None:
        """Log metrics"""

        if logger is not None:
            logger.add_scalar(
                tag="training/reward", scalar_value=self.reward, global_step=episode
            )
            logger.add_scalar(
                tag="training/collisions",
                scalar_value=self.collisions,
                global_step=episode,
            )
            logger.add_scalar(
                tag="training/collected_objects",
                scalar_value=self.collected_objects,
                global_step=episode,
            )
            logger.add_scalar(
                tag="training/steps_per_seconds",
                scalar_value=self.sps,
                global_step=episode,
            )


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


def train(config: Config) -> None:
    """Train an agent"""

    run_dir: str = f"runs/train_{config.learning.algorithm}_{int(time.time())}"

    logger = None
    if config.log_events:
        # Initialize logging
        logger = SummaryWriter(f"{run_dir}")

    # Initialize environment
    env = gym.make("CollabSort-v0", config=config.env)

    # Create agent
    agent = create_agent(config=config, sample_obs=env.observation_space.sample())

    # Training time step (= number of time steps since beginning of training)
    training_step: int = 0

    start_time = time.time()

    # Global loop
    for episode in trange(config.n_episodes, desc="Training progress"):
        # Reset environment and metrics for new episode
        obs, _ = env.reset()
        ep_metrics = EpisodeMetrics()
        ep_over: bool = False

        # Episode loop
        while not ep_over:
            # Agent chooses an action
            action: Action = agent.act(obs=obs, training_step=training_step)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action=action)
            reward: float = float(reward)

            # Use this experience to update agent
            agent.update(
                next_obs=next_obs,
                reward=reward,
                done=terminated or truncated,
            )

            # Update episode metrics
            ep_metrics.reward += reward
            ep_metrics.collisions += info["n_collisions"]
            ep_metrics.collected_objects += info["n_placed_objects"]
            ep_metrics.step += 1
            training_step += 1

            # Move to next state
            obs = next_obs
            ep_over = (
                terminated or truncated or ep_metrics.step >= config.n_steps_episode
            )

        # Log episode data
        ep_metrics.sps = int(training_step / (time.time() - start_time))
        ep_metrics.log(
            logger=logger,
            episode=episode,
        )
        agent.log_episode(logger=logger, episode=episode)

    env.close()

    if config.save_agent:
        agent.save(run_dir=run_dir)


if __name__ == "__main__":  # pragma: no cover
    # Create training configuration from command line args
    config: Config = tyro.cli(Config)

    train(config=config)
