"""
Train an agent.
"""

import time
from dataclasses import dataclass

import gymnasium as gym
import tyro
from gym_collabsort.config import Action
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

from collabsort_agent.config import Config, create_agent, save_cfg


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


def train(config: Config) -> None:
    """Train an agent"""

    train_dir: str = f"runs/train_{config.learning.algorithm}_{int(time.time())}"

    logger = None
    if config.log_events:
        # Initialize logging
        logger = SummaryWriter(f"{train_dir}")

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

            # Move to next state
            training_step += 1
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

    if config.save_state:
        agent.save_state(dir=train_dir)
        save_cfg(config=config, dir=train_dir)


if __name__ == "__main__":  # pragma: no cover
    # Create training configuration from command line args
    config: Config = tyro.cli(Config)

    train(config=config)
