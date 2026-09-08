"""
Train an agent.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import gymnasium as gym
import tyro
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from collabsort_agent.agent_factory import create_agent
from collabsort_agent.config import Config
from collabsort_agent.metrics import EpisodeMetrics


def train(config: Config) -> None:
    """Execute one training run of the agent"""

    # Create directory path for training output
    time_str = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    train_dir = f"runs/train_{time_str}_{config.agent.decision.algorithm}_{config.agent.learning.algorithm}"

    # Create logger
    logger = SummaryWriter(f"{train_dir}", flush_secs=60)

    if config.load_dir is not None:
        if not Path(config.load_dir).is_dir():
            raise NotADirectoryError(f"Invalid loading path '{config.load_dir}'")

        # Load configuration from previously saved run
        config.deserialize(dir=config.load_dir)

    # Create environment
    env = gym.make(id=config.env_id, config=config.env)

    # Create agent
    agent = create_agent(
        config=config,
        sample_obs=env.observation_space.sample(),
        rng=env.np_random,
    )

    if config.load_dir is not None:
        # Load agent state from previously saved run
        agent.deserialize(dir=config.load_dir)

    # Initialize time-related values
    training_step: int = 0  # Number of time steps since beginning of training
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
            action: Action = agent.act(
                obs=obs,
                training_step=training_step,
            )

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
            ep_metrics.agent.reward += reward
            ep_metrics.agent.actions.append(action.value)
            ep_metrics.n_collisions += info["n_collisions"]
            ep_metrics.n_missed_objects += info["n_fallen_objects"]
            ep_metrics.step += 1

            # Move to next state
            training_step += 1
            obs = next_obs
            ep_over = (
                terminated or truncated or ep_metrics.step >= config.n_steps_episode
            )

        ep_metrics.sps = int(training_step / (time.time() - start_time))

        if config.save_output:
            # Log episode metrics
            ep_metrics.log(
                logger=logger,
                episode=episode,
            )

    env.close()
    logger.close()

    if config.save_output:
        # Serialize config and agent state
        config.serialize(dir=train_dir)
        agent.serialize(dir=train_dir)


if __name__ == "__main__":  # pragma: no cover
    # Load configuration from CLI
    config: Config = tyro.cli(Config)

    # Launch the training process
    train(config=config)
