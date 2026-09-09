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


def log_hyperparameters(
    logger: SummaryWriter, config: Config, final_return: float
) -> None:
    """Log main hyperparameters and summary metric for a training run"""

    # Log hyperparameters as a markdown table
    logger.add_text(
        tag="hyperparameters",
        text_string="\n".join(
            (
                "| hyperparameter | value |",
                "| --- | --- |",
                f"| decision_algorithm | {config.agent.decision.algorithm} |",
                f"| learning_algorithm | {config.agent.learning.algorithm} |",
                f"| n_steps_episode | {config.n_steps_episode} |",
                f"| n_episodes | {config.n_episodes} |",
            )
        ),
        global_step=0,
    )

    # Log hyperparameters (kept in the same run directory via run_name=".")
    logger.add_hparams(
        hparam_dict={
            "decision_algorithm": config.agent.decision.algorithm,
            "learning_algorithm": config.agent.learning.algorithm,
        },
        metric_dict={"hparams/final_episodic_return": final_return},
        run_name=".",
    )


def train(config: Config) -> None:
    """Execute one training run of the agent"""

    if config.load_dir is not None:
        if not Path(config.load_dir).is_dir():
            raise NotADirectoryError(f"Invalid loading path '{config.load_dir}'")

        # Load configuration from previously saved run
        config.deserialize(dir=config.load_dir)

    # Create directory path for training output
    time_str = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    train_dir = f"runs/train_{time_str}_{config.agent.decision.algorithm}_{config.agent.learning.algorithm}"

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

    # Summary metric: agent's episodic return for the last completed episode
    final_return: float = 0.0

    # Create logger
    logger = SummaryWriter(f"{train_dir}", flush_secs=60)

    try:
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

                # Update episode metrics
                ep_metrics.agent.reward += reward
                ep_metrics.agent.actions.append(action.value)
                ep_metrics.n_collisions += info["n_collisions"]
                ep_metrics.n_missed_objects += info["n_fallen_objects"]
                ep_metrics.step += 1

                # Use this experience to update agent
                agent.update(
                    next_obs=next_obs,
                    reward=reward,
                    done=terminated or truncated,
                )

                # Move to next state
                training_step += 1
                obs = next_obs
                ep_over = (
                    terminated or truncated or ep_metrics.step >= config.n_steps_episode
                )

            # Compute steps per second for finished episode
            ep_metrics.sps = int(training_step / (time.time() - start_time))

            # Store symmary metric
            final_return = ep_metrics.agent.reward

            if config.save_output:
                # Log episode metrics
                ep_metrics.log(
                    logger=logger,
                    episode=episode,
                )

        if config.save_output:
            log_hyperparameters(logger, config, final_return)

            # Serialize config and agent state
            config.serialize(dir=train_dir)
            agent.serialize(dir=train_dir)

    finally:
        # Always release the environment and flush/close the logger
        env.close()
        logger.close()


if __name__ == "__main__":  # pragma: no cover
    # Load configuration from CLI
    config: Config = tyro.cli(Config)

    # Launch the training process
    train(config=config)
