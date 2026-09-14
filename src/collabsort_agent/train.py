"""
Train an agent.
"""

import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import gymnasium as gym
import tyro
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from collabsort_agent.agent_factory import create_agent
from collabsort_agent.config import Config
from collabsort_agent.metrics import EpisodeMetrics, safe_ratio

# Number of trailing episodes averaged for the run-level hparam metrics
N_LAST_EPISODES = 5


def log_hyperparameters(
    logger: SummaryWriter,
    config: Config,
    recent_metrics: deque[EpisodeMetrics],
) -> None:
    """Log main hyperparameters and summary metrics for a training run."""

    # Log hyperparameters as a markdown table
    logger.add_text(
        tag="hyperparameters",
        text_string="\n".join(
            (
                "| Hyperparameter | Value |",
                "| --- | --- |",
                f"| decision_algorithm | {config.agent.decision.algorithm} |",
                f"| learning_algorithm | {config.agent.learning.algorithm} |",
                f"| n_steps_episode | {config.n_steps_episode} |",
                f"| n_episodes | {config.n_episodes} |",
            )
        ),
        global_step=0,
    )

    # Average summary metrics over the trailing episodes (0.0 if none were
    # completed, e.g. n_episodes=0)
    mean_reward = (
        mean(m.agent.reward + m.robot.reward for m in recent_metrics)
        if recent_metrics
        else 0.0
    )
    mean_agent_reward = (
        mean(m.agent.reward for m in recent_metrics) if recent_metrics else 0.0
    )
    mean_collected_objects_ratio = (
        mean(
            safe_ratio(
                m.agent.n_collected_objects + m.robot.n_collected_objects,
                m.n_objects,
            )
            for m in recent_metrics
        )
        if recent_metrics
        else 0.0
    )

    # Log hyperparameters (kept in the same run directory via run_name=".")
    logger.add_hparams(
        hparam_dict={
            "decision_algorithm": config.agent.decision.algorithm,
            "learning_algorithm": config.agent.learning.algorithm,
            "n_steps_episode": config.n_steps_episode,
            "n_episodes": config.n_episodes,
        },
        metric_dict={
            "hparam/reward": mean_reward,
            "hparam/agent_reward": mean_agent_reward,
            "hparam/collected_objects_ratio": mean_collected_objects_ratio,
        },
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

    # Create logger
    logger = SummaryWriter(f"{train_dir}", flush_secs=60)

    # Metrics for the trailing episodes, used for the run-level hparam
    # summary metrics logged at the end of training
    recent_metrics: deque[EpisodeMetrics] = deque(maxlen=N_LAST_EPISODES)

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

            # Record or compute metrics after end of episode
            ep_metrics.sps = training_step / (time.time() - start_time)
            ep_metrics.n_collisions = info["n_collisions"]
            ep_metrics.n_objects = info["n_objects"]
            ep_metrics.n_missed_objects = info["n_fallen_objects"]
            ep_metrics.agent.n_collected_objects = info["n_agent_placed_objects"]
            ep_metrics.robot.n_collected_objects = info["n_robot_placed_objects"]
            ep_metrics.robot.reward = info["robot_ep_reward"]

            recent_metrics.append(ep_metrics)

            if config.save_output:
                # Log episode metrics
                ep_metrics.log(
                    logger=logger,
                    episode=episode,
                )

        if config.save_output:
            # Serialize config and agent state
            config.serialize(dir=train_dir)
            agent.serialize(dir=train_dir)

            log_hyperparameters(logger, config, recent_metrics)

    finally:
        # Always release the environment and flush/close the logger
        env.close()
        logger.close()


if __name__ == "__main__":  # pragma: no cover
    # Load configuration from CLI
    config: Config = tyro.cli(Config)

    # Launch the training process
    train(config=config)
