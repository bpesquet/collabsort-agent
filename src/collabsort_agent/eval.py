"""
Evaluate a previously trained agent.
"""

import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import tyro
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from collabsort_agent.agent_factory import create_agent
from collabsort_agent.config import Config
from collabsort_agent.metrics import EpisodeMetrics, safe_ratio


def log_summary(
    logger: SummaryWriter,
    load_dir: str,
    episode_metrics: list[EpisodeMetrics],
) -> None:
    """Log run-level summary metrics for an evaluation run."""

    rewards = [m.agent.reward + m.robot.reward for m in episode_metrics]

    logger.add_text(
        tag="hyperparameters",
        text_string="\n".join(
            (
                "| Hyperparameter | Value |",
                "| --- | --- |",
                f"| train_dir | {load_dir} |",
                f"| n_episodes | {len(episode_metrics)} |",
            )
        ),
        global_step=0,
    )

    # Log run-level summary metrics, tracing back to the evaluated training run
    logger.add_hparams(
        hparam_dict={
            "source_train_dir": load_dir,
            "n_episodes": len(episode_metrics),
        },
        metric_dict={
            "hparam/reward": mean(rewards),
            "hparam/reward_std": pstdev(rewards),
            "hparam/agent_reward": mean(m.agent.reward for m in episode_metrics),
            "hparam/collected_objects_ratio": mean(
                safe_ratio(
                    m.agent.n_collected_objects + m.robot.n_collected_objects,
                    m.n_objects,
                )
                for m in episode_metrics
            ),
        },
        run_name=".",
    )


def evaluate(load_dir: str, n_episodes: int = 20, seed: int | None = None) -> None:
    """
    Evaluate a previously trained agent over a number of episodes.

    The agent acts greedily (no exploration, no learning updates); metrics
    are logged to their own run directory, separate from training runs.
    """

    if not Path(load_dir).is_dir():
        raise NotADirectoryError(f"Invalid loading path '{load_dir}'")

    # Load configuration used for training
    config = Config()
    config.deserialize(dir=load_dir)

    # Create directory path for evaluation output, separate from training runs
    time_str = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    eval_dir = f"runs/eval_{time_str}_{config.agent.decision.algorithm}_{config.agent.learning.algorithm}"

    # Create environment
    env = gym.make(id=config.env_id, config=config.env)

    # Create agent and load its trained state
    agent = create_agent(
        config=config, sample_obs=env.observation_space.sample(), rng=env.np_random
    )
    agent.deserialize(dir=load_dir)

    # Create logger, writing to its own directory rather than the training run's
    logger = SummaryWriter(f"{eval_dir}", flush_secs=60)

    episode_metrics: list[EpisodeMetrics] = []
    start_time = time.time()
    total_steps = 0

    try:
        for episode in trange(n_episodes, desc="Evaluation progress"):
            # Seed only the first reset, so the env's own RNG stream still
            # varies episode conditions (object layouts) across the run
            obs, _ = env.reset(seed=seed if episode == 0 else None)
            ep_metrics = EpisodeMetrics()
            ep_over = False

            while not ep_over:
                # Deliberate greedily: no exploration, no learning update
                action: Action = agent.act(
                    obs=obs, training_step=None, deterministic=True
                )

                next_obs, reward, terminated, truncated, info = env.step(action=action)

                ep_metrics.agent.reward += float(reward)
                ep_metrics.agent.actions.append(action.value)
                ep_metrics.step += 1
                total_steps += 1

                obs = next_obs
                ep_over = (
                    terminated or truncated or ep_metrics.step >= config.n_steps_episode
                )

            ep_metrics.sps = total_steps / (time.time() - start_time)
            ep_metrics.n_collisions = info["n_collisions"]
            ep_metrics.n_objects = info["n_objects"]
            ep_metrics.n_missed_objects = info["n_fallen_objects"]
            ep_metrics.agent.n_collected_objects = info["n_agent_placed_objects"]
            ep_metrics.robot.n_collected_objects = info["n_robot_placed_objects"]
            ep_metrics.robot.reward = info["robot_ep_reward"]

            episode_metrics.append(ep_metrics)

            # Log per-episode metrics (reward, collisions, collected objects...).
            # Deliberator/estimator internals (e.g. exploration probability,
            # learning loss) are not logged here: no exploration or learning
            # happens during evaluation.
            ep_metrics.log(logger=logger, episode=episode)

        log_summary(logger=logger, load_dir=load_dir, episode_metrics=episode_metrics)

    finally:
        # Always release the environment and flush/close the logger
        env.close()
        logger.close()


if __name__ == "__main__":  # pragma: no cover
    # Run evaluation with command line args
    tyro.cli(evaluate)
