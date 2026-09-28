"""
Train an agent.
"""

import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import gymnasium as gym
import tyro
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from collabsort_agent.agent import Agent
from collabsort_agent.agent_factory import create_agent
from collabsort_agent.config import Config
from collabsort_agent.eval import eval
from collabsort_agent.metrics import EpisodeMetrics, safe_ratio

# File used to persist the training step count across saved runs
TRAINING_STEP_FILENAME = "training_step.txt"


def log_hyperparameters(
    logger: SummaryWriter,
    config: Config,
    eval_metrics: list[EpisodeMetrics],
) -> None:
    """
    Log main hyperparameters and summary metrics for a training run.

    Summary metrics (reward, agent_reward, collected_objects_ratio) are
    computed only from the end-of-training evaluation phase, not from
    training episodes, since training rewards are contaminated by exploration.
    """

    # Log hyperparameters as a markdown table
    logger.add_text(
        tag="hyperparameters",
        text_string="\n".join(
            (
                "| Hyperparameter | Value |",
                "| --- | --- |",
                f"| decision | {config.agent.decision.algorithm} |",
                f"| learning | {config.agent.learning.algorithm} |",
                f"| n_training_steps | {config.total_steps} |",
                f"| n_eval_steps | {config.n_eval_episodes * config.n_steps_episode} |",
            )
        ),
        global_step=0,
    )

    # Log hyperparameters (kept in the same run directory via run_name=".")
    logger.add_hparams(
        hparam_dict={
            "decision": config.agent.decision.algorithm,
            "learning": config.agent.learning.algorithm,
        },
        # Metrics are only computed during evaluation
        metric_dict={
            "hparam/reward": mean(
                m.agent.reward + m.robot.reward for m in eval_metrics
            ),
            "hparam/agent_reward": mean(m.agent.reward for m in eval_metrics),
            "hparam/collected_objects_ratio": mean(
                safe_ratio(
                    m.agent.n_collected_objects + m.robot.n_collected_objects,
                    m.n_objects,
                )
                for m in eval_metrics
            ),
        },
        run_name=".",
    )


def train_episode(
    agent: Agent,
    env,
    n_steps_episode: int,
    training_step: int,
) -> EpisodeMetrics:
    """
    Run a single training episode (with exploration and learning updates).

    training_step is the global step count at the start of the episode;
    the caller is responsible for advancing it by the returned metrics' step count.
    """

    obs, _ = env.reset()
    ep_metrics = EpisodeMetrics()
    ep_over = False

    while not ep_over:
        # Agent chooses an action
        action: Action = agent.act(
            obs=obs,
            training_step=training_step + ep_metrics.step,
        )

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action=action)
        reward: float = float(reward)

        ep_metrics.agent.reward += reward
        ep_metrics.agent.actions.append(action.value)
        ep_metrics.step += 1

        # Use this experience to update agent
        agent.update(
            next_obs=next_obs,
            reward=reward,
            done=terminated or truncated,
        )

        obs = next_obs
        ep_over = terminated or truncated or ep_metrics.step >= n_steps_episode

    ep_metrics.n_collisions = info["n_collisions"]
    ep_metrics.n_objects = info["n_objects"]
    ep_metrics.n_missed_objects = info["n_fallen_objects"]
    ep_metrics.agent.n_collected_objects = info["n_agent_placed_objects"]
    ep_metrics.robot.n_collected_objects = info["n_robot_placed_objects"]
    ep_metrics.robot.reward = info["robot_ep_reward"]

    return ep_metrics


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

    # Create training environment
    training_env = gym.make(id=config.env_id, config=config.env)

    # Create agent
    agent = create_agent(
        config=config,
        sample_obs=training_env.observation_space.sample(),
        rng=training_env.np_random,
    )

    # Number of time steps since beginning of training
    training_step: int = 0
    if config.load_dir is not None:
        # Load agent state from previously saved run
        agent.deserialize(dir=config.load_dir)

        # Restore training step count from previous training run.
        # Necessary for coherent exploration probability decaying
        training_step_file = Path(config.load_dir) / TRAINING_STEP_FILENAME
        if training_step_file.is_file():
            training_step = int(training_step_file.read_text())

    start_time = time.time()

    # Create logger
    logger = SummaryWriter(f"{train_dir}", flush_secs=60)

    # Separate environment for the end-of-training evaluation phase, created
    # only when that phase actually runs (i.e. save_output is enabled)
    eval_env = None

    try:
        # Global loop
        for episode in trange(config.n_training_episodes, desc="Training progress"):
            ep_metrics = train_episode(
                agent=agent,
                env=training_env,
                n_steps_episode=config.n_steps_episode,
                training_step=training_step,
            )

            # Advance global step count and compute throughput
            training_step += ep_metrics.step
            ep_metrics.sps = training_step / (time.time() - start_time)

            if config.save_output:
                # Log episode metrics
                ep_metrics.log(
                    logger=logger,
                    episode=episode,
                )
                # Log internal agent information
                agent.log_episode(logger, episode)

        if config.save_output:
            # Serialize config and agent state
            config.serialize(dir=train_dir)
            agent.serialize(dir=train_dir)
            # Serialize training step count
            (Path(train_dir) / TRAINING_STEP_FILENAME).write_text(str(training_step))

            # Create separate environment for end-of-training evaluation phase
            eval_env = gym.make(id=config.env_id, config=config.env)

            # Evaluation phase
            eval_metrics = eval(
                agent=agent,
                env=eval_env,
                n_steps_episode=config.n_steps_episode,
                n_episodes=config.n_eval_episodes,
                seed=config.eval_seed,
            )

            log_hyperparameters(logger, config, eval_metrics)

    finally:
        # Always release the environments and flush/close the logger
        training_env.close()
        if eval_env is not None:
            eval_env.close()
        logger.close()


if __name__ == "__main__":  # pragma: no cover
    # Load configuration from CLI
    config: Config = tyro.cli(Config)

    # Launch the training process
    train(config=config)
