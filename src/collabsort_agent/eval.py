"""
Evaluation of an agent, run at the end of a training phase.
"""

from gym_collabsort.config import Action
from tqdm import trange

from collabsort_agent.agent import Agent
from collabsort_agent.metrics import EpisodeMetrics


def eval_episode(
    agent: Agent,
    env,
    n_steps_episode: int,
    seed: int | None,
    deterministic: bool,
) -> EpisodeMetrics:
    """
    Run a single evaluation episode (no learning update).

    If deterministic is True, actions are chosen greedily (no exploration).
    Otherwise, the decision algorithm is used as-is, with its exploration
    level frozen at its latest value.
    """

    obs, _ = env.reset(seed=seed)
    ep_metrics = EpisodeMetrics()
    ep_over = False

    while not ep_over:
        # No learning update. training_step is None: exploration is not decayed
        action: Action = agent.act(
            obs=obs, training_step=None, deterministic=deterministic
        )

        next_obs, reward, terminated, truncated, info = env.step(action=action)

        ep_metrics.agent.reward += float(reward)
        ep_metrics.agent.actions.append(action.value)
        ep_metrics.step += 1

        obs = next_obs
        ep_over = terminated or truncated or ep_metrics.step >= n_steps_episode

    ep_metrics.n_collisions = info["n_collisions"]
    ep_metrics.n_objects = info["n_objects"]
    ep_metrics.n_missed_objects = info["n_fallen_objects"]
    ep_metrics.agent.n_collected_objects = info["n_agent_placed_objects"]
    ep_metrics.robot.n_collected_objects = info["n_robot_placed_objects"]
    ep_metrics.robot.reward = info["robot_ep_reward"]

    return ep_metrics


def eval(
    agent: Agent,
    env,
    n_steps_episode: int,
    n_episodes: int,
    seed: int,
    deterministic: bool,
) -> list[EpisodeMetrics]:
    """
    Run several evaluation episodes and return their metrics, so that
    they reflect actual policy performance rather than exploration-noisy training rewards.

    Only the first episode's reset is seeded, so the env's own RNG stream
    still varies episode conditions (object layouts) across the batch.
    Two runs starting from the same seed draw from that RNG in exactly the same sequence
    and get exactly the same N episodes, just not N identical episodes:
    using a fixed seed at first makes the whole batch reproducible across runs.
    """

    return [
        eval_episode(
            agent=agent,
            env=env,
            n_steps_episode=n_steps_episode,
            seed=seed if episode == 0 else None,
            deterministic=deterministic,
        )
        for episode in trange(n_episodes, desc="Evaluation progress")
    ]
