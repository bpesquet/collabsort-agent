"""
Greedy evaluation of an agent, run at the end of a training phase.
"""

from gym_collabsort.config import Action
from tqdm import trange

from collabsort_agent.agent import Agent
from collabsort_agent.metrics import EpisodeMetrics


def run_eval_episode(
    agent: Agent,
    env,
    n_steps_episode: int,
    seed: int | None = None,
) -> EpisodeMetrics:
    """Run a single greedy evaluation episode (no exploration, no learning update)."""

    obs, _ = env.reset(seed=seed)
    ep_metrics = EpisodeMetrics()
    ep_over = False

    while not ep_over:
        # Deliberate greedily: no exploration, no learning update
        action: Action = agent.act(obs=obs, training_step=None, deterministic=True)

        next_obs, reward, terminated, truncated, info = env.step(action=action)

        ep_metrics.agent.reward += float(reward)
        ep_metrics.step += 1

        obs = next_obs
        ep_over = terminated or truncated or ep_metrics.step >= n_steps_episode

    ep_metrics.n_objects = info["n_objects"]
    ep_metrics.agent.n_collected_objects = info["n_agent_placed_objects"]
    ep_metrics.robot.n_collected_objects = info["n_robot_placed_objects"]
    ep_metrics.robot.reward = info["robot_ep_reward"]

    return ep_metrics


def run_eval_episodes(
    agent: Agent,
    env,
    n_steps_episode: int,
    n_episodes: int,
    seed: int,
) -> list[EpisodeMetrics]:
    """
    Run several greedy evaluation episodes and return their metrics.

    Only the first episode's reset is seeded, so the env's own RNG stream
    still varies episode conditions (object layouts) across the batch; using
    a fixed seed makes the whole batch reproducible across runs, for
    objective comparison between agents.
    """

    return [
        run_eval_episode(
            agent=agent,
            env=env,
            n_steps_episode=n_steps_episode,
            seed=seed if episode == 0 else None,
        )
        for episode in trange(n_episodes, desc="Evaluation progress")
    ]
