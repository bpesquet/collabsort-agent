"""
Unit tests for agent.
"""

import gym_collabsort
import gymnasium as gym
from gym_collabsort.config import Config
from gym_collabsort.envs.env import RenderMode


def test_random_agent() -> None:
    """Test an agent choosing random actions"""

    print(f"Using gym-collabsort {gym_collabsort.__version__}")

    config = Config(n_objects=30)

    env = gym.make("CollabSort-v0", render_mode=RenderMode.HUMAN, config=config)
    env.reset()

    ep_over: bool = False
    while not ep_over:
        _, _, terminated, truncated, _ = env.step(action=env.action_space.sample())
        ep_over = terminated or truncated

    env.close()
