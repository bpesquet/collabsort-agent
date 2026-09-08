"""
Code for demoing a previously trained agent.
"""

from pathlib import Path

import gymnasium as gym
import tyro
from gym_collabsort.config import Action, RenderMode

from collabsort_agent.config import Config
from collabsort_agent.train import create_agent


def demo(load_dir: str) -> None:
    """Demonstrates a previously trained agent"""

    if not Path(load_dir).is_dir():
        raise NotADirectoryError(f"Invalid loading path '{load_dir}'")

    # Load config used for training
    config = Config()
    config.deserialize(dir=load_dir)

    # Switch configuration to demo mode
    config.env.render_mode = RenderMode.HUMAN

    # Initialize environment
    env = gym.make(id=config.env_id, config=config.env)

    # Create agent and load its state from disk
    agent = create_agent(
        config=config, sample_obs=env.observation_space.sample(), rng=env.np_random
    )
    agent.deserialize(dir=load_dir)

    # Reset environment
    obs, _ = env.reset()
    ep_over: bool = False

    # Episode loop
    while not ep_over:
        # Agent chooses an action
        action: Action = agent.act(obs=obs, training_step=0)

        # Take action and observe result
        next_obs, _, terminated, truncated, _ = env.step(action=action)

        # Move to next state
        obs = next_obs
        ep_over = terminated or truncated

    env.close()


if __name__ == "__main__":  # pragma: no cover
    # Run demo with command line args
    tyro.cli(demo)
