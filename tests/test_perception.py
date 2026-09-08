"""
Unit tests for perception.
"""

import gymnasium as gym
import numpy as np
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.config import AgentConfig
from collabsort_agent.perception import Perceiver, PerceptionConfig


def make_perceiver(
    n_perceived_cols: int = 3, n_objects: int = 1
) -> tuple[Perceiver, EnvConfig]:
    """Helper function to create a Percevier object."""

    env_config = EnvConfig(n_objects=n_objects)
    perceiver = Perceiver(
        config=PerceptionConfig(n_perceived_cols=n_perceived_cols),
        treadmill_rows=[env_config.upper_treadmill_row, env_config.lower_treadmill_row],
    )
    return perceiver, env_config


def sample_obs(env_config: EnvConfig) -> dict:
    """Helper function to sample an observation from the environment."""

    env = gym.make(id=AgentConfig().env_id, config=env_config)
    # obs: dict = env.observation_space.sample()
    obs = env.observation_space.sample()
    env.close()

    return obs


def test_state_size() -> None:
    for n_perceived_cols in (1, 3, 6):
        # Create perceiver and sample observation
        perceiver, env_config = make_perceiver(n_perceived_cols=n_perceived_cols)
        obs = sample_obs(env_config=env_config)

        sensory_state = perceiver.get_sensory_state(obs=obs)

        # Check that sensory state is a vector with the expected number of features:
        # - 4 for agent (coords + presence of a picked object + collision penalty state)
        # - 4 for robot (same)
        # - 3 for each perceived position (presence, color and shape of the object)
        assert sensory_state.ndim == 1
        expected_len = (
            4
            + 4
            + (len(perceiver.treadmill_rows) * perceiver.config.n_perceived_cols * 3)
        )
        assert len(sensory_state) == expected_len


def test_state_format() -> None:
    # Create perceiver and sample observation
    perceiver, env_config = make_perceiver(n_perceived_cols=9)
    obs = sample_obs(env_config=env_config)

    sensory_state = perceiver.get_sensory_state(obs=obs)

    # Check agent coordinates
    agent_row, agent_col = obs["self"]["coords"]
    assert sensory_state[0] == agent_row
    assert sensory_state[1] == agent_col

    # Check picked object flag
    assert sensory_state[2] == obs["self"]["picked_object"]

    # Check collision penalty flag
    assert sensory_state[3] == obs["self"]["collision_penalty"]

    # Check robot coordinates
    robot_row, robot_col = obs["robot"]["coords"]
    assert sensory_state[4] == robot_row
    assert sensory_state[5] == robot_col

    # Check picked object flag
    assert sensory_state[6] == obs["robot"]["picked_object"]

    # Check collision penalty flag
    assert sensory_state[7] == obs["robot"]["collision_penalty"]

    # Check object presence flag.
    # Object slots start at index 8, every 3rd value is the presence flag
    presence_indices = range(8, len(sensory_state), 3)
    for i in presence_indices:
        assert sensory_state[i] in (0, 1), (
            f"Presence flag at index {i} should be 0 or 1"
        )

    # Check state consistency for same observation
    s1 = perceiver.get_sensory_state(obs=obs)
    s2 = perceiver.get_sensory_state(obs=obs)
    np.testing.assert_array_equal(s1, s2)


def test_state_content() -> None:
    """Test perception of a predefined, valid observation"""

    # Perceiver with 3 perceived columns and treadmill rows 4 (upper) and 8 (lower)
    perceiver, _ = make_perceiver(n_perceived_cols=3)
    assert perceiver.treadmill_rows == [4, 8]

    # Hand-crafted, valid observation.
    obs = {
        "self": {
            "coords": (11, 4),
            "picked_object": 1,
            "collision_penalty": 0,
        },
        "robot": {
            "coords": (2, 4),
            "picked_object": 0,
            "collision_penalty": 1,
        },
        "moving_objects": (
            {"coords": (4, 6), "color": 2, "shape": 1},
            {"coords": (8, 4), "color": 0, "shape": 0},
            {"coords": (8, 6), "color": 1, "shape": 2},
            {
                "coords": (8, 7),
                "color": 0,
                "shape": 1,
            },  # Not "seen" with 3 perceived columns
        ),
    }

    sensory_state = perceiver.get_sensory_state(obs=obs)

    expected = np.array(
        [
            # Agent: row, col, picked_object, collision_penalty
            11,
            4,
            1,
            0,
            # Robot: row, col, picked_object, collision_penalty
            2,
            4,
            0,
            1,
            # Upper treadmill (row 4)
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            2,
            1,
            # Lower treadmill (row 8)
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            2,
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(sensory_state, expected)
