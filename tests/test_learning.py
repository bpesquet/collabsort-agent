"""
Unit test for learning.
"""

import math

import numpy as np

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN


def test_dqn() -> None:
    """Test DQN algorithm"""

    # Learning hyperparameters
    initial_state = np.array([0, 2, -1, 7])
    n_actions = 4

    dqn = DQN(
        config=LearningConfig(), state_size=len(initial_state), n_actions=n_actions
    )

    action = dqn.choose_action(state=initial_state)
    assert action >= 0 and action < n_actions

    dqn.store_transition(
        state=initial_state, action=action, reward=1, next_state=initial_state * 1.1
    )
    assert len(dqn.replay_buffer) == 1

    dqn.learn()
    assert dqn.epsilon == math.pow(dqn.config.beta, 2)
