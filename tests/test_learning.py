"""
Unit test for learning.
"""

import numpy as np
from numpy.testing import assert_almost_equal

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN
from collabsort_agent.learning.exploration_decay import (
    ExponentialExplorationDecay,
    LinearExplorationDecay,
)


def test_linear_explo_decay() -> None:
    """Test linear exploration decay"""

    config = LearningConfig(exploration_decay_span=0.6)
    total_steps = 1000

    lin_decay = LinearExplorationDecay(config=config, total_steps=total_steps)

    # Assert epsilon value at beginning of decay
    assert lin_decay.get_epsilon(training_step=0) == config.epsilon_start

    # Assert epsilon value at middle of decay
    assert (
        lin_decay.get_epsilon(
            training_step=int(total_steps * config.exploration_decay_span) // 2
        )
        == config.epsilon_min + (config.epsilon_start - config.epsilon_min) / 2
    )

    # Assert epsilon value at end of decay
    assert_almost_equal(
        lin_decay.get_epsilon(
            training_step=int(total_steps * config.exploration_decay_span)
        ),
        config.epsilon_min,
    )

    # Assert epsilon value at end of training
    assert_almost_equal(
        lin_decay.get_epsilon(training_step=total_steps), config.epsilon_min
    )


def test_exponential_explo_decay() -> None:
    """Test exponential exploration decay"""

    config = LearningConfig()
    total_steps = 1000

    exp_decay = ExponentialExplorationDecay(config=config, total_steps=total_steps)

    # Assert epsilon value at beginning of decay
    assert exp_decay.get_epsilon(training_step=0) == config.epsilon_start

    # Assert epsilon value at end of decay
    assert_almost_equal(
        exp_decay.get_epsilon(
            training_step=int(total_steps * config.exploration_decay_span)
        ),
        config.epsilon_min,
        decimal=2,
    )

    # Assert epsilon value at end of training
    assert_almost_equal(
        exp_decay.get_epsilon(training_step=total_steps), config.epsilon_min, decimal=2
    )


def test_dqn() -> None:
    """Test DQN algorithm"""

    # Learning hyperparameters
    config = LearningConfig()
    explo_decay = LinearExplorationDecay(config=config, total_steps=1000)
    initial_state = np.array([0, 2, -1, 7])
    n_actions = 4

    dqn = DQN(
        config=config,
        exploration_decay=explo_decay,
        state_size=len(initial_state),
        n_actions=n_actions,
    )

    action = dqn.choose_action(state=initial_state, training_step=0)
    assert action >= 0 and action < n_actions

    dqn.store_transition(
        state=initial_state, action=action, reward=1, next_state=initial_state * 1.1
    )
    assert len(dqn.replay_buffer) == 1

    dqn.learn()
    assert len(dqn.replay_buffer) == 1
