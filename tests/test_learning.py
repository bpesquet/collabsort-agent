"""
Unit test for learning.
"""

import numpy as np
import torch
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
    initial_state = np.array([0, 2, -1, 7], dtype=np.float32)
    n_actions = 4

    dqn = DQN(
        config=config,
        exploration_decay=LinearExplorationDecay(config=config, total_steps=1000),
        state_size=len(initial_state),
        n_actions=n_actions,
    )

    action = dqn.choose_action(state=initial_state, training_step=0)

    assert isinstance(action, int), f"Expected int, got {type(action)}"
    assert 0 <= action < n_actions

    dqn.store_transition(
        state=initial_state, action=action, reward=1, next_state=initial_state * 1.1
    )
    assert len(dqn.replay_buffer) == 1

    dqn.learn()
    assert len(dqn.replay_buffer) == 1


def test_dqn_target_network_syncs() -> None:
    """Target network should be synced to online network after sync_freq learn() calls"""

    config = LearningConfig(
        target_network_sync_freq=5, batch_size=4, replay_buffer_size=100
    )
    state = np.zeros(4, dtype=np.float32)
    n_actions = 2

    dqn = DQN(
        config=config,
        exploration_decay=LinearExplorationDecay(config=config, total_steps=1000),
        state_size=4,
        n_actions=n_actions,
    )

    # Fill replay buffer above batch_size threshold
    for i in range(config.batch_size):
        dqn.store_transition(state=state, action=0, reward=float(i), next_state=state)

    # Run enough learn steps to trigger at least one sync
    for _ in range(config.target_network_sync_freq):
        dqn.learn()

    # After sync, target and online networks should have identical weights
    for p_online, p_target in zip(
        dqn.q_network.parameters(), dqn.target_network.parameters(), strict=True
    ):
        assert (p_online.data == p_target.data).all(), "Target network not synced"


def test_dqn_save_and_load(tmp_path) -> None:
    """Saving and loading should restore DQN internal state"""

    config = LearningConfig(lr=1e-3)
    state = np.zeros(4, dtype=np.float32)

    dqn = DQN(
        config=config,
        exploration_decay=LinearExplorationDecay(config=config, total_steps=1000),
        state_size=4,
        n_actions=2,
    )

    with torch.no_grad():
        for i, param in enumerate(dqn.q_network.parameters()):
            param.fill_(0.1 * (i + 1))
        for i, param in enumerate(dqn.target_network.parameters()):
            param.fill_(0.2 * (i + 1))

    saved_lr = 5e-4
    dqn.optimizer.param_groups[0]["lr"] = saved_lr
    dqn.epsilon = 0.42

    run_dir = tmp_path / "save_roundtrip"
    dqn.save(dir=str(run_dir))

    restored = DQN(
        config=config,
        exploration_decay=LinearExplorationDecay(config=config, total_steps=1000),
        state_size=len(state),
        n_actions=2,
    )
    restored.load(dir=str(run_dir))

    for p_saved, p_restored in zip(
        dqn.q_network.parameters(), restored.q_network.parameters(), strict=True
    ):
        assert torch.equal(p_saved, p_restored)

    for p_saved, p_restored in zip(
        dqn.target_network.parameters(),
        restored.target_network.parameters(),
        strict=True,
    ):
        assert torch.equal(p_saved, p_restored)

    assert restored.optimizer.param_groups[0]["lr"] == saved_lr
    assert restored.epsilon == dqn.epsilon
