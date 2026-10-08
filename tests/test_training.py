"""
Unit tests for training.
"""

import gymnasium as gym
import pytest
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.agent import AgentConfig
from collabsort_agent.agent_factory import create_agent
from collabsort_agent.config import Config
from collabsort_agent.decision import DecisionConfig
from collabsort_agent.eval import eval
from collabsort_agent.learning import LearningConfig
from collabsort_agent.memory import MemoryConfig
from collabsort_agent.metacognition import MetaConfig
from collabsort_agent.perception import PerceptionConfig
from collabsort_agent.train import get_decision_name, get_meta_name, train


def _make_config() -> Config:
    """Helper to build a minimal, fast training configuration."""

    agent_config = AgentConfig(
        perception=PerceptionConfig(),
        memory=MemoryConfig(),
        # Fully random agent to keep the run cheap and deterministic
        decision=DecisionConfig(epsilon_start=1, epsilon_min=1),
        learning=LearningConfig(),
        meta=MetaConfig(),
    )
    return Config(
        env=EnvConfig(),
        agent=agent_config,
        n_training_episodes=2,
        n_steps_episode=20,
        save_output=False,
    )


def test_total_steps() -> None:
    """The total number of training steps is episodes * steps per episode."""

    config = Config(n_training_episodes=3, n_steps_episode=10)
    assert config.total_steps == 30


def test_train(tmp_path, monkeypatch) -> None:
    """A short training run completes and serializes the config and agent state."""

    monkeypatch.chdir(tmp_path)
    config = _make_config()

    train(config=config)


@pytest.mark.parametrize("confidence_method", ["gap", "bayesian", "qgap"])
def test_train_ard(tmp_path, monkeypatch, confidence_method) -> None:
    """A short training run completes with ARD decisions and each confidence method."""

    monkeypatch.chdir(tmp_path)
    config = _make_config()
    config.agent.decision.algorithm = "ard"
    config.agent.meta.confidence_method = confidence_method

    train(config=config)


@pytest.mark.parametrize("eval_mode", ["greedy", "policy"])
@pytest.mark.parametrize("decision", ["eps", "ard"])
def test_train_with_output(tmp_path, monkeypatch, decision, eval_mode) -> None:
    """A short training run with evaluation and logging writes a seeded run directory."""

    monkeypatch.chdir(tmp_path)
    config = _make_config()
    config.agent.decision.algorithm = decision
    config.eval_mode = eval_mode
    config.n_eval_episodes = 2
    config.save_output = True
    config.training_seed = 123

    train(config=config)

    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("_s123")


def test_train_draws_seed(tmp_path, monkeypatch) -> None:
    """Without an explicit seed, one is drawn so that the run can be reproduced."""

    monkeypatch.chdir(tmp_path)
    config = _make_config()
    assert config.training_seed is None

    train(config=config)

    assert isinstance(config.training_seed, int)


def test_get_decision_name() -> None:
    """The decision name includes the accumulation type for ARD only."""

    config = _make_config()
    assert get_decision_name(config) == "eps"

    config.agent.decision.algorithm = "ard"
    config.agent.decision.accumulation = "euler"
    assert get_decision_name(config) == "ard-euler"


def test_get_meta_name() -> None:
    """The metacognition name is defined for all decision algorithms."""

    config = _make_config()
    assert get_meta_name(config) == "none"

    config.agent.decision.algorithm = "ard"
    config.agent.meta.confidence_method = "qgap"
    assert get_meta_name(config) == "qgap"


@pytest.mark.parametrize("deterministic", [True, False])
def test_eval_records_episode_metrics(deterministic) -> None:
    """Evaluation records the same episode metrics as training."""

    config = _make_config()
    env = gym.make(id=config.env_id, config=config.env)
    agent = create_agent(
        config=config,
        sample_obs=env.observation_space.sample(),
        rng=env.np_random,
    )

    (metrics,) = eval(
        agent=agent,
        env=env,
        n_steps_episode=config.n_steps_episode,
        n_episodes=1,
        seed=0,
        deterministic=deterministic,
    )
    env.close()

    assert metrics.step == config.n_steps_episode
    assert len(metrics.agent.actions) == metrics.step
    assert metrics.n_collisions >= 0
    assert metrics.n_missed_objects >= 0


def test_train_from_pretrained(tmp_path, monkeypatch) -> None:
    """Training can resume from a previously saved config and agent state."""

    monkeypatch.chdir(tmp_path)
    config = _make_config()

    # 1. Save an initial config and agent state to a directory
    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()

    env = gym.make(id=config.env_id, config=config.env)
    agent = create_agent(
        config=config,
        sample_obs=env.observation_space.sample(),
        rng=env.np_random,
    )
    config.serialize(dir=str(pretrained_dir))
    agent.serialize(dir=str(pretrained_dir))
    env.close()

    # 2. Resume training from the pretrained state
    config.load_dir = str(pretrained_dir)
    train(config=config)


def test_save_load_config(tmp_path) -> None:
    """A configuration round-trips through disk serialization."""

    config = _make_config()
    config.serialize(dir=str(tmp_path))

    loaded = Config()
    loaded.deserialize(dir=str(tmp_path))

    assert loaded.agent == config.agent
    assert loaded.env == config.env
