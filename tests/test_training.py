"""
Unit tests for training.
"""

import gymnasium as gym
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.agent import AgentConfig
from collabsort_agent.agent_factory import create_agent
from collabsort_agent.config import Config
from collabsort_agent.decision import DecisionConfig
from collabsort_agent.learning import LearningConfig
from collabsort_agent.memory import MemoryConfig
from collabsort_agent.metacognition import MetaConfig
from collabsort_agent.perception import PerceptionConfig
from collabsort_agent.train import train


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
        n_episodes=2,
        n_steps_episode=20,
        save_output=False,
    )


def test_total_steps() -> None:
    """The total number of training steps is episodes * steps per episode."""

    config = Config(n_episodes=3, n_steps_episode=10)
    assert config.total_steps == 30


def test_train(tmp_path, monkeypatch) -> None:
    """A short training run completes and serializes the config and agent state."""

    monkeypatch.chdir(tmp_path)
    config = _make_config()

    train(config=config)


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
