"""
Unit tests for training.
"""

from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.memory import Config as MemoryConfig
from collabsort_agent.perception import Config as PerceptionConfig
from collabsort_agent.train import Config as TrainingConfig
from collabsort_agent.train import train


def test_random_agent() -> None:
    """Test an agent choosing random actions"""

    train(
        config=TrainingConfig(
            env=EnvConfig(),
            perception=PerceptionConfig(),
            memory=MemoryConfig(),
            # epsilon = 1 => always explore randomly
            learning=LearningConfig(epsilon_start=1, epsilon_min=1),
            n_episodes=10,
            n_steps_episode=100,
            log_events=False,
        )
    )
