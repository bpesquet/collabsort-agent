"""
Unit tests for training.
"""

from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.train import Config as TrainingConfig
from collabsort_agent.train import train


def test_random_agent() -> None:
    """Test an agent choosing random actions"""

    # epsilon = 1 => always explore randomly
    train(
        config=TrainingConfig(
            env=EnvConfig(n_objects=100), learning=LearningConfig(epsilon=1)
        )
    )
