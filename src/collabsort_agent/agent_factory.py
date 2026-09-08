"""
Definitions for instancing an agent.
"""

import numpy as np
from gym_collabsort.config import Action

from collabsort_agent.agent import Agent
from collabsort_agent.config import Config
from collabsort_agent.decision.ard import ARD
from collabsort_agent.decision.decision_rule import WinAllRule
from collabsort_agent.decision.epsilon_greedy import EpsilonGreedy
from collabsort_agent.decision.exploration_decay import (
    ExponentialExplorationDecay,
    LinearExplorationDecay,
)
from collabsort_agent.learning.dd_dqn import DoubleDuelingDQN
from collabsort_agent.learning.double_dqn import DoubleDQN
from collabsort_agent.learning.dqn import DQN
from collabsort_agent.learning.dueling_dqn import DuelingDQN
from collabsort_agent.learning.n_step_learning import NStepLearning
from collabsort_agent.learning.per import PER
from collabsort_agent.learning.q_learning import Qlearning
from collabsort_agent.memory import Memory
from collabsort_agent.metacognition import Hyperparameters
from collabsort_agent.metacognition.confidence import (
    BayesianConfidence,
    GapConfidence,
    TDErrorCalibration,
)
from collabsort_agent.metacognition.controller import MetaController
from collabsort_agent.metacognition.monitoring import MetaMonitoring
from collabsort_agent.perception import Perceiver


def create_agent(config: Config, sample_obs: dict, rng: np.random.Generator) -> Agent:
    """Create an agent with a specific configuration"""

    # Build perception module
    perceiver = Perceiver(
        config=config.agent.perception, treadmill_rows=config.env.treadmill_rows
    )

    # Build memory module
    n_memory_actions: int = 0
    memory_type: str = config.agent.memory.type
    if memory_type == "none":
        memory: Memory = Memory()
        n_memory_actions = len(memory.get_actions())
    else:
        raise ValueError(f"Unrecognized memory type: {memory_type}")

    # Initialize metacognition & dimensions
    hyperparameters = Hyperparameters(
        decision_cfg=config.agent.decision, learning_cfg=config.agent.learning
    )
    meta_ctrl = MetaController(
        config=config.agent.meta,
        learning_cfg=config.agent.learning,
        decision_cfg=config.agent.decision,
        hyperparameters=hyperparameters,
    )

    sample_sensory_state = perceiver.get_sensory_state(obs=sample_obs)
    sample_extended_state = (
        memory.get_extended_state(sensory_state=sample_sensory_state),
    )
    extended_state_size = len(sample_extended_state)
    n_actions = len(Action) + n_memory_actions

    # Dynamic build
    estimator = _build_estimator(
        config.agent.learning.algorithm,
        config,
        n_actions,
        extended_state_size,
        hyperparameters,
    )
    deliberator = _build_deliberator(
        config.agent.decision.algorithm,
        config,
        estimator,
        rng,
        hyperparameters,
        meta_ctrl,
    )

    return Agent(perceiver=perceiver, memory=memory, deliberator=deliberator)


def _build_estimator(
    algo_name: str,
    config: Config,
    n_actions: int,
    state_size: int,
    hyperparameters: Hyperparameters,
):
    """Factory helper to build the value estimator dynamically."""
    c_learn = config.agent.learning

    if algo_name == "ql":
        return Qlearning(
            config=c_learn, n_actions=n_actions, hyperparameters=hyperparameters
        )
    elif algo_name == "dqn":
        return DQN(config=c_learn, n_actions=n_actions, state_size=state_size)
    elif algo_name == "dueling_dqn":
        return DuelingDQN(config=c_learn, n_actions=n_actions, state_size=state_size)
    elif algo_name == "ddqn":
        return DoubleDQN(config=c_learn, n_actions=n_actions, state_size=state_size)
    elif algo_name == "dd_dqn":
        return DoubleDuelingDQN(
            config=c_learn, n_actions=n_actions, state_size=state_size
        )
    elif algo_name == "per":
        return PER(config=c_learn, n_actions=n_actions, state_size=state_size)
    elif algo_name == "n_step":
        return NStepLearning(
            config=c_learn,
            n_actions=n_actions,
            state_size=state_size,
            n_step=c_learn.n_step,
        )

    raise ValueError(f"Unrecognized learning algorithm: {algo_name}")


def _build_deliberator(
    algo_name: str,
    config: Config,
    estimator,
    rng: np.random.Generator,
    hyperparameters: Hyperparameters,
    meta_ctrl: MetaController,
):
    """Factory helper to build the deliberator dynamically."""
    if algo_name == "eps":
        if config.agent.decision.exploration_decay == "lin":
            decay = LinearExplorationDecay(
                config=config.agent.decision, total_steps=config.total_steps
            )
        elif config.agent.decision.exploration_decay == "exp":
            decay = ExponentialExplorationDecay(
                config=config.agent.decision, total_steps=config.total_steps
            )
        else:
            raise ValueError(
                f"Unrecognized exploration decay: {config.agent.decision.exploration_decay}"
            )

        return EpsilonGreedy(
            config=config.agent.decision,
            estimator=estimator,
            exploration_decay=decay,
            rng=rng,
        )

    if algo_name == "ard":
        decision_rule = (
            WinAllRule(rng=rng)
            if config.agent.decision.decision_rule == "win-all"
            else None
        )

        if config.agent.meta.confidence_method == "gap":
            confidence_method = GapConfidence(
                decision_cfg=config.agent.decision, hyperparameters=hyperparameters
            )
        elif config.agent.meta.confidence_method == "bayesian":
            confidence_method = BayesianConfidence(
                decision_cfg=config.agent.decision, hyperparameters=hyperparameters
            )
        else:
            raise ValueError(
                f"Unrecognized confidence method: {config.agent.meta.confidence_method}"
            )
        if config.agent.meta.confidence_calibration_method == "none":
            calibration_method = None
        elif config.agent.meta.confidence_calibration_method == "td_error":
            calibration_method = TDErrorCalibration(config=config.agent.meta)
        else:
            raise ValueError(
                "Unrecognized confidence calibration method: "
                f"{config.agent.meta.confidence_calibration_method}"
            )

        meta_monitoring = MetaMonitoring(
            config=config.agent.meta,
            confidence_method=confidence_method,
            calibration_method=calibration_method,
        )

        return ARD(
            config=config.agent.decision,
            estimator=estimator,
            decision_rule=decision_rule,
            hyperparameters=hyperparameters,
            meta_monitoring=meta_monitoring,
            meta_ctrl=meta_ctrl,
            rng=rng,
        )

    raise ValueError(f"Unrecognized decision algorithm: {algo_name}")
