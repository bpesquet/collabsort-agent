"""
Unit tests for the Advantage Racing Diffusion (ARD) deliberator.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from collabsort_agent.decision import DecisionConfig
from collabsort_agent.decision.ard import ARD
from collabsort_agent.decision.decision_rule import WinAllRule
from collabsort_agent.decision.exploration_decay import (
    ExplorationDecay,
    LinearExplorationDecay,
)
from collabsort_agent.learning import ActionValueEstimator, LearningConfig
from collabsort_agent.metacognition import Hyperparameters, MetaConfig
from collabsort_agent.metacognition.confidence import BayesianConfidence
from collabsort_agent.metacognition.controller import MetaController
from collabsort_agent.metacognition.monitoring import MetaMonitoring


class EstimatorStub(ActionValueEstimator):
    """Estimator stub returning fixed Q-values"""

    def __init__(self, action_values: np.ndarray) -> None:
        super().__init__(config=LearningConfig(), n_actions=len(action_values))

        self._action_values = action_values

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        return self._action_values

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        pass

    def save_state(self, dir: str) -> None:
        pass

    def load_state(self, dir: str) -> None:
        pass


class MetaStub(MetaController):
    """Metacognition stub"""

    def __init__(
        self, decision_cfg: DecisionConfig, hyperparameters: Hyperparameters
    ) -> None:
        super().__init__(
            # No update of hyperparameters
            config=MetaConfig(alpha_rate=0.0, theta_rate=0.0),
            learning_cfg=LearningConfig(),
            decision_cfg=decision_cfg,
            hyperparameters=hyperparameters,
        )
        self.confidence: float = 0.0
        self.rt: float = 0.0

    def update_hyperparameters(self, confidence: float, reaction_time: float) -> None:
        # Store received values for latter assertion
        self.confidence = confidence
        self.rt = reaction_time


class TestARD:
    def test_chosen_action(self, show_plots: bool = False) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_matrix = [[0.0, 1.0, 0.0], [0.5, 1.0, 0.2], [0.05, 0.2, 0.1]]

        for accumulation, q_values in itertools.product(("wald", "euler"), q_matrix):
            q_values = np.array(q_values)
            ard, _ = self._make_ard(
                action_values=q_values,
                config=DecisionConfig(accumulation=accumulation),
            )
            action = ard.choose_action(state=state, training_step=0)

            # Evidence history is only recorded by step-by-step accumulation
            show_plots = show_plots and accumulation == "euler"

            if show_plots:
                # Plot all accumulators
                lines = plt.plot(ard.accumulators.evidence_history.T)

                # Plot decision threashold
                plt.hlines(
                    ard.hyperparameters.theta,
                    xmin=plt.xlim()[0],
                    xmax=plt.xlim()[1],
                    linestyles="dotted",
                )
                plt.annotate(
                    "Decision threshold",
                    xy=(plt.xlim()[1] / 4, ard.hyperparameters.theta * 1.02),
                )

                # Plot decision time
                plt.vlines(
                    len(ard.accumulators.evidence_history[0]) - 1,
                    ymin=plt.ylim()[0],
                    ymax=plt.ylim()[1],
                    linestyles="dotted",
                )

                # Add labels for each accumulator
                for line_i, (i, j) in enumerate(ard.accumulators.action_pairs):
                    lines[line_i].set_label(f"({i},{j})")

                # Change line style for accumulators (0,*) and (2,*).
                # (Should be improved from n_actions != 3)
                for line in lines[:2]:
                    line.set_linestyle("dashed")
                for line in lines[4:]:
                    line.set_linestyle("dashdot")

                plt.xlabel("Time steps (decision)")
                plt.ylabel("Evidence")
                plt.legend()
                plt.title(f"Accumulator race ({len(q_values)} actions)")

                plt.show()

            # Given the q_values, the following assertion should apply in all cases
            assert action == np.argmax(q_values)

    def test_chosen_action_scale_invariance(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([0.0, 2.0, 1.0, 0.5])

        # Same Q-value gaps at very different scales and signs:
        # normalized drift rates should not depend on them
        for offset, scale in [(0.0, 1.0), (-20.0, 1.0), (-200.0, 10.0), (50.0, 10.0)]:
            ard, meta_stub = self._make_ard(
                action_values=offset + scale * q_values, config=DecisionConfig()
            )
            action = ard.choose_action(state=state, training_step=0)

            assert action == np.argmax(q_values)
            # Decision terminated before reaching the maximal decision time
            assert meta_stub.rt < ard.config.max_steps

    def test_advantage_weight_annealing(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([0.5, 1.0, 0.2])
        config = DecisionConfig()
        total_steps = 1000
        decay = LinearExplorationDecay(config=config, total_steps=total_steps)

        ard, _ = self._make_ard(
            action_values=q_values, config=config, exploration_decay=decay
        )

        # Start of training: no advantage, all drift rates equal to V_0
        ard.choose_action(state=state, training_step=0)
        assert ard.w_d == 0.0
        assert np.allclose(ard.accumulators.drift_rates, config.V_0)

        # Middle of decay: advantage weight partially grown, following a power curve
        ard.choose_action(state=state, training_step=decay.decay_steps // 2)
        expected_w_d = config.w_d * 0.5**config.w_d_schedule_power
        assert abs(ard.w_d - expected_w_d) < 1e-6

        # End of decay and beyond: final advantage weight
        for training_step in (decay.decay_steps, total_steps):
            ard.choose_action(state=state, training_step=training_step)
            assert abs(ard.w_d - config.w_d) < 1e-6

        # Demo mode: current weight is kept
        ard.choose_action(state=state, training_step=None)
        assert abs(ard.w_d - config.w_d) < 1e-6

    def test_uniform_choices_at_start_of_exploration(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([0.0, 5.0, 1.0, 2.0])
        config = DecisionConfig()
        decay = LinearExplorationDecay(config=config, total_steps=1000)

        ard, _ = self._make_ard(
            action_values=q_values, config=config, exploration_decay=decay
        )
        actions = [ard.choose_action(state=state, training_step=0) for _ in range(400)]

        # With w_d = 0, every action should be chosen roughly equally often
        counts = np.bincount(actions, minlength=len(q_values))
        assert np.all(counts > 60)

    def test_deterministic_choice(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([-20.0, -19.9, -25.0])
        config = DecisionConfig()
        decay = LinearExplorationDecay(config=config, total_steps=1000)

        ard, meta_stub = self._make_ard(
            action_values=q_values, config=config, exploration_decay=decay
        )
        for _ in range(20):
            # Greedy even at the start of exploration
            action = ard.choose_action(state=state, training_step=0, deterministic=True)
            assert action == np.argmax(q_values)

        # No accumulation took place, so no metacognitive update either
        assert meta_stub.rt == 0.0

    def test_crossing_times(self) -> None:
        ard, _ = self._make_ard(
            action_values=np.zeros(3), config=DecisionConfig(accumulation="wald")
        )
        theta, s, n = 1.0, 0.5, 20000

        # Positive drift: mean first-passage time is theta/v
        times = ard._sample_crossing_times(v=np.full(n, 2.0), theta=theta, s=s)
        assert abs(times.mean() - theta / 2.0) < 0.01

        # Negative drift: threshold reached with probability exp(2*v*theta/s^2)
        v = -0.2
        times = ard._sample_crossing_times(v=np.full(n, v), theta=theta, s=s)
        assert abs(np.isfinite(times).mean() - np.exp(2 * v * theta / s**2)) < 0.02

        # Zero drift: threshold always reached
        times = ard._sample_crossing_times(v=np.zeros(100), theta=theta, s=s)
        assert np.all(np.isfinite(times))

    def test_evidence_at_decision_time(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([0.5, 1.0, 0.2])

        ard, _ = self._make_ard(
            action_values=q_values, config=DecisionConfig(accumulation="wald")
        )
        for _ in range(50):
            action = ard.choose_action(state=state, training_step=0)
            theta = ard.hyperparameters.theta
            winner_evidence = ard.accumulators.evidence[
                ard.accumulators.adv_accs[action]
            ]

            # Slowest accumulator of the winning action is exactly at threshold
            assert abs(winner_evidence.min() - theta) < 1e-9

            # Every other action has at least one accumulator below threshold
            for other in range(len(q_values)):
                if other != action:
                    other_evidence = ard.accumulators.evidence[
                        ard.accumulators.adv_accs[other]
                    ]
                    assert other_evidence.min() < theta

    def test_wald_matches_euler(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([0.0, 1.0, 1.1, 0.5])
        n = 1000

        choice_probs = {}
        for accumulation in ("wald", "euler"):
            # Fixed threshold, noisy choices between actions 1 and 2
            config = DecisionConfig(accumulation=accumulation, w_d=1.0, theta_start=0.3)
            ard, _ = self._make_ard(action_values=q_values, config=config)
            actions = [
                ard.choose_action(state=state, training_step=0) for _ in range(n)
            ]
            choice_probs[accumulation] = np.bincount(actions, minlength=4) / n

        # Same choice probabilities, up to sampling and discretization error
        assert np.all(np.abs(choice_probs["wald"] - choice_probs["euler"]) < 0.06)

    def test_multiple_winners(self) -> None:
        class TwoWinnersRule(WinAllRule):
            def get_winning_actions(self, n_actions, evidence, theta, adv_accs):
                return [2, 3]

        state = np.zeros(5, dtype=np.float32)
        ard, _ = self._make_ard(
            action_values=np.array([0.0, 1.0, 2.0, 3.0]),
            config=DecisionConfig(accumulation="euler"),
        )
        ard.decision_rule = TwoWinnersRule(rng=ard.rng)

        for _ in range(20):
            # Chosen action must be one of the winners, not an index into the winners list
            assert ard.choose_action(state=state, training_step=0) in (2, 3)

    def test_action_values_stored_for_confidence(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([-20.0, -18.0, -19.0])

        ard, _ = self._make_ard(action_values=q_values, config=DecisionConfig())
        ard.choose_action(state=state, training_step=0)

        # Normalized action values: centered, same ordering as raw Q-values
        stored = ard.accumulators.action_values
        assert abs(stored.mean()) < 1e-9
        assert np.array_equal(np.argsort(stored), np.argsort(q_values))

    def test_drift_rates(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([0.5, 1.0, 0.2])

        ard, _ = self._make_ard(action_values=q_values, config=DecisionConfig())
        ard.choose_action(state=state, training_step=0)

        drift_rates = ard._compute_drift_rates_dict(q_values=q_values)
        for i, j in ard.accumulators.action_pairs:
            # v(i,j) = w_d*(Q_i - Q_j) + w_s*(Q_i + Q_j) + V0
            expected_drift_rate = (
                ard.w_d * (q_values[i] - q_values[j])
                + ard.config.w_s * (q_values[i] + q_values[j])
                + ard.config.V_0
            )
            assert abs(drift_rates[i, j] - expected_drift_rate) < 1e-6

    def _make_ard(
        self,
        action_values: np.ndarray,
        config: DecisionConfig,
        exploration_decay: ExplorationDecay | None = None,
    ) -> tuple[ARD, MetaStub]:
        rng = np.random.default_rng(42)

        learning_cfg = LearningConfig()
        hyperparameters = Hyperparameters(
            decision_cfg=config, learning_cfg=learning_cfg
        )
        meta_monitoring = MetaMonitoring(
            config=MetaConfig(),
            confidence_method=BayesianConfidence(
                decision_cfg=config, hyperparameters=hyperparameters
            ),
        )
        meta_stub = MetaStub(decision_cfg=config, hyperparameters=hyperparameters)
        ard = ARD(
            config=config,
            estimator=EstimatorStub(action_values=action_values),
            decision_rule=WinAllRule(rng=rng),
            hyperparameters=hyperparameters,
            meta_monitoring=meta_monitoring,
            meta_ctrl=meta_stub,
            rng=rng,
            exploration_decay=exploration_decay,
        )
        return ard, meta_stub


if __name__ == "__main__":
    # Standalone execution
    TestARD().test_chosen_action(show_plots=True)
