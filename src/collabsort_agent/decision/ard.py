"""
Advantage Racing Diffusion definitions.
"""

from statistics import NormalDist

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import DecisionConfig, Deliberator
from collabsort_agent.decision.accumulators import Accumulators
from collabsort_agent.decision.decision_rule import DecisionRule
from collabsort_agent.decision.exploration_decay import ExplorationDecay
from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.metacognition import Hyperparameters
from collabsort_agent.metacognition.controller import MetaController
from collabsort_agent.metacognition.monitoring import MetaMonitoring


class ARD(Deliberator):
    """
    Advantage Racing Diffusion algorithm for decision making.

    Translates Q-values into an action selection with associated confidence
    via evidence accumulation. Each of the n actions has n-1 "advantage" accumulators versus all others.

    Inspired by Miletic2021 https://elifesciences.org/articles/63055
    """

    def __init__(
        self,
        config: DecisionConfig,
        estimator: ActionValueEstimator,
        decision_rule: DecisionRule,
        hyperparameters: Hyperparameters,
        meta_monitoring: MetaMonitoring,
        meta_ctrl: MetaController,
        rng: np.random.Generator,
        exploration_decay: ExplorationDecay | None = None,
    ) -> None:
        super().__init__(config=config, estimator=estimator, rng=rng)

        self.decision_rule = decision_rule
        self.exploration_decay = exploration_decay
        self.hyperparameters = hyperparameters
        self.meta_monitoring = meta_monitoring
        self.meta_ctrl = meta_ctrl

        # Init accumulators
        self.accumulators = Accumulators(n_actions=estimator.n_actions)

        # Running estimate of the Q-value spread across actions, used for normalization.
        self.q_scale: float | None = None

        # Current weight of the advantage term.
        self.w_d: float = self.config.w_d

    def choose_action(
        self,
        state: np.ndarray,
        training_step: int | None,
        deterministic: bool = False,
    ) -> int:
        """Choose the action to perform."""

        action_values = self.estimator.get_action_values(state=state)

        n_actions = len(action_values)
        if n_actions == 1:
            return 0  # Only one possible action

        if deterministic:
            # No exploration and no metacognitive update: greedily choose the best known action
            return int(np.argmax(action_values).item())

        # Update advantage weight from the exploration schedule.
        # In demo mode (training_step is None), the current weight is kept.
        if self.exploration_decay is not None and training_step is not None:
            self.w_d = self._compute_advantage_weight(training_step=training_step)

        # Compute drift rates for all accumulators
        if self.config.normalize_q_values:
            action_values = self._normalize_q_values(action_values)
        self.accumulators.action_values = action_values
        drift_rates = self._compute_drift_rates(action_values)
        self.accumulators.drift_rates = drift_rates

        # Run the accumulator race.
        # chosen_action is -1 if no action won within the maximal decision time.
        if self.config.accumulation == "wald":
            chosen_action, rt = self._race_wald(
                drift_rates=drift_rates, n_actions=n_actions
            )
        else:
            chosen_action, rt = self._race_euler(
                drift_rates=drift_rates, n_actions=n_actions
            )

        min_actions_evidence = self.accumulators.min_evidence(
            actions=list(range(n_actions))
        )

        # Fallback: no action chosen within max_steps.
        # Select action whose slowest accumulator has highest value.
        if chosen_action == -1:
            chosen_action = np.argmax(min_actions_evidence).item()

        # Compute second best action (non-winning action whose slowest accumulator has highest value)
        min_actions_evidence[chosen_action] = -np.inf
        runnerup_action = np.argmax(min_actions_evidence).item()

        # Compute decision confidence and adjust hyperparameters
        confidence = self.meta_monitoring.compute_decision_confidence(
            chosen_action=chosen_action,
            runnerup_action=runnerup_action,
            reaction_time=rt,
            accumulators=self.accumulators,
        )
        self.meta_ctrl.update_hyperparameters(confidence=confidence, reaction_time=rt)

        return chosen_action

    def _race_euler(self, drift_rates: np.ndarray, n_actions: int) -> tuple[int, float]:
        """
        Simulate the accumulator race step by step (Euler-Maruyama scheme).

        Slower than _race_wald, but records the whole evidence history (useful for plotting).
        Return the chosen action (-1 if none within max_steps) and the reaction time (in steps).
        """

        # Reset evidence and history.
        # History is preallocated for max_steps columns and trimmed to the
        # actual decision length below, to avoid reallocating/copying the
        # whole array at each accumulation step.
        self.accumulators.evidence = self.accumulators.empty_evidence()
        self.accumulators.evidence_history = np.zeros(
            (self.accumulators.n_accumulators, self.config.max_steps), dtype=float
        )

        chosen_action = -1
        rt = float(self.config.max_steps)

        # Evidence accumulation loop
        for t in range(1, self.config.max_steps + 1):
            # Compute accumulation noise
            noise = self.rng.normal(
                loc=self.config.noise_mean,
                scale=self.config.noise_std,
                size=self.accumulators.n_accumulators,
            )

            # Accumulate evidence
            self.accumulators.evidence += drift_rates * self.config.dt + noise

            # No lower bound on evidence, as in the racing diffusion models of Miletic2021

            # Add new evidence to history
            self.accumulators.evidence_history[:, t - 1] = self.accumulators.evidence

            winning_actions = self.decision_rule.get_winning_actions(
                n_actions=n_actions,
                evidence=self.accumulators.evidence,
                theta=self.hyperparameters.theta,
                adv_accs=self.accumulators.adv_accs,
            )

            if winning_actions:
                if len(winning_actions) > 1:
                    # More than one action have seen all their advantage accumulators cross the threshold.
                    # Select action whose slowest accumulator has highest value.
                    min_winners_evidence = self.accumulators.min_evidence(
                        actions=winning_actions
                    )
                    chosen_action = winning_actions[
                        np.argmax(min_winners_evidence).item()
                    ]

                elif len(winning_actions) == 1:
                    # Only one action has seen all its advantage accumulators cross the threshold
                    chosen_action = winning_actions[0]

                rt = float(t)
                break

        # Trim history to the actual number of accumulation steps taken
        self.accumulators.evidence_history = self.accumulators.evidence_history[
            :, : int(rt)
        ]

        return chosen_action, rt

    def _race_wald(self, drift_rates: np.ndarray, n_actions: int) -> tuple[int, float]:
        """
        Run the accumulator race by sampling exactly the first-passage time of each accumulator.

        Without a lower bound, the first-passage time of a diffusion with positive drift v,
        noise s and threshold theta follows a Wald (inverse Gaussian) distribution
        of mean theta/v and shape theta^2/s^2 (Miletic2021). With the win-all rule,
        an action wins as soon as all its advantage accumulators have reached the threshold.

        This is the continuous-time version of _race_euler: no discretization, and no
        simulation loop. Evidence is only sampled at decision time (no history).
        Return the chosen action (-1 if none within the maximal decision time)
        and the reaction time (in steps of duration dt, for consistency with _race_euler).
        """

        theta = self.hyperparameters.theta
        dt = self.config.dt

        # Diffusion coefficient and drift rates equivalent to the discrete scheme of _race_euler
        s = self.config.noise_std / np.sqrt(dt)
        v = drift_rates + self.config.noise_mean / dt

        crossing_times = self._sample_crossing_times(v=v, theta=theta, s=s)

        # Win-all rule: an action wins when its slowest advantage accumulator reaches threshold
        action_times = np.array(
            [
                crossing_times[self.accumulators.adv_accs[action]].max()
                for action in range(n_actions)
            ]
        )
        chosen_action = int(np.argmin(action_times))

        max_decision_time = self.config.max_steps * dt
        if action_times[chosen_action] > max_decision_time:
            # No decision within maximal decision time
            chosen_action = -1
            decision_time = max_decision_time
        else:
            decision_time = float(action_times[chosen_action])

        # Evidence of all accumulators at decision time
        self.accumulators.evidence = self._sample_evidence(
            crossing_times=crossing_times, v=v, t=decision_time, theta=theta, s=s
        )
        self.accumulators.evidence_history = self.accumulators.evidence[:, np.newaxis]

        return chosen_action, decision_time / dt

    def _sample_crossing_times(
        self, v: np.ndarray, theta: float, s: float
    ) -> np.ndarray:
        """
        Return sampled first-passage times through threshold theta for diffusions
        starting at 0 with drift rates v and noise s (np.inf if never reached).
        """

        crossing_times = np.full(v.shape, np.inf)

        # Positive drift: threshold is always reached, after a Wald-distributed time
        positive = v > 0
        crossing_times[positive] = self.rng.wald(
            mean=theta / v[positive], scale=(theta / s) ** 2
        )

        # Zero drift: threshold is always reached, after a Lévy-distributed time
        zero = v == 0
        crossing_times[zero] = (theta / s) ** 2 / self.rng.standard_normal(
            np.count_nonzero(zero)
        ) ** 2

        # Negative drift: threshold is reached with probability exp(2*v*theta/s^2).
        # If so, the first-passage time is Wald-distributed with drift |v|.
        hit_probs = np.exp(2 * np.minimum(v, 0) * theta / s**2)
        negative_hits = (v < 0) & (self.rng.random(v.shape) < hit_probs)
        crossing_times[negative_hits] = self.rng.wald(
            mean=theta / -v[negative_hits], scale=(theta / s) ** 2
        )

        return crossing_times

    def _sample_evidence(
        self,
        crossing_times: np.ndarray,
        v: np.ndarray,
        t: float,
        theta: float,
        s: float,
    ) -> np.ndarray:
        """Return sampled evidence values at time t, consistent with the sampled crossing times"""

        evidence = np.empty_like(v)

        # Accumulators which reached threshold: free diffusion since crossing time
        crossed = crossing_times <= t
        elapsed = t - crossing_times[crossed]
        evidence[crossed] = (
            theta
            + v[crossed] * elapsed
            + s * np.sqrt(elapsed) * self.rng.standard_normal(elapsed.shape)
        )

        # Other accumulators: evidence conditioned on the threshold not having been reached yet
        for k in np.flatnonzero(~crossed):
            evidence[k] = self._sample_uncrossed_evidence(
                v=float(v[k]), t=t, theta=theta, s=s
            )

        return evidence

    def _sample_uncrossed_evidence(
        self, v: float, t: float, theta: float, s: float
    ) -> float:
        """
        Return a sample of evidence x at time t for a diffusion (drift v, noise s)
        which has not reached threshold theta before t.

        x is drawn from a normal distribution truncated below theta, then accepted with
        the probability that a Brownian bridge from 0 to x stays below theta:
        1 - exp(-2*theta*(theta - x) / (s^2 * t)).
        """

        distribution = NormalDist(mu=v * t, sigma=s * np.sqrt(t))
        max_prob = distribution.cdf(theta)

        x = theta
        # Near-certain crossing: evidence can only be (just) below threshold
        if max_prob < 1e-12:
            return x

        for _ in range(100):
            prob = min(max_prob * (1.0 - self.rng.random()), 1.0 - 1e-12)
            x = distribution.inv_cdf(prob)
            if self.rng.random() < 1.0 - np.exp(-2 * theta * (theta - x) / (s**2 * t)):
                break

        return x

    def update_calibration(self, td_error: float) -> None:
        """Recalibrate decision confidence using the outcome of the last decision (extension 5)"""

        self.meta_monitoring.update_calibration(td_error=td_error)

    def _normalize_q_values(self, q_values: np.ndarray) -> np.ndarray:
        """
        Return Q-values centered on their mean and divided by a running estimate
        of their spread across actions.

        This makes drift rates independent of the learned Q-value scale (which
        grows with rewards and discount factor), while preserving differences
        in spread between states.
        """

        spread = float(np.std(q_values))
        if self.q_scale is None:
            self.q_scale = spread
        else:
            decay = self.config.q_scale_decay
            self.q_scale = decay * self.q_scale + (1.0 - decay) * spread

        return (q_values - np.mean(q_values)) / (self.q_scale + 1e-8)

    def _compute_advantage_weight(self, training_step: int) -> float:
        """
        Return the advantage weight for a training step, following the exploration schedule.
        """

        assert self.exploration_decay is not None

        """The weight grows from 0 when exploration probability is at its start value
        (all drift rates equal to V_0: uniformly random choices)
        to config.w_d when exploration probability reaches its minimum,
        following a power curve of the decay progress."""

        epsilon = self.exploration_decay.get_epsilon(training_step=training_step)
        epsilon_range = self.config.epsilon_start - self.config.epsilon_min
        if epsilon_range <= 0:
            return self.config.w_d

        progress = (self.config.epsilon_start - epsilon) / epsilon_range
        return self.config.w_d * progress**self.config.w_d_schedule_power

    def _compute_drift_rates(self, q_values: np.ndarray) -> np.ndarray:
        """
        Return the drift rates for all accumulators. Shape: (n_accumulators,).

        v(i,j) = w_d*(Q_i - Q_j) + w_s*(Q_i + Q_j) + V0

        w_d is the current (possibly annealed) advantage weight.
        """

        pairs = np.array(self.accumulators.action_pairs)
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        return (
            self.w_d * (q_values[i_idx] - q_values[j_idx])
            + self.config.w_s * (q_values[i_idx] + q_values[j_idx])
            + self.config.V_0
        )

    def _compute_drift_rates_dict(
        self, q_values: np.ndarray
    ) -> dict[tuple[int, int], float]:
        """Return {(i,j): drift_rate}. Used for debugging."""

        v = self._compute_drift_rates(q_values)
        return {
            pair: float(v[k]) for k, pair in enumerate(self.accumulators.action_pairs)
        }

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

        logger.add_scalar(
            tag="decision/decision_threshold",
            scalar_value=self.hyperparameters.theta,
            global_step=episode,
        )
        logger.add_scalar(
            tag="decision/advantage_weight",
            scalar_value=self.w_d,
            global_step=episode,
        )
        self.meta_monitoring.log_episode(logger=logger, episode=episode)

    def save_state(self, dir: str) -> None:
        # TODO save state for ARD
        pass

    def load_state(self, dir: str) -> None:
        # TODO load state for ARD
        pass
