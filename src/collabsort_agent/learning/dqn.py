"""
Deep Q-Learning algorithm.
"""

import random
from collections import deque
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from collabsort_agent.learning.exploration_decay import ExplorationDecay
from collabsort_agent.learning.learning import Config as LearningConfig
from collabsort_agent.learning.learning import LearningAlgorithm


def get_device() -> torch.device:
    """Return accelerated device if available, or fall back to CPU"""

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class QNetwork(nn.Module):
    """Neural network for Q-value estimation over all actions"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple = (100, 100),
    ) -> None:
        super().__init__()

        # Create network layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features=prev_size, out_features=hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(in_features=prev_size, out_features=output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.net(x)


class DQN(LearningAlgorithm):
    """Deep Q-Learning algorithm"""

    def __init__(
        self,
        config: LearningConfig,
        exploration_decay: ExplorationDecay,
        state_size: int,
        n_actions: int,
    ) -> None:
        super().__init__(config=config)

        self.config = config
        self.exploration_decay = exploration_decay
        self.n_actions = n_actions

        self.device = get_device()

        # Create Q-network for estimating action values
        self.q_network = QNetwork(input_size=state_size, output_size=n_actions).to(
            self.device
        )

        # Use SmoothL1Loss (Huber) rather than MSELoss.
        # DQN targets can have large variance; Huber loss is less sensitive to
        # outlier rewards (acts like MAE for large errors, MSE for small ones).
        self.loss_fn = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(
            params=self.q_network.parameters(), lr=self.config.lr
        )

        # Create target network with fixed parameters (stabilizes training)
        self.target_network = QNetwork(input_size=state_size, output_size=n_actions).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Target network must not accumulate gradients
        self.target_network.eval()

        # Create replay buffer for training the Q-network
        self.replay_buffer: deque = deque(maxlen=self.config.replay_buffer_size)

        # Step counter used to decide when to sync the target network
        self.learning_step: int = 0

        # Current exploration probability (set on first choose_action call)
        self.epsilon: float = self.config.epsilon_start

        # Recorded loss values (used for logging)
        self.losses: list[float] = []

        # Average Q-values (used for logging)
        self.mean_q_values: list[float] = []

    def choose_action(self, state: np.ndarray, training_step: int) -> int:
        # Update exploration probability
        self.epsilon = self.exploration_decay.get_epsilon(training_step=training_step)

        # With probability epsilon: explore (choose a random action)
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))

        # With probability (1-epsilon): exploit (greedily choose the best known action)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        # Store transition in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self) -> None:
        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample a batch of past experiences from replay buffer
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)

        # Obtain PyTorch tensors from NumPy arrays.
        # torch.from_numpy avoids allocating new memory
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(
            1
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(
            self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Clamp actions to valid range
        actions = torch.clamp(actions, 0, self.n_actions - 1)

        # Compute action values for the current states
        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())

        # Using target_network (not q_network) to compute Q-targets.
        # Using q_network here would defeat the purpose of the target network: the
        # same network would be used both to generate targets and to be updated,
        # creating a moving-target problem that destabilises training.
        with torch.no_grad():
            q_next = self.target_network(next_states).max(1)[0]
            # Q_target = r + gamma * max_a' Q_target(s', a') * (1 - done)
            q_target = rewards + self.config.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_values, q_target)
        self.losses.append(loss.item())

        # Update Q-network parameters through a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients.
        # max_norm=10 is a common conservative bound for DQN.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Periodically sync the target network with the online network.
        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

        logger.add_scalar(
            tag="learning/exploration_probability",
            scalar_value=self.epsilon,
            global_step=episode,
        )
        logger.add_scalar(
            tag="learning/td_loss",
            scalar_value=mean(self.losses),
            global_step=episode,
        )
        logger.add_scalar(
            tag="learning/q_values",
            scalar_value=mean(self.mean_q_values),
            global_step=episode,
        )

        # Reset episode data
        self.losses.clear()
        self.mean_q_values.clear()

    def save(self, run_dir: str) -> None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        file_path = f"{run_dir}/learning.pth"
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            file_path,
        )

    def load(self, run_dir: str) -> None:
        file_path = f"{run_dir}/learning.pth"
        checkpoint = torch.load(file_path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint["epsilon"])

        # Target network is only used for inference during target computation.
        self.target_network.eval()
