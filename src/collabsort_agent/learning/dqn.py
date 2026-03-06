"""
Deep Q-Learning algorithm.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from collabsort_agent.learning.exploration_decay import ExplorationDecay
from collabsort_agent.learning.learning import Config as LearningConfig
from collabsort_agent.learning.learning import LearningAlgorithm


def get_device() -> torch.device:
    """Return accelerated device if available, or fail back to CPU"""

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

        # Create Q-network for estimating actions values
        self.q_network = QNetwork(input_size=state_size, output_size=n_actions).to(
            self.device
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_network.parameters(), lr=self.config.lr
        )

        # Create target network with fixed parameters (useful to stabilize training)
        self.target_network = QNetwork(input_size=state_size, output_size=n_actions).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Create replay buffer for training the Q-network
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)

    def choose_action(self, state: np.ndarray, training_step: int) -> int:
        # Update exploration probability
        self.epsilon = self.exploration_decay.get_epsilon(training_step=training_step)

        # With probability epsilon: explore (choose a random action)
        if np.random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # With probability (1-epsilon): exploit (greedily choose the best known action)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return torch.argmax(q_values, dim=1).cpu().numpy()

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

        # Convert batch data to PyTorch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Clamp actions to valid range
        actions = torch.clamp(actions, 0, self.n_actions - 1)

        # Compute action values for the batch data
        q_values = self.q_network(states).gather(1, actions).squeeze(1)

        # Q_target = r + γ * max_a' Q(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.q_network(next_states).max(1)[0]
            q_target = rewards + self.config.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_values, q_target)

        # Update Q-network parameters through a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def log(self, logger: SummaryWriter, episode: int) -> None:
        """Log information for an episode"""

        logger.add_scalar(
            tag="learning/exploration_probability",
            scalar_value=self.epsilon,
            global_step=episode,
        )
