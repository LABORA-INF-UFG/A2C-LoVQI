import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Camadas compartilhadas
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Stream de Valor
        self.fc_value = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)  # V(s)

        # Stream de Vantagem
        self.fc_advantage = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, action_size)  # A(s, a)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # Cálculo do Valor
        value = torch.relu(self.fc_value(x))
        value = self.value(value)

        # Cálculo da Vantagem
        advantage = torch.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)

        # Combinação Dueling: Q(s, a) = V(s) + (A(s, a) - média(A(s, a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DuelingDDQNAgent:
    def __init__(self, state_size, action_space, device, gamma=0.99, epsilon_max=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=64):
        self.state_size = state_size
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.device = device

        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.policy_network = DuelingDQN(state_size, self.n_actions).to(device)
        self.target_network = DuelingDQN(state_size, self.n_actions).to(device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = deque(maxlen=100_000)

    def update_target_network(self):
        """ Atualiza a rede alvo copiando os pesos da rede principal. """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def get_action(self, state):
        """ Seleciona a ação usando e-greedy. """
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_network(state_tensor)
            return self.action_space[torch.argmax(q_values).item()]

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay_experience(self):
        """ Treina a rede neural usando o Double DQN. """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Double DQN update
        with torch.no_grad():
            best_actions = torch.argmax(self.policy_network(next_states), dim=1)
            q_targets_next = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        q_values = self.policy_network(states)
        actions_indices = torch.tensor([self.action_space.index(a) for a in actions]).to(self.device)
        q_expected = q_values.gather(1, actions_indices.unsqueeze(1)).squeeze(1)

        loss = self.criterion(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
