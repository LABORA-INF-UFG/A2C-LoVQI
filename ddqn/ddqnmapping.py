import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple
import heapq
from ns3gym import ns3env

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Estrutura de transição para o buffer de replay
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# Rede Dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Camada de valor do estado
        self.value_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Saída é escalar
        )

        # Camada de vantagem
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)  # Saída é o número de ações
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_layer(features)
        advantages = self.advantage_layer(features)
        # Combinação do valor e vantagem
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# Replay Buffer com Prioritized Experience Replay (PER)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, transition, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)  # Adiciona na memória
            self.priorities.append(priority)
        else:
            # Substitui o elemento mais antigo
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]
        weights = (1 / len(self.buffer) / probabilities[indices]) ** beta
        weights /= weights.max()
        return indices, transitions, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


# Agente DDQN com PER
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=1e-3, batch_size=64, tau=0.005,
                 update_freq=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_freq = update_freq

        # Redes principais e alvo
        self.policy_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Otimizador
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Buffer de replay com PER
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        self.steps_done = 0

    def select_action(self, state, epsilon=0.1):
        """Escolha de ação com política ε-greedy."""
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done, td_error):
        """Adiciona uma transição no Replay Buffer com prioridade."""
        self.memory.add((state, action, reward, next_state, done), td_error)

    def update_target_network(self):
        """Atualiza a rede-alvo usando soft-update."""
        for target_param, local_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def train(self, beta=0.4):
        if len(self.memory.buffer) < self.batch_size:
            return

        indices, transitions, weights = self.memory.sample(self.batch_size, beta)
        weights = torch.FloatTensor(weights).to(device)

        batch = Transition(*zip(*transitions))
        state_batch = torch.FloatTensor(batch.state).to(device)
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(device)
        done_batch = torch.FloatTensor(batch.done).to(device)

        # Cálculo do valor Q-Atual
        q_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)

        # Alvo usando Double DQN
        with torch.no_grad():
            max_actions = self.policy_network(next_state_batch).argmax(dim=1)
            next_q_values = self.target_network(next_state_batch).gather(1, max_actions.unsqueeze(-1)).squeeze(-1)
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Erro Temporal-Diferenciado (TD Error)
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Função de perda ponderada
        loss = (weights * (q_values - target_q_values) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Atualiza a rede-alvo periodicamente
        if self.steps_done % self.update_freq == 0:
            self.update_target_network()

        self.steps_done += 1

