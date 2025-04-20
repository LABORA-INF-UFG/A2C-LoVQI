import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class DQNMapping:
    def __init__(self, ns3_env, dim_grid, action_space, n_vants, state_size, gamma=0.99, epsilon_max=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001,
                 replay_buffer_size=None, batch_size=24, device='cpu'):

        # Configuração do dispositivo para GPU ou CPU
        self.device = device

        # Parâmetros do ambiente
        self.ns3_env = ns3_env
        self.dim_grid = dim_grid
        self.action_space = action_space
        self.n_vants = n_vants
        self.state_size = state_size
        self.n_actions = len(action_space)

        # Inicializa estados e atributos da classe
        self.state = None  # Estado atual
        self.gamma = gamma

        # Controle de exploração
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Define the replay buffer size proportional to state complexity
        if replay_buffer_size is None:
            # Example: buffer size = 10 times the state_space size (upper limit of 100,000)
            replay_buffer_size = min(100_000, 10 * state_size)
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        self.batch_size = batch_size

        # Configuração da rede neural
        self.learning_rate = learning_rate
        self.policy_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.update_target_network()  # Sincroniza pesos inicialmente

        # Função de perda e otimizador
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def _build_model(self):
        """Cria e retorna a arquitetura da rede neural."""
        input_dim = int(self.state_size)
        # Garante que input_dim corresponde ao grid achatado
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        ).to(self.device)
        return model

    def update_target_network(self):
        """Atualiza os pesos da rede de destino (target network)."""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Armazena experiências no buffer de replay."""
        # Create the experience tuple
        experience = (state, action, reward, next_state, done)
        # Compute the initial TD-error dynamically as |reward + gamma * max(Q_next) - Q_current|
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Q-value of current state-action pair (Q_current)
            q_values = self.policy_network(state_tensor)
            q_current = q_values[0, self.action_space.index(action)].item()

            # Q-value of the next state (max Q_next)
            q_next = 0 if done else torch.max(self.target_network(next_state_tensor)).item()

        # TD-error calculation
        initial_error = abs(reward + self.gamma * q_next - q_current)

        # Add the experience along with the computed error to the ReplayBuffer
        self.replay_buffer.add(experience, error=initial_error)

        # Add the experience to the ReplayBuffer
        self.replay_buffer.add(experience, error=initial_error)

    def get_action(self, state):
        """Define a ação a ser tomada (exploração ou exploração)."""
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)  # Escolha aleatória (exploração)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Garante formato [1, state_dim]
            state_tensor = state_tensor.to(self.device)
            q_values = self.policy_network(state_tensor)
            return self.action_space[torch.argmax(q_values).item()]  # Escolha baseada em Q (greedy)

    def get_loss(self, next_state, reward, action):
        # Atualiza o agente usando experiência armazenada
        criterion = torch.nn.MSELoss().to(self.device)
        # if len(self.replay_buffer) < self.batch_size:
        #     loss = self.replay_experience()
        # else:
        with torch.no_grad():
            next_state_ = torch.from_numpy(next_state).float().to(self.device)
            newQ = self.policy_network(next_state_.unsqueeze(0).to(self.device))
            m = torch.nn.Softmax(dim=1).to(self.device)
            newQ = m(newQ)
        maxQ = torch.max(newQ).to(self.device)
        Y = reward + (self.gamma * maxQ)
        Y = Y.detach().to(self.device)  # target value
        next_state_ = torch.from_numpy(next_state).float().to(self.device)
        qval = self.policy_network(next_state_.unsqueeze(0).to(self.device))
        iAction = self.action_space.index(action)
        Y_pred = qval.squeeze()[iAction]  # predicted
        loss = criterion(Y_pred, Y)
        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def replay_experience(self):
        """Treina a rede neural usando minibatches do buffer de replay."""
        if len(self.replay_buffer.buffer) < self.batch_size:
            return  # Certifique-se de ter exemplos suficientes

        batch, indices, weights = self.replay_buffer.sample(self.batch_size)

        # Processa os estados, ações, recompensas e próximos estados
        states, targets, td_errors = [], [], []
        for (state, action, reward, next_state, done), weight in zip(batch, weights):
            state = np.array(state).flatten()
            next_state = np.array(next_state).flatten()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            q_values = self.policy_network(state_tensor).detach().numpy()[0]
            q_next = self.target_network(next_state_tensor).detach().numpy()[0]

            # Atualiza o valor de Q para a ação tomada
            target = np.copy(q_values)
            action_index = self.action_space.index(action)
            if done:
                td_error = reward - q_values[action_index]
                target[action_index] = reward
            else:
                td_error = reward + self.gamma * np.max(q_next) - q_values[action_index]
                target[action_index] = reward + self.gamma * np.max(q_next)

            td_errors.append(td_error)
            states.append(state)
            targets.append(target)

        # Atualiza as prioridades no buffer
        self.replay_buffer.update_priorities(indices, td_errors)

        # Converte listas em tensores para treinamento
        states_tensor = torch.FloatTensor(np.array(states))
        targets_tensor = torch.FloatTensor(np.array(targets))

        # Treinamento da política atual
        self.optimizer.zero_grad()
        predictions = self.policy_network(states_tensor)
        loss = self.criterion(predictions, targets_tensor)
        loss.backward()
        self.optimizer.step()

    def reset_epsilon(self):
        """Redefine o epsilon para o valor máximo."""
        self.epsilon = self.epsilon_max

    def update(self, state, action, reward, next_state, done):
        """Atualização do passo do DQN."""
        self.remember(state, action, reward, next_state, done)
        self.replay_experience()  # Aprendizado
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decaimento do epsilon

    def state_from_coordinates(self, x, y):
        """Transforma coordenadas em representação de estado."""
        state = np.zeros(self.dim_grid)
        state[x, y] = 1
        return state.flatten()

    def coordinates_from_state(self, state):
        """Transforma o estado em coordenadas."""
        index = np.argmax(state)
        return divmod(index, self.dim_grid)

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha  # Controle de importância dos erros no PER.

    def add(self, experience, error):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(error ** self.alpha)
        else:
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(experience)
            self.priorities.append(error ** self.alpha)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        importance_sampling_weights = (1 / (len(self.buffer) * probabilities[indices])) ** beta
        importance_sampling_weights /= importance_sampling_weights.max()

        return experiences, importance_sampling_weights, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error ** self.alpha




