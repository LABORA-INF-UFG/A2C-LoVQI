import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PPOMapping:
    def __init__(self, ns3_env, state_size, action_space, n_vants, dim_grid, gamma=0.99, lambdaa=0.95,
                 clip_epsilon=0.2, learning_rate=3e-4, entropy_coef=0.01, batch_size=64, epochs=10, device='cpu'):
        """
        Classe Proximal Policy Optimization (PPO).
        Args:
            state_size: Tamanho do estado (entrada da rede).
            action_space: Espaço de ações disponíveis.
            gamma: Fator de desconto para recompensa futura.
            lambdaa: Parâmetro de suavização para GAE (General Advantage Estimation).
            clip_epsilon: Valor de clipping usado no PPO.
            learning_rate: Taxa de aprendizado para otimizadores.
            batch_size: Tamanho do minibatch para atualização.
            epochs: Número de épocas para treinar em cada atualização.
            entropy_coef: Hiperparâmetro para coeficiente de entropia na perda.
            device: CPU ou GPU para execução.
        """
        self.device = device
        self.ns3_env = ns3_env
        self.state_size = state_size
        self.action_space = action_space
        self.n_vants = n_vants
        self.dim_grid = dim_grid
        self.gamma = gamma
        self.lambdaa = lambdaa
        self.clip_epsilon = clip_epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.entropy_coef = entropy_coef

        # Redes (Actor e Critic)
        self.policy_network = self._build_policy_network().to(self.device)
        self.value_network = self._build_value_network().to(self.device)

        # Otimizadores
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        # Memória para armazenar as trajetórias
        self.memory = []

    def select_action(self, state):
        """Seleciona uma ação com base na política."""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs = self.policy_network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()


    def _build_policy_network(self):
        """Constrói a rede de política (Actor)."""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.action_space)),
            nn.Softmax(dim=-1)  # Gera probabilidades para cada ação
        )

    def _build_value_network(self):
        """Constrói a rede de valor (Critic)."""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Saída escalar para valor estado-ação
        )

    def remember(self, state, action, reward, next_state, done, log_prob):
        """
        Armazena uma transição na memória (trajetória para atualização do PPO).
        """
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def _compute_gae(self, rewards, values, next_values, dones):
        """Calcula a Vantagem Generalizada (GAE) e os retornos."""
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambdaa * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)

    def update(self):
        """Atualiza as redes utilizando os dados armazenados."""
        # Organizar a memória
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)
        self.memory = []  # Limpar memória após atualização

        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

        # Calcular valores estimados e vantagens
        values = self.value_network(states_tensor).squeeze()
        next_values = torch.cat(
            (values[1:], torch.tensor([0.0]).to(self.device))  # Valor do próximo estado
        )  # Adiciona zero no último próximo-estado
        advantages, returns = self._compute_gae(rewards_tensor, values.detach(), next_values.detach(), dones)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalização das vantagens

        # Atualizações por minibatches
        for _ in range(10):  # Número de épocas por atualização
            probs = self.policy_network(states_tensor)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

            # Clipped Objective (PPO)
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            # Atualiza Policy Network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Atualiza Value Network
            value_loss = nn.MSELoss()(self.value_network(states_tensor).squeeze(), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
