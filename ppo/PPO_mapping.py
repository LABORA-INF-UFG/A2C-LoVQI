import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class PPOMapping:
    def __init__(self, ns3_env, state_size, action_space,
                 n_vants, dim_grid, gamma=0.99, lambdaa=0.95,
                 clip_epsilon=0.2, actor_learning_rate=3e-4, critic_learning_rate=2e-4,
                 entropy_coef=0.01, batch_size=5000, memory_limit=100000, epochs=20,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 weight_decay = 1e-5 ,device='cpu'):
        self.device = torch.device(device)
        self.ns3_env = ns3_env
        self.state_size = state_size
        self.action_space = action_space
        self.n_vants = n_vants
        self.dim_grid = dim_grid
        self.gamma = gamma
        self.lambdaa = lambdaa
        self.clip_epsilon = clip_epsilon
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size
        self.memory_limit=memory_limit
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon  # Taxa de exploração inicial
        self.epsilon_decay = epsilon_decay  # Taxa de decaimento
        self.epsilon_min = epsilon_min  # Taxa mínima de exploração
        self.weigth_decay = weight_decay # Taxa de decaimento para regularização de L2

        # Redes (Policy e Value)
        self.policy_network = self._build_policy_network().to(self.device)
        self.value_network = self._build_value_network().to(self.device)

        # Otimizadores com regularização L2
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.actor_learning_rate,
            weight_decay=self.weigth_decay  # Regularização L2
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=self.critic_learning_rate,
            weight_decay=self.weigth_decay  # Regularização L2
        )

        # Memória para armazenar as trajetórias
        self.memory = []

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(probs)

        # Seleção com política principal (1 - epsilon) e exploração (epsilon)
        if random.random() < self.epsilon:
            action = torch.randint(0, len(self.action_space), (1,)).to(self.device)
        else:
            action = action_dist.sample()

        # Decay do epsilon para reduzir exploração ao longo do tempo
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item()

    def _build_policy_network(self, n_hidden_layers=128):
        """Constrói a rede de política (Actor) com inicialização de He."""
        policy_net = nn.Sequential(
            nn.Linear(self.state_size, n_hidden_layers),
            nn.ReLU(),
            nn.Linear(n_hidden_layers, n_hidden_layers),
            nn.ReLU(),
            nn.Linear(n_hidden_layers, len(self.action_space)),
            nn.Softmax(dim=-1)  # Gera probabilidades para cada ação
        )

        # Aplicar inicialização de He para camadas lineares
        for layer in policy_net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                init.zeros_(layer.bias)

        return policy_net

    def _build_value_network(self, n_hidden_layers=128):
        """Constrói a rede de valor (Critic) com inicialização de He."""
        value_net = nn.Sequential(
            nn.Linear(self.state_size, n_hidden_layers),
            nn.ReLU(),
            nn.Linear(n_hidden_layers, n_hidden_layers),
            nn.ReLU(),
            nn.Linear(n_hidden_layers, 1)  # Saída escalar para valor estado-ação
        )

        # Aplicar inicialização de He para camadas lineares
        for layer in value_net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                init.zeros_(layer.bias)

        return value_net

    def remember(self, state, action, reward, next_state, done, log_prob):
        # Armazena uma transição na memória (trajetória para atualização do PPO).
        self.memory.append((state, action, reward, next_state, done, log_prob))
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)

    def sample_batch(self):
        # Amostra um lote aleatório da memória, com tamanho definido por batch_size.
        # Garante que o tamanho do lote não exceda o tamanho da memória
        batch_size = min(self.batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)  # Retorna um lote aleatório


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

    def update(self, epochs=None):
        # Divide o conteúdo da memória em lotes e processa iterativamente
        num_batches = len(self.memory) // self.batch_size
        num_batches += 1 if len(self.memory) % self.batch_size != 0 else 0
        m_policy_loss = []
        m_value_loss = []
        for i in range(num_batches):
            batch = self.sample_batch()
            states, actions, rewards, next_states, dones, old_log_probs = zip(*batch)
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

            policy_losses = []
            value_losses = []

            # Atualizações por minibatches
            for _ in range(epochs):  # Número de épocas por atualização
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
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
            m_policy_loss.append(np.mean(policy_losses))
            m_value_loss.append(np.mean(value_losses))
        lp = np.mean(m_policy_loss)
        lv = np.mean(m_value_loss)
        return lp, lv

def save_checkpoint(actor_model, critic_model, actor_optimizer, critic_optimizer, filename):
    """Salva o estado atual do modelo e otimizadores."""
    checkpoint = {
        'actor_state_dict': actor_model.state_dict(),
        'critic_state_dict': critic_model.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint salvo em {filename}")

def load_checkpoint(actor_model, critic_model, actor_optimizer, critic_optimizer, filename):
    """Carrega o estado do modelo e otimizadores de um arquivo."""
    checkpoint = torch.load(filename)
    actor_model.load_state_dict(checkpoint['actor_state_dict'])
    critic_model.load_state_dict(checkpoint['critic_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def save_metrics(results_file_name, best_positions_file_name, movements_file_name, execution_times,
                 episode_mean_rewards, episode_mean_q_rewards, episode_mean_actor_loss, episode_mean_critic_loss, best_episode,
                 best_step, best_reward, best_positions, best_info, print_movements, episodes_movements, file_mode="a"):
    # Saving results metrics
    with open(results_file_name, file_mode) as file:
        if file_mode == "w":
            file.write("episodio,tempo,reward,q_reward,policy_loss,value_loss\n")
        for idx, (time_elapsed, reward, q_reward, actor_loss, critic_loss) in enumerate(
                zip(execution_times, episode_mean_rewards, episode_mean_q_rewards, episode_mean_actor_loss, episode_mean_critic_loss)):
            file.write(f"{idx + 1},{time_elapsed:.4f},{reward:.4f},{q_reward:.4f},{actor_loss:.4f},{critic_loss:.4f}\n")
    # Saving best positions
    with open(best_positions_file_name, file_mode) as file:
        if file_mode == "w":
            file.write("episode,step,reward,positions,info\n")
        file.write(f"{best_episode},{best_step},{best_reward},{best_positions},{best_info}\n")
    # Saving movements if enabled
    if print_movements:
        with open(movements_file_name, file_mode) as file:
            if file_mode == "w":
                file.write("episodio,step,state,next_state,reward,q_reward,info,action,step_size\n")
            for idx, (episode, episode_movements) in enumerate(episodes_movements.items()):
                for step, state, next_state, reward, q_reward, info, action, step_size in episode_movements:
                    file.write(
                        f"{idx + episode},{step},{state},{next_state},{reward},{q_reward},{info},{action},{step_size}\n")