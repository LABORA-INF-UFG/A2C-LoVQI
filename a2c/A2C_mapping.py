import numpy as np
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, n_hidden_layers=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, n_hidden_layers),
            nn.Tanh(),
            nn.Linear(n_hidden_layers, n_hidden_layers),
            nn.Tanh(),
            nn.Linear(n_hidden_layers, n_actions),
            nn.Softmax(dim=-1)
        )
        self.initialize_weights()

    def forward(self, X):
        return self.model(X)

    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # Melhor para estabilidade


class Critic(nn.Module):
    def __init__(self, state_dim, n_hidden_layers=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, n_hidden_layers),
            nn.ReLU(),
            nn.Linear(n_hidden_layers, n_hidden_layers),
            nn.ReLU(),
            nn.Linear(n_hidden_layers, 1)
        )
        self.initialize_weights()

    def forward(self, X):
        return self.model(X)

    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # Melhor para estabilidade


class RewardNormalizer:
    def __init__(self, alpha=0.99, epsilon=1e-8):
        """ Inicializa a normalização com média móvel exponencial """
        self.mean = 0
        self.var = 1
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def normalize(self, reward):
        """ Normaliza a recompensa com média e variância móveis """
        self.mean = self.alpha * self.mean + (1 - self.alpha) * reward
        self.var = self.alpha * self.var + (1 - self.alpha) * (reward - self.mean) ** 2
        return (reward - self.mean) / (np.sqrt(self.var) + self.epsilon)


def save_checkpoint(actor_model, critic_model, actor_optimizer, critic_optimizer, filename):
    """Salva o estado atual do modelo e otimizadores."""
    checkpoint = {
        'actor_state_dict': actor_model.state_dict(),
        'critic_state_dict': critic_model.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    # print(f"Checkpoint salvo em {filename}")


def load_checkpoint(actor_model, critic_model, actor_optimizer, critic_optimizer, filename):
    """Carrega o estado do modelo e otimizadores de um arquivo."""
    checkpoint = torch.load(filename)
    actor_model.load_state_dict(checkpoint['actor_state_dict'])
    critic_model.load_state_dict(checkpoint['critic_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def save_metrics(results_file_name, best_positions_file_name, movements_file_name, execution_times,
                 episode_mean_rewards, episode_mean_q_rewards, episode_mean_actor_loss, episode_mean_critic_loss, best_episode,
                 best_step, best_reward, best_positions, best_info, print_movements, episodes_movements, file_mode="w"):
    # Saving results metrics
    with open(results_file_name, file_mode) as file:
        if file_mode == "w":
            file.write("episodio,tempo,reward,q_reward,actor_loss,critic_loss\n")
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
                file.write("episodio,step,state,next_state,reward,info,action,step_size\n")
            for idx, (episode, episode_movements) in enumerate(episodes_movements.items()):
                for step, state, next_state, reward, info, action, step_size in episode_movements:
                    file.write(
                        f"{episode},{step},{state},{next_state},{reward},{info},{action},{step_size}\n")


