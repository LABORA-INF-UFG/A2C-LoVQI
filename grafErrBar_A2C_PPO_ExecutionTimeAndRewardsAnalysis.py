import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminhos para os diretórios de entrada e saída
input_A2C_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/new_tr/"
input_PPO_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/ppo"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Parâmetros experimentais
devices_list = [50, 100, 200]
gateways = 3
seeds = range(1, 12)
pv = 144


# Função para processar dados de um algoritmo
def process_data(input_dir, algorithm_prefix):
    results = {'episode': [], 'mean_time': [], 'time_ci': [], 'mean_q_reward': [], 'q_reward_ci': []}

    max_episodes = 300
    for episode in range(1, max_episodes + 1):
        episode_times = []
        episode_q_rewards = []

        for devices in devices_list:
            all_times = []
            all_q_rewards = []

            for seed in seeds:
                input_file = os.path.join(
                    input_dir,
                    f"{algorithm_prefix}_results_{pv}V_{gateways}G_{devices}D_{seed}{'SS.dat' if algorithm_prefix == 'A2C' else 'S.dat'}"
                )

                # Verifique se o arquivo existe antes de tentar lê-lo
                if os.path.exists(input_file):
                    data = pd.read_csv(
                        input_file,
                        names=['episodio', 'tempo', 'reward', 'q_reward', 'actor_loss', 'critic_loss'],
                        skiprows=1
                    )

                    # Filtrar somente o episódio atual
                    data = data[data['episodio'] == episode]

                    all_times.extend(data['tempo'].tolist())
                    all_q_rewards.extend(data['q_reward'].tolist())

            # Calcular a média para cada número de dispositivos no episódio
            if all_times and all_q_rewards:
                episode_times.append(sum(all_times) / len(all_times))
                episode_q_rewards.append(sum(all_q_rewards) / len(all_q_rewards))

        # Calcular a média e intervalo de confiança (95%) dos dispositivos para o episódio
        if episode_times and episode_q_rewards:
            mean_time = sum(episode_times) / len(episode_times)
            mean_q_reward = sum(episode_q_rewards) / len(episode_q_rewards)

            time_ci = 1.96 * (pd.Series(episode_times).std() / len(episode_times) ** 0.5)
            q_reward_ci = 1.96 * (pd.Series(episode_q_rewards).std() / len(episode_q_rewards) ** 0.5)

            results['episode'].append(episode)
            results['mean_time'].append(mean_time)
            results['time_ci'].append(time_ci)
            results['mean_q_reward'].append(mean_q_reward)
            results['q_reward_ci'].append(q_reward_ci)

    return results
colors = {
    "DQN": "tab:red",
    "DDQN": "tab:blue",
    "PPO": "tab:green",
    "A2C": "tab:orange"
}

# Processar dados A2C e PPO
a2c_results = process_data(input_A2C_dir, "A2C")
ppo_results = process_data(input_PPO_dir, "PPO")

# Gerar gráfico comparativo
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico 1: Tempo médio de execução por episódio
axs[0].plot(a2c_results['episode'], a2c_results['mean_time'], label='A2C', color=colors['A2C'])
axs[0].errorbar(a2c_results['episode'], a2c_results['mean_time'],
                yerr=a2c_results['time_ci'],  color=colors['A2C'], elinewidth=0.1)
axs[0].plot(ppo_results['episode'], ppo_results['mean_time'], label='PPO', color=colors['PPO'])
axs[0].errorbar(ppo_results['episode'], ppo_results['mean_time'],
                yerr=ppo_results['time_ci'],  color=colors['PPO'], elinewidth=0.1)
axs[0].set_title('Tempo Médio por Episódio', fontsize=14)
axs[0].set_xlabel('Episódios', fontsize=12)
axs[0].set_ylabel('Tempo Médio (s)', fontsize=12)
axs[0].legend()
axs[0].grid(True)

# Gráfico 2: Recompensa acumulada média por episódio
axs[1].plot(a2c_results['episode'], a2c_results['mean_q_reward'], label='A2C', color=colors['A2C'])
axs[1].errorbar(a2c_results['episode'], a2c_results['mean_q_reward'],
                yerr=a2c_results['q_reward_ci'],  color=colors['A2C'], elinewidth=0.1)
axs[1].plot(ppo_results['episode'], ppo_results['mean_q_reward'], label='PPO', color=colors['PPO'])
axs[1].errorbar(ppo_results['episode'], ppo_results['mean_q_reward'],
                yerr=ppo_results['q_reward_ci'],  color=colors['PPO'], elinewidth=0.1)
axs[1].set_title('Recompensa Acumulada Média por Episódio', fontsize=14)
axs[1].set_xlabel('Episódios', fontsize=12)
axs[1].set_ylabel('Recompensa Média', fontsize=12)
axs[1].legend()
axs[1].grid(True)

# Ajustar layout e salvar o gráfico
plt.tight_layout()
output_file = os.path.join(output_dir, 'comparison_execution_time_rewards.png')
plt.savefig(output_file)
plt.show()