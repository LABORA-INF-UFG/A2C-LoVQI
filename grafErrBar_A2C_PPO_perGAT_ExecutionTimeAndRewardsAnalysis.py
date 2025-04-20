import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminhos para os diretórios de entrada e saída
input_A2C_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/new_tr/"
input_PPO_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/ppo"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Parâmetros experimentais
gateways_list = [2, 3, 4, 5]
devices = 200
seeds = range(1, 21)
pv = 144


# Correção na função process_data
def process_data(input_dir, algorithm_prefix):
    results = {'episode': [], 'mean_time': {}, 'time_ci': {}, 'mean_q_reward': {}, 'q_reward_ci': {}}

    # Inicializar listas para cada número de gateways
    for gateways in gateways_list:
        results['mean_time'][gateways] = []
        results['time_ci'][gateways] = []
        results['mean_q_reward'][gateways] = []
        results['q_reward_ci'][gateways] = []

    max_episodes = 300
    for episode in range(1, max_episodes + 1):
        episode_times = []
        episode_q_rewards = []

        for gateways in gateways_list:
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

            # Calcular a média para este número de gateways
            if all_times:
                episode_times.append(sum(all_times) / len(all_times))
            else:
                episode_times.append(0)  # Fallback para evitar erros

            if all_q_rewards:
                episode_q_rewards.append(sum(all_q_rewards) / len(all_q_rewards))
            else:
                episode_q_rewards.append(0)  # Fallback para evitar erros

        # Atualizar resultados para este episódio
        results['episode'].append(episode)
        for idx, gateways in enumerate(gateways_list):
            mean_time = episode_times[idx]
            mean_q_reward = episode_q_rewards[idx]

            time_ci = 1.96 * (
                pd.Series(episode_times).std() / len(episode_times) ** 0.5 if len(episode_times) > 1 else 0)
            q_reward_ci = 1.96 * (
                pd.Series(episode_q_rewards).std() / len(episode_q_rewards) ** 0.5 if len(episode_q_rewards) > 1 else 0)

            results['mean_time'][gateways].append(mean_time)
            results['time_ci'][gateways].append(time_ci)
            results['mean_q_reward'][gateways].append(mean_q_reward)
            results['q_reward_ci'][gateways].append(q_reward_ci)

    return results


# Processar dados A2C e PPO
a2c_results = process_data(input_A2C_dir, "A2C")
ppo_results = process_data(input_PPO_dir, "PPO")

# Gerar gráfico comparativo
colors = {
    "DQN": "tab:red",
    "DDQN": "tab:blue",
    "PPO": "tab:green",
    "A2C": "tab:orange"
}
device_lines = {
    2: "dotted",
    3: "dashdot",
    4: "dashed",
    5: "solid"
}

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Tempo médio de execução por episódio
for gateways in gateways_list:
    axs[0].plot(a2c_results['episode'], a2c_results['mean_time'][gateways],
                label=f'A2C {gateways}G', color=colors['A2C'], linestyle=device_lines[gateways])
    axs[0].errorbar(a2c_results['episode'], a2c_results['mean_time'][gateways],
                    yerr=a2c_results['time_ci'][gateways], color=colors['A2C'],
                    linestyle=device_lines[gateways], elinewidth=0.2, alpha=0.5)
    axs[0].plot(ppo_results['episode'], ppo_results['mean_time'][gateways],
                label=f'PPO {gateways}G', color=colors['PPO'], linestyle=device_lines[gateways])
    axs[0].errorbar(ppo_results['episode'], ppo_results['mean_time'][gateways],
                    yerr=ppo_results['time_ci'][gateways], color=colors['PPO'],
                    linestyle=device_lines[gateways], elinewidth=0.2, alpha=0.5)
axs[0].set_title('Tempo de Execução por Episódio', fontsize=14)
axs[0].set_xlabel('Episódios', fontsize=12)
axs[0].set_ylabel('Tempo de Execução (s)', fontsize=12)
axs[0].legend()
axs[0].grid(True)

# Gráfico 2: Recompensa acumulada média por episódio
for gateways in gateways_list:
    axs[1].plot(a2c_results['episode'], a2c_results['mean_q_reward'][gateways],
                label=f'A2C {gateways}G', color=colors['A2C'], linestyle=device_lines[gateways])
    axs[1].errorbar(a2c_results['episode'], a2c_results['mean_q_reward'][gateways],
                    yerr=a2c_results['q_reward_ci'][gateways], color=colors['A2C'],
                    linestyle=device_lines[gateways], elinewidth=0.2, alpha=0.5)
    axs[1].plot(ppo_results['episode'], ppo_results['mean_q_reward'][gateways],
                label=f'PPO {gateways}G', color=colors['PPO'], linestyle=device_lines[gateways])
    axs[1].errorbar(ppo_results['episode'], ppo_results['mean_q_reward'][gateways],
                    yerr=ppo_results['q_reward_ci'][gateways], color=colors['PPO'],
                    linestyle=device_lines[gateways], elinewidth=0.2, alpha=0.5)

axs[1].set_title('Recompensa Acumulada por Episódio', fontsize=14)
axs[1].set_xlabel('Episódios', fontsize=12)
axs[1].set_ylabel('Recompensa Acumulada', fontsize=12)
axs[1].legend()
axs[1].grid(True)

# Ajustar layout e salvar o gráfico
plt.tight_layout()
output_file = os.path.join(output_dir, 'comparison_execution_time_rewards.png')
plt.savefig(output_file)
plt.show()