import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parser de argumentos para entrada de parâmetros
parser = argparse.ArgumentParser(description='PLOTS of PPO for n UAVs')
parser.add_argument('--v', type=int, help='Number of Virtual Positions')
parser.add_argument('--g', type=int, help='Number of Gateways (e.g., 3 5 7)')
parser.add_argument('--d', type=int, nargs='+', help='List of Devices')
parser.add_argument('--l', type=bool, help='Log Scale')
args = parser.parse_args()

vp = args.v
devices_list = args.d  # Lista de gateways
gp = args.g
log = args.l

# Caminhos para os diretórios de entrada e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/ppo"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Criação de uma figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Variáveis para rastrear se dados foram encontrados para pelo menos um gateway
data_found = False
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'X', 'd', 'P', '|', '_']


# Configurações dos gráficos
title_size = 20
label_size = 18
tick_size = 16

# Iterar pela lista de gateways fornecidos
for dp in devices_list:
    seed_data = []

    # Iterar pelos valores de seed
    for seed in range(1, 21):  # Você pode ajustar o range conforme o necessário
        input_file = os.path.join(input_dir, f'OLD/PPO_results_{vp}V_{gp}G_{dp}D_{seed}S-OLD.dat')

        if not os.path.exists(input_file):
            print(f"Arquivo não encontrado: {input_file}")
            continue  # Pule para o próximo seed

        # Ler o arquivo .dat usando pandas removendo a linha de cabeçalho
        data = pd.read_csv(input_file,
                           names=['episodio', 'tempo', 'reward', 'q_reward', 'actor_loss', 'critic_loss'],
                           skiprows=1)

        # Adicionar os dados lidos à lista
        seed_data.append(data)

        # Atualizar a flag indicando que encontramos dados
        data_found = True

    # Verifica se dados foram encontrados
    if not seed_data:
        continue

    # Combina os dados de todas as seeds em um único DataFrame
    all_data = pd.concat(seed_data, ignore_index=True)

    # Calcula a média por episódio agrupando por número de gateways
    mean_data = all_data.groupby('episodio').mean().reset_index()
    mean_data = mean_data[mean_data['episodio'] <= 300]

    if mean_data is not None:
        # Plot 1: Tempo de execução médio
        axs[0][0].plot(mean_data["episodio"], mean_data["tempo"], linestyle='-', label=f'{dp} Devices')
        axs[0][0].set_title(f"Tempo de Execução Médio", fontsize=title_size)
        axs[0][0].set_xlabel("Episódio ", fontsize=label_size)
        axs[0][0].set_ylabel("Tempo de Execução", fontsize=label_size)
        axs[0][0].tick_params(axis='both', labelsize=tick_size)
        if log:
            axs[0][0].set_yscale('log')
        axs[0][0].legend(fontsize=tick_size)
        axs[0][0].grid()

        # Plot 4: Recompensa acumulada média
        axs[0][1].plot(mean_data["episodio"], mean_data["q_reward"], linestyle='-', label=f'{dp} Devices')
        axs[0][1].set_title(f"Recompensa Acumulada Média", fontsize=title_size)
        axs[0][1].set_xlabel("Episódio ", fontsize=label_size)
        axs[0][1].set_ylabel("Recompensa", fontsize=label_size)
        axs[0][1].tick_params(axis='both', labelsize=tick_size)
        if log:
            axs[0][1].set_yscale('log')
        axs[0][1].legend(fontsize=tick_size)
        axs[0][1].grid()

        # Plot 2: Actor Loss médio
        axs[1][0].plot(mean_data["episodio"], mean_data["actor_loss"], linestyle='--', label=f'{dp} Devices')
        axs[1][0].set_title(f"Perda média do Ator", fontsize=title_size)
        axs[1][0].set_xlabel("Episódio ", fontsize=label_size)
        axs[1][0].set_ylabel("Perdas", fontsize=label_size)
        axs[1][0].tick_params(axis='both', labelsize=tick_size)
        if log:
            axs[1][0].set_yscale('log')
        axs[1][0].legend(fontsize=tick_size)
        axs[1][0].grid()

        # Plot 3: Critic Loss médio
        axs[1][1].plot(mean_data["episodio"], mean_data["critic_loss"], linestyle='-', label=f'{dp} Devices')
        axs[1][1].set_title(f"Perda média do Critico", fontsize=title_size)
        axs[1][1].set_xlabel("Episódio ", fontsize=label_size)
        axs[1][1].set_ylabel("Perdas", fontsize=label_size)
        axs[1][1].tick_params(axis='both', labelsize=tick_size)
        if log:
            axs[1][1].set_yscale('log')
        axs[1][1].legend(fontsize=tick_size)
        axs[1][1].grid()



# Ajusta o layout para evitar sobreposição
if data_found:
    plt.tight_layout()

    # Nome do arquivo de saída
    output_file = os.path.join(output_dir, f"graf_PPO_results_{vp}V_{gp}G_devices_avg.png")

    # Salva o gráfico como imagem
    fig.savefig(output_file)
    print(f"Gráfico salvo em: {output_file}")

    plt.show()
else:
    print("Nenhum dado encontrado para os arquivos especificados.")