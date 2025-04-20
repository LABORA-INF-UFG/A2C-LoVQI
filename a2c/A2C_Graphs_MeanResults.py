import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Caminhos de entrada e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/new_tr/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Parser de argumentos para entrada de parâmetros
parser = argparse.ArgumentParser(description='PLOTS of A2C for n UAVs')
parser.add_argument('--v', type=int, help='Number of Virtual Positions')
parser.add_argument('--g', type=int, nargs='+', help='List of Gateways (e.g., 3 5 7)')
parser.add_argument('--d', type=int, help='Number of Devices')
parser.add_argument('--l', type=bool, help='Log Scale')
args = parser.parse_args()

vp = args.v
gps = args.g  # Lista de gateways
dp = args.d
log = args.l

# Configurações dos gráficos
font_size = 18
label_size = 18
tick_size = 16
markers = ['o', 's', 'D', '^', 'x']

# Para cada número de gateways fornecido
for gp in gps:
    all_seeds_data = []  # Lista para armazenar os dados de todas as seeds combinadas

    # Varre os arquivos de dados por seed
    for seed in range(1, 6):  # Assumindo seeds de 1 a 5
        input_file = os.path.join(input_dir, f'A2C_results_{vp}V_{gp}G_{dp}D_{seed}SS.dat')

        if os.path.exists(input_file):
            # Carrega os dados
            data = pd.read_csv(input_file,
                               names=['episodio', 'tempo', 'reward', 'q_reward', 'actor_loss', 'critic_loss'],
                               skiprows=1)
            all_seeds_data.append(data)

    # Verifica se dados foram encontrados
    if not all_seeds_data:
        print(f"Nenhum dado encontrado para {gp} Gateways!")
        continue

    # Combina os dados de todas as seeds em um único DataFrame
    all_data = pd.concat(all_seeds_data, ignore_index=True)

    # Calcula a média por episódio agrupando por número de gateways
    mean_data = all_data.groupby('episodio').mean().reset_index()

    # Plota as curvas
    fig, axs = plt.subplots(figsize=(10, 6))

    metrics = {'q_reward': "Recompensa", 'actor_loss':"Perda do Ator", 'critic_loss':"Perda do Crítico"}
    for i, metric in enumerate(metrics):
        axs.plot(mean_data['episodio'], mean_data[metric],
                 label=metrics[metric])

    # Configurações do gráfico
    axs.set_title(f'Curvas Médias - {gp} Gateways', fontsize=font_size)
    axs.set_xlabel('Episódios', fontsize=label_size)
    axs.set_ylabel('Valores', fontsize=label_size)
    axs.tick_params(axis='both', labelsize=tick_size)
    axs.grid(True)
    axs.legend(fontsize=label_size)
    if log:
        axs.set_yscale('log')

    # Salva a figura
    output_file = os.path.join(output_dir, f"graf_A2C_results_{vp}V_{gp}G_{dp}D_avg.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Gráfico salvo em: {output_file}")