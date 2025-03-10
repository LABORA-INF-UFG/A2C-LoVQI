import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

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

# Caminhos para os diretórios de entrada e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Criação de uma figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Variáveis para rastrear se dados foram encontrados
data_found = False
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'X', 'd', 'P', '|', '_']

# Selecionar apenas o valor de 3 gateways
gp = 3
all_seeds_data = []

# Iterar pelos valores de seed de 1 a 10
for seed in range(1, 11):
    input_file = os.path.join(input_dir, f'A2C_results_{vp}V_{gp}G_{dp}D_{seed}SS.dat')

    if not os.path.exists(input_file):
        print(f"Arquivo não encontrado: {input_file}")
        continue  # Pule para o próximo seed

    # Lê o arquivo .dat usando pandas removendo a linha de cabeçalho
    data = pd.read_csv(input_file,
                       names=['episodio', 'tempo', 'reward', 'q_reward', 'actor_loss', 'critic_loss'],
                       skiprows=1)

    # Adicionar flag indicando que encontramos dados
    data_found = True

    # Plotar os dados das sementes individuais para cada curva
    # Plot 1: Tempo de execução
    axs[0].plot(data["episodio"], data["tempo"], linestyle='-', label=f'Seed {seed}')
    axs[0].set_title(f"Tempo de Execução para {gp} Gateways")
    axs[0].set_xlabel("Episódio")
    axs[0].set_ylabel("Tempo de Execução")
    axs[0].legend()
    axs[0].grid()

    # Plot 2: Loss (actor e critic)
    axs[1].plot(data["episodio"], data["actor_loss"], linestyle='--', label=f'Actor Loss Seed {seed}')
    axs[1].plot(data["episodio"], data["critic_loss"], linestyle='-', label=f'Critic Loss Seed {seed}')
    axs[1].set_title(f"Loss para {gp} Gateways")
    axs[1].set_xlabel("Episódio")
    axs[1].set_ylabel("Loss")
    if log:
        axs[1].set_yscale('log')
    axs[1].set_ylim(bottom=-50000, top=1e8)
    # axs[1].legend()
    axs[1].grid()

    # Plot 3: Recompensa acumulada
    axs[2].plot(data["episodio"], data["q_reward"], linestyle='-', label=f'Seed {seed}')
    axs[2].set_title(f"Recompensa Acumulada para {gp} Gateways")
    axs[2].set_xlabel("Episódio")
    axs[2].set_ylabel("Recompensa Acumulada")
    axs[2].legend()
    axs[2].grid()

# Ajusta o layout para evitar sobreposição
if data_found:
    plt.tight_layout()

    # Nome do arquivo de saída
    output_file = os.path.join(output_dir, f"graf_A2C_results_{vp}V_{gp}G_{dp}D_seeds.png")

    # Salva o gráfico como imagem
    fig.savefig(output_file)
    print(f"Gráfico salvo em: {output_file}")

    plt.show()
else:
    print("Nenhum dado encontrado para os arquivos com seed para 3 Gateways.")