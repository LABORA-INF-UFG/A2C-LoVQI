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
gateways_list = args.g  # Lista de gateways
dp = args.d
log = args.l

# Caminhos para os diretórios de entrada e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/new_tr/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Criação de uma figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(1, 4, figsize=(28, 7))

# Variáveis para rastrear se dados foram encontrados para pelo menos um gateway
data_found = False
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'X', 'd', 'P', '|', '_']


# Função para calcular a média de todas as seeds por episódio
def calculate_average(data_list):
    if not data_list:
        return None
    combined_df = pd.concat(data_list)  # Concatenar todos os DataFrames
    mean_df = combined_df.groupby('episodio').mean().reset_index()  # Agrupar por episódio e calcular a média
    return mean_df

font_size = 18
label_size = 18
tick_size = 16

# Iterar pela lista de gateways fornecidos
for gp in gateways_list:
    seed_data = []

    # Iterar pelos valores de seed
    for seed in range(1, 10):  # Você pode ajustar o range conforme o necessário
        input_file = os.path.join(input_dir, f'A2C_results_{vp}V_{gp}G_{dp}D_{seed}SS.dat')

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

    # Calcular as médias de todas as seeds
    mean_data = calculate_average(seed_data)

    if mean_data is not None:
        # Plot 1: Tempo de execução médio
        axs[0].plot(mean_data["episodio"], mean_data["tempo"], linestyle='-', label=f'{gp} Gateways')
        axs[0].set_title(f"Tempo de Execução Médio", fontsize=font_size)
        axs[0].set_xlabel("Episódio", fontsize=label_size)
        axs[0].set_ylabel("Tempo de Execução", fontsize=label_size)
        axs[0].tick_params(axis='both', labelsize=tick_size)
        if log:
            axs[0].set_yscale('log')
        axs[0].legend()
        axs[0].grid()

        # Plot 2: Actor Loss médio
        axs[1].plot(mean_data["episodio"], mean_data["actor_loss"], linestyle='--', label=f'{gp} Gateways')
        axs[1].set_title(f"Perda Média do Ator", fontsize=font_size)
        axs[1].set_xlabel("Episódio", fontsize=label_size)
        axs[1].set_ylabel("Perda Acumulada", fontsize=label_size)
        axs[1].tick_params(axis='both', labelsize=tick_size)
        if log:
            axs[1].set_yscale('log')
        axs[1].legend()
        axs[1].grid()

        # Plot 3: Critic Loss médio
        axs[2].plot(mean_data["episodio"], mean_data["critic_loss"], linestyle='-', label=f'{gp} Gateways')
        axs[2].set_title(f"Perda Média do Critico", fontsize=font_size)
        axs[2].set_xlabel("Episódio", fontsize=label_size)
        axs[2].set_ylabel("Perda Acumulada", fontsize=label_size)
        axs[2].tick_params(axis='both', labelsize=tick_size)
        axs[2].set_ylim(
            bottom=0,
            top=1.5
        )
        # if log:
        #     axs[2].set_yscale('log')
        axs[2].legend()
        axs[2].grid()

        # Plot 4: Recompensa acumulada média
        axs[3].plot(mean_data["episodio"], mean_data["q_reward"], linestyle='-', label=f'{gp} Gateways')
        axs[3].set_title(f"Recompensa Acumulada Média", fontsize=font_size)
        axs[3].set_xlabel("Episódio", fontsize=label_size)
        axs[3].set_ylabel("Recompensa Acumulada", fontsize=label_size)
        axs[3].tick_params(axis='both', labelsize=tick_size)
        if log:
            axs[3].set_yscale('log')
        axs[3].legend()
        axs[3].grid()

# Ajusta o layout para evitar sobreposição
if data_found:
    plt.tight_layout()

    # Nome do arquivo de saída
    output_file = os.path.join(output_dir, f"graf_A2C_results_{vp}V_gateways_avg_{dp}D.png")

    # Salva o gráfico como imagem
    fig.savefig(output_file)
    print(f"Gráfico salvo em: {output_file}")

    plt.show()
else:
    print("Nenhum dado encontrado para os arquivos especificados.")