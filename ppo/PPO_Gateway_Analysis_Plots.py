import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parser de argumentos para entrada de parâmetros
parser = argparse.ArgumentParser(description='PLOTS of PPO for n UAVs')
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

for gp in gps:
    # Lista para armazenar os DataFrames de cada execução
    all_seeds_data = []

    # Iterar pelos valores de seed de 1 a 10
    for seed in range(1, 6):
        input_file = os.path.join(input_dir, f'PPO_results_{vp}V_{gp}G_{dp}D_{seed}S.dat')

        if not os.path.exists(input_file):
            print(f"Arquivo não encontrado: {input_file}")
            continue  # Pule para o próximo seed

        # Lê o arquivo .dat usando pandas removendo a linha de cabeçalho
        data = pd.read_csv(input_file,
                           names=['episodio', 'tempo', 'reward', 'q_reward', 'actor_loss', 'critic_loss'],
                           skiprows=1,
                           # dtype={'episodio': int, 'tempo': float, 'reward': float, 'q_reward': float,
                           #        'actor_loss': float, 'critic_loss': float}
                           )
        all_seeds_data.append(data)

    # Se nenhum arquivo for encontrado para este gateway, prossiga para o próximo
    if not all_seeds_data:
        print(f"Nenhum dado encontrado para {gp} Gateways.")
        continue

    data_found = True

    # Concatenar os dados das 10 seeds no eixo dos dados individuais
    combined_data = pd.concat(all_seeds_data)

    # Substituir valores NaN por 0 no combined_data
    combined_data.fillna(0, inplace=True)
    # Substituir strings "nan" por 0
    combined_data = combined_data.replace("nan", 0)

    # Calcular a média agrupada por episódio
    mean_data = combined_data.groupby("episodio").mean().reset_index()

    # Adicionar o título geral da figura
    plt.title(f"PPO for {gp} Gateways")

    # Obter a cor automática baseada no matplotlib
    color = axs[0]._get_lines.get_next_color()

    # Plot 1: Tempo de execução (média)
    axs[0].plot(mean_data["episodio"], mean_data["tempo"], linestyle='-', label=f'{gp} Gateways', color=color)
    axs[0].set_title(f"Tempo de Execução por Episódio {'(Escala Log)' if log else ''}")
    axs[0].set_xlabel("Episódio")
    axs[0].set_ylabel("Tempo de Execução")
    if log:
        axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid()

    # Plot 2: Loss acumulada (com mesmo esquema de cores)
    axs[1].plot(mean_data["episodio"], mean_data["actor_loss"], linestyle='--', label=f'Actor Loss ({gp} Gateways)',
                color=color)
    axs[1].plot(mean_data["episodio"], mean_data["critic_loss"], linestyle='-', label=f'Critic Loss ({gp} Gateways)',
                color=color)
    axs[1].set_title(f"Loss Acumulada por Episódio {'(Escala Log)' if log else ''}")
    axs[1].set_xlabel("Episódio")
    axs[1].set_ylabel("Loss Acumulada")
    if log:
        axs[1].set_yscale('log')  # Aplicar escala log no eixo Y
    axs[1].legend()
    axs[1].grid()

    # Plot 3: Recompensa acumulada (média)
    axs[2].plot(mean_data["episodio"], mean_data["q_reward"], linestyle='-', label=f'{gp} Gateways', color=color)
    axs[2].set_title(f"Recompensa Acumulada por Episódio {'(Escala Log)' if log else ''}")
    axs[2].set_xlabel("Episódio")
    axs[2].set_ylabel("Recompensa Acumulada")
    # if log:
        # axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].grid()

# Ajusta o layout para evitar sobreposição
if data_found:
    plt.suptitle(
        f"Resultados do PPO para {vp} Posições Candidatas, {' '.join(map(str, gps))} Gateways e {dp} Dispositivos",
        fontsize=16)
    plt.tight_layout()

    # Nome do arquivo de saída
    output_file = os.path.join(output_dir, f"graf_PPO_results_{vp}V_{'_'.join(map(str, gps))}G_{dp}D.png")

    # Salva o gráfico como imagem
    fig.savefig(output_file)
    print(f"Gráfico salvo em: {output_file}")

    plt.show()
else:
    print("Nenhum dado encontrado para os gateways fornecidos. Verifique os arquivos de entrada.")
