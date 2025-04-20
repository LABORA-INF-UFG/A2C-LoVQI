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

# Caminhos para o arquivo de entrada e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Criação de uma figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Variáveis para rastrear se dados foram encontrados
data_found = False
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'X', 'd', 'P', '|', '_']

for gp in gps:
    # Caminho do arquivo de entrada e saída para cada quantidade de gateways
    input_file = os.path.join(input_dir, f'A2C_results_{vp}V_{gp}G_{dp}DD.dat')

    if not os.path.exists(input_file):
        print(f"Arquivo não encontrado para {gp} Gateways: {input_file}")
        continue  # Avançar para o próximo valor de gateway

    # Lê o arquivo .dat usando pandas
    data = pd.read_csv(input_file)
    data_found = True

    # Adiciona o título geral da figura
    plt.title(f"A2C for {gp} Gateways")

    # Obter a cor automática baseada no matplotlib
    color = axs[0]._get_lines.get_next_color()

    # Plot 1: Tempo de execução
    axs[0].plot(data["episodio"], data["tempo"], linestyle='-', label=f'{gp} Gateways', color=color)
    axs[0].set_title(f"Tempo de Execução por Episódio {'(Escala Log)' if log else ''}")
    axs[0].set_xlabel("Episódio")
    axs[0].set_ylabel("Tempo de Execução")
    if log:
        axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid()

    # Plot 2: Loss acumulada (com mesmo esquema de cores)
    axs[1].plot(data["episodio"], data["actor_loss"], linestyle='--', label=f'Actor Loss ({gp} Gateways)',
                color=color, marker=markers[gps.index(gp)], markevery=50-gp*5, markersize=12 - gp * 3,
                markerfacecolor='none')
    axs[1].plot(data["episodio"], data["critic_loss"], linestyle='-', label=f'Critic Loss ({gp} Gateways)',
                color=color)
    axs[1].set_title(f"Loss Acumulada por Episódio {'(Escala Log)' if log else ''}")
    axs[1].set_xlabel("Episódio")
    axs[1].set_ylabel("Loss Acumulada")
    
    # if log:
    # axs[1].set_yscale('log')
        # axs[1].set_ylim(bottom=40000, top=-60000)
    axs[1].legend()
    axs[1].grid()

    # Plot 3: Recompensa acumulada
    axs[2].plot(data["episodio"], data["q_reward"], linestyle='-', label=f'{gp} Gateways', color=color)
    axs[2].set_title(f"Recompensa Acumulada por Episódio {'(Escala Log)' if log else ''}")
    axs[2].set_xlabel("Episódio")
    axs[2].set_ylabel("Recompensa Acumulada")
    if log:
        axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].grid()

# Ajusta o layout para evitar sobreposição
if data_found:
    plt.suptitle(f"Resultados do A2C para {vp} Posições Candidatas, {' '.join(map(str, gps))} Gateways e {dp} Dispositivos",
                 fontsize=16)
    plt.tight_layout()

    # Nome do arquivo de saída
    output_file = os.path.join(output_dir, f"graf_A2C_results_{vp}V_{'_'.join(map(str, gps))}G_{dp}D.png")

    # Salva o gráfico como imagem
    fig.savefig(output_file)
    print(f"Gráfico salvo em: {output_file}")

    plt.show()
else:
    print("Nenhum dado encontrado para os gateways fornecidos. Verifique os arquivos de entrada.")