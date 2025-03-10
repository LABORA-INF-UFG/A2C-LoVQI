import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parser de argumentos para entrada de parâmetros
parser = argparse.ArgumentParser(description='PLOTS of DDQN for n UAVs')
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

for gp in gps:
    # Caminho do arquivo de entrada e saída para cada quantidade de gateways
    input_file = os.path.join(input_dir, f'DDQN_results_{vp}V_{gp}G_{dp}D.dat')

    if not os.path.exists(input_file):
        print(f"Arquivo não encontrado para {gp} Gateways: {input_file}")
        continue  # Avançar para o próximo valor de gateway

    # Lê o arquivo .dat usando pandas
    data = pd.read_csv(input_file)
    data_found = True

    # Remover os episódios duplicados
    data = data.drop_duplicates(subset=['episodio'], keep='first')

    plt.title(f"DDQN for {gp} Gateways")

    # Plot 1: Tempo de execução
    axs[0].plot(data["episodio"], data["tempo"], linestyle='-', label=f'{gp} Gateways')
    axs[0].set_title(f"Tempo de Execução por Episódio {'(Escala Log)' if log else ''}")
    axs[0].set_xlabel("Episódio")
    axs[0].set_ylabel("Tempo de Execução")
    if log:
        axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid()

    # Plot 2: Loss acumulada
    axs[1].plot(data["episodio"], data["loss"], linestyle='-', label=f'{gp} Gateways')
    axs[1].set_title(f"Loss Acumulada por Episódio {'(Escala Log)' if log else ''}")
    axs[1].set_xlabel("Episódio")
    axs[1].set_ylabel("Loss Acumulada")
    if log:
        axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid()

    # Plot 3: Recompensa acumulada
    axs[2].plot(data["episodio"], data["qualified_reward"], linestyle='-', label=f'{gp} Gateways')
    axs[2].set_title(f"Recompensa Acumulada por Episódio {'(Escala Log)' if log else ''}")
    axs[2].set_xlabel("Episódio")
    axs[2].set_ylabel("Recompensa Acumulada")
    # if log:
    #     axs[2].set_yscale('log')
    # axs[2].legend()
    # axs[2].grid()

    # Plot 4: Recompensa qualificada acumulada
    # axs[3].plot(data["episodio"], data["qualified_reward"], linestyle='-', label=f'{gp} Gateways')
    # axs[3].set_title(f"Recompensa Q-Acumulada por Episódio {'(Escala Log)' if log else ''}")
    # axs[3].set_xlabel("Episódio")
    # axs[3].set_ylabel("Recompensa Q-Acumulada")
    if log:
        axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].grid()

# Ajusta o layout para evitar sobreposição
if data_found:
    plt.suptitle(f"Resultados do DDQN para {vp} Posições Candidatas, {' '.join(map(str, gps))} Gateways e {dp} Dispositivos",
                 fontsize=16)
    plt.tight_layout()

    # Nome do arquivo de saída
    output_file = os.path.join(output_dir, f"graf_DDQN_results_{vp}V_{'_'.join(map(str, gps))}G_{dp}D.png")

    # Salva o gráfico como imagem
    fig.savefig(output_file)
    print(f"Gráfico salvo em: {output_file}")

    plt.show()
else:
    print("Nenhum dado encontrado para os gateways fornecidos. Verifique os arquivos de entrada.")