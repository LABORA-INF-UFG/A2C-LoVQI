import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='PLOTS of DQN for n UAVs')
parser.add_argument('--v', type=int, help='Number of Virtual Positions')
parser.add_argument('--g', type=int, help='Number of Gateways')
parser.add_argument('--d', type=int, help='Number of Devices')
parser.add_argument('--l', type=bool, help='Log Scale')
args = parser.parse_args()

vp = args.v
gp = args.g
dp = args.d
log = args.l

# Caminhos para o arquivo de entrada e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Nome da figura de saída
input_file = os.path.join(input_dir, f'DQN_results_{vp}V_{gp}G_{dp}DD.dat')  # Nome do arquivo de entrada
output_file = os.path.join(output_dir, f"graf_DQN_results_{vp}V_{gp}G_{dp}DD.png")

# Lê o arquivo .dat usando pandas
# O arquivo possui o seguinte cabeçalho: "episodio, tempo_execucao, sum_loss, avg_loss, sum_reward, avg_reward"
if os.path.exists(input_file):
    data = pd.read_csv(input_file)
else:
    print("Arquivo não encontrado. Verifique e tente novamente.")
    exit(1)

# Remover os episódios duplicados
data = data.drop_duplicates(subset=['episodio'], keep='first')
# filtrar episódios de 200 a 400
# data = data[(data['episodio'] >= 200) & (data['episodio'] <= 400)]

# Criação de uma figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(1, 3, figsize=(21, 5))

# 1. Tempo de execução por episódio (escala log no eixo Y)
axs[0].plot(data["episodio"], data["tempo"], linestyle='-', color='b', label='Tempo de Execução')
axs[0].set_title(f"Tempo de Execução por Episódio")
axs[0].set_xlabel("Episódio")
axs[0].set_ylabel("Tempo de Execução")
if log:
    axs[0].set_yscale('log')  # Escala logarítmica no eixo Y
axs[0].legend()
axs[0].grid()

# 2. Loss acumulada por episódio (escala log no eixo Y)
axs[1].plot(data["episodio"], data["loss"], linestyle='-', color='r', label='Loss Acumulada')
axs[1].set_title(f"Perda Acumulada por Episódio")
axs[1].set_xlabel("Episódio")
axs[1].set_ylabel("Perda Acumulada")
if log:
    axs[1].set_yscale('log')  # Escala logarítmica no eixo Y
axs[1].legend()
axs[1].grid()

# 3. Recompensa qualificada acumulada por episódio (escala log no eixo Y)
axs[2].plot(data["episodio"], data["qualified_reward"], linestyle='-', color='g', label='Recompensa Q-Acumulada')
axs[2].set_title(f"Recompensa Acumulada por Episódio")
axs[2].set_xlabel("Episódio")
axs[2].set_ylabel("Recompensa Acumulada")
if log:
    axs[2].set_yscale('log')  # Escala logarítmica no eixo Y
axs[2].legend()
axs[2].grid()

# Ajusta o layout para evitar sobreposição
plt.tight_layout()

# Salva o gráfico como imagem
fig.savefig(output_file)
print(f"Gráfico salvo em: {output_file}")

plt.show()
