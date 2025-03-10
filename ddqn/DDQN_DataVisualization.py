import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='PLOTS of DQN for n UAVs')
parser.add_argument('--v', type=int, help='Number of Virtual Positions')
parser.add_argument('--g', type=int, help='Number of Gateways')
parser.add_argument('--d', type=int, help='Number of Devices')
args = parser.parse_args()

vp = args.v
gp = args.g
dp = args.d

# Caminhos para o arquivo de entrada e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Nome da figura de saída
input_file = os.path.join(input_dir, f'DQN_results_{vp}V_{gp}G_{dp}D.dat')  # Nome do arquivo de entrada
output_file = os.path.join(output_dir, f"graf_DQN_results_{vp}V_{gp}G_{dp}D.png")

# Lê o arquivo .dat usando pandas
# O arquivo possui o seguinte cabeçalho: "episodio,tempo_execucao,sum_loss,avg_loss,sum_reward,avg_reward"
data = pd.read_csv(input_file)

# Definir o tamanho da janela para a média móvel
rolling_window = 10  # Altere o valor da janela, se necessário

# Aplicar a média móvel nas colunas desejadas
data['tempo_execucao_ma'] = data['tempo'].rolling(window=rolling_window).mean()
data['sum_loss_ma'] = data['loss'].rolling(window=rolling_window).mean()
data['sum_reward_ma'] = data['reward'].rolling(window=rolling_window).mean()

# Criação de uma figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 1. Tempo de execução por episódio (média móvel e escala log no eixo Y)
axs[0].plot(data["episodio"], data["tempo_execucao_ma"], marker='o', linestyle='-', color='b',
            label='Média Movel - Tempo Execução')
axs[0].set_title("Tempo de Execução por Episódio (Média Móvel)")
axs[0].set_xlabel("Episódio")
axs[0].set_ylabel("Tempo de Execução")
axs[0].set_yscale('log')  # Escala logarítmica no eixo Y
axs[0].legend()
axs[0].grid()

# 2. Loss acumulada por episódio (média móvel e escala log no eixo Y)
axs[1].plot(data["episodio"], data["sum_loss_ma"], marker='o', linestyle='-', color='r',
            label='Média Móvel - Loss Acumulada')
axs[1].set_title("Loss Acumulada por Episódio (Média Móvel)")
axs[1].set_xlabel("Episódio")
axs[1].set_ylabel("Loss Acumulada")
axs[1].set_yscale('log')  # Escala logarítmica no eixo Y
axs[1].legend()
axs[1].grid()

# 3. Recompensa acumulada por episódio (média móvel e escala log no eixo Y)
axs[2].plot(data["episodio"], data["sum_reward_ma"], marker='o', linestyle='-', color='g',
            label='Média Móvel - Recompensa Acumulada')
axs[2].set_title("Recompensa Acumulada por Episódio (Média Móvel)")
axs[2].set_xlabel("Episódio")
axs[2].set_ylabel("Recompensa Acumulada")
axs[2].set_yscale('log')  # Escala logarítmica no eixo Y
axs[2].legend()
axs[2].grid()

# Ajusta o layout para evitar sobreposição
plt.tight_layout()

# Salva o gráfico como imagem
fig.savefig(output_file)
print(f"Gráfico salvo em: {output_file}")