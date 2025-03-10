import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t


# Função para calcular o intervalo de confiança de 95%
def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data, axis=0)
    error_margin = sem(data, axis=0) * t.ppf((1 + confidence) / 2, n - 1)  # 95% IC
    return mean, error_margin


# Argumentos de entrada
parser = argparse.ArgumentParser(description='PLOTS of A2C for multiple seeds')
parser.add_argument('--v', type=int, help='Number of Virtual Positions', required=True)
parser.add_argument('--g', type=int, help='Number of Gateways', required=True)
parser.add_argument('--d', type=int, help='Number of Devices', required=True)
args = parser.parse_args()

vp = args.v
gp = args.g
dp = args.d

# Caminhos para leitura e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"
output_file = os.path.join(output_dir, f"graf_A2C_results_{vp}V_{gp}G_{dp}D.png")

# Armazena os dados dos arquivos
n_seeds = 18  # Número de seeds/arquivos
columns = ["episodio", "tempo", "reward", "actor_loss", "critic_loss"]
all_data = {col: [] for col in columns}

# Identifica o tamanho máximo de episódios para padronizar
max_len = 0

# Lê os arquivos
for seed in range(1, n_seeds):
    filename = f"A2C_results_{vp}V_{gp}G_{dp}D_{seed}S.dat"
    file_path = os.path.join(input_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo {file_path} não encontrado!")

    # Lê os dados e remove valores não numéricos
    data = pd.read_csv(file_path, names=columns)

    for col in columns:
        # Converte valores não numéricos para NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Atualiza o tamanho máximo de episódios
    max_len = max(max_len, len(data))

    # Salva os dados das métricas individualmente por seed
    for col in columns:
        all_data[col].append(data[col].values)

# Padroniza os dados para tamanhos iguais
for col in all_data:
    standardized_data = []
    for array in all_data[col]:
        # Converte o array para float antes de padronizar o tamanho
        array = array.astype(float)
        # Se o array for menor que max_len, preenche com NaNs
        if len(array) < max_len:
            padded_array = np.pad(array, (0, max_len - len(array)), constant_values=np.nan)
        else:
            padded_array = array
        standardized_data.append(padded_array)
    all_data[col] = np.array(standardized_data)  # Converte para numpy array

# Define os episódios (supondo que todos os arquivos tenham os mesmos episódios após padronização)
episodios = np.arange(1, max_len + 1)

# Calcula a média e o intervalo de confiança para cada métrica
mean_tempo, ci_tempo = compute_confidence_interval(all_data["tempo"])
mean_reward, ci_reward = compute_confidence_interval(all_data["reward"])
mean_actor_loss, ci_actor_loss = compute_confidence_interval(all_data["actor_loss"])
mean_critic_loss, ci_critic_loss = compute_confidence_interval(all_data["critic_loss"])

# Criação da figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Gráfico 1: Tempo médio de execução com barras de erro (intervalo de confiança)
axs[0].errorbar(episodios, mean_tempo, yerr=ci_tempo, label="Tempo Médio", color="b", fmt='-o', capsize=3)
axs[0].set_title("Tempo Médio de Execução")
axs[0].set_xlabel("Episódios")
axs[0].set_ylabel("Tempo (s)")
axs[0].set_yscale("log")
axs[0].grid()
axs[0].legend()

# Gráfico 2: Perdas acumuladas (Actor Loss e Critic Loss) com barras de erro
axs[1].errorbar(episodios, mean_actor_loss, yerr=ci_actor_loss, label="Actor Loss Média", color="r", fmt='-o',
                capsize=3)
axs[1].errorbar(episodios, mean_critic_loss, yerr=ci_critic_loss, label="Critic Loss Média", color="orange", fmt='-o',
                capsize=3)
axs[1].set_title("Perdas Acumuladas")
axs[1].set_xlabel("Episódios")
axs[1].set_ylabel("Loss")
axs[1].set_yscale("log")
axs[1].grid()
axs[1].legend()

# Gráfico 3: Recompensas acumuladas com barras de erro
axs[2].errorbar(episodios, mean_reward, yerr=ci_reward, label="Recompensa Média", color="g", fmt='-o', capsize=3)
axs[2].set_title("Recompensas Acumuladas")
axs[2].set_xlabel("Episódios")
axs[2].set_ylabel("Recompensa")
axs[2].set_yscale("log")
axs[2].grid()
axs[2].legend()

# Ajusta o layout
plt.tight_layout()

# Salva o gráfico como imagem
fig.savefig(output_file)
print(f"Gráfico salvo em: {output_file}")

# Mostra o gráfico
plt.show()