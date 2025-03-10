import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
n_seeds = 8  # Número de seeds/arquivos
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

# Calcula a média para cada métrica
mean_tempo = np.nanmean(all_data["tempo"], axis=0)
mean_reward = np.nanmean(all_data["reward"], axis=0)
mean_actor_loss = np.nanmean(all_data["actor_loss"], axis=0)
mean_critic_loss = np.nanmean(all_data["critic_loss"], axis=0)

# Criação da figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Gráfico 1: Tempo médio de execução com execuções individuais
for seed_data in all_data["tempo"]:
    axs[0].plot(episodios, seed_data, color="b", alpha=0.2)  # Linhas individuais com transparência
axs[0].plot(episodios, mean_tempo, label="Tempo Médio", color="b", linestyle="-")  # Linha sólida para a média
axs[0].set_title("Tempo Médio de Execução")
axs[0].set_xlabel("Episódios")
axs[0].set_ylabel("Tempo (s)")
# axs[0].set_yscale("log")
axs[0].grid()
axs[0].legend()

# Gráfico 2: Perdas acumuladas (Actor Loss e Critic Loss) com execuções individuais
for seed_data in all_data["actor_loss"]:
    axs[1].plot(episodios, seed_data, color="r", alpha=0.2)  # Actor loss individual com transparência
for seed_data in all_data["critic_loss"]:
    axs[1].plot(episodios, seed_data, color="orange", alpha=0.2)  # Critic loss individual com transparência
axs[1].plot(episodios, mean_actor_loss, label="Actor Loss Média", color="r", linestyle="-")  # Linha sólida média
axs[1].plot(episodios, mean_critic_loss, label="Critic Loss Média", color="orange", linestyle="-")  # Linha sólida média
axs[1].set_title("Perdas Acumuladas")
axs[1].set_xlabel("Episódios")
axs[1].set_ylabel("Loss")
# axs[1].set_yscale("log")
axs[1].grid()
axs[1].legend()

# Gráfico 3: Recompensas acumuladas com execuções individuais
for seed_data in all_data["reward"]:
    axs[2].plot(episodios, seed_data, color="g", alpha=0.2)  # Linhas individuais com transparência
axs[2].plot(episodios, mean_reward, label="Recompensa Média", color="g", linestyle="-")  # Linha sólida para a média
axs[2].set_title("Recompensas Acumuladas")
axs[2].set_xlabel("Episódios")
axs[2].set_ylabel("Recompensa")
# axs[2].set_yscale("log")
axs[2].grid()
axs[2].legend()

# Ajusta o layout
plt.tight_layout()

# Salva o gráfico como imagem
fig.savefig(output_file)
print(f"Gráfico salvo em: {output_file}")

# Mostra o gráfico
plt.show()