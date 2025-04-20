import os
import pandas as pd
import matplotlib.pyplot as plt

# Diretórios de entrada e saída
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Arquivos de dados de entrada para os algoritmos
file_paths = {
    "PPO": os.path.join(input_dir, "ppo/PPO_results.dat"),
    "A2C": os.path.join(input_dir, "a2c/A2C_results.dat"),
    "DDQN": os.path.join(input_dir, "ddqn/DDQN_results.dat"),
    "DQN": os.path.join(input_dir, "dqn/DQN_results.dat"),
}

# Variáveis para armazenar os resultados médios
results = {
    "PPO": {"episodios": [], "tempo": [], "recompensa": []},
    "A2C": {"episodios": [], "tempo": [], "recompensa": []},
    "DDQN": {"episodios": [], "tempo": [], "recompensa": []},
    "DQN": {"episodios": [], "tempo": [], "recompensa": []},
}

# Leitura e processamento dos dados
for method, file_path in file_paths.items():
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        continue

    # Lê o arquivo de dados usando pandas
    data = pd.read_csv(file_path)

    # Remove episódios duplicados
    data = data.drop_duplicates(subset=["episodio"], keep="first")

    # Calcula médias agrupando por episódio
    mean_data = data.groupby("episodio").mean()

    # Armazena os dados processados
    results[method]["episodios"] = mean_data.index.tolist()
    results[method]["tempo"] = mean_data["tempo"].tolist()
    results[method]["recompensa"] = mean_data["q_reward"].tolist()

# Criação do gráfico comparativo
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Subgráfico 1: Tempo de execução
for method, data in results.items():
    if data["episodios"]:  # Verifica se há dados para o método
        axs[0].plot(data["episodios"], data["tempo"], label=method)

axs[0].set_title("Comparação de Tempo de Execução", fontsize=20)
axs[0].set_xlabel("Episódios", fontsize=18)
axs[0].set_ylabel("Tempo de Execução (s)", fontsize=18)
axs[0].legend(fontsize=16)
axs[0].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
axs[0].tick_params(axis='both', which='major', labelsize=18)

# Subgráfico 2: Recompensas acumuladas
for method, data in results.items():
    if data["episodios"]:  # Verifica se há dados para o método
        axs[1].plot(data["episodios"], data["recompensa"], label=method)

axs[1].set_title("Comparação de Recompensas Acumuladas", fontsize=20)
axs[1].set_xlabel("Episódios", fontsize=18)
axs[1].set_ylabel("Recompensas Acumuladas", fontsize=18)
axs[1].legend(fontsize=16)
axs[1].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
axs[1].tick_params(axis='both', which='major', labelsize=18)

# Ajusta o layout do gráfico e salva como imagem
plt.tight_layout()
output_file = os.path.join(output_dir, "comparativo_algoritmos.png")
plt.savefig(output_file)
print(f"Gráfico salvo em: {output_file}")

# Exibe o gráfico na tela
plt.show()