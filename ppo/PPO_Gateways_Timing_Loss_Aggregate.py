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
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/ppo"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

# Criação de uma figura com 3 subgráficos lado a lado
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Variáveis para rastrear se dados foram encontrados
data_found = False
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'X', 'd', 'P', '|', '_']

# Selecionar apenas o valor de 3 gateways
# gp = 5
all_seeds_data = []
font_size = 18
label_size = 18
tick_size = 16

# Iterar pelos valores de gateways fornecidos
for gp in gps:
    for seed in range(1, 21):  # Iterar pelos valores de seed de 1 a 10
        input_file = os.path.join(input_dir, f'OLD/PPO_results_{vp}V_{gp}G_{dp}D_{seed}S-OLD.dat')

        if not os.path.exists(input_file):
            print(f"Arquivo não encontrado: {input_file}")
            continue  # Pule para o próximo seed

        # Lê o arquivo .dat usando pandas removendo a linha de cabeçalho e filtra para os primeiros 100 episódios
        data = pd.read_csv(input_file,
                           names=['episodio', 'tempo', 'reward', 'q_reward', 'actor_loss', 'critic_loss'],
                           skiprows=1)

        # Filtrar os dados para apenas os primeiros 100 episódios
        data = data[data['episodio'] <= 300]

        all_seeds_data.append(data)

    # Verifica se dados foram encontrados
    if not all_seeds_data:
        print(f"Nenhum dado encontrado para {gp} Gateways!")
        continue

    # Combina os dados de todas as seeds em um único DataFrame
    all_data = pd.concat(all_seeds_data, ignore_index=True)

    # Calcula a média por episódio agrupando por número de gateways
    mean_data = all_data.groupby('episodio').mean().reset_index()

    # Normalizar actor_losses por z-score
    # Calcula a média e o desvio padrão da coluna 'actor_loss'
    # mean = mean_data['actor_loss'].mean()
    # std = mean_data['actor_loss'].std()
    # # Aplica a fórmula de Z-score
    # mean_data['actor_loss'] = mean_data['actor_loss'].apply(lambda x: (x - mean) / std)

    # Adicionar flag indicando que encontramos dados
    data_found = True


    # Plotar os dados médios por episódio
    # Plot 1: Tempo de execução médio
    axs[0][0].plot(mean_data["episodio"], mean_data["tempo"], linestyle='-', label=f'{gp} VANTs')
    axs[0][0].set_title(f"Tempo Médio de Execução", fontsize=font_size)
    axs[0][0].set_xlabel("Episódio", fontsize=label_size)
    axs[0][0].set_ylabel("Tempo de Execução", fontsize=label_size)
    axs[0][0].legend(fontsize=tick_size)
    axs[0][0].tick_params(axis='both', which='major', labelsize=tick_size)
    axs[0][0].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    # Plot 3: Recompensa acumulada
    axs[0][1].plot(mean_data["episodio"], mean_data["q_reward"], linestyle='-', label=f'{gp} VANTs')
    axs[0][1].set_title(f"Recompensa Acumulada Média", fontsize=font_size)
    axs[0][1].set_xlabel("Episódio", fontsize=label_size)
    axs[0][1].set_ylabel("Recompensa", fontsize=label_size)
    axs[0][1].legend(fontsize=tick_size)
    axs[0][1].tick_params(axis='both', which='major', labelsize=tick_size)
    axs[0][1].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    # Plot 2: Loss (actor e critic)
    axs[1][0].plot(mean_data["episodio"], mean_data["actor_loss"], linestyle='--', label=f'{gp} VANTs')
    axs[1][0].set_title(f"Perda Média do Ator", fontsize=font_size)
    axs[1][0].set_xlabel("Episódio", fontsize=label_size)
    axs[1][0].set_ylabel("Perdas", fontsize=label_size)
    axs[1][0].legend(fontsize=tick_size)
    # axs[1][0].set_yscale('log')
    axs[1][0].tick_params(axis='both', which='major', labelsize=tick_size)
    axs[1][0].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    axs[1][1].plot(mean_data["episodio"], mean_data["critic_loss"], linestyle='-', label=f'{gp} VANTs')
    axs[1][1].set_title(f"Perda Média do Critico", fontsize=font_size)
    axs[1][1].set_xlabel("Episódio", fontsize=label_size)
    axs[1][1].set_ylabel("Perdas", fontsize=label_size)
    # if log:
    #     axs[1][1].set_yscale('log')
    axs[1][1].legend(fontsize=tick_size)
    axs[1][1].tick_params(axis='both', which='major', labelsize=tick_size)
    axs[1][1].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)



# Ajusta o layout para evitar sobreposição
if data_found:
    plt.tight_layout()

    # Nome do arquivo de saída
    output_file = os.path.join(output_dir, f"graf_PPO_results_{vp}V_{gps}G_{dp}D_avg.png")

    # Salva o gráfico como imagem
    fig.savefig(output_file)
    print(f"Gráfico salvo em: {output_file}")

    plt.show()
else:
    print("Nenhum dado encontrado para os arquivos com seed para 3 Gateways.")