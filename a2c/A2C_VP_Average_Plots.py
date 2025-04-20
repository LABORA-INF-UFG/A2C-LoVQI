import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def calculate_episode_averages(data, metrics, seeds):
    """
    Calcula a média de cada métrica por episódio com base nos valores de todos os seeds.

    :param data: Dicionário contendo os dados organizados por episódio e por seed.
    :param metrics: Lista de métricas para análise.
    :param seeds: Lista de seeds usados nos dados.
    :return: DataFrame contendo a média das métricas por episódio.
    """
    # Organiza os dados por episódios
    aggregated_data = {}
    for seed in seeds:
        for episode, values in data[seed].items():
            if episode <= 300:  # Filtra apenas os primeiros 300 episódios
                if episode not in aggregated_data:
                    aggregated_data[episode] = {metric: [] for metric in metrics}
                for metric in metrics:
                    if metric in values:
                        aggregated_data[episode][metric].append(values[metric])

    # Calcula a média de cada métrica por episódio
    avg_data = {
        episode: {metric: pd.Series(values).mean() for metric, values in metrics_data.items()}
        for episode, metrics_data in aggregated_data.items()
    }

    # Cria um DataFrame para facilitar o acesso aos dados
    return pd.DataFrame(avg_data).transpose().sort_index()


def main():
    # Configuração de argumentos do script
    parser = argparse.ArgumentParser(description="Processa e gera gráficos dos resultados A2C.")
    parser.add_argument("--v", nargs='+', type=int, required=True,
                        help="Lista do número de posições virtuais (VPs).")
    parser.add_argument("--d", type=int, required=True, help="Numero de devices.")
    parser.add_argument("--g", type=int, required=True, help="Numero de gateways.")
    args = parser.parse_args()

    # Configuração básica
    gp = args.g  # Número fixo de Gateways
    dp = args.d  # Número fixo de devices
    seeds = list(range(1, 5))  # Supõe-se que há 4 seeds
    metrics = ['tempo', 'actor_loss', 'critic_loss', 'q_reward']  # Métricas analisadas
    data = {vp: {seed: {} for seed in seeds} for vp in args.v}

    input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/new_tr/"
    output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

    # Processa os dados de entrada
    for vp in args.v:
        for seed in seeds:
            input_file = os.path.join(input_dir, f'A2C_results_{vp}V_{gp}G_{dp}D_{seed}SS.dat')
            if os.path.exists(input_file):
                try:
                    df = pd.read_csv(input_file)
                    for _, row in df.iterrows():  # Itera pelas linhas do DataFrame
                        episode = int(row['episodio'])  # Organiza os dados por episódio
                        if episode <= 300:  # Filtra apenas os primeiros 300 episódios
                            if episode not in data[vp][seed]:
                                data[vp][seed][episode] = {metric: 0 for metric in metrics}
                            for metric in metrics:
                                data[vp][seed][episode][metric] += row[metric]
                except Exception as e:
                    print(f"Erro ao processar {input_file}: {e}")
            else:
                print(f"Arquivo não encontrado: {input_file}")

    
    # Calcula as médias dos episódios para os diferentes VPs
    mean_data = {vp: calculate_episode_averages(data[vp], metrics, seeds) for vp in args.v}

    # Gerar gráficos para cada métrica
    fig, axs = plt.subplots(1, len(metrics), figsize=(28, 7))
    axs = axs.flatten()
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'x']

    for metric, ax in zip(metrics, axs):
        for i, vp in enumerate(args.v):
            ax.plot(
                mean_data[vp].index,
                mean_data[vp][metric],
                label=f"VP={vp}",
                linestyle="--"
            )
        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.set_xlabel("Episódios")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    # Salvar gráfico em arquivo
    output_file = os.path.join(output_dir, f"A2C_Gateway_Average_Plots_{gp}G_{dp}D.png")
    plt.savefig(output_file)
    print(f"Gráfico salvo em: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()