import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, t


def load_data(input_dir, pc, gw, dv, algorithm):
    """
    Carrega os dados dos arquivos correspondentes a um algoritmo específico.
    """
    file_map = {
        "DQN": f"DQN_results_{pc}V_{gw}G_{dv}DD.dat",
        "DDQN": f"DDQN_results_{pc}V_{gw}G_{dv}D.dat",
        "PPO": f"PPO_results_{pc}V_{gw}G_{dv}D_1S.dat",
        "A2C": f"A2C_results_{pc}V_{gw}G_{dv}D_1SS.dat"
    }

    filename = f"{file_map[algorithm]}"
    file_path = os.path.join(input_dir, filename)

    try:
        if algorithm in ["DQN", "DDQN"]:
            cols = ["episodio", "tempo", "reward", "qualified_reward", "loss"]
        elif algorithm == "PPO":
            cols = ["episodio", "tempo", "reward", "q_reward", "policy_loss", "value_loss"]
        elif algorithm == "A2C":
            cols = ["episodio", "tempo", "reward", "q_reward", "actor_loss", "critic_loss"]
        else:
            raise ValueError(f"Algoritmo desconhecido: {algorithm}")

        data = pd.read_csv(file_path, header=1, names=cols)
        return data[:120]
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        return None


def process_data(data, metric):
    """
    Processa os dados para calcular os valores acumulados de uma métrica.
    Adiciona cálculo de média, intervalo de confiança e desvio padrão.
    """
    if data is None:
        return None

    # Calculate the accumulated mean
    mean_data = data[metric].expanding().mean()

    # Calculate confidence interval
    ci = (
        mean_data
        - t.ppf(0.975, df=len(data[metric]) - 1) * sem(data[metric]),
        mean_data
        + t.ppf(0.975, df=len(data[metric]) - 1) * sem(data[metric])
    )

    return {
        "mean": mean_data,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "std_dev": data[metric].expanding().std(),
    }


def plot_results(accum_data, algorithms, metric_labels, gateways, output_dir, pc, dv, title_size=18, tick_size=16,
                 label_size=18):
    """
    Gera gráficos comparativos com base nos dados acumulados para os diferentes algoritmos e gateways.
    """

    fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    styles = ['-', '--']  # Diferentes traços para diferentes conjuntos de gateways
    colors = {
        "DQN": "tab:red",
        "DDQN": "tab:blue",
        "PPO": "tab:green",
        "A2C": "tab:orange"
    }
    labels = {'A2C': 'NSE-A2C',
              'PPO': 'RG2E-PPO',
              'DQN': 'SR-DQN',
              'DDQN': 'DA-DDDQN'}
    # # Plot do tempo médio acumulado
    axes.set_title("Tempo de Execução", fontsize=title_size)
    axes.set_xlabel("Episódio", fontsize=label_size)
    axes.set_ylabel("Tempo", fontsize=label_size)
    axes.tick_params(axis='both', labelsize=tick_size)
    # axes.set_yscale("log")  # Set y-axis to logarithmic scale
    
    for i, gw_set in enumerate(gateways):
        for algorithm in algorithms:
            data = accum_data[(algorithm, gw_set)]
            if data is not None:
                # Gráfico do tempo
                # Obter valores para plot com barras de erro
                mean = data["tempo"]["mean"]
                ci_lower = data["tempo"]["ci_lower"]
                ci_upper = data["tempo"]["ci_upper"]

                # Plot with error bars
                axes.errorbar(
                    range(len(mean)),
                    mean,
                    yerr=[mean - ci_lower, ci_upper - mean],
                    label=f"{labels[algorithm]} ({gw_set} GW)",
                    linestyle=styles[i], elinewidth=0.1,
                    color=colors[algorithm],
                    capthick=0.1)
                # # Gráfico da recompensa
                # axes[1].plot(data["reward"], label=f"{algorithm} ({gw_set} GW)", linestyle=styles[i],
                #              color=colors[algorithm])

    # Ajustes e legendas
    axes.legend(fontsize=tick_size, loc='upper left', ncol=2)
    axes.grid()
    axes.set_xticks(range(0, 120, 20))
    # Salvar gráficos como arquivos de imagem
    output_path = os.path.join(output_dir, f"resultadosTempo_{pc}V_{dv}D.png")
    plt.savefig(output_path)
    plt.show()


def main(input_dir, output_dir, pc, dv):
    """
    Função principal para carregar dados, processar e gerar gráficos.
    """
    algorithms = ["DQN", "DDQN", "PPO", "A2C"]
    gateways = [2, 4]  # Conjuntos de gateways

    metric_labels = {"A2C":["tempo", "q_reward"],
                     "PPO":["tempo", "q_reward"],
                     "DQN":["tempo", "qualified_reward"],
                     "DDQN":["tempo", "qualified_reward"]
    }  # Métricas a processar (tempo e recompensa) por algoritmo

    # Carregar todos os dados
    accum_data = {}
    for gw in gateways:
        for algorithm in algorithms:
            data = load_data(input_dir, pc, gw, dv, algorithm)
            if data is not None:
                accum_data[(algorithm, gw)] = {
                    "tempo": process_data(data, metric_labels[algorithm][0]),
                    "reward": process_data(data,  metric_labels[algorithm][1])
                }

    # Gerar gráficos
    plot_results(accum_data, algorithms, metric_labels, gateways, output_dir, pc, dv)


# Parâmetros fornecidos pelo usuário
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/tempo_exec/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"
pc = 64  # Posições candidatas
dv = 100  # Número de dispositivos

main(input_dir, output_dir, pc, dv)