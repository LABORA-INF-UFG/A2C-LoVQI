import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(input_dir, pc, gw, dv, algorithm):
    """
    Carrega os dados dos arquivos correspondentes a um algoritmo específico.
    """
    file_map = {
        "DQN": f"treinamento/DQN_results_{pc}V_{gw}G_{dv}DD.dat",
        "DDQN": f"treinamento/DDQN_results_{pc}V_{gw}G_{dv}D.dat",
        "PPO": f"treinamento/ppo/PPO_results_{pc}V_{gw}G_{dv}D_1S.dat",
        "A2C": f"new_tr/A2C_results_{pc}V_{gw}G_{dv}D_1SS.dat"
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
        return data[:100]
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        return None


def process_data(data, metric):
    """
    Processa os dados para calcular os valores acumulados de uma métrica.
    """
    if data is None:
        return None
    return data[metric].expanding().mean()


def plot_results(accum_data, algorithms, metric_labels, gateways, output_dir, pc, dv, title_size=18, tick_size=16,
                 label_size=18):
    """
    Gera gráficos comparativos com base nos dados acumulados para os diferentes algoritmos e gateways.
    """

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    styles = ['-', '--']  # Diferentes traços para diferentes conjuntos de gateways
    colors = {
        "DQN": "tab:red",
        "DDQN": "tab:blue",
        "PPO": "tab:green",
        "A2C": "tab:orange"
    }
    labels = {'a2c': 'NSE-A2C',
              'ppo': 'RG2E-PPO',
              'dqn': 'SR-DQN',
              'ddqn': 'DA-DDDQN'}

    # # Plot do tempo médio acumulado
    axes.set_title("Tempo de Execução", fontsize=title_size)
    axes.set_xlabel("Episódio", fontsize=label_size)
    axes.set_ylabel("Tempo", fontsize=label_size)
    axes.tick_params(axis='both', labelsize=tick_size)

    for i, gw_set in enumerate(gateways):
        for algorithm in algorithms:
            data = accum_data[(algorithm, gw_set)]
            if data is not None:
                # Gráfico do tempo
                axes.plot(data["tempo"], label=f"{labels[algorithm]} ({gw_set} GW)", linestyle=styles[i],
                          color=colors[algorithm])
                # # Gráfico da recompensa
                # axes[1].plot(data["reward"], label=f"{algorithm} ({gw_set} GW)", linestyle=styles[i],
                #              color=colors[algorithm])

    # Ajustes e legendas
    axes.legend(fontsize=tick_size)
    axes.grid()
    axes.legend()
    axes.grid()
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
input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/"
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"
pc = 64  # Posições candidatas
dv = 100  # Número de dispositivos

main(input_dir, output_dir, pc, dv)