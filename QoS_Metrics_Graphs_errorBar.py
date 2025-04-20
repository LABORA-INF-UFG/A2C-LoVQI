import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

def plot_qos_metrics(base_path, methods, seeds, virtual_positions, devices, gateways):
    """
    Lê os arquivos e gera gráficos para as métricas de delay, throughput e QoS.

    :param base_path: Caminho base da pasta contendo os arquivos.
    :param methods: Lista de métodos (por exemplo, ["a2c", "ppo", "dqn", "ddqn"]).
    :param seeds: Lista de seeds usadas nos arquivos.
    :param virtual_positions: Lista de números de posições virtuais.
    :param devices: Lista de números de dispositivos.
    :param gateways: Lista de números de gateways.
    """
    # Configurações
    labels = {'a2c': 'NSE-A2C',
              'ppo': 'RG2E-PPO',
              'dqn': 'SR-DQN',
              'ddqn': 'DA-DDDQN'}
    colors = {
        "dqn": "tab:red",
        "ddqn": "tab:blue",
        "ppo": "tab:green",
        "a2c": "tab:orange"
    }
    title_size = 22
    label_size = 20
    tick_size = 18

    # Função para calcular intervalo de confiança de 95%
    def confidence_interval(data):
        if len(data) > 1:
            return sem(data) * t.ppf((1 + 0.95) / 2, df=len(data) - 1)
        else:
            return 0

    # Preparar armazenamento de resultados
    results_vp = {metric: {method: {g: [] for g in gateways} for method in methods} for metric in
                  ["delay", "throughput", "qos"]}
    results_devices = {metric: {method: {g: [] for g in gateways} for method in methods} for metric in
                       ["delay", "throughput", "qos"]}

    # Iterar pelos métodos e processar os arquivos
    for method in methods:
        for g in gateways:
            for v in virtual_positions:
                metric_data = {metric: [] for metric in ["delay", "throughput", "qos"]}
                for s in seeds:
                    for d in devices:
                        file_name = f"{method}_QoSPerGw_{s}s_{v}V_{g}Gx{d}D.dat"
                        file_path = os.path.join(base_path, method, file_name)
                        if os.path.exists(file_path):
                            # Ler o arquivo com o delimitador correto
                            df = pd.read_csv(file_path, names=["seed", "gwid", "delay", "throughput", "qos"],
                                             delimiter=",")

                            # Limpar dados inválidos
                            for col in ["delay", "throughput", "qos"]:
                                df[col] = pd.to_numeric(df[col], errors="coerce")

                            # Verificar e calcular apenas se houver dados válidos
                            if not df["delay"].isna().all():
                                metric_data["delay"].append(df["delay"].mean())
                            if not df["throughput"].isna().all():
                                metric_data["throughput"].append(df["throughput"].mean())
                            if not df["qos"].isna().all():
                                metric_data["qos"].append(df["qos"].mean())
                for metric in metric_data:
                    results_vp[metric][method][g].append(np.mean(metric_data[metric]) if metric_data[metric] else 0)


            # Analisar por dispositivos
            for d in devices:
                metric_data = {metric: [] for metric in ["delay", "throughput", "qos"]}
                for s in seeds:
                    for v in virtual_positions:
                        file_name = f"{method}_QoSPerGw_{s}s_{v}V_{g}Gx{d}D.dat"
                        file_path = os.path.join(base_path, method, file_name)
                        if os.path.exists(file_path):
                            # Ler o arquivo
                            df = pd.read_csv(file_path, names=["seed", "gwid", "delay", "throughput", "qos"],
                                             delimiter=",")
                            # Limpar dados inválidos
                            for col in ["delay", "throughput", "qos"]:
                                df[col] = pd.to_numeric(df[col], errors="coerce")

                            # Verificar e calcular apenas se houver dados válidos
                            if not df["delay"].isna().all():
                                metric_data["delay"].append(df["delay"].mean())
                            if not df["throughput"].isna().all():
                                metric_data["throughput"].append(df["throughput"].mean())
                            if not df["qos"].isna().all():
                                metric_data["qos"].append(df["qos"].mean())
                # Fazer média para cada metric
                for metric in metric_data:
                    results_devices[metric][method][g].append(
                        np.mean(metric_data[metric]) if metric_data[metric] else 0)

    # Criar gráficos
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    metrics = ["delay", "throughput", "qos"]
    titles = ["Atraso", "Vazão", "QoS"]

    # Define line styles for different gateway numbers
    line_styles = {2: 'o-', 4: 'o--'}

    # Gráficos por Virtual Positions
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        for method in methods:
            for g in gateways:
                values = results_vp[metric][method][g]
                ci = [confidence_interval(values)] * len(virtual_positions) if values else [0] * len(virtual_positions)
                ax.errorbar(virtual_positions, values, yerr=ci, label=f"{labels[method]} {g} GW",
                            color=colors[method], fmt=line_styles[g])
        ax.set_title(f"Média de {titles[i]} x Posições Virtuais", fontsize=title_size)
        ax.set_xlabel("Posições Virtuais (v)", fontsize=label_size)
        ax.set_ylabel(titles[i], fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(fontsize=tick_size)
        ax.grid()

    # Gráficos por Dispositivos
    for i, metric in enumerate(metrics):
        ax = axes[1, i]
        for method in methods:
            for g in gateways:
                values = results_devices[metric][method][g]
                ci = [confidence_interval(values)] * len(devices) if values else [0] * len(devices)
                ax.errorbar(devices, values, yerr=ci, label=f"{labels[method]} {g} GW",
                            color=colors[method], fmt=line_styles[g])
        ax.set_title(f"Média de {titles[i]} x Dispositivos", fontsize=title_size)
        ax.set_xlabel("Dispositivos (d)", fontsize=label_size)
        ax.set_ylabel(titles[i], fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(fontsize=tick_size)
        ax.grid()

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'grafQoS_allMethods.png')
    plt.savefig(output_file)
    plt.show()


# Parâmetros
methods = ["a2c", "ppo"]#, "dqn", "ddqn"]
seeds = [1, 2, 3, 4]
n_virtual_positions = [49, 64, 81, 100, 121, 144]
n_gateways = [2, 4]
n_devices = [50, 100, 150, 200]
caminho_dados = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/res"

# Chamar a função para gerar os gráficos
plot_qos_metrics(caminho_dados, methods, seeds, n_virtual_positions, n_devices, n_gateways)