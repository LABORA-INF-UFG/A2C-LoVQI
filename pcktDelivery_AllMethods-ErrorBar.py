import os
import pandas as pd
import matplotlib.pyplot as plt
output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

def generate_packet_delivery_graphs(base_path, methods, seeds, virtual_positions, devices, gateways, title_size=18, tick_size=16,
                 label_size=18):
    """
    Gera gráficos de entrega de pacotes por método DRL considerando dados fornecidos.

    :param base_path: Caminho base da pasta contendo os arquivos.
    :param methods: Lista de métodos (por exemplo, ["a2c", "ppo"]).
    :param seeds: Lista de seeds usadas nos arquivos.
    :param virtual_positions: Lista de números de posições virtuais.
    :param devices: Lista de números de dispositivos.
    """
    # Fix 1: Criar dicionário para armazenar taxas agregadas separado por <v>, <d>, e <g>
    delivery_rate_v = {m: {g: [0] * len(virtual_positions) for g in gateways} for m in methods}
    delivery_rate_d = {m: {g: [0] * len(devices) for g in gateways} for m in methods}
    colors = {
        "dqn": "tab:red",
        "ddqn": "tab:blue",
        "ppo": "tab:green",
        "a2c": "tab:orange"
    }

    # Iterar sobre métodos
    for method in methods:
        # Iterar sobre posições virtuais (<v>) e dispositivos (<d>)
        for vi, v in enumerate(virtual_positions):
            gateways_ = [4] if method in ["dqn", "ddqn"] else gateways
            for g in gateways_:
                rates = []
                for d in devices:
                    for s in seeds:
                        # Criar o caminho do arquivo
                        file_name = f"{method}_PcktDlv_{s}s_{v}G_{d}D.dat" if method in ["dqn", "ddqn"] else f"{method}_PcktDlv_{s}s_{v}V_{g}Gx{d}D.dat"
                        file_path = os.path.join(base_path, method, file_name)
                        if os.path.exists(file_path):
                            # Ler o arquivo
                            df = pd.read_csv(file_path, skiprows=1, names=["seed", "sent", "received"])
                            sent = df["sent"].values[0]
                            received = df["received"].values[0]
                            rate = (received / sent) * 100  # Taxa em porcentagem
                            rates.append(rate)
                            if method in ["dqn", "ddqn"]:
                                print(f"File found: {file_path}")
                        # else:
                        #     print(f"File not found: {file_path}")
                # Fix 2: Calcular a média geral para cada posição virtual <v>, separado por gateway <g>
                if rates:
                    mean_rate = sum(rates) / len(rates)
                    std_err = (pd.Series(rates).std() / (len(rates) ** 0.5)) if len(rates) > 1 else 0
                    ci_95 = 1.96 * std_err  # 95% confidence interval
                    delivery_rate_v[method][g][vi] = (mean_rate, ci_95)

        # Agregar por número de dispositivos <d>
        for di, d in enumerate(devices):
            gateways_ = [4] if method in ["dqn", "ddqn"] else gateways
            for g in gateways_:
                rates = []
                for v in virtual_positions:
                    for s in seeds:
                        # Criar o caminho do arquivo
                        file_name = f"{method}_PcktDlv_{s}s_{v}G_{d}D.dat" if method in ["dqn", "ddqn"] else f"{method}_PcktDlv_{s}s_{v}V_{g}Gx{d}D.dat"
                        file_path = os.path.join(base_path, method, file_name)
                        if os.path.exists(file_path):
                            # Ler o arquivo
                            df = pd.read_csv(file_path, skiprows=1, names=["seed", "sent", "received"])
                            sent = df["sent"].values[0]
                            received = df["received"].values[0]
                            rate = (received / sent) * 100  # Taxa em porcentagem
                            rates.append(rate)
                            if method in ["dqn", "ddqn"]:
                                print(f"File found: {file_path}")
                        # else:
                        #     print(f"File not found: {file_path}")
                # Fix 3: Calcular a média geral para cada dispositivo <d>, separado por gateway <g>
                if rates:
                    mean_rate = sum(rates) / len(rates)
                    std_err = (pd.Series(rates).std() / (len(rates) ** 0.5)) if len(rates) > 1 else 0
                    ci_95 = 1.96 * std_err  # 95% confidence interval
                    delivery_rate_d[method][g][di] = (mean_rate, ci_95)

    # Criar os gráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    lines = {2: "-", 4: "--"}
    labels = {'a2c': 'NSE-A2C',
              'ppo': 'RG2E-PPO',
              'dqn': 'SR-DQN',
              'ddqn': 'DA-DDDQN'}

    # Gráfico Taxa de entrega de pacotes x Número de Posições Virtuais
    ax1 = axes[0]
    for method in methods:
        gateways_ = [4] if method in ["dqn", "ddqn"] else gateways
        for g in gateways_:
            values = delivery_rate_v[method][g]
            non_zero_indices = [i for i, val in enumerate(values) if val != 0]
            virtual_positions_non_zero = [virtual_positions[i] for i in non_zero_indices]
            rates_non_zero = [values[i] for i in non_zero_indices]
            mean_rates = [delivery_rate_v[method][g][vi][0] for vi in non_zero_indices]
            ci_95_intervals = [delivery_rate_v[method][g][vi][1] for vi in non_zero_indices]
            if method in ["dqn", "ddqn"]:
                # ax1.plot(virtual_positions_non_zero, mean_rates, marker="o",
                #          label=f"{labels[method]} {g} GW)", color=colors.get(method, "black"), linestyle=lines[g])
                ax1.errorbar(virtual_positions_non_zero, mean_rates, yerr=ci_95_intervals,
                             fmt="o", label=f"{labels[method]} {g} GW)",
                             color=colors.get(method, "black"), linestyle=lines[g], capsize=5)
            else:
                # ax1.plot(virtual_positions, delivery_rate_v[method][g], marker="o",
                #          label=f"{labels[method]} {g} GW", color=colors.get(method, "black"), linestyle=lines[g])
                ax1.errorbar(virtual_positions, mean_rates, yerr=ci_95_intervals,
                             fmt="o", label=f"{labels[method]} {g} GW",
                             color=colors.get(method, "black"), linestyle=lines[g], capsize=5)
    ax1.set_title("Taxa de Entrega de Pacotes x Nº de Pos. Candidatas", fontsize=title_size)
    ax1.set_xlabel("Número de Posições Virtuais", fontsize=label_size)
    ax1.set_ylabel("Taxa de Entrega de Pacotes (%)", fontsize=label_size)
    ax1.tick_params(axis='both', labelsize=tick_size)
    ax1.set_xticks(virtual_positions)
    ax1.legend()
    ax1.grid()

    # Gráfico Taxa de entrega de pacotes x Número de Dispositivos
    ax2 = axes[1]
    for method in methods:
        gateways_ = [4] if method in ["dqn", "ddqn"] else gateways
        for g in gateways_:
            values = delivery_rate_d[method][g]
            non_zero_indices = [i for i, val in enumerate(values) if val != 0]
            devices_non_zero = [devices[i] for i in non_zero_indices]
            rates_non_zero = [values[i] for i in non_zero_indices]
            mean_rates = [delivery_rate_d[method][g][di][0] for di in non_zero_indices]
            ci_95_intervals = [delivery_rate_d[method][g][di][1] for di in non_zero_indices]
            if method in ["dqn", "ddqn"]:
                # ax2.plot(devices_non_zero, mean_rates, marker="o",
                #          label=f"{labels[method]} {g} GW", color=colors.get(method, "black"), linestyle=lines[g])
                ax2.errorbar(devices_non_zero, mean_rates, yerr=ci_95_intervals,
                             fmt="o", label=f"{labels[method]} {g} GW",
                             color=colors.get(method, "black"), linestyle=lines[g], capsize=5)
            else:
                # ax2.plot(devices, delivery_rate_d[method][g], marker="o",
                #          label=f"{labels[method]} {g} GW", color=colors.get(method, "black"), linestyle=lines[g])
                ax2.errorbar(devices, mean_rates, yerr=ci_95_intervals,
                             fmt="o", label=f"{labels[method]} {g} GW",
                             color=colors.get(method, "black"), linestyle=lines[g], capsize=5)
    ax2.set_title("Taxa de Entrega de Pacotes x Nº de Dispositivos", fontsize=title_size)
    ax2.set_xlabel("Número de Dispositivos", fontsize=label_size)
    ax2.set_ylabel("Taxa de Entrega de Pacotes (%)", fontsize=label_size)
    ax2.tick_params(axis='both', labelsize=tick_size)
    ax2.set_ylabel("Taxa de Entrega de Pacotes (%)")
    ax2.set_xticks(devices)
    ax2.legend()
    ax2.grid()

    # Mostrar gráficos
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'grafPDR_allMethods.png')
    plt.savefig(output_file)
    plt.show()


# Parâmetros de entrada
metodos = ["a2c", "ppo", "dqn", "ddqn"]
seeds = [1, 2, 3, 4]
n_virtual_positions = [49, 64, 81, 100, 121, 144]
n_gateways = [2, 4]
n_devices = [50, 100, 150, 200]
caminho_dados = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/res"

# Chamar a função para gerar gráficos
generate_packet_delivery_graphs(caminho_dados, metodos, seeds, n_virtual_positions, n_devices, n_gateways)
