import os
import numpy as np
import pandas as pd
from itertools import product
import time


def process_movements(pc, gw, dv, s, episode):
    # Caminhos para os diretórios de entrada e saída
    input_dir = "/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento/"
    output_dir = "/home/rogerio/git/IoT-J2024/plots/img/"

    # Nome do arquivo de entrada
    filename = f"PPO_episodes_movements_{pc}V_{gw}G_{dv}D_{s}S.dat"
    file_path = os.path.join(input_dir, filename)

    # Verifica se o arquivo existe
    if not os.path.exists(file_path):
        print(f"Arquivo {file_path} não encontrado.")
        return

    # Movimentos possíveis
    movements = ["↑", "↓", "→", "←", "X"]  # Norte, Sul, Leste, Oeste, Parado

    # Gera todas as combinações possíveis de movimentos para os gateways
    all_combinations = list(product(movements, repeat=gw))

    # Lê o arquivo
    try:
        data = pd.read_csv(file_path, sep=",")
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    # Verifica colunas necessárias
    required_columns = [
        "episodio",
        "step",
        "state",
        "next_state",
        "reward",
        "q_reward",
        "info",
        "action",
        "step_size",
    ]
    if not all(column in data.columns for column in required_columns):
        print(f"O arquivo não contém todas as colunas necessárias: {required_columns}")
        return

    # Filtro para o episódio informado
    data = data[data["episodio"] == episode]
    if data.empty:
        print(f"Nenhum dado encontrado para o episódio {episode}.")
        return

    # Prepara a matriz de visualização do trajeto
    grid_size = int(np.sqrt(pc))  # Calcula a dimensão do grid baseado na raiz quadrada de `pc`
    grid = np.full((grid_size, grid_size), "º", dtype=str)

    # Controle para marcar posições iniciais e finais
    initial_positions = {}
    final_positions = {}

    # Processa os dados do episódio para preencher a matriz passo a passo
    for idx, row in data.iterrows():

        step = int(row["step"])  # Obtém o passo atual
        step_size = int(row["step_size"])  # Obtém o tamanho do grid do arquivo

        # Processa "state"
        raw_state = row["state"]
        if pd.isna(raw_state) or raw_state.strip() == "":
            print("Aviso: Estado inválido, pulando esta linha.")
            continue

        raw_state = raw_state[1:] if raw_state.startswith(";") else raw_state
        _state = (
            raw_state.replace("[", "")
            .replace(";", ",")
            .replace("]", "")
            .replace(",45", "")
            .strip("()")
            .split(",")
        )

        try:
            state = tuple(
                (step_size - 1) - (coord // step_size) for coord in map(int, _state)
            )  # Converte para coordenadas redimensionadas (x, y)
        except ValueError as ve:
            print(f"Erro ao processar o estado {_state}: {ve}")
            continue

        # Armazena a posição inicial
        if step == 1:
            for g in range(gw):
                x, y = state[g * 2: g * 2 + 2]
                initial_positions[g] = (x % grid_size, y % grid_size)

        # Processa "next_state"
        raw_next_state = row["next_state"]
        if pd.isna(raw_next_state) or raw_next_state.strip() == "":
            print("Aviso: Próximo estado inválido, pulando esta linha.")
            continue

        raw_next_state = raw_next_state[1:] if raw_next_state.startswith(";") else raw_next_state
        _nstate = (
            raw_next_state.replace("[", "")
            .replace(";", ",")
            .replace("]", "")
            .replace(",45", "")
            .strip("()")
            .split(",")
        )

        try:
            next_state = tuple(
                (step_size - 1) - (coord // step_size) for coord in map(int, _nstate)
            )  # Converte para coordenadas redimensionadas (x, y)
        except ValueError as ve:
            print(f"Erro ao processar o próximo estado {_nstate}: {ve}")
            continue

        # Traduz o índice da ação para sua combinação correspondente
        action_index = int(row["action"])
        gateway_movement = all_combinations[action_index]

        # Adicionar as setas representando a direção do movimento de forma individual
        for g, move in enumerate(gateway_movement):
            x, y = state[g * 2: g * 2 + 2]  # Extrai a posição (x, y) do gateway atual
            nx, ny = next_state[g * 2: g * 2 + 2]  # Extrai (x, y) do próximo estado do gateway
            grid[x % grid_size, y % grid_size] = move

            # Atualiza a posição final do gateway
            final_positions[g] = (nx % grid_size, ny % grid_size)

        # Exibe a matriz no final de cada passo
        print(f"Matriz no Passo {step}:")
        for row in grid:
            print(" ".join(row))

        # Adiciona as coordenadas dos gateways abaixo da matriz
        print("\nCoordenadas dos Gateways (State): ", _state)
        print("Coordenadas dos Gateways (Next State): ", _nstate)
        print("-" * 40)
        time.sleep(0.5)  # Pausar brevemente para observar a movimentação passo a passo

    # Marca as posições iniciais ('I') e finais ('F')
    for g, (x, y) in initial_positions.items():
        grid[x, y] = "I"
    for g, (x, y) in final_positions.items():
        grid[x, y] = "F"

    # Exibe a matriz final
    print(f"Matriz Final dos Gateways para o Episódio {episode}:")
    for row in grid:
        print("".join(row))

    # Salva a matriz em um arquivo no diretório de saída
    output_file = os.path.join(output_dir, f"trajeto_{pc}V_{gw}G_{dv}D_{s}S_ep{episode}.txt")
    with open(output_file, "w") as f:
        for row in grid:
            f.write(" ".join(row) + "\n")

    print(f"Matriz salva em: {output_file}")


# Exemplo de chamada com o episódio especificado
process_movements(pc=144, gw=2, dv=200, s=17, episode=3)