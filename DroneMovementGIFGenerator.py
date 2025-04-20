import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from matplotlib.animation import FuncAnimation

# Parâmetros gerais (substitua conforme necessário)
AREA_LIMIT = (144, 144)  # Limites da área (x, y)
STEP_SIZE = 1  # Tamanho do passo entre frames

# Funções auxiliares

def load_file(path):
    """Carrega os dados do arquivo especificado."""
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Erro ao carregar arquivo {path}: {e}")
        return None


def load_devices(data):
    """Carrega as informações de dispositivos (drones) dos dados."""
    try:
        devices = data.groupby('device_id').apply(
            lambda df: df[['x', 'y']].values.tolist()
        ).to_dict()
        return devices
    except Exception as e:
        print(f"Erro ao processar dispositivos: {e}")
        return None


def generate_combined_gif(devices_positions, gif_path, step_size=STEP_SIZE):
    """
    Função que gera um GIF combinando os movimentos dos dispositivos.

    Parâmetros:
        devices_positions: dict
            Dicionário onde a chave é o frame (tempo) e o valor são
            as posições (x, y) de cada dispositivo (lista de tuplas).
        gif_path: str
            Caminho onde o GIF final será salvo.
        step_size: int
            Tamanho do intervalo (passo) entre os frames.
    """

    frames = list(devices_positions.keys())
    num_drones = len(devices_positions[frames[0]])

    # Configuração do gráfico
    fig, ax = plt.subplots()
    ax.set_xlim([0, AREA_LIMIT[0]])
    ax.set_ylim([0, AREA_LIMIT[1]])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Drone Movement")

    # Frames para o GIF
    def update(frame_idx):
        """
        Atualiza o gráfico para o frame atual.

        Parâmetros:
            frame_idx: int
                Índice do frame atual no ciclo de animação.
        """
        ax.cla()  # Limpa o gráfico para atualizar

        # Configurar limites e rótulos após limpar
        ax.set_xlim([0, AREA_LIMIT[0]])
        ax.set_ylim([0, AREA_LIMIT[1]])
        ax.set_title(f"Movimento dos Drones - Frame {frames[frame_idx]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        current_frame = frames[frame_idx]
        if frame_idx + 1 < len(frames):
            next_frame = frames[frame_idx + 1]
        else:
            return  # Último frame, sem próximo para calcular vetor

        current_positions = devices_positions[current_frame]
        next_positions = devices_positions[next_frame]

        for drone_id in range(num_drones):
            # Obtém a posição atual e a próxima
            x, y = current_positions[drone_id]
            x_next, y_next = next_positions[drone_id]

            # Plota o drone como ponto
            ax.plot(x, y, "bo", label="Drone" if drone_id == 0 else "")  # 'bo' = ponto azul

            # Calcula a direção do movimento (vetor)
            dx = x_next - x
            dy = y_next - y

            # Plota a seta representando o vetor de movimento
            ax.quiver(
                x, y, dx, dy, angles="xy", scale_units="xy", scale=1, color="r", alpha=0.7
            )

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100)
    anim.save(gif_path, writer="imagemagick")
    plt.close()


# Função principal

def main():
    parser = argparse.ArgumentParser(description="Geração de GIF de movimento dos drones.")
    parser.add_argument("--file_path", type=str, required=True, help="Caminho para o arquivo CSV com os dados.")
    parser.add_argument("--gif_path", type=str, required=True, help="Caminho para salvar o arquivo GIF de saída.")
    args = parser.parse_args()

    # Carregar dados do arquivo
    data = load_file(args.file_path)
    if data is None:
        print("Erro ao carregar os dados do arquivo. Encerrando o programa.")
        return

    # Carregar e processar informações dos dispositivos
    devices = load_devices(data)
    if devices is None:
        print("Erro ao carregar informações dos dispositivos. Encerrando o programa.")
        return

    # Gerar o GIF
    try:
        generate_combined_gif(devices, args.gif_path, step_size=STEP_SIZE)
        print(f"GIF gerado com sucesso: {args.gif_path}")
    except Exception as e:
        print(f"Erro ao gerar o GIF: {e}")


if __name__ == "__main__":
    main()