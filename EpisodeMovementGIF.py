import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import math

parser = argparse.ArgumentParser(description='MOVEMENTS of DQN for n UAVs')
parser.add_argument('--v', type=int, help='Number of Virtual Positions')
parser.add_argument('--g', type=int, help='Number of Gateways')
parser.add_argument('--d', type=int, help='Number of Devices')
parser.add_argument('--a', type=str, help='Algorithm Method [DQN, A2C, PPO]')
parser.add_argument('--l', type=bool, help='Log Scale')
args = parser.parse_args()

CANDIDATE_POSITIONS = args.v
GATEWAYS = args.g
DEVICES = args.d
METHOD = args.a
log = args.l

# Parameters of the area
AREA_LIMIT = [0, 0, 20000, 20000]
FILE_PATH = '/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ml/treinamento'
GIF_PATH = '/home/rogerio/git/IoT-J2024/plots/img'
DEVICES_PATH = '/home/rogerio/git/ns-allinone-3.42/ns-3.42/scratch/ql-uav-deployment/data/ed/'
FILE_NAME = f'{FILE_PATH}/{METHOD}_episodes_movements_{CANDIDATE_POSITIONS}V_{GATEWAYS}G_{DEVICES}D.dat'
DEVICES_FILE_NAME = f'{DEVICES_PATH}/d_pos_7s+{DEVICES}d.dat'
STEP_SIZE = 20000 / math.sqrt(CANDIDATE_POSITIONS)


# Function to load and process the file
def load_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            values = line.strip().split(',')
            episode = int(values[0])
            step = int(values[1])
            state = values[2].replace('[', '').replace(']', '').strip().split(';')
            next_state = values[3].replace('[', '').replace(']', '').strip().split(';')
            reward = float(values[4])
            info = values[5].replace('[', '').replace(']', '').strip().split(' ')
            data.append((episode, step, state, next_state, reward, info))
    return data

def load_devices(file_path):
    devices = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            values = line.strip().split(' ')
            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
            devices.append((x, y, z))
    return devices


# Function to verify if a drone went out of bounds
def is_out_of_area(coord, action):
    x, y = coord
    return (action[0] * STEP_SIZE + x < AREA_LIMIT[0] or
            action[0] * STEP_SIZE + x > AREA_LIMIT[2] or
            action[1] * STEP_SIZE + y < AREA_LIMIT[1] or
            action[1] * STEP_SIZE + y > AREA_LIMIT[3])


# GIF creation function for all episodes with pauses in "OutOfArea" and "Collision"
def generate_combined_gif(data, devices, output_path):
    # Obtaining all unique episodes
    all_episodes = sorted(set(d[0] for d in data))

    # Graph setup
    fig, ax = plt.subplots()
    ax.set_xlim(AREA_LIMIT[0], AREA_LIMIT[2])
    ax.set_ylim(AREA_LIMIT[1], AREA_LIMIT[3])
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")

    # Create an expanded combined list to include pauses (repetitions)
    expanded_frames = []
    for frame_data in data:
        episode = frame_data[0]
        step = frame_data[1]
        state = frame_data[2]
        info = frame_data[5]

        is_collision = any("Collision" in item for item in info)  # Check for collision
        is_outofarea = any("OutOfArea" in item for item in info)

        # Add repeated frames based on the condition
        if is_collision:
            expanded_frames.extend([frame_data] * 6)  # Add frame 12 times
        elif is_outofarea:
            expanded_frames.extend([frame_data] * 2)  # Add frame 4 times
        else:
            expanded_frames.append(frame_data)  # Add frame only once

    # Function to update the frame
    def update(frame_index):
        ax.clear()
        ax.set_xlim(AREA_LIMIT[0], AREA_LIMIT[2])
        ax.set_ylim(AREA_LIMIT[1], AREA_LIMIT[3])

        # Get the current frame data
        frame_data = expanded_frames[frame_index]
        episode = frame_data[0]
        step = frame_data[1]
        state = frame_data[2]
        info = frame_data[5]

        # Update title and print debugging
        is_collision = any("Collision" in item for item in info)
        is_outofarea = any("OutOfArea" in item for item in info)
        ax.set_title(f"Episode {episode} - Step {step + 1}")
        print(f"Episode {episode} - Step {step + 1} - Collision: {is_collision} - OutOfArea: {is_outofarea}")

        # Add markers to the graph
        markers = []
        for i in range(0, len(state), 3):
            try:
                coords = list(map(float, state[i:i+3]))
                if len(coords) < 3:
                    print(f"Skipping invalid data at frame {frame_index}: {state}")
                    continue
                x, y, z = coords

                if is_collision:
                    marker, = ax.plot(x, y, marker='*', color='red', markersize=12)  # Mark when there's a collision
                elif is_outofarea:
                    marker, = ax.plot(x, y, marker='*', color='white', markersize=14, markeredgecolor='red')  # Out of area
                else:
                    marker, = ax.plot(x, y, marker='*', color='green', markersize=12)  # Normal drone
                markers.append(marker)
            except ValueError as e:
                print(f"Failed to parse coordinates: {state[i:i+3]} at frame {frame_index}. Error: {e}")

        # Adicionar os pontos dos devices com bolas azuis
        for x, y, _ in devices:
            ax.plot(x, y, '.', color='blue', markersize=3)

        return markers

    # Create animation for all combined frames
    ani = FuncAnimation(fig,
                        update,
                        frames=len(expanded_frames),  # Use the number of expanded frames
                        repeat=False,
                        interval=2500,
                        blit=True)

    # Save as GIF using Pillow
    ani.save(output_path, writer='pillow', fps=10)


# Main
if __name__ == "__main__":
    data = load_file(FILE_NAME)
    devices = load_devices(DEVICES_FILE_NAME)
    GIF_NAME = f'{GIF_PATH}/{METHOD}_Movements_{CANDIDATE_POSITIONS}V_{GATEWAYS}G_{DEVICES}D.gif'
    print("Generating combined GIF for all episodes with pauses on 'OutOfArea' and 'Collision'...")
    generate_combined_gif(data, devices, GIF_NAME)
    print(f"Combined GIF generated at: {GIF_NAME}")