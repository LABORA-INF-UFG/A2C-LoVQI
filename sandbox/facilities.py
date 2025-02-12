import os

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Save the Q-table to a file
def save_qtable(path='', q_table=None):
    # Save the Q-table to a file
    np.save(f'{path}/QTable.npy', q_table)


# Plot the average reward for each episode
def plot_avg_reward(path='', output='screen', means_reward=None, m=None):
    if means_reward is None:
        means_reward = []
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(np.arange(len(means_reward)), means_reward, linestyle='solid', color='blue', linewidth=2)
    ax.set_title(f'Agents: {m.n_agents} , Grid: {m.dim_grid}x{m.dim_grid}, Movements per Agent: {m.n_actions}')
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Avg Reward")
    if output == 'file':
        plt.savefig(f'{path}/avg_reward.png')
    plt.show()


def plot_losses(path='', output='screen', losses=None):
    if losses is None:
        losses = []
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(np.arange(len(losses)), losses, linestyle='solid', color='blue', linewidth=2)
    ax.set_title(f'Losses')
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    if output == 'file':
        plt.savefig(f'{path}/losses.png')
    plt.show()


#
def plot_visited_states(path='', output='file', episodes_visited_states=None, states=0):
    lsts = np.array(np.zeros(0), dtype='float64')
    # creates a unique list with all visited states for all episodes and steps (for histogram analysis)
    for v in episodes_visited_states:
        for i in v :
            lsts.append(v)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title(f'Visited States')
    ax.set_xlabel("States")
    ax.set_ylabel("Frequency")
    lsts = np.float64(lsts)
    plt.hist(lsts, bins=states)
    plt.ylim(0, states)
    if output == 'file':
        plt.savefig(f'{path}/visited_states.png')


def gif_visited_states(path='', devices_file='', episodes_visited_states=None, m=None):
    # Create an animated gif with the visited positions for all episodes and steps
    # (for visualization purposes)
    # get cwd
    # Read the devices file
    devices = pd.read_csv(devices_file, sep=" ", names=['x', 'y', 'z'])
    for index, visited in enumerate(episodes_visited_states):
        # Save only 20% of the episodes
        if index % 5 != 0:
            continue
        for i, visited_state in enumerate(visited):
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.set_xlabel("X-Axis")
            ax.set_ylabel("Y-Axis")
            ax.set_xlim(0, m.dim_grid * m.step_size)
            ax.set_ylim(0, m.dim_grid * m.step_size)
            ax.set_xticks(np.arange(0, m.dim_grid * m.step_size, m.step_size))
            ax.set_yticks(np.arange(0, m.dim_grid * m.step_size, m.step_size))
            ax.grid(which='both', alpha=0.5)
            positions = m.coordinates_from_state(visited_state)
            ax.scatter(devices['x'], devices['y'], color='blue', s=50, marker='o')
            for j in range(0, len(positions), 3):
                ax.scatter(positions[j], positions[j + 1], color='red', s=100, marker='*')
            plt.title(f'Visited State {i}: Agents: {len(positions)}, Movements per Agent: {len(visited)}')
            plt.savefig(f'{path}/visited_{i}.png')
            plt.close()

        #  Create a GIF with all images from each episode
        images = []
        for i in range(len(visited)):
            images.append(
                imageio.imread(f'{path}/visited_{i}.png'))
        imageio.mimsave(f'{path}/episode_e{index}.gif',
                        images, duration=0.75)
        # remove all png files in path
        [os.remove(f'{path}/visited_{i}.png') for i in range(len(visited))]


def plot_exploration(path='', output='file', expl=None, episodes=0):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title(f'Exploration')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Exploration")
    ex = np.array(expl)
    count_ones = np.sum(ex == 1, axis=1)
    ax.plot(np.arange(episodes), count_ones, linestyle='solid', color='blue', linewidth=2)
    if output == 'file':
        plt.savefig(f'{path}/exploration.png')
    plt.show()


def get_collision(target_position):
    # Check for collisions
    for i in range(0, len(target_position), 3):
        for j in range(i + 3, len(target_position), 3):
            if target_position[i] == target_position[j] and target_position[i + 1] == target_position[j + 1]:
                return True
    return False
