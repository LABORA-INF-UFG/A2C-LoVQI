import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def getStatus(df):
    # drop all the lines which all columns are zeros
    df = df[(df.T != 0).any()]
    # Get the visited states
    # visited_states = index of lines in df
    visited_states = df.index
    # Get the max reward
    max_reward = df.max().max()
    # Get the min reward
    min_reward = df.min().min()
    # Get the mean reward
    mean_reward = df.mean().mean()
    return visited_states, max_reward, min_reward, mean_reward

path = 'data/12x15_20240429-2201/'

# Get the qtable file names with .npy extension on data/ directory
qtable_files = os.listdir(path)
qtable_files = [file for file in qtable_files if file.endswith('.npy')]

# Get the visited states, max reward, min reward and mean reward for each qtable file
for qtable_file in qtable_files:
    visited_states, max_reward, min_reward, mean_reward = getStatus(pd.DataFrame(np.load(f'{path}{qtable_file}')))
    print(f'\nFile: {qtable_file} ' + '---------' * 5)
    print(f'Visited states for {qtable_file}: {visited_states}')
    print(f'Max Reward for {qtable_file}: {max_reward}')
    print(f'Min Reward for {qtable_file}: {min_reward}')
    print(f'Mean Reward for {qtable_file}: {mean_reward}')
    print('---------' * 8)




