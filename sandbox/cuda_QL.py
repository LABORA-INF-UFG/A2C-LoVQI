import argparse
import itertools
import os
import random
from datetime import datetime

import numpy as np
from codetiming import Timer
from ns3gym import ns3env
from torch import multiprocessing

import facilities as f
# Change to ParallelMapping for CPU execution
# from parallel_mapping import ParallelMapping
# Change to CUDAMapping for GPU execution
from cuda_mapping import CUDAMapping as ParallelMapping

"""
Created on Sun Apr 1 15:00:00 2023
Modified on Sun Apr 1 15:00:00 2023
Authors: Rogério S. Silva, Renan R. Oliveira, and Lucas
Email: {rogerio.sousa, renan.oliveira}@ifg.edu.br,
"""

__author__ = ("Rogério S. Silva, Renan R. Oliveira, Lucas Tadeu"
              "Antonio Oliveira-JR, and Kleber V. Cardoso")
__copyright__ = ("Copyright (c) 2024, "
                 "Instituto de Informática - Universidade Federal de Goiás - UFG, and"
                 "Instituto Federal de Goiás - IFG")
__version__ = "0.1.0"
__email__ = ("{rogerio.sousa, renan.oliveira}@ifg.edu.br, lucastsc@gmail.com"
             "{antonio, kleber}@inf.ufg.br")

execution_time = Timer(text="Execution time: {0:.2f} seconds")

# Create a parser
parser = argparse.ArgumentParser(description='Numba Parallel QL for n UAVs')
# Add arguments to the parser
parser.add_argument('--v', type=int, help='Verbose mode')
parser.add_argument('--gr', type=int, help='Grid dimension (dim_grid x dim_grid)')
parser.add_argument('--dv', type=int, help='Number of Devices')
parser.add_argument('--gw', type=int, help='Number of Gateways')
parser.add_argument('--ep', type=int, help='Number of episodes')
parser.add_argument('--st', type=int, help='Number of steps')
parser.add_argument('--ss', type=int, help='Start NS-3 Simulation')
parser.add_argument('--out', type=int, help='Plot the results')
parser.add_argument('--out_term', type=int, help='Output type [file or screen] for results plotting')
parser.add_argument('--progress', type=int, help='Show progress bar')
# Parse the arguments
args = parser.parse_args()

# ns-3 environment
port = 0
devSeed = 6
nDevices = args.dv
nGateways = args.gw
simTime = 600  # seconds
stepTime = 600  # seconds

startSim = args.ss
debug = 0
dim_grid = args.gr
movements = ['up', 'right', 'down', 'left', 'stay']
action_space = list(itertools.product(movements, repeat=nGateways))
position_space = list(itertools.combinations(range(dim_grid * dim_grid), nGateways))
verbose = False if args.v == 0 else True
output = 'file' if args.out_term == 1 else 'screen'
plots_outputs = args.out

# Q-learning settings
episodes = args.ep
steps = args.st


# Shared memory Q_table
def init_arr(q_table=None, q_table_lock=None,
             exploration=None, expo_lock=None,
             episodes_visited_states=None, episodes_lock=None,
             improvements=None, improvements_lock=None,
             means_reward=None, means_reward_lock=None):
    globals()['q_table'] = np.frombuffer(q_table, dtype='double').reshape(len(position_space), len(action_space))
    globals()['arr_lock'] = q_table_lock
    globals()['exploration'] = exploration
    globals()['expo_lock'] = expo_lock
    globals()['episodes_visited_states'] = (np.frombuffer(episodes_visited_states, dtype='int32').
                                            reshape(episodes, steps))
    globals()['episodes_lock'] = episodes_lock
    globals()['improvements'] = improvements
    globals()['improvements_lock'] = improvements_lock
    globals()['means_reward'] = means_reward
    globals()['means_reward_lock'] = means_reward_lock


def stepper(t_r, ep=0):
    global q_table, q_table_lock, exploration, expo_lock, episodes_visited_states, episodes_visited_states_lock
    rw = 0
    for step in t_r:
        # Get new action from Exploration X Exploitation strategy
        action, ex = m.get_action(m.actual_state)
        m.actual_action = action
        # Get the reward for the current state and action
        coord, rw, d, inf = m.ns3_env.step(action)
        # convert coordinates from ns3 to state
        new_state = m.state_from_coordinates(coord)

        # Update Q-table
        m.qtable = q_table
        m.update(rw, new_state)
        q_table = m.qtable

        episodes_visited_states[ep][step] = new_state

        m.accumulated_rewards += rw
        m.actual_state = new_state
        m.message_from_ns3 = inf

        exploration[step + ep * steps] = ex

    return rw


def episoder(ep):
    global q_table, q_table_lock, improvements, improvements_lock, mean_rws, mean_rewards_lock
    env = ns3env.Ns3Env(port=port,
                        startSim=startSim,
                        simSeed=random.randint(1, 200),
                        simArgs={"--nDevices": nDevices,
                                 "--nGateways": nGateways,
                                 "--vgym": 1,
                                 "--verbose": 0,
                                 "--devSeed": devSeed},
                        debug=debug)
    m.ns3_env = env
    state_coord, reward, done, info = m.ns3_env.get_state()
    m.actual_state = m.state_from_coordinates(state_coord)  # convert coordinates to state
    # Update Q-table for the first state

    with q_table_lock:
        m.qtable = q_table
        m.update(reward, m.actual_state)
        q_table = m.qtable

    start_qos_episode = reward
    m.accumulated_rewards = reward

    m.reset_epsilon()

    #  Execute all steps for the episode
    final_qos_episode = stepper(range(steps), ep)

    # Check if the QoS improved
    with improvements_lock:
        if final_qos_episode > start_qos_episode:
            improvements[ep] = 1
        else:
            improvements[ep] = 0

    # Save the mean reward for the episode
    with means_reward_lock:
        means_reward[ep] = m.accumulated_rewards / (steps + 1)
    env.close()


if __name__ == '__main__':
    execution_time.start()
    # Create a shared memory Q-tables
    q_table = multiprocessing.Array('d', np.zeros(len(position_space) * len(action_space), dtype='double'), lock=False)
    exploration = multiprocessing.Array('i', np.zeros(steps * episodes, dtype='int'), lock=False)
    episodes_visited_states = multiprocessing.Array('i', np.zeros(steps * episodes, dtype='int32'), lock=False)
    improvements = multiprocessing.Array('i', np.zeros(episodes, dtype='int'), lock=False)
    mean_rws = multiprocessing.Array('d', np.zeros(episodes, dtype='double'), lock=False)
    q_table_lock = multiprocessing.Lock()
    expo_lock = multiprocessing.Lock()
    episodes_visited_states_lock = multiprocessing.Lock()
    improvements_lock = multiprocessing.Lock()
    mean_rewards_lock = multiprocessing.Lock()

    m = ParallelMapping(ns3_env=None,
                        dim_grid=dim_grid,
                        n_agents=nGateways,
                        n_actions=len(action_space),
                        action_space=action_space,
                        initial_state=0,
                        n_states=len(position_space),
                        state_positions=position_space,
                        epsilon=0.999,
                        epsilon_min=0.1,
                        epsilon_decay=0.999,
                        alpha=0.2,
                        gamma=0.9,
                        q_table=q_table)

    # Create a pool of processes
    pool = multiprocessing.Pool(multiprocessing.cpu_count(), initializer=init_arr,
                                initargs=(q_table, q_table_lock, exploration, expo_lock, episodes_visited_states,
                                          episodes_visited_states_lock, improvements, improvements_lock, mean_rws,
                                          mean_rewards_lock)).map(episoder, range(episodes))

    q_table = np.frombuffer(q_table, dtype='double').reshape(len(position_space), len(action_space))
    episodes_visited_states = np.frombuffer(episodes_visited_states, dtype='int32').reshape(episodes, steps)
    exploration = np.frombuffer(exploration, dtype='int')
    improvements = np.frombuffer(improvements, dtype='int')
    mean_rws = np.frombuffer(mean_rws, dtype='double')

    # Print the results
    if verbose:
        print(
            f'For {episodes} episodes, there was {sum(improvements)} '
            f'improvements ({round(sum(improvements) * 100 / episodes, 2)}%), '
            f'{episodes - sum(improvements)} worse results '
            f'({round((episodes - sum(improvements)) * 100 / episodes, 2)}%)'
        )
    # Create a directory to save the images
    dt_hr = datetime.now().strftime("%Y%m%d-%H%M")
    path = 'data/{}x{}_{}'.format(episodes, steps, dt_hr)
    devices_file = "data/ed/endDevices_LNM_Placement_{}s+{}d.dat".format(devSeed, nDevices)
    os.makedirs(path, exist_ok=True)

    # Save the Q-table to a file
    f.save_qtable(path=path, q_table=q_table)
    if plots_outputs == 1:
        f.save_qtable(path=path, q_table=q_table)
        f.plot_avg_reward(path=path,
                          means_reward=mean_rws,
                          output=output,
                          m=m)

        f.plot_visited_states(path=path,
                              output=output,
                              episodes_visited_states=episodes_visited_states,
                              states=m.n_states)

        f.gif_visited_states(path=path,
                             devices_file=devices_file,
                             episodes_visited_states=episodes_visited_states,
                             m=m)

        # f.plot_exploration(path=path,
        #                    output=output,
        #                    expl=exploration,
        #                    episodes=episodes)

    execution_time.stop()
