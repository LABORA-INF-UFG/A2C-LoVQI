import argparse
import itertools
import os
import random
import numpy as np
import torch
from sandbox import facilities as f
from codetiming import Timer
from ns3gym import ns3env
from tqdm import tqdm
from datetime import datetime
from qlmapping import QLMapping

"""
Created on Sun Apr 1 15:00:00 2023
Modified on Sun Apr 1 15:00:00 2023
Authors: Rogério S. Silva, Renan R. Oliveira, and Lucas, and Xavier Sebastião, and Cleyber
Email: {rogerio.sousa, renan.oliveira}@ifg.edu.br,
"""

__author__ = ("Rogério S. Silva, Renan R. Oliveira, Lucas Tadeu, Xavier Sebastião, "
              "Cleyber Bezerra, Antonio Oliveira-JR, and Kleber V. Cardoso")
__copyright__ = ("Copyright (c) 2024, "
                 "Instituto de Informática - Universidade Federal de Goiás - UFG, and"
                 "Instituto Federal de Goiás - IFG")
__version__ = "0.1.0"
__email__ = ("{rogerio.sousa, renan.oliveira}@ifg.edu.br, lucastsc@gmail.com"
             "{cleyber.bezerra, xavierpaulino, antonio, kleber}@inf.ufg.br")

execution_time = Timer(text="Execution time: {0:.2f} seconds")
execution_time.start()
if not torch.cuda.is_available():
    raise Exception('CUDA is not available. Aborting.')

# Create a parser
parser = argparse.ArgumentParser(description='QL for n UAVs')
# Add arguments to the parser
parser.add_argument('--v', type=int, help='Verbose mode')
parser.add_argument('--pr', type=int, help='Port number')
parser.add_argument('--gr', type=int, help='Grid dimension (gr x gr)')
parser.add_argument('--sz', type=int, help='Area side size')
parser.add_argument('--dv', type=int, help='Number of Devices')
parser.add_argument('--gw', type=int, help='Number of Gateways')
parser.add_argument('--ns', type=int, help='NS3 seed')
parser.add_argument('--ep', type=int, help='Number of episodes')
parser.add_argument('--st', type=int, help='Number of steps')
parser.add_argument('--ss', type=int, help='Start NS-3 Simulation')
parser.add_argument('--out', type=int, help='Plot the results')
parser.add_argument('--out_term', type=int, help='Output type [file or screen] for results plotting')
parser.add_argument('--progress', type=int, help='Show progress bar')
parser.add_argument('--so', type=int, help='Start Optimal')
args = parser.parse_args()

# ns-3 environment
port = args.pr  # port number must be changed to 0 to use a random port
sim_seed = args.ns
nDevices = args.dv
nGateways = args.gw
start_sim = args.ss
start_optimal = args.so if args.so != None else 0
debug = 0 if start_sim == 1 else 1
dim_grid = args.gr
area_side = args.sz  # meters
simTime = 600  # seconds
stepTime = 600  # seconds
simArgs = {"--nDevices": nDevices,
           "--nGateways": nGateways,
           "--vgym": 1,
           "--verbose": 0,
           "--simSeed": sim_seed,
           "--startOptimal": start_optimal,
           "--virtualPositions": math.pow(dim_grid, 2),
           "--areaSide": area_side}

movements = ['up', 'right', 'down', 'left', 'stay']
action_space = list(itertools.product(movements, repeat=nGateways))
position_space = list(itertools.combinations(range(dim_grid * dim_grid), nGateways))
verbose = False if args.v == 0 else True
output = 'file' if args.out_term == 1 else 'screen'
plots_outputs = args.out

# Q-learning settings
episodes = args.ep
steps = args.st
improvements = []
mean_rws = [0]

m = QLMapping(ns3_env=None,
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
              gamma=0.9)

episodes_visited_states = np.zeros((episodes, steps), dtype=int)
collisions = 0
exploration = [[0] * steps] * episodes
for ep in range(episodes):
    seed = random.randint(1, 200)
    env = ns3env.Ns3Env(port=port, startSim=start_sim, simSeed=sim_seed, simArgs=simArgs, debug=debug,
                        spath="/home/rogerio/git/ns-allinone-3.42/ns-3.42/")
    m.ns3_env = env
    # state_coordinates = m.ns3_env.reset()  # Restart the environment and get the initial state
    state_coordinates, reward, done, info = m.ns3_env.get_state()
    state = m.state_from_coordinates(state_coordinates)  # convert coordinates to state
    m.state = state  # Set initial state
    # _, reward, done, info = m.ns3_env.get_state()
    start_qos_episode = reward
    if verbose:
        print(f'\nQos start state (episode {ep}): {start_qos_episode} at {state_coordinates}')
        print(info)
    m.reset_epsilon()
    visited_states = []
    if args.progress == 1:
        t_range = tqdm(range(steps), desc="Progress: ", unit=" steps")
    else:
        t_range = range(steps)
    sum_rewards = reward
    exploit = 0
    for step in t_range:
        # Get new action from Exploration X Exploitation strategy
        action, ex = m.get_action(state)
        if ex == 1:
            exploration[ep][step] = ex
            exploit += 1
        # Get the reward for the current state and action
        state_coordinates, reward, done, info = m.ns3_env.step(action)
        visited_states.append(state)
        # convert coordinates from ns3 to state
        new_state = m.state_from_coordinates(state_coordinates)
        # Update Q-table
        m.update(state, action, reward, new_state)
        sum_rewards += reward
        state = new_state

    episodes_visited_states[ep] = visited_states
    final_qos_episode = reward
    if verbose:
        print(f'Qos final state (episode {ep}): {final_qos_episode} exploits: {exploit}/{steps}%')
    if final_qos_episode > start_qos_episode:
        print(True)
        improvements.append(1)
    else:
        print(False)
        improvements.append(0)

    print('---------' * 5)

    # # Média móvel da recompensa no término de cada episodio
    mean_rws.append(sum_rewards / steps)
    # #     clear_output(wait=True)
    if verbose:
        print(f"episode: {ep:0{5}}/{episodes} - R: {sum_rewards / steps:.{8}f}")
    env.close()

# Print the results
if verbose:
    print(
        f'For {episodes} episodes, there was {sum(improvements)} '
        f'improvements ({round(sum(improvements) * 100 / episodes, 2)}%), '
        f'{episodes - sum(improvements)} worse results '
        f'({round((episodes - sum(improvements)) * 100 / episodes, 2)}%), '
        f'and {collisions} collisions ({collisions / episodes}%).'
    )
# Create a directory to save the images
dt_hr = datetime.now().strftime("%Y%m%d-%H%M")
path = 'data/{}x{}_{}'.format(episodes, steps, dt_hr)
devices_file = "data/ed/endDevices_LNM_Placement_{}s+{}d.dat".format(seed, nDevices)
os.makedirs(path, exist_ok=True)

# Save the Q-table to a file
if plots_outputs == 1:
    f.save_qtable(path=path, q_table=m.qtable)
    f.plot_avg_reward(path=path,
                      means_reward=mean_rws,
                      output=output,
                      m=m)

    # f.plot_visited_states(path=path,
    #                       output=output,
    #                       episodes_visited_states=episodes_visited_states,
    #                       states=m.states)

    # f.gif_visited_states(path=path,
    #                      devices_file=devices_file,
    #                      episodes_visited_states=episodes_visited_states,
    #                      m=m)

    # f.plot_exploration(path=path,
    #                    output=output,
    #                    expl=exploration,
    #                    episodes=episodes)

execution_time.stop()
