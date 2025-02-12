import argparse
import itertools
import os
import random
from datetime import datetime
import torch
import numpy as np
from codetiming import Timer
from ns3gym import ns3env
from tqdm import tqdm

import facilities as f
from dqn.dqnmapping import DQNMapping

"""
Created on Sun Apr 1 15:00:00 2023
Modified on Sun Apr 1 15:00:00 2023
Authors: Rogério S. Silva, Renan R. Oliveira, and Lucas Tadeu
Email: {rogerio.sousa, renan.oliveira}@ifg.edu.br,
"""

__author__ = ("Rogério S. Silva, Renan R. Oliveira, Lucas Tadeu, "
              "Antonio Oliveira-JR, and Kleber V. Cardoso")
__copyright__ = ("Copyright (c) 2024, "
                 "Instituto de Informática - Universidade Federal de Goiás - UFG, and"
                 "Instituto Federal de Goiás - IFG")
__version__ = "0.1.0"
__email__ = ("{rogerio.sousa, renan.oliveira}@ifg.edu.br, lucastsc@gmail.com"
             "{antonio, kleber}@inf.ufg.br")

execution_time = Timer(text="Execution time: {0:.2f} seconds")

# Create a parser
parser = argparse.ArgumentParser(description='DQN for n UAVs')
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
args = parser.parse_args()

execution_time.start()

# ns-3 environment
port = 0
devSeed = 6
nDevices = args.dv
nGateways = args.gw
simTime = 600  # seconds
stepTime = 600  # seconds
simArgs = {"--nDevices": nDevices,
           "--nGateways": nGateways,
           "--vgym": 1,
           "--verbose": 0,
           "--devSeed": devSeed}

startSim = args.ss
debug = 0
dim_grid = args.gr
movements = ['up', 'right', 'down', 'left', 'stay']
# action_space has all possible movements for the gateways combinations
action_space = list(itertools.product(movements, repeat=nGateways))
# position_space has the all possible gateways positions in the grid
position_space = list(itertools.combinations(range(dim_grid * dim_grid), nGateways))
# state_space has the indexes of the position_space
state_space = np.zeros((1, len(position_space)))

verbose = False if args.v == 0 else True
output = 'file' if args.out_term == 1 else 'screen'
plots_outputs = args.out

# DQN settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
l1 = len(position_space)
l2 = 150
l3 = 100
l4 = len(action_space)
batch_size = 32
memory_size = 1000
memory_counter = 0
replace_target_iter = 100
learning_steps_counter = 0
learning_rate = 1e-3
epsilon = 0.9
epsilon_max = 0.9
epsilon_increment = 0.001
gamma = 0.9
target_replace_iter = 100
n_actions = len(action_space)
n_states = len(position_space)
episodes = args.ep
steps = args.st

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

improvements = []
mean_rws = [0]

m = DQNMapping(ns3_env=None,
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
losses = [0]
losses_episode = [0]

# episodes loop for training
for ep in range(episodes):
    seed = random.randint(1, 200)
    env = ns3env.Ns3Env(port=port, startSim=startSim, simSeed=devSeed, simArgs=simArgs, debug=debug)
    m.ns3_env = env
    # state_coordinates = m.ns3_env.reset()  # Restart the environment and get the initial state
    state_coordinates, reward, done, info = m.ns3_env.get_state()
    state = m.state_from_coordinates(state_coordinates)  # convert coordinates to state
    m.state = state  # Set initial state
    # _, reward, done, info = m.ns3_env.get_state()
    start_qos_episode = reward
    ep_state = np.zeros((1, len(position_space)))
    ep_state[0, state] = 1
    t_ep_state = torch.from_numpy(ep_state).float()
    m.reset_epsilon()
    visited_states = []
    if args.progress == 1:
        t_range = tqdm(range(steps), desc="Progress: ", unit=" steps")
    else:
        t_range = range(steps)
    sum_rewards = reward
    exploit = 0
    mean_loss_episode = 0
    for step in t_range:
        model.train()
        # Predict Q-value from state
        q_value = model(t_ep_state)  # torch format
        q_value_ = q_value.data.numpy()  # numpy format

        # random action (random < epsilon) or best action (random > epsilon) - exploration vs exploitation
        action, ex = m.get_action(q_value_)
        if ex == 1:
            exploration[ep][step] = ex
            exploit += 1
        # Get the reward for the current state and action
        state_coordinates, reward, done, info = m.ns3_env.step(action)
        visited_states.append(state)
        # convert coordinates from ns3 to state
        new_state = m.state_from_coordinates(state_coordinates)

        newstate = np.zeros((1, len(position_space)))
        newstate[0, new_state] = 1
        newstate = torch.from_numpy(newstate).float()

        sum_rewards += reward

        with torch.inference_mode():
            newQ = model(newstate)

        maxQ = torch.max(newQ)
        Y = reward + (m.gamma * maxQ)

        Y = torch.Tensor([Y]).detach()  # target value
        Y_pred = q_value.squeeze()[action]  # predicted

        loss = loss_fn(Y_pred, Y)

        losses.append(loss.item())
        mean_loss_episode = ((step+1) * losses_episode[-1] + loss.item())/(step+2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    losses_episode.append(mean_loss_episode)
    mean_rws.append(sum_rewards / steps)
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
devices_file = "data/ed/endDevices_LNM_Placement_{}s+{}d.dat".format(devSeed, nDevices)
os.makedirs(path, exist_ok=True)

# Save the Q-table to a file
if plots_outputs == 1:
    f.save_qtable(path=path, q_table=m.qtable)
    f.plot_avg_reward(path=path,
                      means_reward=mean_rws,
                      output=output,
                      m=m)

    f.plot_visited_states(path=path,
                          output=output,
                          episodes_visited_states=episodes_visited_states,
                          states=m.states)

    f.gif_visited_states(path=path,
                         devices_file=devices_file,
                         episodes_visited_states=episodes_visited_states,
                         m=m)

    # f.plot_exploration(path=path,
    #                    output=output,
    #                    expl=exploration,
    #                    episodes=episodes)

execution_time.stop()
