#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:33:32 2020
Modified on Sat Oct  9 17:00:00 2021
Author: Rogério S. Silva
Email: rogerio.sousa@ifg.edu.br
"""
__author__ = "Rogério S. Silva"
__copyright__ = "Copyright (c) 2023, NumbERS - Federal Institute of Goiás, Inhumas - IFG"
__version__ = "0.1.0"
__email__ = "rogerio.sousa@ifg.edu.br"

import argparse
import itertools
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from ns3gym import ns3env
from colorama import Fore, Back, Style

from QLearning import QLearning

# S.O. parameters
parser = argparse.ArgumentParser()
parser.add_argument("area_side")  # The area side in points (e.g., 10 points = 10000/10=1000 meters between points)
parser.add_argument("nDevices")  # The number of devices deployed in the area
parser.add_argument("nGateways")  # The number of gateways deployed in the area
parser.add_argument("seed")  # The seed for the random number generator
parser.add_argument("path")  # The path to save the results
args = parser.parse_args()
cwd = os.getcwd()
resFolder = "{}/".format(cwd) + args.path

# Gym settings
port = 5555  # The port to connect to the ns-3 environment via ns3gym
startSim = 1  # 0: start the ns-3 simulation manually, 1: start the simulation automatically
debug = 0  # 0: no debug, 1: debug mode
simTime = 600  # seconds
stepTime = 600  # seconds

# Environment settings
nDevices = int(args.nDevices)
# seed x #GW: [1,3], [2,1], [3,2], [4,2], [5,2], [6,2], [7,2], [8,1], [9,1], [10,2]
# TODO: Refactoring dynamic nGateways in NS-3 code
nGateways = int(args.nGateways)
# TODO: Refactoring dynamic seed in NS-3 and Agent.py code
seed = int(args.seed)
area_side = int(args.area_side)
matrix_size = area_side ** 2
state = 0  # Initial state
# Arguments for the NS-3 simulation. It affects the execution environment topology and seed
simArgs = {"--nDevices": nDevices,
           "--nGateways": nGateways,
           "--seed": seed}

# Hiperparameters for the Q-Learning algorithm
epsilon = 0.8
epsilon_min = 0.4
alpha = 0.2
gamma = 0.9
epsilon_decay_rate = 0.999

# Environment connection
env = ns3env.Ns3Env(port=port, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
# Capture environment parameters
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space, ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

# Generate all possible positions (combinations) for the gateways (drones)
positions = list(itertools.combinations(range(matrix_size), nGateways))
ns_positions = [[((pos // matrix_size) * 1000 + 500, (pos % matrix_size) * 1000 + 500, 45) for pos in position] for
                position in positions]

# Generate all possible states for the gateways (drones), states are the index of the positions (e.g.: 0 -> (0,1))
states = [i for i, pos in enumerate(positions)]
# Generate all possible actions (permutations) for the set of gateways (drones)
movements = ['Up', 'Right', 'Down', 'Left', 'Stopped']
actions = list(itertools.product(movements, repeat=nGateways))

# Training parameters
num_episodes = 10
max_steps = 10  # per episode
action = -1
rewards = []

# Q-Learning algorithm
try:
    ql = QLearning(epsilon=epsilon,
                   epsilon_decay=epsilon_decay_rate,
                   epsilon_min=0.4,
                   alpha=alpha,
                   gamma=gamma,
                   init=0,
                   dim_grid=area_side)

    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        obs, reward, done, info = env.get_state()
        sum_reward = 0
        for step in range(max_steps):

            ql.print_state(action, obs, reward, done, info, step)

            if random.uniform(0, 1) < epsilon:
                # exploration
                action = env.action_space.sample()
            else:
                # exploitation
                action = np.argmax(ql.qtable[state, :])

            # print(f"Action: {action}")
            obs, reward, done, info = env.step(action)

            # new_state = get_state_QIndex(nGateways, obs, area_side, area_side, 1)

            # Q-learning
            old_value = ql.qtable[state, action]  # Value of the chosen action in the current state
            next_max = np.max(ql.qtable[new_state])  # Maximum value of the next state

            # Q-learning bellman equation
            # Q(s, a) = (1 - α) * Q(s, a) + α * (r + γ * max(Q(s', a')))
            # * Q(s, a) is the old value (current Q-value estimate of state s and action a).
            # * α is the learning rate.
            # * r is the reward for taking action a in state s.
            # * γ is the discount factor.
            # * max(Q(s', a')) is the estimate of the optimal future value.
            # Here, old_value is Q(s, a), new_value is the updated Q(s, a),
            # next_max is max(Q(s', a')).
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            ql.qtable[state, action] = new_value
            # Update to our new state
            state = new_state
            # print(f"State: {state}")
            # TODO: Refactoring collision detection and penalties
            collision = False
            if collision:
                reward = -2
                break
            sum_reward = sum_reward + reward

        rewards.append(sum_reward)
        env.reset()

        # Decrease epsilon
        epsilon = np.exp(-epsilon_decay_rate * episode)

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    np.save("qtable_100ex50st.npy", qtable)
    env.close()
    print("Done")
    # Plot rewards
    average_reward = []
    for idx in range(len(rewards)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 50:
            avg_list = rewards[:idx + 1]
        else:
            avg_list = rewards[idx - 49:idx + 1]
        average_reward.append(np.average(avg_list))
    # Plot
    plt.plot(rewards)
    plt.plot(average_reward)
