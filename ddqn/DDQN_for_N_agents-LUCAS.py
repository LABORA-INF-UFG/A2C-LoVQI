#!/usr/bin/env python
# coding: utf-8

# How to use this file:
# 
# This file implements the Deep QLearning for N drones in a predefined grid space.
# 
# For your use case, you need to:
# 
# - Change the NUMBER_ACTION_COLUMNS properly for your case (e.g for actions 0,...24, the value must be 25. For 0,...124, the value must be 125, and so on)
# - Set the hyperparameters properly inside Map \_init\_() class.
# - Change any other values in order to customize the process, e.g number of episodes and steps.

# In[1]:


import pandas as pd
import numpy as np
import random
import ast
from itertools import combinations
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import sys
import pickle
import os
import time


# In[2]:


print(torch.cuda.is_available())
print(torch.cuda.device_count())


# In[3]:


d = pd.read_csv(f'rewards/reward_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}.dat', sep = ' ')

# the action columns (0,1,..24) are in string mode. Below, we change it to int mode, to make future manipulations easy.
d = d.rename({col:int(col) for col in d.columns if d[col].dtype == 'object'}, axis='columns')

AGENTS = int(sys.argv[2])
NUMBER_ACTION_COLUMNS = 5 ** AGENTS # total number of possible actions

# the table's body are in string mode (e.g. '[945,0.95379]'). Turning it into lists to make future manipulations easy.
for col in range(NUMBER_ACTION_COLUMNS):# from 0 to 24 (for 2 agents and 5 possible actions each)
    d[col] = d[col].apply(ast.literal_eval)

d


# In[ ]:


class Map:
    def __init__(self,
                 dim_grid=10, # means the grid is 10x10
                 actions_per_agent=5, # each agent is capable of up,right,down,left and stopped movements
                 agents=AGENTS, # total number of agents in the grid
                 state=0, # initial state, starts at state 0 (means there is a first position for all agents)
                 alpha=0.2, # Q-learning algorithm learning rate
                 gamma=0.9, # gamma is the discount factor. It is multiplied by the estimation of the optimal future value.
                 epsilon=1, # epsilon handles the exploration/exploitation trade-off (e.g. epsilon < 0.4 means 40% exploration and 60% exploitation)
                 epsilon_min=0.5, # minimum allowed epsilon. Epsilon will change (reduce) with decay_epsilon function. At begginings, it means more exploration than exploitation.
                 epsilon_decay=0.999 # epsilon will decay at each step. For 1000 steps and decay 0.999, for example, epsilon will decay a factor by 0.367 of it initial value.
                ):
        self.dim_grid = dim_grid
        self.actions_per_agent = actions_per_agent
        self.agents = agents
        self.state = state
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon
        self.epsilon_decay = epsilon_decay
        self.init()

    # for a grid 10x10 and 2 agents, for example, total stated are C(10x10,2) = 4950 states
    def init(self):
        self.states = len(list(combinations([i for i in range(self.dim_grid*self.dim_grid)],self.agents)))

    # gives the next state given the current state for a given action
    def next_state(self, current_state, action):
        return d.loc[current_state,action][0]

    # gives the current qos for a current state
    def current_qos(self,current_state):
        return d.loc[current_state,'qos']

    # gives the qos of the next state, given the current state and a given action
    def next_qos(self, current_state, action):
        return d.loc[current_state,action][1]

    # epsilon will return to it's initial value for each episode
    def resetEpsilon(self):
        self.epsilon = self.epsilon_max

    # attribute a new value to epsilon after a decay
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # if current qos is less than the next qos, we have a reward. Otherwise, we have a penalty.
    def reward(self,current_state,action):
        return self.next_qos(current_state, action)

    def actionResults(self,state, action):
        newstate = self.next_state(state,action)
        reward = self.reward(state,action)
        return newstate, reward


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = Map()

# model for 2 agents in a 10x10 grid
# l1 = 4950
possible_states = m.states
possible_actions = m.actions_per_agent**m.agents

l1 = possible_states
l2 = 150
l3 = 100
l4 = possible_actions

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
).to(device)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


rewards = [0]
k = 0

improvements = []

episodes = 300
steps = 500

losses = [0]
losses_episode = [0]

episodes_visited_states = {}

execution_times = []

for ep in range(episodes):
    start = time.time()
    random_state = random.choice(d['state'])
    
    visited_states = [random_state]

    # state is [0,0,...,state,0,0,...,0]
    state = np.zeros((1, possible_states))
    state[0, random_state] = 1
    state = torch.from_numpy(state).float().to(device)  # Mova o tensor para a GPU

    start_qos_episode = m.current_qos(torch.argmax(state[0]).item())
    print(f'Qos start state (episode {ep}): {start_qos_episode}')

    # Reestabeleça epsilon para o valor máximo para decréscimos sucessivos
    m.resetEpsilon()
    steps_rewards = []
    steps_losses = []
    for step in range(steps):
        model.train()
        qval = model(state)  # predicted qvalue for state (torch format)
        qval_ = qval.data.cpu().numpy()  # predicted qvalue in numpy format

        # ação aleatória (random < epsilon) ou melhor ação (random > epsilon) - exploração vs exploração
        m.epsilon = max(m.epsilon * m.epsilon_decay, m.epsilon_min)
        if random.random() < m.epsilon:
            action = np.random.randint(0, possible_actions)
        else:
            action = np.argmax(qval_)

        newstate, reward = m.actionResults(torch.argmax(state[0]).item(), action)
        steps_rewards.append(reward)
        visited_states.append(newstate)

        # newstate is [0,0,...,newstate,0,0,...,0]
        state2 = np.zeros((1, possible_states))
        state2[0, newstate] = 1
        state2 = torch.from_numpy(state2).float().to(device)  # Mova o tensor para a GPU

        with torch.no_grad():
            newQ = model(state2)

        maxQ = torch.max(newQ)
        Y = reward + (m.gamma * maxQ)

        Y = Y.detach().to(device)  # target value
        Y_pred = qval.squeeze()[action]  # predicted

        loss = loss_fn(Y_pred, Y)

        steps_losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = state2
        
    end = time.time()
    episode_time = end - start
    execution_times.append(episode_time)
    
    episodes_visited_states[ep] = visited_states

    final_qos_episode = m.current_qos(torch.argmax(state2[0]).item())
    print(f'Qos final state (episode {ep}): {final_qos_episode}')

    if final_qos_episode > start_qos_episode:
        print(True)
        improvements.append(1)
    else:
        print(False)
        improvements.append(0)

    print('---------' * 5)

    # Média móvel da recompensa no final de cada episódio
    mean_reward = ((k + 1) * rewards[-1] + np.mean(steps_rewards)) / (k + 2)
    print(f'ep:{ep} reward:{np.mean(steps_rewards)} mean_reward:{mean_reward}')
    rewards.append(mean_reward)
    
    mean_loss_episode = ((k + 1) * losses_episode[-1] + np.mean(steps_losses)) / (k + 2)
    losses_episode.append(mean_loss_episode)
    k = k + 1
    clear_output(wait=True)
    print(f"episode: {ep:0{5}}/{episodes} - R: {mean_reward:.{8}f} - loss: {mean_loss_episode}")


# In[ ]:


del rewards[0]
del losses_episode[0]


# In[ ]:


def save_file(filename_base,somelist):
    directory = f'results/{filename_base}_{sys.argv[2]}Gw'
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    
    with open(f'results/{filename_base}_{sys.argv[2]}Gw/{filename_base}_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}.txt', "w") as arquivo:
        # Escrever todos os itens da lista no arquivo, cada um em uma nova linha
        arquivo.write("\n".join(map(str, somelist)) + "\n")

save_file('DQN rewards', rewards)
save_file('DQN losses', losses_episode)
save_file('DQN elapsed_time',execution_times)


# In[ ]:


all_visited_states = [state for ep, list_visited_states in episodes_visited_states.items() for state in list_visited_states]
all_visited_states = list(set(all_visited_states))

# testing the policy with random states
best_states = []
for n in range(5):
    state = random.choice(all_visited_states)
    max_qos = 0
    print(f'\n\nSTART STATE-->{state}')
    for step in range(100):
        if random.random() < 0.8: # following the policy
            
            state_vect = np.zeros((1,possible_states))
            state_vect[0,state] = 1
            state_vect = torch.from_numpy(state_vect).float().to(device)
            
            qval = model(state_vect) # predicted qvalue for state (torch format)
            qval_ = qval.data.cpu().numpy() # predicted qvalue in numpy format
            action = np.argmax(qval_)

            state = m.next_state(state,action)
        else: # off-policy, trying to avoid loops
            state = all_visited_states[random.randint(0, len(all_visited_states) - 1)]
            
        qos = d.loc[d['state'] == state,'qos'].values[0]
        if qos > max_qos:
            max_qos = qos
            best_state = state
    best_states.append(best_state)
    print(best_state,max_qos)


# In[ ]:


save_file('DQN best_states', best_states)   


# In[ ]:


print(f'For {episodes} episodes, there was {sum(improvements)} improvements ({round(sum(improvements)*100/episodes,2)}%) and {episodes-sum(improvements)} worse results ({round((episodes-sum(improvements))*100/episodes,2)}%)')


# In[ ]:


# print(Y_pred)
# print(Y)


# In[ ]:


def plot_and_save(mylist,title,ylabel,xlabel):
    plt.plot(np.arange(len(mylist)), mylist)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    
    fig_path = f'figs/{sys.argv[2]}Gw/{title}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    plt.savefig(f'{fig_path}/reward_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}.png')
    
    plt.show()
    
plot_and_save(rewards,'DQN Rewards','Avg Reward','Episodes')
plot_and_save(losses_episode,'DQN Losses','Avg Loss','Episodes')


# In[ ]:


# plt.figure(figsize=(6,3))
# plt.plot(losses_episode)
# plt.xlabel("Episodes",fontsize=16)
# plt.ylabel("Loss_episode",fontsize=16)
# plt.plot()

# fig_path = f'figs/{sys.argv[2]}Gw/losses'
# if not os.path.exists(fig_path):
#     os.makedirs(fig_path)
    
# plt.savefig(f'{fig_path}/loss_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}.png')


# In[ ]:


# ###############
# fig, ax = plt.subplots(figsize=(6, 3))
# ax.plot(np.arange(len(rewards)),rewards, linestyle = 'solid', color='blue', linewidth=2)
# ax.set_title(f'Agents: {m.agents} , Grid: {m.dim_grid}x{m.dim_grid}, Movements per Agent: {m.actions_per_agent}')
# ax.set_xlabel("Episodes")
# ax.set_ylabel("Avg Reward")

# fig_path = f'figs/{sys.argv[2]}Gw/rewards'
# if not os.path.exists(fig_path):
#     os.makedirs(fig_path)
    
# plt.savefig(f'{fig_path}/reward_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}.png')

# plt.show()


# In[ ]:


# top_qos_states = d['qos'].sort_values(ascending=False)[:10]

# # testing the policy with random states
# for n in range(5):
#     state = random.choice(d['state'])
#     print(f'\n\nSTART STATE-->{state}')
#     for step in range(50):
#         if random.random() < 0.8: # following the policy
            
#             state_vect = np.zeros((1,possible_states))
#             state_vect[0,random_state] = 1
#             state_vect = torch.from_numpy(state_vect).float().to(device)
            
#             qval = model(state_vect) # predicted qvalue for state (torch format)
#             qval_ = qval.data.cpu().numpy() # predicted qvalue in numpy format
#             action = np.argmax(qval_)

#             state = m.next_state(state,action)
#         else: # off-policy, trying to avoid loops
#             action = np.random.randint(0,possible_actions)
#             state = m.next_state(state,action)
            
#         if state in top_qos_states:
#             print(f"\033[1;33;40m{state}\033[0m", end=' ')
#         else:
#             print(state, end=' ')


# In[ ]:


# lsts = []
# # creates a unique list with all visited states for all episodes and steps (for histogram analysis)
# for v in episodes_visited_states.values():
#     lsts.extend(v)


# In[ ]:


# def max_repetitions(lista):
#     # Counts each element occurence in the list
#     contagem = Counter(lista)
    
#     # Finds the element that most occurs and how many times it occurs
#     max_repetido, max_repeticoes = contagem.most_common(1)[0]
    
#     return max_repetido, max_repeticoes


# In[ ]:


# top_10_qos = list(d['qos'].sort_values(ascending=False)[:10].keys())

# plt.figure(figsize=(10,6))
# N, bins, patches = plt.hist(lsts,bins=m.states)
# for b in range(m.states):
#     if b in top_10_qos:
#         patches[b].set_fc('red')

# plt.title("Histogram")
# plt.xlabel("State")
# plt.yscale('log')
# plt.ylabel("Occurrences")
# # plt.xlim(4500,5000)
# # plt.ylim(0,10000)
# plt.show()

