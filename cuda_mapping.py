import random

import numpy as np


class CUDAMapping:
    def __init__(self, ns3_env, dim_grid=10, n_actions=5, action_space=None, n_agents=2, initial_state=0,
                 n_states=0, state_positions=None, alpha=0.2, gamma=0.9, q_table=None,
                 epsilon=1, epsilon_min=0.5, epsilon_decay=0.999, accum_rewards=0):
        self.qtable = None  # Q-table
        self.ns3_env = ns3_env  # NS-3 environment
        self.dim_grid = dim_grid  # area side, means the grid is 10x10
        self.step_size = 1000  # step size for each movement (UAVs move 1000 meters each time)
        self.n_agents = n_agents  # number of agents (UAVs) in the grid
        self.n_actions = n_actions  # number of possible actions of each agent
        self.actual_action = 0  # current action of the agent
        self.action_space = action_space  # space of all possible actions
        self.initial_state = initial_state  # initial state of UAVs in the grid obtained from the optimizer
        self.actual_state = initial_state  # current state of UAVs in the grid
        self.n_states = n_states  # number of states
        self.state_positions = state_positions  # list of all possible positions for all agents
        self.alpha = alpha  # Q-learning algorithm learning rate
        self.gamma = gamma  # Gamma is the discount factor. It is multiplied by the estimation
        # of the optimal future value.
        self.epsilon = epsilon  # epsilon handles the exploration/exploitation trade-off
        self.epsilon_min = epsilon_min  # minimum allowed epsilon. Epsilon will change (reduce)
        # with decay_epsilon function. At beginnings, it means more exploration than exploitation.
        self.epsilon_max = epsilon  # maximum allowed epsilon. (e.g. epsilon < 0.4,
        # means 40% exploration and 60% exploitation)
        self.epsilon_decay = epsilon_decay  # epsilon will decay at each step. For 1000 steps
        # and decay 0.999, for example, epsilon will decay a factor by 0.367 of it initial value.
        self.qtable = q_table
        self.accumulated_rewards = accum_rewards  # accumulated rewards for each episode
        self.message_from_ns3 = None

    # epsilon will return to it's initial value for each episode
    def reset_epsilon(self):
        self.epsilon = self.epsilon_max

    def get_max_q(self, state):
        return max(self.qtable[state])

    # given a state, times it'll be a random action,
    # times it'll be the best action. It depends on epsilon value.
    def get_action(self, state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if random.random() < self.epsilon:
            return self.ns3_env.action_space.sample(), 0
            # actions = [i for i in range(self.actions ** self.n_agents)]
            # return random.choice(actions), 0
        else:
            return np.argmax(self.qtable[state]), 1

    # given a state and a given action, it updates the qtable and returns
    # the new state and the reward for the given movement in that state.
    # Q(s,a) = (1-alpha) * Q(s,a) + alpha(reward + gamma*max(Q(s',a')))
    def update(self, a_reward, n_state):
        self.qtable[self.actual_state][self.actual_action] = (
            (1 - self.alpha) * self.qtable[self.actual_state][self.actual_action]
            + self.alpha * (a_reward + self.gamma * self.get_max_q(n_state)))

    def state_from_coordinates(self, c_pos):
        # Transform coordinates into state_positions
        # Separate coordinates into (x,y)-tuples
        # ex: [0, 1, 2, 3, 4, 5, 6, 7, 8] = > [(0, 1), (3, 4), (6, 7)]
        coord_pos = [((c_pos[i] - 500) // 1000, (c_pos[i + 1] - 500) // 1000) for i in range(0, len(c_pos), 3)]
        # Transform drones coordinates into state coordinates
        state_pos = [x * self.dim_grid + y for x, y in coord_pos]

        if len(list(set(state_pos))) != len(state_pos):
            return -1
        #  Transform state_positions into state
        state = self.state_positions.index(tuple(sorted(state_pos)))
        return state

    def coordinates_from_state(self, state):
        state_pos = self.state_positions[state]
        # Transform state coordinates into drones coordinates
        coord_pos = [(x // self.dim_grid * 1000 + 500, x % self.dim_grid * 1000 + 500, 45) for x in state_pos]
        # Transform state_positions into coordinates
        c_pos = [x for tup in coord_pos for x in tup]
        return c_pos
