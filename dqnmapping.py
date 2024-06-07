import math
import random
import numpy as np


class DQNMapping:
    def __init__(self, ns3_env, dim_grid=10, n_actions=5, action_space=None, n_agents=2, initial_state=0,
                 n_states=0, state_positions=None, alpha=0.2, gamma=0.9, step_size=1000, start_position=500,
                 epsilon=1, epsilon_min=0.5, epsilon_decay=0.999, accum_rewards=0, area_side=20000):
        self.ns3_env = ns3_env  # NS-3 environment
        self.dim_grid = dim_grid  # area side, means the grid is 10x10
        self.step_size = step_size  # step size for each movement (UAVs move 1000 meters each time)
        self.area_side = area_side  # area size (20000 meters)
        self.start_pos = start_position  # initial position for each UAV
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
        self.accumulated_rewards = accum_rewards  # accumulated rewards for each episode
        self.message_from_ns3 = None

    # epsilon will return to it's initial value for each episode
    def reset_epsilon(self):
        self.epsilon = self.epsilon_max

    # gives the current qos for a current state
    def current_qos(self):
        return self.ns3_env.get_state()[1]

    # given a state, times it'll be a random action, times it'll be the best action. It depends on epsilon value.
    def get_action(self, q_val):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if random.random() < self.epsilon:
            return self.ns3_env.action_space.sample(), 0
            # actions = [i for i in range(self.actions ** self.n_agents)]
            # return random.choice(actions), 0
        else:
            return np.argmax(q_val), 1

    def state_from_coordinates(self, c_pos):
        # UAVs are deployed in the grid on virtual positions (positioning (x,y,z) possibilities)
        # To define the state, we need to transform these coordinates into a state
        # The coordinates are in a set of 3D-coord representing all UAVs (x1, y1, z1,..., xn, yn, zn) from simulator
        # The states are in the linear format (s1, s2, ..., sn), where s_i is the state of the i-th UAV, i=1,2,...,n
        # The agent state positions representation is in a list, where each element is a tuple with the all drones
        # to deploy, e.g. (a,b,c) means that the first drone is in position a, the second in position b and the third
        # in position c. The state_positions list is a list of all possible states for all agents.
        # represents an agent state. The state_positions list is a list of all possible states for all agents.
        # Thus, to transform the Drones coordinates into a state, we need to:
        # 1. Transform drones coordinates into state coordinates
        # 2. Transform state coordinates into state

        # Transform drones coordinates into state coordinates
        coord_pos = [((c_pos[i] - self.start_pos) // self.step_size, (c_pos[i+1] - self.start_pos) // self.step_size)
                     for i in range(0, len(c_pos), 3)]
        # Transform state coordinates into state tuple
        state_pos = [x * self.dim_grid + y for x, y in coord_pos]
        # Check if there are repeated positions (drone collisions)
        if len(list(set(state_pos))) != len(state_pos):
            return -1
        #  Transform state_positions into state
        state = self.state_positions.index(tuple(sorted(state_pos)))
        return state

    def coordinates_from_state(self, state):
        state_pos = self.state_positions[state]
        # Transform state coordinates into drones coordinates
        coord_pos = [(x // self.dim_grid * self.step_size + self.start_pos,
                      x % self.dim_grid * self.step_size + self.start_pos, 45) for x in state_pos]
        # Transform state_positions into coordinates
        c_pos = [x for tup in coord_pos for x in tup]
        return c_pos
