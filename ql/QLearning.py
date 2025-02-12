import random

from colorama import Fore


class QLearning:
    def __init__(self, epsilon=0.8, epsilon_decay=0.999, epsilon_min=0.4, alpha=0.1, gamma=0.9, init=0, dim_grid=10,
                 actions=['Up', 'Right', 'Down', 'Left', 'Stopped']):
        self.qtable = None
        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma
        self.dim_grid = dim_grid
        self.actions = actions
        self.init = init
        self.init_q_table()

    def init_q_table(self):
        self.epsilon = self.epsilon_max
        self.qtable = [[[self.init] * len(self.actions) for j in range(self.dim_grid)] for i in range(self.dim_grid)]

    def reset_epsilon(self):
        self.epsilon = self.epsilon_max

    def print_q_table(self):
        print("Pos\t|\tUp\t|\tRight\t|\tDown\t|\tLeft\t|\tStopped\t|")
        for i in range(self.dim_grid):
            for j in range(self.dim_grid):
                print("%d,%d\t|\t%d\t|\t%d\t|\t%d\t|\t%d\t|\t%d\t|" % (
                    i, j, self.qtable[i][j][0], self.qtable[i][j][1], self.qtable[i][j][2], self.qtable[i][j][3],
                    self.qtable[i][j][4]))

    def print_policy(self):
        print("----" * self.dim_grid + "-")
        for i in range(self.dim_grid - 1, -1, -1):
            print("|", end="")
            for j in range(self.dim_grid):
                best = self.get_best_action([i, j])
                print(" " + "↑→↓←*"[best] + " |", end="")
            print("\n" + ("----" * self.dim_grid) + "-")

    def get_max_q(self, pos):
        return max(self.qtable[pos[0]][pos[1]])

    def get_best_action(self, pos):
        qs = self.qtable[pos[0]][pos[1]]
        m = max(qs)
        bests = [i for i, j in enumerate(qs) if j == m]
        return random.choice(bests)

    def get_random_action(self):
        return int(random.random() * len(self.actions))

    def get_action(self, pos):
        self.decay_epsilon()
        if random.random() < self.epsilon:
            return self.get_random_action()
        else:
            return self.get_best_action(pos)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update(self, oldpos, action, newpos, reward, final):
        self.qtable[oldpos[0]][oldpos[1]][action] += self.alpha * (
                reward + self.gamma * self.get_max_q(newpos) - self.qtable[oldpos[0]][oldpos[1]][action])

    def get_state_index(self, obs):
        return (obs[0] * self.dim_grid * self.dim_grid) + (obs[1] * self.dim_grid * self.dim_grid) + obs[2] + obs[3]

    def print_state(self, action, obs, reward, done, info, step):
        if reward > 0:
            print(
                Fore.LIGHTGREEN_EX + f"Step: {step} Action: {action} "
                                     f"---obs: {obs}, reward: {reward}, done: {done}, info: {info}")
        else:
            if reward == -1:
                print(
                    Fore.LIGHTWHITE_EX + f"Step: {step} Action: {action} "
                                         f"---obs: {obs}, reward: {reward}, done: {done}, info: {info}")
            else:
                print(
                    Fore.RED + f"Step: {step} Action: {action} "
                               f"---obs: {obs}, reward: {reward}, done: {done}, info: {info}")

