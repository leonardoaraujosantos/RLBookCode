import random
from collections import defaultdict
import numpy as np

"""
Implement Q Learning Agent
"""
class Q_Agent():
    def __init__(self, env, gamma, alpha=0.2, e_greedy_prob=0.5):
        self.gamma = gamma
        self.env = env
        self.q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        self.alpha = alpha
        self.e_greedy_prob = e_greedy_prob

    def choose_action(self, state):
        if random.random() < self.e_greedy_prob:
            # randomly select action from state
            action = np.random.choice(len(self.q_vals[state]))
        else:
            # greedily select action from state
            action = np.argmax(self.q_vals[state])
        return action

    def update_q(self, cur_state, action, reward, next_state):
        new_max_q = np.max(self.q_vals[next_state])
        new_value = reward + self.gamma * new_max_q

        old_value = self.q_vals[cur_state][action]
        self.q_vals[cur_state][action] = old_value + self.alpha * (new_value - old_value)