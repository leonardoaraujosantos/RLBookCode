import random
import numpy as np

"""
Implement Q Learning Agent, the advantage of this TD (Model-Free) method
is that we don't need to wait for a sequence of actions to update our 
knowledge.
"""
class Q_Agent():
    def __init__(self, env, gamma, alpha=0.2, e_greedy_prob=0.5, e_greedy_decay=0.01):
        self.gamma = gamma
        self.env = env
        num_actions = env.action_space.n
        num_states = np.prod([state.n for state in env.observation_space])
        rows, cols = (num_states, num_actions)

        # Create the state-action table
        self.q_val_table = [[0.] * cols for _ in range(rows)]

        self.alpha = alpha
        self.e_greedy_prob = e_greedy_prob
        self.e_greedy_decay = e_greedy_decay

    def choose_action(self, state):
        """
        Choose an action for a particular state using epsilon-greedy technique
        :param state: Current state
        :return: Random or greedy action
        """
        if random.random() < self.e_greedy_prob:
            # randomly select action from state
            action = np.random.choice(len(self.q_val_table[state]))
        else:
            # greedily select action from state
            action = np.argmax(self.q_val_table[state])
        return action

    def update_q_table(self, cur_state, action, reward, next_state):
        """
        Update the Q-Table (Bellman Update)
        :param cur_state: Current State
        :param action: Action chosen from the current state
        :param reward: Reward for doing the action on the particular state(current)
        :param next_state: Next state returned from the environment
        :return: None
        """
        new_max_q = max(self.q_val_table[next_state])
        new_value = reward + self.gamma * new_max_q

        old_value = self.q_val_table[cur_state][action]
        self.q_val_table[cur_state][action] = old_value + self.alpha * (new_value - old_value)

        # Decay epsilon_greedy
        self.e_greedy_prob = self.exp_decay(self.e_greedy_prob)

    def exp_decay(self, value):
        y = value * (1 - self.e_greedy_decay)
        return y
