#!/usr/bin/env pybricks-micropython
from crawler_lego_env import CrawlingRobotEnv
from agent import Q_Agent
from pybricks.tools import print

# Number seem to converge in simulation
num_iterations_train = 400
# Bigger values decay faster
e_greedy_decay = 1. / num_iterations_train
initial_e_greedy_prob = 1.0

if __name__ == '__main__':
    # Initialize environemnt
    env = CrawlingRobotEnv(invert_reward=False)
    current_state = env.reset()
    agent = Q_Agent(env, gamma=0.9, alpha=0.2, e_greedy_prob=initial_e_greedy_prob, e_greedy_decay=e_greedy_decay)
    print(agent.q_val_table)

    # Train
    for steps in range(num_iterations_train):
        action = agent.choose_action(current_state)
        next_state, reward, done, info = env.step(action)
        agent.update_q_table(current_state, action, reward, next_state)
        print('steps:', steps, 'Reward:', reward, 'next_state:', next_state.e_greedy_prob, 'action:', action)
        current_state = next_state

    # Evaluate
    print(agent.q_val_table)
    # Only act greedly ...
    agent.e_greedy_prob = 0.1
    while True:
        # Greedly run actions without learn anymore
        action = agent.choose_action(current_state)
        next_state, reward, done, info = env.step(action)
        print('Greedy action:', action, 'Reward State:', reward)

