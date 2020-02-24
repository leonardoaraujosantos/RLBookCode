#!/usr/bin/env pybricks-micropython
from crawler_lego_env import CrawlingRobotEnv
from agent import Q_Agent

# Number seem to converge in simulation
num_iterations_train = 400
# Bigger values decay faster
e_greedy_decay = 1. / num_iterations_train
initial_e_greedy_prob = 1.0

if __name__ == '__main__':
    # Initialize environemnt
    env = CrawlingRobotEnv()
    current_state = env.reset()
    agent = Q_Agent(env, gamma=0.9, alpha=0.2, e_greedy_prob=initial_e_greedy_prob, e_greedy_decay=e_greedy_decay)

    # Train
    for steps in range(num_iterations_train):
        action = agent.choose_action(current_state)
        next_state, reward, done, info = env.step(action)
        agent.update_q_table(current_state, action, reward, next_state)
        current_state = next_state

    # Evaluate
    # Only act greedly ...
    agent.e_greedy_prob = 0
    while True:
        # Greedly run actions without learn anymore
        action = agent.choose_action(current_state)
        next_state, reward, done, info = env.step(action)

