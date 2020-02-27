#!/usr/bin/env pybricks-micropython

from crawler_lego_env import CrawlingRobotEnv
from agent import Q_Agent

try:
    from pybricks.tools import print
    running_on_lego = True
except ModuleNotFoundError:
    running_on_lego = False

# Number seem to converge in simulation
num_iterations_train = 300
# Bigger values decay faster
e_greedy_decay = 1. / num_iterations_train
initial_e_greedy_prob = 1.0

if __name__ == '__main__':
    # Initialize environemnt
    env = CrawlingRobotEnv(invert_reward=False, run_on_lego=running_on_lego, step_angle=45)
    current_state = env.reset()
    agent = Q_Agent(env, gamma=0.9, alpha=0.2, e_greedy_prob=initial_e_greedy_prob, e_greedy_decay=e_greedy_decay)
    print(agent.q_val_table)

    # Train
    for steps in range(num_iterations_train):
        action = agent.choose_action(current_state)
        current_state_str = str(env)
        next_state, reward, done, info = env.step(action)
        next_state_str = env.state_idx_to_str(next_state)
        action_str = env.action_idx_to_str(action)
        agent.update_q_table(current_state, action, reward, next_state)
        print('steps:', steps, '\n\tcurrent_state:', current_state_str, '\n\tACTION:', action_str, '\n\tnext_state:',
              next_state_str, '\n\treward:', reward, '\nprob:', agent.e_greedy_prob)
        print('-' * 20)
        # Don't forget to update your state otherwise the robot will be stuck
        current_state = next_state

    # Evaluate
    print(agent.q_val_table)
    # Only act greedly ...
    agent.e_greedy_prob = 0
    sum_rewards = 0
    num_steps_eval = 20
    for steps in range(num_steps_eval):
        # Greedly run actions without learn anymore
        current_state_str = str(env)
        action = agent.choose_action(current_state)
        next_state, reward, done, info = env.step(action)
        action_str = env.action_idx_to_str(action)
        next_state_str = env.state_idx_to_str(next_state)
        print('steps:', steps, '\n\tcurrent_state:', current_state_str, '\n\tACTION:', action_str, '\n\tnext_state:',
              next_state_str, '\n\treward:', reward, '\nprob:', agent.e_greedy_prob)
        print('-' * 20)
        sum_rewards += reward
        # Don't forget to update your state otherwise the robot will be stuck
        current_state = next_state
    print('Sum of rewards in %d steps: %d' % (num_steps_eval, sum_rewards))

