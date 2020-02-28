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
e_greedy_decay = 1.0 / num_iterations_train
# Initial agent action probability (just try things at random)
initial_e_greedy_prob = 1.0
# Number of iterations before check statitics of reward
num_steps_eval = num_iterations_train//10

if __name__ == '__main__':
    # Initialize environment
    env = CrawlingRobotEnv(invert_reward=True, run_on_lego=running_on_lego, step_angle=40)
    current_state = env.reset()
    agent = Q_Agent(env, gamma=0.9, alpha=0.2, e_greedy_prob=initial_e_greedy_prob, e_greedy_decay=e_greedy_decay)
    print(agent.q_val_table)

    # Train
    sum_rewards = 0
    sum_rewards_vec = []
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
        sum_rewards += reward
        if steps % num_steps_eval == 0:
            sum_rewards_vec.append(sum_rewards)
            print('\t\t*******Sum of rewards in %d steps: %d' % (num_steps_eval, sum_rewards))
            sum_rewards = 0


    # Save Q-table to a file
    print('Q TABLE')
    print('-' * 20)
    print(agent.q_val_table)
    print('-' * 20)

    print('Sampled Reward')
    print('-' * 20)
    print(env.sampled_reward_function)
    print('-' * 20)

    print('Sampled MDP')
    print('-' * 20)
    print(env.sampled_transition_function)
    print('-' * 20)
    try:
        with open("./q_val_table.txt", 'w') as f:
            print(agent.q_val_table, file=f)
    except:
        print('Failed to save to a Q-Value table to file')

    # Save the sampled Reward function
    try:
        with open("./reward_dict.txt", 'w') as f:
            print(env.sampled_reward_function, file=f)
    except:
        print('Failed to save to a Reward dictionary to file')

    # Save the sampled MDP function
    try:
        with open("./transition_function_dict.txt", 'w') as f:
            print(env.sampled_transition_function, file=f)
    except:
        print('Failed to save to a Reward dictionary to file')

    # Evaluate
    # Only act greedly ...
    agent.e_greedy_prob = 0
    sum_rewards = 0
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
        # Accumulate rewards
        sum_rewards += reward
        # Don't forget to update your state otherwise the robot will be stuck
        current_state = next_state
    print('Sum of rewards in %d steps: %d' % (num_steps_eval, sum_rewards))

