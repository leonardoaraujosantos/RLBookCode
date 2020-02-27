#!/usr/bin/env pybricks-micropython
from crawler_lego_env import CrawlingRobotEnv
from agent import Q_Agent

from pybricks import ev3brick as brick
from pybricks.ev3devices import Motor, InfraredSensor
from pybricks.parameters import Port, Stop, Button
from pybricks.tools import print
import utils_motor
import random
import time

# Play a beep sound
brick.sound.beep()
print('Should display on VisualStudio')
seedling = int(round(time.time()))
random.seed(seedling)

# Initialize environment
# If we invert the reward during training the robot should change direction
env = CrawlingRobotEnv(step_angle=45, invert_reward=True)
current_state = env.reset()
agent = Q_Agent(env, gamma=0.9, alpha=0.2)

# Do the right sequence (1,5,2,4,0)
# 0: LEG NEUTRAL
# 1: LEG UP
# 2: LEG DOWN
# 3: FEET NEUTRAL
# 4: FEET UP
# 5: FEET DOWN
print('Distance:', env.read_sensor())
# Backward
#list_actions = [1, 4, 2, 5, 1, 4, 2, 5, 1, 4, 2, 5, 1, 4, 2, 5,1, 4, 2, 5, 1, 4, 2, 5, 0, 3]
# Forward
list_actions = [1, 5, 2, 4, 1, 5, 2, 4, 1, 5, 2, 4, 1, 5, 2, 4, 1, 5, 2, 4, 1, 5, 2, 4, 0, 3]

# Exercise action, check if we have a positive sum of rewards
sum_reward = 0
for steps, action in enumerate(list_actions):
    current_state_str = str(env)
    next_state, reward, done, info = env.step(action)
    action_str = env.action_idx_to_str(action)
    next_state_str = env.state_idx_to_str(next_state)
    print('steps:', steps, '\n\tcurrent_state:', current_state_str, '\n\tACTION:', action_str, '\n\tnext_state:',
    next_state_str, '\n\treward:', reward, '\nprob:', agent.e_greedy_prob)
    print('-' * 20)
    sum_reward += reward
print('Sum of rewards:', sum_reward)
if sum_reward > 0:
    print('Good')
else:
    print('Bad ....')

print('Distance:', env.read_sensor())
