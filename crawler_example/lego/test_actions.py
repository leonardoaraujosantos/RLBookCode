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
env = CrawlingRobotEnv(step_angle=30, invert_reward=True)
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
#list_actions = [2,5,1,2,5,1,2,5,1,2,5,1,2,5,1]

# Forward
list_actions = [1, 5, 2, 4, 1, 5, 2, 4, 1, 5, 2, 4, 1, 5, 2, 4, 1, 5, 2, 4, 1, 5, 2, 4, 0, 3]

# Exercise action, check if we have a positive sum of rewards
sum_reward = 0
for action in list_actions:
    next_state, reward, done, info = env.step(action)
    print('Action:', action, 'Reward:', reward, 'next_state:', next_state)
    sum_reward += reward
print('Sum of rewards:', sum_reward)
if sum_reward > 0:
    print('Good')
else:
    print('Bad ....')
