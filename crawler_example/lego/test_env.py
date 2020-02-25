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

# Number seem to converge in simulation
num_iterations_train = 400
# Bigger values decay faster
e_greedy_decay = 1. / num_iterations_train
initial_e_greedy_prob = 1.0

# Play a beep sound
brick.sound.beep()
print('Should display on VisualStudio')
seedling = int(round(time.time()))
random.seed(seedling)


# Initialize environemnt
env = CrawlingRobotEnv(step_angle=45)
current_state = env.reset()
agent = Q_Agent(env, gamma=0.9, alpha=0.2, e_greedy_prob=initial_e_greedy_prob, e_greedy_decay=e_greedy_decay)
action = 0

# Do the right sequence (1,5,2,4,0)
for step in range(4):
    next_state, reward, done, info = env.step(1)
    print('Action:',1 , 'Reward:', reward, 'next_state:', next_state)
    next_state, reward, done, info = env.step(5)
    print('Action:',5 , 'Reward:', reward, 'next_state:', next_state)
    next_state, reward, done, info = env.step(2)
    print('Action:',2 , 'Reward:', reward, 'next_state:', next_state)
    next_state, reward, done, info = env.step(4)
    print('Action:',4 , 'Reward:', reward, 'next_state:', next_state)
    next_state, reward, done, info = env.step(0)
    print('Action:',0 , 'Reward:', reward, 'next_state:', next_state)


"""
while True:
    # Move arm/leg depending on the button pressed
    if Button.UP in brick.buttons():
        action += 1
    elif Button.DOWN in brick.buttons():
        action -= 1
    next_state, reward, done, info =  env.step(action)
    print('State:', next_state, 'action:', action)
"""