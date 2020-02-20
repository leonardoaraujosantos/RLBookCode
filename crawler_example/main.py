from agent import Q_Agent
from crawler_env import CrawlingRobotEnv

all_rewards = 0
# On this reference environment the action space is 4 (simpler than lego)
# Also it's state space is a bit more complex, including velocities, arm positions, etc...
env = CrawlingRobotEnv(render=False)
current_state = env.reset()
agent = Q_Agent(env, gamma=0.9, alpha=0.2)
total_reward = 0
num_iterations_train = 900000

# Get the action space
print('Robot action space:', env.action_space.n)
print('Robot state-space:', env.observation_space)

# Training
i = 0
while i < num_iterations_train:
    i = i + 1
    action = agent.choose_action(current_state)
    next_state, reward, done, info = env.step(action)
    agent.update_q_table(current_state, action, reward, next_state)
    current_state = next_state
    total_reward += reward

    # Evaluate
    if i % 5000 == 0:
        print("average_reward in last 5000 steps", total_reward / i)
        # Stop training if total reward is big enough
        if (total_reward / i) > 1.3:
            break
        average_reward = 0
        env.render = False


# Evaluating
env = CrawlingRobotEnv(render=True)
current_state=env.reset()
total_reward = 0
# Force epsilon-greedy to always use the max Q
agent.e_greedy_prob = 0

i = 0
while True:
    i = i+1
    action = agent.choose_action(current_state)
    next_state, reward, done, info = env.step(action)
    current_state = next_state
    total_reward += reward

    # Evaluate
    if i % 5000 == 0:
        print("average_reward in last 5000 steps", total_reward / 5000)
        i = 0
        average_reward = 0
        env.render = True
