import gymnasium as gym
from battlesnake_env import BattleSnake
import numpy as np
###########################################
#         Stage 1 - Initialization
###########################################

# create the cartpole environment
# create an instance of our custom environment
env = BattleSnake()

# initialize the environment
env.reset()
env.render()

terminated = False
while not terminated:
    # choose a random action
    action = np.random.randint(0,4) #env.action_space.sample()

    # take the action and get the information from the environment
    new_state, reward, terminated, truncated, info = env.step(action)

    # show the current position and reward
    env.render(action=action, reward=reward)

# # run for 10 episodes
# for episode in range(10):
#
#   # put the environment into its start state
#   env.reset()
#
# ###########################################
# #            Stage 2 - Execution
# ###########################################
#
#   # run until the episode completes
#   terminated = False
#   while not terminated:
#
#     # show the environment
#     env.render()
#
#     # choose a random action
#     action = env.action_space.sample()
#
#     # take the action and get the information from the environment
#     observation, reward, terminated, truncated, info = env.step(action)
#
#
# ###########################################
# #           Stage 3 - Termination
# ###########################################
#
# # terminate the environment
# env.close()
