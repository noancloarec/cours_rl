import os
import random
from pprint import pprint
from typing import cast

import ffmpeg
from gym.wrappers import RecordVideo
from gymnasium import Env
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.spaces import Discrete

from play_frozen_lake import  GameStat
import numpy as np
from numpy import ndarray

from video import merge_videos

type Observation = int
type Action = int
type Move = tuple[Observation, Action]
type Moves = list[Move]

MAP_SIZE = 5
env = gym.make("FrozenLake-v1", render_mode="rgb_array", desc = generate_random_map(MAP_SIZE, p=.6, seed=2),is_slippery=False)
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda i:i%100==0)
def play_episode(env:Env, q_table:ndarray, epsilon:float)-> tuple[float, Moves]:
    episode_q_table = q_table.copy()
    observation, _ = env.reset()
    moves = []
    while True:
        best_action = chose_best_action(episode_q_table[observation])
        # If the best action is not better than other actions
        if episode_q_table[observation, best_action]==0:
            best_action = env.action_space.sample()
        # Or more randomly
        if random.uniform(0, 1) < epsilon:
            best_action = env.action_space.sample()

        episode_q_table[observation, best_action] = -1 # Suggest we do not do it again

        moves.append((cast(Observation, observation), best_action))
        observation, reward, terminated, truncated, info = env.step(best_action)
        # print(env.render())
        if terminated or truncated:
            #reward = compute_reward(observation, MAP_SIZE)
            # print(reward)
            return cast(float, reward), moves

def compute_reward(observation: Observation, map_size:int):
    # The reward depends on the distance from the agent and the goal
    agent_coordinate = np.array([observation//map_size, observation%map_size])
    goal_coordinate = np.array([map_size-1, map_size-1])
    distance = np.linalg.norm(goal_coordinate - agent_coordinate)
    normalized_distance = distance / np.linalg.norm(goal_coordinate)
    return 1 - normalized_distance

def chose_best_action(q_row:ndarray)-> Action:
    candidates = np.argwhere(q_row==q_row.max()).flatten()
    return np.random.choice(candidates)

def refresh_q_table(q_table:ndarray, reward: float, moves: Moves, learning_rate=.8, discount_factor=.95)->None:
    adaptive_reward = reward
    # if moves[:-1] == [MAP_SIZE-1, MAP_SIZE-1]:
    #     moves_to_reward = moves
    # else:
    #     moves_to_reward = moves[:-1]

    for observation, action in reversed(moves):
        q_table[observation, action] += learning_rate * (adaptive_reward - q_table[observation, action])
        adaptive_reward*=discount_factor

def learn_to_play(env: Env)->np.ndarray:
    observation_space = cast(Discrete, env.observation_space)

    action_space = cast(Discrete, env.action_space)
    q_table = np.zeros(shape=(observation_space.n,action_space.n))

    for _ in range(50000):

        reward, moves = play_episode(env, q_table, .1)
        refresh_q_table(q_table, reward, moves, learning_rate=.8)
    print(q_table)
    merge_videos("videos")




if __name__ == '__main__':
    print(learn_to_play(env))