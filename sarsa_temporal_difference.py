import random
import shutil
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
from gym.wrappers import RecordVideo
from gymnasium import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.spaces import Discrete
from numpy import ndarray
from tqdm import tqdm

from video import merge_videos

type State = int
type Action = int
type Move = tuple[State, Action]
type Moves = list[Move]


def play_episode(env: Env, q_table: ndarray, epsilon: float) -> ndarray:
    state, _ = env.reset()
    while True:
        best_action = policy(q_table, state)
        if random.uniform(0, 1) < epsilon:
            best_action = env.action_space.sample()
        move = (cast(State, state), best_action)
        next_state, reward, terminated, truncated, info = env.step(best_action)
        # if reward > 0:
        #     print(state)
        #     print(env.render())
        refresh_q_table(q_table, reward, move, next_state=next_state, learning_rate=.1, discount_factor=.95)
        state = next_state
        if terminated or truncated:
            return q_table


def policy(q_table: ndarray, state: int) -> Action:
    q_values_per_action = q_table[state]
    candidates = np.argwhere(q_values_per_action == q_values_per_action.max()).flatten()
    return np.random.choice(candidates)


def refresh_q_table(q_table: ndarray, reward: float, move: Move, next_state: State, learning_rate=.8,
                    discount_factor=.95) -> None:
    # SARSA means q table, on-policy. I.e. the same policy for rewarding actions that the one we used to take actions,
    # In practice it consists in retributing the move we've taken
    state, action = move

    next_state_action = policy(q_table, next_state)
    next_state_value = q_table[next_state, next_state_action]
    discounted_value_of_next_state = discount_factor * next_state_value
    q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discounted_value_of_next_state - q_table[state, action])
    # if reward != 0:
    #     print(f"Retributed {state = }, {action = } because {reward =}")


def learn_to_play_temporal_difference(env: Env):
    # Monte carlo means playing an entire episode before refreshing the q-table
    observation_space = cast(Discrete, env.observation_space)

    action_space = cast(Discrete, env.action_space)
    q_table = np.zeros(shape=(observation_space.n, action_space.n))
    NB_EPISODS = 100_000
    for i in range(NB_EPISODS):
        epsilon = ((NB_EPISODS - i) /(2* NB_EPISODS)) ** 2+.03 # Will decrease from 1 to 0 to prioritize exploitation versus exploration as training goes
        q_table = play_episode(env, q_table, epsilon)
        # print(q_table)
    merge_videos("videos")


if __name__ == '__main__':
    shutil.rmtree((Path(__file__).parent / "videos"), ignore_errors=True)
    MAP_SIZE = 5
    # env = gym.make("FrozenLake-v1", render_mode="rgb_array", desc=generate_random_map(MAP_SIZE,p=.7, seed=2),
    #                is_slippery=False)
    # env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda i: i % 1_000 == 0)

    learn_to_play_temporal_difference(env)
