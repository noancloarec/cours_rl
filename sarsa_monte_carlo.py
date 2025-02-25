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

type Observation = int
type Action = int
type Move = tuple[Observation, Action]
type Moves = list[Move]


def play_episode(env: Env, q_table: ndarray, epsilon: float) -> tuple[float, Moves]:
    state, _ = env.reset()
    moves = []
    while True:
        best_action = policy(q_table, state)
        if random.uniform(0, 1) < epsilon:
            best_action = env.action_space.sample()

        moves.append((cast(Observation, state), best_action))
        state, reward, terminated, truncated, info = env.step(best_action)
        # print(f"{state = }, {reward = }, {terminated = }, {truncated = }, {info =}")
        if terminated or truncated:
            return cast(float, reward), moves


def policy(q_table: ndarray, state: int) -> Action:
    q_values_per_action = q_table[state]
    candidates = np.argwhere(q_values_per_action == q_values_per_action.max()).flatten()
    return np.random.choice(candidates)


def refresh_q_table(q_table: ndarray, reward: float, moves: Moves, learning_rate=.8, discount_factor=.95) -> None:
    # SARSA means q table, on-policy. I.e. the same policy for rewarding actions that the one we used to take actions,
    # In practice it consists in retributing the moves we've taken
    for i, (state, action) in enumerate(reversed(moves)):
        q_table[state, action] += learning_rate * (reward * discount_factor ** i - q_table[state, action])


def learn_to_play_monte_carlo(env: Env):
    # Monte carlo means playing an entire episode before refreshing the q-table
    observation_space = cast(Discrete, env.observation_space)

    action_space = cast(Discrete, env.action_space)
    q_table = np.zeros(shape=(observation_space.n, action_space.n))

    for _ in tqdm(range(100)):
        reward, moves = play_episode(env, q_table, .1)
        refresh_q_table(q_table, reward, moves, learning_rate=.8)
    print(q_table)
    merge_videos("videos")


if __name__ == '__main__':
    shutil.rmtree((Path(__file__).parent / "videos"), ignore_errors=True)
    MAP_SIZE = 5
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", desc=generate_random_map(MAP_SIZE,p=.7, seed=2),
                   is_slippery=False)
    # env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    # env = RecordVideo(env, video_folder="videos", episode_trigger=lambda i: True)

    learn_to_play_monte_carlo(env)
