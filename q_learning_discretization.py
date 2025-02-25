import random
import shutil
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
from gym.wrappers import RecordVideo
from gymnasium import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.spaces import Discrete, Box
from numpy import ndarray
from tqdm import tqdm

from video import merge_videos

type State = int
type Action = int
type Move = tuple[State, Action]
type Moves = list[Move]

NB_VALUES_PER_CONTINUOUS_VAR = 10

def play_episode(env: Env, q_table: ndarray, epsilon: float) -> ndarray:
    state, _ = env.reset()
    state = discretize_to_int(env.observation_space, NB_VALUES_PER_CONTINUOUS_VAR, state )
    total_reward = 0
    while True:
        best_action = policy(q_table, state)
        if random.uniform(0, 1) < epsilon:
            best_action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(best_action)
        next_state = discretize_to_int(env.observation_space, NB_VALUES_PER_CONTINUOUS_VAR, next_state)

        move = (cast(State, state), best_action)
        refresh_q_table(q_table, reward, move, next_state=next_state, learning_rate=.1, discount_factor=.95)
        state = next_state
        total_reward+=reward
        if terminated or truncated:
            return q_table, total_reward


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
    observation_space = cast(Box, env.observation_space)
    nb_different_observations = discretize_to_int(observation_space, NB_VALUES_PER_CONTINUOUS_VAR, observation_space.high)

    action_space = cast(Discrete, env.action_space)
    q_table = np.zeros(shape=(nb_different_observations+1, action_space.n))
    print(f"Q table shape : {q_table.shape}")
    NB_EPISODS = 30_000
    for i in range(NB_EPISODS):
        q_table, total_reward = play_episode(env, q_table, .1)
        if i%30==0:
            print(f"Episode {i} , {total_reward =}")
    merge_videos("videos")


def discretize(observation_space: Box, nb_values_per_continuous_var: int, observation: ndarray) -> ndarray:
    discretized = np.trunc(((observation - observation_space.low) / ( observation_space.high - observation_space.low))*nb_values_per_continuous_var).astype(int)
    discretized[discretized>=nb_values_per_continuous_var] -=1
    return discretized

def discretize_to_int(observation_space: Box, nb_values_per_continuous_var: int, observation: ndarray):
    discretized_array = discretize(observation_space,nb_values_per_continuous_var,observation)
    as_str = ''.join(str(n) for n in discretized_array)
    return int(as_str, nb_values_per_continuous_var)

def test_discretize():
    observation_space = Box(np.array([-2.5, - 2.5, - 10., - 10., - 6.2831855, - 10.,
                                      - 0., - 0.]), np.array([2.5, 2.5, 10., 10., 6.2831855, 10.,
                                                              1., 1.]))
    observation = np.array([-1, -1, 1, 1, -1, -1, 0, 0])
    nb_values_per_continuous_var = 2
    expected_discretization = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    actual_discretization = discretize(observation_space, nb_values_per_continuous_var, observation)
    assert(actual_discretization == expected_discretization).all()


if __name__ == '__main__':
    shutil.rmtree((Path(__file__).parent / "videos"), ignore_errors=True)
    MAP_SIZE = 5
    # env = gym.make("FrozenLake-v1", render_mode="rgb_array", desc=generate_random_map(MAP_SIZE,p=.7, seed=2),
    #                is_slippery=False)
    # env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    # env = gym.make("Taxi-v3", render_mode="rgb_array")
    env = gym.make("LunarLander-v3",render_mode="rgb_array", continuous=False,               enable_wind=False,)
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda i: i%500==0)

    learn_to_play_temporal_difference(env)
