import random
import shutil
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
from gym.wrappers import RecordVideo
from gymnasium import Env
from gymnasium.spaces import Discrete
from keras import Sequential
from keras.src.layers import Dense
from numpy import ndarray

from video import merge_videos

type State = int
type Action = int
type Move = tuple[State, Action]
type Moves = list[Move]

discount_factor = .95


def get_model(nb_state_params: int, nb_actions: int) -> Sequential:
    model = Sequential()
    model.add(Dense(10, input_shape=nb_state_params, activation="relu"))
    model.add(Dense(nb_actions, activation="relu"))
    model.compile()
    return model


def play_episode(env: Env, q_table: ndarray, epsilon: float, model: Sequential) -> ndarray:
    state, _ = env.reset()
    while True:

        q_values = model.predict(state)
        best_action = np.argmax(q_values)

        if random.uniform(0, 1) < epsilon:
            best_action = env.action_space.sample()

        move = (cast(State, state), best_action)
        next_state, reward, terminated, truncated, info = env.step(best_action)

        if terminated:
            q_target = reward
        else:
            best_q_value_at_next_state = np.max(model.predict(next_state))
            q_target = reward + discount_factor * best_q_value_at_next_state
        target_q_values = np.copy(q_values)
        target_q_values[best_action] = q_target
        # Target_q_values est le y-train, il faudra stocker les x-y dans un batch et entrainer le modèle une fois le batch rempli

        state = next_state
        if terminated or truncated:
            return q_table


def argmax_policy(q_table: ndarray, state: int) -> Action:
    q_values_per_action = q_table[state]
    candidates = np.argwhere(q_values_per_action == q_values_per_action.max()).flatten()
    return np.random.choice(candidates)


def refresh_q_table(q_table: ndarray, reward: float, move: Move, next_state: State, learning_rate=.8,
                    discount_factor=.95) -> None:
    # SARSA means q table, on-policy. I.e. the same policy for rewarding actions that the one we used to take actions,
    # In practice it consists in retributing the move we've taken
    state, action = move

    next_state_action = argmax_policy(q_table, next_state)
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
        epsilon = ((NB_EPISODS - i) / (
                2 * NB_EPISODS)) ** 2 + .03  # Will decrease from 1 to 0 to prioritize exploitation versus exploration as training goes
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
