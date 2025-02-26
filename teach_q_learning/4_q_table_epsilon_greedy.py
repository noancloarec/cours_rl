from random import randint

import numpy as np
from matplotlib import pyplot as plt

from mouse_env import MouseEnv, display_move, pause_if_necessary, display_q_table

def policy_greedy(state:int, q_table: np.ndarray, epsilon:float):
    if np.random.sample() < epsilon:
        return randint(0,3)
    candidates= np.argwhere(q_table[state] == q_table[state].max())
    return np.random.choice(candidates.flatten())

def main():
    env = MouseEnv()
    env.render()

    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.int8)
    display_q_table(q_table)
    pause_if_necessary()
    position = 0
    rewards_per_episode = []

    NB_EPISODES = 1000
    for i in range(NB_EPISODES):
        position = 0
        cumulated_reward = 0

        while True:
            # Chose the action to perform randomly
            action = policy_greedy(position, q_table, 0.2)

            # Perform the action on the env
            position_after_move, reward, terminated, moves = env.step(action)
            display_move(action, position_after_move, reward, moves, env)
            cumulated_reward+=reward


            # Pause the loop to give time to the teacher to explain what happened
            pause_if_necessary()

            # Fill the q-table
            q_table[position, action] = reward
            display_q_table(q_table)

            position = position_after_move

            # Pause the loop to give time to the teacher to explain what happened
            pause_if_necessary()

            # If the game is over (after 4 moves), reset the env
            if terminated:
                print("GAME OVER")
                env.reset()
                pause_if_necessary()
                env.render()
                pause_if_necessary()
                break
        rewards_per_episode.append(cumulated_reward)
    plt.plot(rewards_per_episode)
    plt.show()


if __name__ == '__main__':
    main()
