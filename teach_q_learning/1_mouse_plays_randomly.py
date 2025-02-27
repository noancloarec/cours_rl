from random import randint

from mouse_env import MouseEnv, display_move, pause_if_necessary
from matplotlib import pyplot as plt

def main():
    env = MouseEnv()
    env.render()
    pause_if_necessary()
    NB_EPISODES = 100
    rewards_per_episode = []
    for i in range(NB_EPISODES):
        cumulated_reward = 0
        while True:
            # Chose the action to perform randomly
            action = randint(0, 3)
            # Perform the action on the env
            position, reward, terminated, nb_moves = env.step(action)
            display_move(action, position, reward, nb_moves, env)
            cumulated_reward+=reward


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
