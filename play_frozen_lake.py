import gymnasium as gym
from time import sleep
from pydantic import BaseModel, Field
from tqdm import tqdm

# Initialise the environment

env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)


class GameStat(BaseModel):
    states: list = Field(exclude=True)
    tries: int
    successes: int


def play_game(nb_games: int) -> GameStat:
    states = []
    games_played = 0
    successes = 0
    observation, info = env.reset(seed=42)
    with tqdm(total=nb_games) as progress_bar:
        while games_played < nb_games:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            states.append(observation)
            if terminated or truncated:
                games_played += 1
                progress_bar.update(1)
                if observation == 15:
                    successes += 1
                observation, info = env.reset(seed=42)
    return GameStat(states=states, tries=nb_games, successes=successes)


if __name__ == '__main__':
    print(play_game(10000).model_dump())