from enum import IntEnum
from random import randint

from gymnasium.spaces import Discrete
from pandas import DataFrame
from py_markdown_table.markdown_table import markdown_table


def get_position(state: int) -> tuple[int, int]:
    return (state // 3, state % 3)


class ActionEnum(IntEnum):
    HAUT = 0
    DROITE = 1
    BAS = 2
    GAUCHE = 3

def get_arrow(action:ActionEnum|int):
    match action:
        case ActionEnum.HAUT:
            return "⬆️"
        case ActionEnum.BAS:
            return "⬇️"
        case ActionEnum.GAUCHE:
            return "⬅️"
        case ActionEnum.DROITE:
            return "➡️"

class MouseEnv:
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Discrete(9)
        self.cheese_position = [0, 1]
        self.lots_of_cheese_position = [2, 2]
        self.reset()

    def reset(self):
        self.mouse_position = [0, 0]
        self.moves_played = 0

    def render(self):
        # Place a dot at the given coordinate
        grid = [['      ' for _ in range(3)] for _ in range(3)]
        row, col = self.mouse_position
        # Place a dot at the given coordinate
        grid[row][col] = '  🐭  '
        row_cheese, col_cheese = self.cheese_position
        grid[row_cheese][col_cheese] = '  🧀  '
        row_lots_of_cheese, col_lots_of_cheese = self.lots_of_cheese_position
        grid[row_lots_of_cheese][col_lots_of_cheese] = '🧀🧀🧀'
        if self.mouse_position == self.cheese_position:
            grid[row_cheese][col_cheese] = ' 🐭🧀 '
        if self.mouse_position == self.lots_of_cheese_position:
            grid[row_lots_of_cheese][col_lots_of_cheese] = '🧀🧀🧀🐭'

        # Print the grid with pipes separating cells
        for r in range(3):
            for c in range(3):
                print(grid[r][c], end=' ')
                if c < 2:
                    print('|', end=' ')
            print()  # New line after each row
            print()  # New line after each row

    def observation(self):
        return self.mouse_position[0] * 3 + self.mouse_position[1]

    def step(self, action: int):
        prior_position = self.mouse_position.copy()
        match action:
            case ActionEnum.HAUT if self.mouse_position[0] >= 1:
                self.mouse_position[0] -= 1
            case ActionEnum.BAS if self.mouse_position[0] < 2:
                self.mouse_position[0] += 1
            case ActionEnum.DROITE if self.mouse_position[1] < 2:
                self.mouse_position[1] += 1
            case ActionEnum.GAUCHE if self.mouse_position[1] >= 1:
                self.mouse_position[1] -= 1
        if prior_position != self.cheese_position and self.mouse_position == self.cheese_position:
            reward = 1
        elif prior_position != self.lots_of_cheese_position and self.mouse_position == self.lots_of_cheese_position:
            reward = 3
        else:
            reward = 0
        self.moves_played += 1
        return self.observation(), reward, self.moves_played >= 4, self.moves_played
steps_before_pause = 0
def pause_if_necessary():
    global steps_before_pause
    if steps_before_pause <= 0:
        try:
            steps_before_pause = int(input())
        except ValueError:
            steps_before_pause = 1
    steps_before_pause -=1

def display_move(action, position, reward, moves, env):
    print("-" * 30)
    print(f"Action: {get_arrow(action)} , recompense : {reward}, position : {position}, nombre de mouvements total : {moves}")
    env.render()

def display_q_table(q_table):
    print("------ Q-Table ------")
    df = DataFrame({"  "+get_arrow(i) : q_table[:,i] for i in range(4)} )
    print(df.to_markdown())

