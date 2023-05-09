# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Lenička & Kevička",
        "color": "#888888",
        "head": "default",
        "tail": "default",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    board_state = build_matrix(game_state)

    my_head = game_state["you"]["head"]
    matrix = -np.clip(board_state-2, -1, 0)
    # matrix[my_head["y"], my_head["x"]] = 2
    grid = Grid(matrix=matrix)

    start_pos = grid.node(**my_head)
    end_pos = grid.node(**game_state["board"]["food"][0])

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start_pos, end_pos, grid)

    next_idx = path[1]
    print(next_idx)

    action = next_move_to_string((start_pos.x, start_pos.y), next_idx)
    print(action)

    return {"move": action}


def next_move_to_string(current_idx, next_idx):
    actions = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}

    for key in actions:
        coord = tuple(sum(x) for x in zip(current_idx, actions[key]))
        if coord == next_idx:
            return key


def build_matrix(game_state):
    FOOD = 1
    SNAKE = 10

    board = game_state["board"]

    board_state = np.zeros((board["height"], board["width"]))

    # maybe be smarter about setting numbers, check id / name / shout / squad
    for snake in board["snakes"]:
        for body in snake["body"]:
            board_state[body["y"], body["x"]] = SNAKE
        board_state[snake["head"]["y"], snake["head"]["x"]] = SNAKE + 1

        SNAKE += 10

    for food in board["food"]:
        board_state[food["y"], food["x"]] = FOOD

    # currently disregard hazards
    return board_state


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
