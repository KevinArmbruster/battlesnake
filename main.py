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
import argparse
import json
import random
import typing
import numpy as np
from source.MXNetEnv.inference.inference_src.predict import *
from source.BattlesnakeGym.battlesnake_gym.snake_gym import Game_state_parser
from source.BattlesnakeGym.battlesnake_gym.snake_gym import Snakes
from source.BattlesnakeGym.battlesnake_gym.snake_gym import Food

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

    snakes, food, _ = Game_state_parser(game_state).parse()
    others = snakes.get_snake_51_map(excluded_snakes=[0])
    us = snakes.get_snake_51_map(excluded_snakes=np.arange(1, len(snakes.get_snakes())))
    food_map = food.get_food_map()


    state = [food_map, us, others] # np.stack((food_map, us, others), axis=-1)

    data = {"state":state,
            "snake_id":0.0,#float(game_state["you"]["id"]),
            "turn_count": game_state["turn"],
            "health": game_state["you"]["health"]}


    print("model loaded")
    response_body, output_content_type = transform_fn(model, data, None, "json")
    print(f"response = {response_body}")
    return response_body


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

# def loadModel():
#     from source.MXNetEnv.training.training_src.networks.agent import MultiAgentsCollection
#
#     parser = argparse.ArgumentParser(
#         description='Train a DQN agent for the Battlesnake.IO')
#
#     parser.add_argument('--seeds', nargs="+", type=int, default=[0, 666, 15],
#                         help='seed for randomiser. Code will run n times given the number of seeds')
#
#     # Gym configurations
#     parser.add_argument('--map_size', type=str, default="[11, 11]",
#                         help='Size of the battlesnake map, default (15, 15)')
#     parser.add_argument('--number_of_snakes', type=int, default=4, help='Number of snakes')
#
#     # Representation configurations
#     parser.add_argument('--snake_representation', type=str, default="bordered-51s",
#                         help="how to represent the snakes and the gym, default bordered-51s, options: [\"flat-num\", \"bordered-num\", \"flat-51s\", \"bordered-51s\"])")
#     parser.add_argument('--state_type', type=str, default="one_versus_all",
#                         help='Output option of the state, default: layered, options: ["layered", "one_versus_all"]')
#
#     # Training configurations
#     parser.add_argument('--buffer_size', type=int, default=5000,
#                         help='Buffer size (default: 5000)')
#     parser.add_argument('--update_every', type=int, default=20,
#                         help='Episodes to update network (default 20)')
#     parser.add_argument('--lr_start', type=float, default=0.0005,
#                         help='Starting learning rate (default: 0.0005)')
#     parser.add_argument('--lr_step', type=int, default=5e5,
#                         help='Number of steps for learning rate decay (default: 50k)')
#     parser.add_argument('--lr_factor', type=float, default=0.5,
#                         help='Factor to decay learning rate (default: 0.5)')
#     parser.add_argument('--gamma', type=float, default=0.95,
#                         help='discount factor (default: 0.95)')
#     parser.add_argument('--tau', type=float, default=1e-3,
#                         help='soft update factor (default: 1e-3)')
#     parser.add_argument('--batch_size', type=int, default=64,
#                         help='Batch size (default: 64)')
#     parser.add_argument('--episodes', type=int, default=200,
#                         help='Number of espidoes (default: 100000)')
#     parser.add_argument('--max_t', type=int, default=1000,
#                         help='Max t (default: 1k)')
#     parser.add_argument('--eps_start', type=float, default=1.0,
#                         help='Episilon start (default: 1.0)')
#     parser.add_argument('--eps_end', type=float, default=0.01,
#                         help='Episilon end (default: 0.01)')
#     parser.add_argument('--eps_decay', type=float, default=0.995,
#                         help='Episilon decay (default: 0.995)')
#     parser.add_argument('--warmup', type=float, default=0,
#                         help='Warmup (default: 0)')
#
#     # Network configurations
#     path = "/home/vinlenka/KTH/multi-agentAI/assignment4/battlesnake/source/MXNetEnv/inference/pretrained_models/Model-11x11/local-0000.params"
#
#     parser.add_argument('--load', default=path, help='Load from param file')
#     parser.add_argument('--load_only_conv_layers', default=False,
#                         help='Boolean to define if only the convolutional layers should be loaded')
#     parser.add_argument('--qnetwork_type', default="attention",
#                         help='Type of q_network. Options: ["concat", "attention"]')
#     parser.add_argument('--starting_channels', type=int, default=6,
#                         help='starting channels for qnetwork')
#     parser.add_argument('--number_of_conv_layers', type=int, default=3,
#                         help='Number of conv. layers for qnetwork concat')
#     parser.add_argument('--number_of_dense_layers', type=int, default=2,
#                         help='Number of dense layers for qnetwork concat')
#     parser.add_argument('--depthS', type=int, default=10,
#                         help='depth of the embeddings for the snake ID for qnetwork attention')
#     parser.add_argument('--depth', type=int, default=200,
#                         help='depth of the embeddings for the snake health for qnetwork attention')
#     parser.add_argument('--number_of_hidden_states', type=int, default=128,
#                         help='Number of hidden states in the qnetwork')
#     parser.add_argument('--kernel_size', type=int, default=3,
#                         help='kernel size for the qnetwork')
#     parser.add_argument('--repeat_size', type=int, default=3,
#                         help='Size to repeat input states')
#     parser.add_argument('--activation_type', type=str, default="softrelu",
#                         help='Activation for qnetwork')
#     parser.add_argument('--sequence_length', type=int, default=2,
#                         help='Number of states to feed sequencially feed in')
#
#     # Logging information
#     parser.add_argument('--print_score_steps', type=int, default=100,
#                         help='Steps to print score (default: 100)')
#     parser.add_argument('--models_to_save', type=str, default='all',
#                         help='select which models to save options ["all", "local"] (default: all)')
#     parser.add_argument('--save_only_best_models', type=bool, default=False,
#                         help='Save only the best models')
#     parser.add_argument('--save_model_every', type=int, default=100,
#                         help='Steps to save the model')
#     parser.add_argument('--model_dir', type=str, default=None)
#     parser.add_argument('--render_steps', type=int, default=1000,
#                         help='Steps to render (default: 1000)')
#     parser.add_argument('--should_render', action='store_true',
#                         help='render the environment to generate a gif in /gifs')
#     parser.add_argument('--writer', action='store_true',
#                         help='should write to tensorboard')
#     parser.add_argument('--print_progress', action='store_true',
#                         help='should print every progressive step')
#     parser.add_argument('--run_name', type=str, default="run",
#                         help='Run name to save reward (default: run+seed)')
#
#     args = parser.parse_args()
#
#     map_size = args.map_size
#     if args.state_type == "layered":
#         state_depth = 1 + args.number_of_snakes
#     elif args.state_type == "one_versus_all":
#         state_depth = 3
#     if "bordered" in args.snake_representation:
#         state_shape = (map_size[0]+2, map_size[1]+2, state_depth)
#     else:
#         state_shape = (map_size[0], map_size[1], state_depth)
#     agent_params = (0, "modelDir",
#                     args.load, args.load_only_conv_layers,
#                     args.models_to_save,
#                     # State configurations
#                     args.state_type, state_shape, args.number_of_snakes,
#
#                     # Learning configurations
#                     args.buffer_size, args.update_every,
#                     args.lr_start, args.lr_step, args.lr_factor,
#                     args.gamma, args.tau, args.batch_size,
#
#                     # Network configurations
#                     args.qnetwork_type, args.sequence_length,
#                     args.starting_channels, args.number_of_conv_layers,
#                     args.number_of_dense_layers, args.number_of_hidden_states,
#                     args.depthS, args.depth,
#                     args.kernel_size, args.repeat_size,
#                     args.activation_type)
#
#     agent = MultiAgentsCollection(*agent_params)
#     return agent


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    model = model_fn(model_dir="source/MXNetEnv/inference/pretrained_models/Model-11x11")
    run_server({"info": info, "start": start, "move": move, "end": end})
