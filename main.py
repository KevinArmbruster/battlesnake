import random
import typing
from snake_basic_behaviors import *
from snake_strategy_behaviors import *


# ----------------------------------------------------------------------------
# SERVER STUFF
# -----------------------------------------------------------------------------


def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Lenička & Kevička",
        "color": "#00FFFF",
        "head": "do-sammy",
        "tail": "coffee",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:
    # start_time = time.time()

    minimax_move = miniMaxEntry(game_state)

    if minimax_move:
        next_move = minimax_move

    else:
        print("Minimax failed! Using stupid 1 step look ahead")

        is_move_safe = {
            "up": DEFAULT_MOVE_VALUE,
            "down": DEFAULT_MOVE_VALUE,
            "left": DEFAULT_MOVE_VALUE,
            "right": DEFAULT_MOVE_VALUE
        }

        preventBack(game_state, is_move_safe)
        outOfBounds(game_state, is_move_safe)
        selfCollision(game_state, is_move_safe)
        collision(game_state, is_move_safe)

        safe_moves = {}
        available_moves = []
        for move, moveValue in is_move_safe.items():
            if moveValue >= DEFAULT_MOVE_VALUE:
                safe_moves[f"{move}"] = moveValue
                available_moves.append(move)

        if len(safe_moves) > 0:
            next_move = random.choice(list(safe_moves.keys()))

        else:
            next_move = "down"

    print(f"MOVE {game_state['turn']}: {next_move}, SNAKE HEALTH: {game_state['you']['health']}")
    # print(f"Total time {time.time() - start_time:.3f}s")
    return {"move": next_move}


if __name__ == "__main__":
    from server import run_server

    run_server({
        "info": info,
        "start": start,
        "move": move,
        "end": end
    })
