import copy
from collections import deque
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid as AStarGrid
from pathfinding.finder.a_star import AStarFinder

# ----------------------------------------------------------------------------
# SNAKE STRATEGY BEHAVIOR
# -----------------------------------------------------------------------------

BOARD_EMPTY = 0
BOARD_FOOD = 1
BOARD_HEAD = 2

# Generates a copy of current game board and another board that tracks snake head positions
def createBoardState(game_state):
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    foods = game_state["board"]["food"]
    all_snake = game_state["board"]["snakes"]

    # id = corresponding snake body
    # snake head is represented in head_board as the corresponding snake id

    # 2d array, containing board state, with heads as BOARD_HEAD, and bodies as id's
    board_copy = [[BOARD_EMPTY for _ in range(board_width)] for _ in range(board_height)]
    # 2d array, containing only the heads as id's
    head_board = [[BOARD_EMPTY for _ in range(board_width)] for _ in range(board_height)]

    for food in foods:
        food_x = food["x"]
        food_y = board_height - 1 - food["y"]
        board_copy[food_y][food_x] = BOARD_FOOD

    for snake in all_snake:
        snake_id = snake["id"]
        snake_head = snake["head"]

        for body in snake["body"]:
            body_x = body["x"]
            body_y = board_height - 1 - body["y"]
            if (body == snake_head):
                board_copy[body_y][body_x] = BOARD_HEAD
                head_board[body_y][body_x] = snake_id
            else:
                board_copy[body_y][body_x] = snake_id

    board_state = {
        "state_board": board_copy,
        "head_board": head_board
    }

    return board_state


# Create an array of snakes, each snake is a dict containing id, head and body coord
def snakeState(game_state):
    snakes = game_state["board"]["snakes"]
    board_height = game_state["board"]["height"]

    snake_state = []
    for snake in snakes:
        snake_id = snake["id"]
        health = snake["health"]
        snake_head = {"x": snake["head"]["x"],
                      "y": board_height - 1 - snake["head"]["y"]}
        snake_body = []
        for body in snake["body"]:
            body_x = body["x"]
            body_y = body["y"]
            snake_body.append({"x": body_x, "y": board_height - 1 - body_y})

        snake_state.append({
            "id": snake_id,
            "head": snake_head,
            "body": snake_body,
            "health": health
        })

    return snake_state


# Create an entire copy of the current game state, including current board, snakes and curr snake id
def createGameState(game_state, curr_snake_id):
    game_state_copy = {}

    game_state_copy["turn"] = game_state["turn"]
    game_state_copy["board"] = createBoardState(game_state)
    game_state_copy["snakes"] = snakeState(game_state)
    game_state_copy["curr_snake_id"] = curr_snake_id
    # game_state_copy["partner_snake_id"] = curr_snake_id  # TODO fill

    return game_state_copy


def updateSnakeHeadCoords(new_snake_state, curr_snake_index, x_coord, y_coord):
    new_snake_state[curr_snake_index]["head"]["x"] = x_coord
    new_snake_state[curr_snake_index]["head"]["y"] = y_coord
    new_snake_state[curr_snake_index]["body"][0]["x"] = x_coord
    new_snake_state[curr_snake_index]["body"][0]["y"] = y_coord


def updateSnakeBodyCoords(new_snake_state, curr_snake_index, body_index, x_coord, y_coord):
    new_snake_state[curr_snake_index]["body"][body_index]["x"] = x_coord
    new_snake_state[curr_snake_index]["body"][body_index]["y"] = y_coord


def updateSnakeHealth(new_snake_state, curr_snake_index, isAlive, hasAte):
    if (hasAte):
        new_snake_state[curr_snake_index]["health"] = 100   # FULL
    elif (isAlive):
        new_snake_state[curr_snake_index]["health"] -= 1    # STEP
    else:
        new_snake_state[curr_snake_index]["health"] = 0     # DEAD

    return new_snake_state[curr_snake_index]["health"]


# Update the snake's movement location in the new board and head state, also updates snake state's coords
def moveForward(new_board_state, new_head_state, new_snake_state, curr_snake_id, curr_snake_index, curr_snake_body,
                head_x, head_y, is_tail):
    prev_x, prev_y = None, None

    body_index = 0
    for body in curr_snake_body:
        curr_x = body["x"]
        curr_y = body["y"]

        if (not is_tail):
            new_board_state[curr_y][curr_x] = 0

        if (body_index == 0):
            new_board_state[head_y][head_x] = 2
            new_head_state[head_y][head_x] = curr_snake_id
            updateSnakeHeadCoords(new_snake_state, curr_snake_index, head_x, head_y)
        else:
            if (new_head_state[prev_y][prev_x] == curr_snake_id):
                new_head_state[prev_y][prev_x] = "0"
            if (body_index > 0):
                updateSnakeBodyCoords(new_snake_state, curr_snake_index,
                                      body_index, prev_x, prev_y)
            new_board_state[prev_y][prev_x] = curr_snake_id

        prev_x = curr_x
        prev_y = curr_y
        body_index += 1


def snakeStateFoodGrow(new_snake_state, curr_snake_index):  # TODO maybe error, only attaches new tail, but same pos as old
    last_body = new_snake_state[curr_snake_index]["body"][-1]
    last_x, last_y = last_body["x"], last_body["y"]
    new_snake_state[curr_snake_index]["body"].append({"x": last_x, "y": last_y})


def removeKilledSnakeBodyFromStateBoards(new_board_state, new_head_state, new_snake_state, snake_index):
    snake_body = new_snake_state[snake_index]["body"]
    snake_head = new_snake_state[snake_index]["head"]

    for body in snake_body:
        body_x = body["x"]
        body_y = body["y"]

        if (body == snake_head):
            new_head_state[body_y][body_x] = BOARD_EMPTY
        new_board_state[body_y][body_x] = BOARD_EMPTY


# Remove killed snake from the snake state list and board
def removeKilledSnake(new_board_state, new_head_state, new_snake_state, snake_index):
    removeKilledSnakeBodyFromStateBoards(new_board_state, new_head_state, new_snake_state, snake_index)
    new_snake_state.pop(snake_index)


# Find snake corresponding to the given current ID and return its info
def findCurrentSnake(new_snake_state, curr_snake_id):
    curr_snake_index = 0
    curr_snake_length = 0
    curr_snake_body = None
    curr_snake_health = 0
    curr_snake_tail = None

    for i in range(len(new_snake_state)):
        curr_snake = new_snake_state[i]
        if (curr_snake["id"] == curr_snake_id):
            curr_snake_index = i
            curr_snake_body = curr_snake["body"]
            curr_snake_length = len(curr_snake_body)
            curr_snake_health = curr_snake["health"]
            curr_snake_tail = curr_snake_body[-1]

    return curr_snake_index, curr_snake_length, curr_snake_body, curr_snake_health, curr_snake_tail


# Creates a deep copy of current game state to create a new game state
def createNewGameStateAndAdvanceTurn(game_state, curr_snake_id):
    new_game_state = copy.deepcopy(game_state)
    new_board_state = new_game_state["board"]["state_board"]
    new_head_state = new_game_state["board"]["head_board"]
    new_snake_state = new_game_state["snakes"]
    new_game_state["turn"] = game_state["turn"] + 1
    new_game_state["curr_snake_id"] = curr_snake_id

    return new_game_state, new_board_state, new_head_state, new_snake_state


# Find current snake's head coordinates
def findHeadCoord(width, height, new_head_state, curr_snake_id):
    head_x, head_y = None, None

    for y in range(height):
        for x in range(width):
            if (new_head_state[y][x] == curr_snake_id):
                head_x = x
                head_y = y
                break
        if (head_x != None):
            break

    return head_x, head_y


# Update head coordinate to its future head coordinate after move
def updateHeadCoord(x, y, move):
    if (move == "up"):
        y = y - 1
    elif (move == "down"):
        y = y + 1
    elif (move == "left"):
        x = x - 1
    elif (move == "right"):
        x = x + 1

    return x, y


# Creates a new version of game state with the move and the correspondent snake
def makeMove(game_state, curr_snake_id, move):
    board_width = len(game_state["board"]["state_board"][0])
    board_height = len(game_state["board"]["state_board"])

    # new game state to update, change the id to current snake
    new_game_state, new_board_state, new_head_state, new_snake_state = createNewGameStateAndAdvanceTurn(game_state, curr_snake_id)

    # Our snake's head coordinates
    head_x, head_y = findHeadCoord(board_width, board_height, new_head_state, curr_snake_id)

    # Current snake does not exist
    if head_x is None or head_y is None:
        return None

    # Update head coordinate value to destination after move is applied
    head_x, head_y = updateHeadCoord(head_x, head_y, move)

    # Acquire current snake info
    curr_snake_index, curr_snake_length, curr_snake_body, curr_snake_health, curr_snake_tail = findCurrentSnake(new_snake_state, curr_snake_id)

    # Check if snake destination hits border
    if not (0 <= head_x < board_width and 0 <= head_y < board_height):
        removeKilledSnake(new_board_state, new_head_state, new_snake_state, curr_snake_index)
        # updateSnakeHealth(new_snake_state, curr_snake_index, False, False)
        return new_game_state

    destination_cell = new_board_state[head_y][head_x]
    destination_cell_head = new_head_state[head_y][head_x]

    # Checks if snake runs into another snake or edge boundary
    if destination_cell not in [BOARD_EMPTY, BOARD_FOOD]:

        # Check if collision is with the head of a snake
        if destination_cell == BOARD_HEAD and destination_cell_head != BOARD_EMPTY:
            destination_snake_length = 0
            destination_snake_body = None
            destination_snake_index = 0

            # Find the snake the current snake is about to collide with
            for snake in new_snake_state:
                if snake["id"] == destination_cell_head:
                    destination_snake_body = snake["body"]
                    destination_snake_length = len(destination_snake_body)
                    break

                destination_snake_index += 1

            # Our size is bigger and we kill the another snake
            if destination_snake_length < curr_snake_length:

                # Remove destination snake from game board and snake state
                removeKilledSnake(new_board_state, new_head_state, new_snake_state, destination_snake_index)

                # Index might have changed when snake is removed
                curr_snake_index, _, _, _, _ = findCurrentSnake(new_snake_state, curr_snake_id)

                # Snake moves forward and updates all coords in new game state
                moveForward(new_board_state, new_head_state, new_snake_state,
                            curr_snake_id, curr_snake_index, curr_snake_body, head_x, head_y, False)

                curr_health = updateSnakeHealth(
                    new_snake_state, curr_snake_index, True, False)

                # check if our snake ran out of health
                if (curr_health <= 0):
                    removeKilledSnake(
                        new_board_state, new_head_state, new_snake_state, curr_snake_index)

            # Our snake is smaller or same size
            else:
                removeKilledSnake(new_board_state, new_head_state,
                                  new_snake_state, curr_snake_index)

                # Index might have changed when snake is removed
                destination_snake_index, _, _, _, _ = findCurrentSnake(new_snake_state, destination_snake_index)

                # Same size case
                if (destination_snake_length == curr_snake_length):
                    removeKilledSnake(
                        new_board_state, new_head_state, new_snake_state, destination_snake_index)

                # updateSnakeHealth(
                #     new_snake_state, curr_snake_index, False, False)

        else:
            if (destination_cell == curr_snake_id):
                if (head_x == curr_snake_tail["x"] and head_y == curr_snake_tail["y"]):
                    # Snake moves forward and updates all coords in new game state
                    moveForward(new_board_state, new_head_state, new_snake_state,
                                curr_snake_id, curr_snake_index, curr_snake_body, head_x, head_y, True)

                    curr_health = updateSnakeHealth(
                        new_snake_state, curr_snake_index, True, False)

                    # check if our snake ran out of health
                    if (curr_health <= 0):
                        removeKilledSnake(
                            new_board_state, new_head_state, new_snake_state, curr_snake_index)
                else:
                    removeKilledSnake(new_board_state, new_head_state,
                                      new_snake_state, curr_snake_index)

            else:
                removeKilledSnake(new_board_state, new_head_state,
                                  new_snake_state, curr_snake_index)

        return new_game_state

    # Snake move to a cell with food
    elif destination_cell == BOARD_FOOD:

        # Check if theres a competitor for the targeted food and if that comptetitor is bigger or equal size
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for dir in directions:
            curr_x = head_x + dir[0]
            curr_y = head_y + dir[1]
            if (curr_x < 0 or curr_y < 0 or curr_x >= board_width or curr_y >= board_height or new_head_state[curr_y][
                curr_x] == curr_snake_id):
                continue
            if (new_head_state[curr_y][curr_x] not in [curr_snake_id, 0] and new_board_state[curr_y][curr_x] == 2):
                _, other_snake_length, _, _, _ = findCurrentSnake(
                    new_snake_state, new_head_state[curr_y][curr_x])
                if (other_snake_length >= curr_snake_length):
                    removeKilledSnake(new_board_state, new_head_state,
                                      new_snake_state, curr_snake_index)

                    return new_game_state

        # Snake moves forward and updates all coords in new game state
        moveForward(new_board_state, new_head_state, new_snake_state,
                    curr_snake_id, curr_snake_index, curr_snake_body, head_x, head_y, False)

        updateSnakeHealth(new_snake_state, curr_snake_index, True, True)

        # add a new body part
        snakeStateFoodGrow(new_snake_state, curr_snake_index)

        return new_game_state

    # Snake's regular movement to empty spaces
    else:

        # Snake moves forward and updates all coords in new game state
        moveForward(new_board_state, new_head_state, new_snake_state, curr_snake_id, curr_snake_index, curr_snake_body, head_x, head_y, False)

        curr_health = updateSnakeHealth(new_snake_state, curr_snake_index, True, False)

        # Check if snake ran out of health
        if curr_health <= 0:
            removeKilledSnake(new_board_state, new_head_state, new_snake_state, curr_snake_index)

        return new_game_state


# Calculate available space current game state snake has
def floodFill(game_state, curr_snake_head, curr_snake_body, curr_snake_tail):
    curr_snake_x = curr_snake_head["x"]
    curr_snake_y = curr_snake_head["y"]

    snake_tail_x = curr_snake_tail["x"]
    snake_tail_y = curr_snake_tail["y"]

    board_state = game_state["board"]["state_board"]
    board_width = len(board_state[0])
    board_height = len(board_state)
    visited = copy.deepcopy(board_state)

    have_eaten = False
    before_tail = curr_snake_body[-2]

    if (before_tail["x"] == snake_tail_x and before_tail["y"] == snake_tail_y):
        have_eaten = True

    for y in range(board_height):
        for x in range(board_width):
            if (board_state[y][x] in [BOARD_EMPTY, BOARD_FOOD]
                    or (y == snake_tail_y and x == snake_tail_x) and not have_eaten):
                visited[y][x] = False
            else:
                visited[y][x] = True

    visited[curr_snake_y][curr_snake_x] = False
    space, is_tail_reachable = fill(visited, board_width, board_height, curr_snake_x, curr_snake_y, snake_tail_x, snake_tail_y)

    if is_tail_reachable:
        return space, True

    return space - 1, False


# Recursive function of floodfill
def fill(visited, width, height, head_x, head_y, tail_x, tail_y):
    queue = deque([(head_x, head_y)])
    counter = 0
    is_tail_reachable = False

    while queue:
        head_x, head_y = queue.popleft()
        if (0 <= head_x < width and 0 <= head_y < height and not visited[head_y][head_x]):
            if (head_x == tail_x and head_y == tail_y):
                is_tail_reachable = True

            visited[head_y][head_x] = True
            counter += 1
            queue.extend([(head_x + 1, head_y), (head_x - 1, head_y), (head_x, head_y + 1), (head_x, head_y - 1)])

    return counter, is_tail_reachable


# Returns boolean depending on if snake state does not contain given id, snake is deleted when it is dead
def isGameOver(game_state, snake_id):
    if (snake_id is None):
        return False

    if (game_state is None):
        return True

    snake_state = game_state["snakes"]

    for snake in snake_state:
        if (snake["id"] == snake_id):
            return False
    return True


def isOnEdge(head_x, head_y, board_width, board_height):
    return (head_x == 0 or head_y == 0 or head_x == board_width - 1 or head_y == board_height - 1)


# Search through snake state and find curr snake, snakes on edge and average length
def snakeInfoLoop(game_state, curr_snake_id, board_width, board_height):
    curr_snake_head = None
    curr_snake_body = None
    curr_snake_size = 0
    curr_snake_health = 0
    curr_snake_tail = None
    other_edge_snakes = []

    biggest_size = 0

    for snake in game_state["snakes"]:
        head_x = snake["head"]["x"]
        head_y = snake["head"]["y"]

        if (snake["id"] == curr_snake_id):
            curr_snake_health = snake["health"]
            curr_snake_head = snake["head"]
            curr_snake_body = snake["body"]
            curr_snake_size = len(curr_snake_body)
            curr_snake_tail = snake["body"][-1]
        else:
            curr_size = len(snake["body"])
            biggest_size = max(curr_size, biggest_size)

            if isOnEdge(head_x, head_y, board_width, board_height):
                other_edge_snakes.append(snake)

    return curr_snake_head, curr_snake_body, curr_snake_tail, curr_snake_size, curr_snake_health, biggest_size, other_edge_snakes


def closestFoodDistanceManhattan(board_state, width, height, head_x, head_y):
    closest_food = float("inf")

    for y in range(height):
        for x in range(width):
            if board_state[y][x] == 1:
                food_distance = abs(
                    head_x - x) + abs(head_y - y)
                closest_food = min(food_distance, closest_food)

    return closest_food


# TODO Astar distance instead of Manhattan distance
def closestFoodDistanceAstar(board_state, width, height, head_x, head_y):  # TODO FIX THIS
    closest_food = float("inf")

    # convert board state to grid
    grid1 = np.zeros((height, width))
    foods = []
    for y in range(height):
        for x in range(width):
            if board_state[y][x] in [BOARD_EMPTY]:
                grid1[y, x] = 1

            if board_state[y][x] == BOARD_FOOD:
                foods.append((x, y))

    grid = AStarGrid(matrix=grid1)

    start_pos = grid.node(x=head_x, y=head_y)

    for food in foods:
        end_pos = grid.node(**food)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start_pos, end_pos, grid)

    for y in range(height):
        for x in range(width):
            if board_state[y][x] == 1:
                food_distance = abs(head_x - x) + abs(head_y - y)
                closest_food = min(food_distance, closest_food)

    return closest_food


# Determines if current head coordinates are on cells one before edge
def isOnEdgeBorder(head_x, head_y, board_height, board_width):
    return (head_x == 1 or head_y == 1 or head_x == board_width - 2 or head_y == board_height - 2)


# Determines if the provided cell is a safe_cell
def isSafeCell(board_state, x, y, safe_cells):
    return board_state[y][x] in safe_cells


# Prevents our snake from being in a position that i will get itself edge killed
def edgeKillDanger(board_state, board_width, board_height, head_x, head_y, main_snake_id):
    edge_kill_danger_weight = -400
    safe_cells = [0, 1, main_snake_id]

    if (not isOnEdge(head_x, head_y, board_height, board_width)):
        return 0

    if (head_x == 0):
        if (not isSafeCell(board_state, head_x + 1, head_y, safe_cells)):
            return edge_kill_danger_weight

    elif (head_x == board_width - 1):
        if (not isSafeCell(board_state, head_x - 1, head_y, safe_cells)):
            return edge_kill_danger_weight

    elif (head_y == 0):
        if (not isSafeCell(board_state, head_x, head_y + 1, safe_cells)):
            return edge_kill_danger_weight

    elif (head_y == board_height - 1):
        if (not isSafeCell(board_state, head_x, head_y - 1, safe_cells)):
            return edge_kill_danger_weight

    return 0


# Return edge kill value of current snake
def edgeKillValue(board_state, board_width, board_height, head_x, head_y, other_edge_snakes, main_snake_id):
    main_snake_edge_kill_weight = -5000
    other_snake_edge_kill_weight = 40

    if (isOnEdgeBorder(head_x, head_y, board_width, board_height)):
        for snake in other_edge_snakes:
            curr_edge_kill_weight = other_snake_edge_kill_weight
            edge_head_x = snake["head"]["x"]
            edge_head_y = snake["head"]["y"]

            # If the snake on the edge is our main snake
            if (snake["id"] == main_snake_id):
                curr_edge_kill_weight = main_snake_edge_kill_weight

            if ((head_x == 1 and edge_head_x == 0) or (head_x == board_width - 2 and edge_head_x == board_width - 1)):
                if (head_x == 1 and edge_head_x == 0):
                    if (snake["body"][1]["y"] < edge_head_y):
                        if (head_y > edge_head_y and board_state[edge_head_y][edge_head_x + 1] == main_snake_id):
                            return curr_edge_kill_weight

                    elif (snake["body"][1]["y"] > edge_head_y):
                        if (head_y < edge_head_y and board_state[edge_head_y][edge_head_x + 1] == main_snake_id):
                            return curr_edge_kill_weight
                elif ((head_x == board_width - 2 and edge_head_x == board_width - 1)):
                    if (snake["body"][1]["y"] < edge_head_y):
                        if (head_y > edge_head_y and board_state[edge_head_y][edge_head_x - 1] == main_snake_id):
                            return curr_edge_kill_weight

                    elif (snake["body"][1]["y"] > edge_head_y):
                        if (head_y < edge_head_y and board_state[edge_head_y][edge_head_x - 1] == main_snake_id):
                            return curr_edge_kill_weight

            elif ((head_y == 1 and edge_head_y == 0) or (
                    head_y == board_height - 2 and edge_head_y == board_height - 1)):
                if ((head_y == 1 and edge_head_y == 0)):
                    if (snake["body"][1]["x"] < edge_head_x):
                        if (head_x > edge_head_x and board_state[edge_head_y + 1][edge_head_x] == main_snake_id):
                            return curr_edge_kill_weight

                    elif (snake["body"][1]["x"] > edge_head_x):
                        if (head_x < edge_head_x and board_state[edge_head_y + 1][edge_head_x] == main_snake_id):
                            return curr_edge_kill_weight
                elif (head_y == board_height - 2 and edge_head_y == board_height - 1):
                    if (snake["body"][1]["x"] < edge_head_x):
                        if (head_x > edge_head_x and board_state[edge_head_y - 1][edge_head_x] == main_snake_id):
                            return curr_edge_kill_weight

                    elif (snake["body"][1]["x"] > edge_head_x):
                        if (head_x < edge_head_x and board_state[edge_head_y - 1][edge_head_x] == main_snake_id):
                            return curr_edge_kill_weight

    return 0


# Finds the closest smallest snake distance as well as returning head collision values
def headCollisionInfo(game_state, head_x, head_y, curr_snake_size, curr_snake_id, main_snake_id):
    smallest_snake_distance = float("inf")
    other_head_losing_weight = -10000
    main_head_losing_weight = float("inf")
    other_head_equal_weight = -10000

    curr_head_losing_weight = 0

    for snake in game_state["snakes"]:
        curr_head_x = snake["head"]["x"]
        curr_head_y = snake["head"]["y"]

        if (snake["id"] == curr_snake_id):
            continue

        if (len(snake["body"]) < curr_snake_size and (curr_snake_size - len(snake["body"])) >= 1):
            curr_snake_distance = abs(
                head_x - curr_head_x) + abs(head_y - curr_head_y)
            smallest_snake_distance = min(
                smallest_snake_distance, curr_snake_distance)

        # If current snake size is smaller or equal
        if ((abs(head_x - curr_head_x) + abs(head_y - curr_head_y) < 2)):

            # If current snake size is smaller
            if (len(snake["body"]) > curr_snake_size):
                curr_head_losing_weight = other_head_losing_weight

                # If current snake is going up against our main snake
                if (curr_snake_id != main_snake_id and snake["id"] == main_snake_id):
                    curr_head_losing_weight = main_head_losing_weight

            # If current snake size is equal
            elif (len(snake["body"]) == curr_snake_size):
                curr_head_losing_weight = other_head_equal_weight

            # elif (curr_snake_size - len(snake["body"]) < 2):
            #     curr_head_losing_weight = other_head_equal_weight

    return smallest_snake_distance, curr_head_losing_weight


def evaluateCurrentGameState(game_state, depth, main_snake_id, curr_snake_id, current_turn):
    curr_weight = 0

    opponent_death_weight = float("inf")
    available_space_weight = 0.5
    outer_bound_weight = -12
    center_control_weight = 5
    head_kill_weight = 0
    food_weight = 25
    snake_size_weight = 15
    more_turn_weight = 20

    LOW_HEALTH = 40
    low_health_penalty = -60
    DANGER_HEALTH = 20
    danger_health_penalty = -120

    # If the game state given somehow does not exist
    if game_state is None:
        if curr_snake_id == main_snake_id:
            return float("-inf")
        else:
            return float("inf")

    # Check if our main snake has died
    if isGameOver(game_state, main_snake_id):
        return float("-inf")

    # Check if the current snake has died (not main snake)
    if curr_snake_id != main_snake_id and isGameOver(game_state, curr_snake_id):
        return opponent_death_weight

    board_state = game_state["board"]["state_board"]
    board_width = len(board_state[0])
    board_height = len(board_state)

    # Find current snake as well as average snake size and snakes that are on the edge
    curr_snake_head, curr_snake_body, curr_snake_tail, curr_snake_size, curr_snake_health, biggest_size, other_edge_snakes = snakeInfoLoop(
        game_state, curr_snake_id, board_width, board_height)

    # Lowers score if snake's health is getting too low
    if (curr_snake_health < DANGER_HEALTH):
        curr_weight += danger_health_penalty
    elif (curr_snake_health < LOW_HEALTH):
        curr_weight += low_health_penalty

    if len(game_state["snakes"]) > 2:
        center_control_weight = 0

    # Add weight the bigger the snake is, currently + 15 for each growth
    curr_weight += curr_snake_size * snake_size_weight

    # FloodFill determines available space for current snake to move, add space weight
    available_space, is_tail_reachable = floodFill(game_state, curr_snake_head, curr_snake_body, curr_snake_tail)
    curr_weight += available_space * available_space_weight

    if (available_space < 2 and not is_tail_reachable):
        return -10000
    elif (available_space < curr_snake_size // 4 and not is_tail_reachable):
        return -800
    elif (available_space < curr_snake_size // 1.5 and not is_tail_reachable):
        return -400

    # Current snake head coordinates
    head_x = curr_snake_head["x"]
    head_y = curr_snake_head["y"]

    # Closest distance to food, add weight scaling depending on how close is curr snake to food
    closest_food_distance = closestFoodDistanceManhattan(board_state, board_width, board_height, head_x, head_y)
    curr_weight += food_weight / (closest_food_distance + 1)

    # Prevent us from being edge killed
    curr_weight += edgeKillDanger(board_state, board_width, board_height, head_x, head_y, main_snake_id)

    # Add weight if snake is on edge of board
    if isOnEdge(head_x, head_y, board_width, board_height):
        curr_weight += outer_bound_weight

    curr_weight += current_turn * more_turn_weight

    # curr_weight *= depth_discount_factor * depth
    if curr_snake_id == main_snake_id:
        return curr_weight
    else:
        return curr_weight * -1


def miniMax(game_state, depth, curr_snake_id, main_snake_id, previous_snake_id, alpha, beta, current_turn):

    # If given game_state reached an end or depth has reached zero, return game_state score
    if depth == 0 or isGameOver(game_state, previous_snake_id):
        return evaluateCurrentGameState(game_state, depth, main_snake_id, previous_snake_id, current_turn), None

    # get the id of the next snake that we're gonna minimax
    curr_index = 0
    for index, snake in enumerate(game_state["snakes"]):
        if snake["id"] == curr_snake_id:
            curr_index = index
            break

    # Select the next snake id inside the snake array
    next_snake_id = game_state["snakes"][(curr_index + 1) % len(game_state["snakes"])]["id"]

    moves = ["up", "down", "right", "left"]

    if curr_snake_id == main_snake_id:  # TODO max if itself or teammate 1. either run separate or handle both, but 1 can die so have to handle that
        ### max step
        highest_value = float("-inf")
        best_move = None

        for move in moves:
            # Current snake takes a possible move
            new_game_state = makeMove(game_state, curr_snake_id, move)

            curr_val, _ = miniMax(new_game_state, depth - 1, next_snake_id, main_snake_id, curr_snake_id, alpha, beta, current_turn + 1)

            if curr_val > highest_value:
                best_move = move
                highest_value = curr_val

            alpha = max(alpha, curr_val)

            if alpha >= beta:
                break

        return highest_value, best_move

    else:
        ### min step
        min_value = float("inf")
        best_move = None

        for move in moves:
            # Current snake takes a possible move
            new_game_state = makeMove(game_state, curr_snake_id, move)

            curr_val, _ = miniMax(new_game_state, depth - 1, next_snake_id, main_snake_id, curr_snake_id, alpha, beta, current_turn)

            if min_value > curr_val:
                best_move = move
                min_value = curr_val

            beta = min(curr_val, beta)

            if beta <= alpha:
                break

        return min_value, best_move


# Main function
def miniMax_value(game_state, safe_moves):
    current_game_state = createGameState(game_state, game_state["you"]["id"])
    current_turn = game_state["turn"]

    snakes_num = len(game_state["board"]["snakes"])

    if snakes_num == 4:
        depth = 5
    elif snakes_num == 3:
        depth = 5
    elif snakes_num == 2:
        depth = 5
    else:
        depth = 5

    result_value, best_move = miniMax(current_game_state, depth, game_state["you"]["id"], game_state["you"]["id"], None, float("-inf"), float("inf"), current_turn)
    print(f"Minimax value: {result_value}, Best move: {best_move}")

    if best_move is not None:
        if best_move in safe_moves:
            safe_moves[best_move] += result_value

    return best_move
