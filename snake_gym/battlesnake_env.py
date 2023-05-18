import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from enum import IntEnum
from simu_model.simugame import SimuGame
from basic_model.direction import Direction
from tools.mapper import simu_to_gamestate
from service.snake_controller import SnakeController, ControllerMode
from simu_model.guiboard import GuiBoard
import time

class Actions(IntEnum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    NONE = -1

    # get the enum name without the class
    def __str__(self): return self.name


class BattleSnake(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, **kwargs):
        super().__init__()
        # dimensions of the grid
        self.width = kwargs.get('width', 11)
        self.height = kwargs.get('height', 11)

        # define the maximum x and y values
        self.max_x = self.width - 1
        self.max_y = self.height - 1

        # there are 4 possible actions: move 0:up, 1:left, 2:down, 3:right -1:None
        self.action_space = Discrete(5)

        # the observation will be the coordinates of Baby Robot
        self.observation_space = MultiDiscrete([self.width, self.height])

        # Snake's position in the grid
        # self.x = np.random.randint(0, self.width)
        # self.y = np.random.randint(0, self.height)

        self.alive = True
        self.simu_game = SimuGame(11, 1)
        self.x, self.y = self.simu_game.board.snake_list[0].body_list[0]
        print(self.x, self.y)

    # def _snake_start_pos(self):
    #     return np.random.randint(0, self.width), np.random.randint(0, self.height)

    # def step(self, action):
    #     obs = np.array([self.x, self.y])
    #     reward = -1
    #     terminated = True
    #     truncated = False
    #     info = {}
    #     return obs, reward, terminated, truncated, info
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset Baby Robot's position in the grid
        self.x, self.y = self.x, self.y = self.simu_game.board.snake_list[0].body_list[0]
        info = {}
        return np.array([self.x, self.y]), info
    #
    # def render(self, **kwargs):
    #     pass

    def take_action(self, action):
        ''' apply the supplied action '''

        # move in the direction of the specified action
        if action == Actions.UP:
            self.y += 1
        elif action == Actions.DOWN:
            self.y -= 1
        elif action == Actions.LEFT:
            self.x -= 1
        elif action == Actions.RIGHT:
            self.x += 1
        else:
            pass

        # make sure the move stays on the grid
        if self.x < 0 or self.y < 0 or self.x > self.max_x or self.y > self.max_y:
            self.alive=False

    def step(self, action):

        # take the action and update the position
        # print(f"in step: {self.x} , {self.y}")
        # self.take_action(action)

        controllers = []
        for _ in self.simu_game.board.snake_list:
            controllers.append(SnakeController(ControllerMode.random))

        desired_directions = []
        for i in range(len(self.simu_game.board.snake_list)):
            if not self.simu_game.board.snake_list[i].alive():
                desired_directions.append(Direction.NONE)

                continue
            gamestate = simu_to_gamestate(self.simu_game.board, i)
            move = controllers[i].move(gamestate)
            desired_directions.append(move)

            self.simu_game.board.desired_directions = desired_directions
            self.simu_game.board.step()
            # print("LENGTH: %d" % self.board.snake_list[0].length)
            print(f"DIRECTION: {self.simu_game.board.desired_directions}")# : {Actions(self.simu_game.board.desired_directions[0])}")
            # print("BODY LIST: {}".format(self.board.snake_list[0].body_list))
            # print("HEALTH: %d" % self.board.snake_list[0].health)
            # print(self.board)
            # self.board.render()
            # time.sleep(0.05)


        # print("OBS :", self.board.board_map())
        obs = np.array([self.x, self.y])

        # set the 'terminated' flag if we've reached the exit
        terminated = not self.simu_game.board.snake_list[0].alive()# True if self.alive is False else False
        truncated = False

        # get -1 reward for dying
        # - except at the terminal state which has zero reward
        reward = -1 if terminated else 1

        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        # print(kwargs)
        # print(f"Snake at x:{self.x}, y:{self.y}")
        print(self.simu_game.board)
        for i in range(len(self.simu_game.board.snake_list)):
            print(f"snake on board is {self.simu_game.board.snake_list[i].body_list}")
            self.simu_game.board.render()
            time.sleep(1)

        try:
            print(f"{Actions(kwargs['action']): <5}: ({self.x},{self.y}) reward = {kwargs['reward']}")
            # self.simu_game.update(kwargs['action'])
        except:
            pass








