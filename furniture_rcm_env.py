import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from config import *
from typing import Optional, Literal, List
from enum import Enum


register(
    id='furniture-rcm-v0',
    entry_point='furniture_rcm:FurnitureRcmEnv',
)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def as_array(self):
        return np.array([self.x, self.y])
    
    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"


class Furniture:
    def __init__(
        self,
        code = 'DEFAUL',
        W = 400,
        D = 400,
        x = None,
        y = None
    ):
        self.code = code
        self.W = W
        self.D = D
        self.center = Point(x, y)
        self.p1 = Point(x - W/2, y + D/2)  # top-left
        self.p2 = Point(x + W/2, y - D/2)  # bottom-right
        self.front_direction = np.array([1, 0])
        """
                1
            W   1    Using
                1

                D           
        """

    def reset(self):
        self.p1 = Point(self.center.x - self.W/2, self.center.y + self.D/2)
        self.p2 = Point(self.center.x + self.W/2, self.center.y - self.D/2)
        self.front_direction = np.array([1, 0])
    
    def rotate(self, k: Literal[0, 1, 2, 3] = 0):
        k2vector = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        k2deg = {
            0: 0,
            1: 90,
            2: 180,
            3: 270
        }

        self.front_direction = k2vector[k]
        angle_deg = k2deg[k]
        theta = np.deg2rad(angle_deg)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        corners = [
            self.p1.as_array(),                                # top-left
            np.array([self.p1.x, self.p2.y]),                  # bottom-left
            self.p2.as_array(),                                # bottom-right
            np.array([self.p2.x, self.p1.y])                   # top-right
        ]

        # Dịch về gốc, xoay, dịch về lại tâm
        c = self.center.as_array()
        rotated = [R @ (p - c) + c for p in corners]

        # Lấy lại bounding box sau khi xoay
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        self.p1 = Point(min(xs), max(ys))  # top-left sau xoay
        self.p2 = Point(max(xs), min(ys))  # bottom-right sau xoay

    def __repr__(self):
        return f"{self.code}    center{self.center}     p1{self.p1}     p2{self.p2}     front_direction{self.front_direction}"


class OccupancyMap:
    def __init__(self, door, N=N_MIN, M=M_MIN, g_size: Optional[int] = GRID_SIZE):
        """
            [0,0,0,0,1],
            [0,0,0,0,0],
        M-  [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
                |
                N
        """
        self.door = self.door
        self.N = N
        self.M = M
        self.g_size = g_size
        self.n_sq = self.N / self.g_size
        self.m_sq = self.M / self.g_size
        self.grid = np.zeros((self.g_size, self.g_size), dtype=int)        

    def add_furniture(self, grid_i, grid_j, y, n, m):
        for i in range(m):
            for j in range(n):
                grid[grid_i + i][grid_j + j] = 1
    
    def reset(self):
        self.grid = np.zeros((self.g_size, self.g_size), dtype=int)
    

class FurnitureRcmEnv(gym.Env):
    metadata = {'render_modes': ['human', 'terminal'], 'render_fps': 1}

    def __init__(self, render_mode='terminal', N=N_MIN, M=M_MIN, furnitures: List[Furniture]= [], g_size=GRID_SIZE):
        self.N = N
        self.M = M
        self.furnitures_lobby = furnitures
        self.furnitures_inroom = []
        self.g_size = g_size

        max_room_sz = max(self.N, self.M)
        x_space = spaces.Box(low=0, high=max_room_sz, shape=(), dtype=float)
        y_space = spaces.Box(low=0, high=max_room_sz, shape=(), dtype=float)
        k_space = spaces.Discrete(4)
        W_space = spaces.Box(low=0, high=max_room_sz, shape=(), dtype=int)
        D_space = spaces.Box(low=0, high=max_room_sz, shape=(), dtype=int)

        object_space = spaces.Tuple((x_space, y_space, W_space, D_space, k_space))
        map_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.g_size, self.g_size),
            dtype=np.int32
        )

        self.action_space = spaces.Tuple((x_space, y_space, k_space))
        self.observation_space = spaces.Tuple((object_space, object_space, map_space))


    def _get_observation(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self):
        pass

    def _calculate_reward(self):
        pass

    

        



    
if __name__ == '__main__':
    grid = np.zeros((32, 32), dtype=int)