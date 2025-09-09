import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from config import *
from typing import Optional, Literal, List
from enum import Enum
import copy
import math

register(
    id='furniture-rcm-v0',
    entry_point='furniture_rcm:FurnitureRcmEnv',
)

"""
y
y
y
y
y
y
xxxxxxxxxxxxxxxxxxxx

Tọa độ của phòng tuân thủ theo đúng hệ trục Oxy bình thường
Các tình toán vị trí đồ đạc trong phòng sử dụng đơn vị minimet (mm)
Chiều dài (N) và chiều rộng (M) được sử dụng đơn vị minimet(mm)
Render sử dụng đơn vị pixel (px)
Quy đổi 1px = PIXEL2M mm

"""

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def as_array(self):
        return np.array([self.x, self.y])
    
    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

class Door:
    def __init__(self, W: int, D: int, x: int, y: int):
        self.W = W
        self.x = x
        self.y = y


class FurnitureType(Enum):
    CARBINET = 0
    TEACHER_TABLE = 1
    ANOTHER = 2


class Furniture:
    def __init__(
        self,
        code = 'DEFAULT',
        W = 0,
        D = 0,
        type = FurnitureType.ANOTHER,
        clearances: List[int] = [0, 0, 0, 0]  # Khoảng trống đặt đồ theo các hướng lần lượt right, up, left, down
    ):
        if clearances is None or len(clearances) != 4:
            raise ValueError("clearances must be a list of four integers representing right, up, left, down clearances.")
        self.code = code
        self.W = W
        self.D = D
        self.type = type
        self.normal_vec = np.array([1, 0])
        self.clearances = clearances  # right, up, left, down
        self.reset()
        """
                1
            W   1    Using
                1

                D           
        """

    @property
    def area(self) -> int:
        return self.W * self.D
    

    @property
    def accessibility_area(self) -> int:
        D_ac = self.p_ac2.x - self.p_ac1.x
        W_ac = self.p_ac1.y - self.p_ac2.as_array
        return W_ac * D_ac - self.area


    def set_position(self, x, y):
        self.center = Point(x, y)
        self.p1 = Point(x - self.W/2, y + self.D/2)  # top-left
        self.p2 = Point(x + self.W/2, y - self.D/2)  # bottom-right
        self.p_ac1 = Point(self.p1.x - self.clearances[2], self.p1.y + self.clearances[1])  # top-left after clearance   (point access area 1)
        self.p_ac2 = Point(self.p2.x + self.clearances[0], self.p2.y - self.clearances[3])  # bottom-right after clearance    (point access area 2)


    def reset(self):
        x, y = 0, 0
        self.set_position(x, y)
        self.normal_vec = np.array([1, 0])
    

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

        self.normal_vec = k2vector[k]
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

        access_points = [
            self.p_ac1.as_array(),
            self.p_ac2.as_array()
        ]

        # Dịch về gốc, xoay, dịch về lại tâm
        c = self.center.as_array()
        rotated = [R @ (p - c) + c for p in corners]
        rotated_access = [R @ (p - c) + c for p in access_points]

        # Lấy lại bounding box sau khi xoay
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        self.p1 = Point(min(xs), max(ys))  # top-left sau xoay
        self.p2 = Point(max(xs), min(ys))  # bottom-right sau xoay

        # Cập nhật lại các điểm access sau khi xoay
        self.p_ac1 = Point(rotated_access[0][0], rotated_access[0][1])
        self.p_ac2 = Point(rotated_access[1][0], rotated_access[1][1])


    def __repr__(self):
        return f"{self.code}    center{self.center}     p1{self.p1}     p2{self.p2}     normal_vec{self.normal_vec}"


class OccupancyMap:
    def __init__(self, N=N_MIN, M=M_MIN, g_size: Optional[int] = GRID_SIZE):
        """
            [0,0,0,0,1],
            [0,0,0,0,0],
        M-  [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
                |
                N
        """
        self.N = N
        self.M = M
        self.g_size = g_size
        self.n_sq = self.N / self.g_size
        self.m_sq = self.M / self.g_size
        self.grid = np.zeros((self.g_size, self.g_size), dtype=int)        


    def add_furniture(self, furniture: Furniture):
        # TODO: Test lại xem logic hàm này có đúng chưa
        x1 = round(furniture.p1.x / self.n_sq)
        y1 = round((self.M - furniture.p1.y) / self.m_sq)
        x2 = round(furniture.p2.x / self.n_sq)
        y2 = round((self.M - furniture.p2.y) / self.m_sq)
        self.grid[y1:y2+1, x1:x2+1] = 1


    def reset(self):
        self.grid = np.zeros((self.g_size, self.g_size), dtype=int)
    

class FurnitureRcmEnv(gym.Env):
    metadata = {'render_modes': ['human', 'terminal'], 'render_fps': 1}

    def __init__(self, furnitures: List[Furniture], door: Door,render_mode='terminal', N=N_MIN, M=M_MIN, g_size=GRID_SIZE):
        if not furnitures:
            raise ValueError("ERROR: The furnitures list is empty")
        self.N = N
        self.M = M
        self.occupancy_map = OccupancyMap(N, M, g_size)
        self.furnitures = furnitures
        self.furnitures.sort(key=lambda x: x.area, reverse=True)
        self.furnitures_inroom = []
        self.cur_index = 0
        self.g_size = g_size
        self.door = door
        
        # reward checkpoit
        self.total_violation_ratio = 0
        self.total_dot_nf_nw = 0


        # Config space
        max_room_sz = max(self.N, self.M)
        x_space = spaces.Box(low=0, high=max_room_sz, shape=(), dtype=float)
        y_space = spaces.Box(low=0, high=max_room_sz, shape=(), dtype=float)
        k_space = spaces.Discrete(4)
        type_space = spaces.Discrete(len(FurnitureType))
        W_space = spaces.Box(low=0, high=max_room_sz, shape=(), dtype=int)
        D_space = spaces.Box(low=0, high=max_room_sz, shape=(), dtype=int)
        object_space = spaces.Tuple((type_space, W_space, D_space))
        map_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.g_size, self.g_size),
            dtype=np.int32
        )
        self.action_space = spaces.Tuple((x_space, y_space, k_space))
        self.observation_space = spaces.Tuple((object_space, object_space, map_space))


    def is_out_of_bounds(self, furniture: Furniture):
        if (
            furniture.p1.x <= 0 or 
            furniture.p1.y >= self.M or 
            furniture.p2.x >= self.N or 
            furniture.p2.y <= 0):
            return True
        return False
    

    def is_overlapping(self, furniture: Furniture):
        for f in self.furnitures_inroom:
            if not(
                furniture.p_ac2.x   <=  f.p1.x or
                furniture.p_ac1.x   >=  f.p2.x or
                furniture.p_ac1.y   <=  f.p2.y or
                furniture.p_ac2.y   >=  f.p1.y
            ):
                return True
        return False


    def is_valid_position(self, furniture: Furniture):
        return not self.is_out_of_bounds(furniture) and not self.is_overlapping(furniture)


    def _get_observation(self):
        cur_furniture = copy.deepcopy(self.furnitures[self.cur_index])
        next_furniture = Furniture()
        if self.cur_index + 1 < len(self.furnitures):
            next_furniture = copy.deepcopy(self.furnitures[self.cur_index + 1])

        cur_furniture_info = (cur_furniture.type, cur_furniture.W, cur_furniture.D)
        next_furniture_info = (next_furniture.type, next_furniture.W, next_furniture.D)
        return cur_furniture_info, next_furniture_info, self.occupancy_map.grid


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.furnitures_lobby = copy.deepcopy(self.furnitures)
        self.furnitures_inroom = []
        self.occupancy_map.reset()
        self.cur_index = 0
        self.total_violation_ratio = 0
        return self._get_observation(), {}


    def step(self, x: int, y: int, k: Literal[0, 1, 2, 3]):
        reward = 0
        terminated = False
        cur_furniture = copy.deepcopy(self.furnitures[self.cur_index])
        cur_furniture.set_position(x, y)
        cur_furniture.rotate(k)

        if self.is_valid_position(cur_furniture):
            self.occupancy_map.add_furniture(cur_furniture)
            self.furnitures_inroom.append(cur_furniture) # Thêm ngay nội thất hiện tại để đủ nội thất trong phòng rồi mới tính reward
            reward = self.calculate_reward(cur_furniture)
            if self.cur_index == len(self.furnitures) - 1:
                terminated = True
            else:
                self.cur_index += 1
        else:
            reward = -PENALTY
            terminated = True
        return self._get_observation(), reward, terminated, False, {}


    def calculate_reward(self, cur_furniture: Furniture) -> float:
        r_a = self.accessibility_reward(cur_furniture)
        r_v = self.visibility_reward(cur_furniture)
        r_path = self.reachable_reward(cur_furniture)

        return 


    def accessibility_reward(self, cur_furniture: Furniture) -> float:
        violation_area = 0

        for f_inroom in self.furnitures_inroom:
            violation_area += self.violation_area(cur_furniture, f_inroom)
        violation_ratio = violation_area / cur_furniture.accessibility_area
        self.total_violation_ratio += violation_ratio
        F = len(self.furnitures_inroom)
        return 1 - 2 * self.total_violation_ratio / F


    def violation_area(f_current: Furniture, f_inroom: Furniture):
        """
            Hàm tính phần diện tích bị giao vào phần access của nội thất hiện tại
            đối với phần nội thất của nội thất trong phòng
        """
        w = min(f_current.p_ac2.x, f_inroom.p2.x) - max(f_current.p_ac1.x, f_inroom.p1.x)
        if w <= 0:
            return 0
        h = min(f_current.p_ac1.y, f_inroom.p1.y) - max(f_current.p_ac2.y, f_inroom.p2.y)
        if h <= 0:
            return 0
        return float(w * h)
    

    def visibility_reward(self, cur_furniture: Furniture):
        n_wall = self.find_wall_normal_vec(cur_furniture)
        n_f = cur_furniture.normal_vec
        dot_nf_nw = np.dot(n_f, n_wall)
        self.total_dot_nf_nw += dot_nf_nw
        F = len(self.furnitures_inroom)
        return - self.total_dot_nf_nw / F


    def find_wall_normal_vec(self, cur_furniture: Furniture):
        d_left   = cur_furniture.center.x
        d_right  = self.N - cur_furniture.center.x
        d_bottom = cur_furniture.center.y
        d_top    = self.M - cur_furniture.center.y

        n_wall = np.array([1, 0])
        d_min = min(d_left, d_right, d_bottom, d_top)
        if d_min == d_left:
            n_wall = np.array([1, 0])
        elif d_min == d_right:
            n_wall = np.array([-1, 0])
        elif d_min == d_bottom:
            n_wall = np.array([0, 1])
        else:
            n_wall = np.array([0, -1])
        return n_wall
    

    def reachable_reward(self, cur_furniture: Furniture):
        door_grid_x = math.floor(self.door.x / self.occupancy_map.n_sq)
        door_grid_y = math.floor(self.door.y / self.occupancy_map.m_sq)
        
        d_door = self.find_shortest_path(cur_furniture)
        d_delta = math.sqrt(self.N * self.N + self.M * self.M)
        if d_door != INF_POS:
            I_f = 1
            e_Kf = math.exp(- (d_door * d_door / d_delta * d_delta))
        else:
            I_f = 0
            e_Kf = 0


    
    def find_shortest_path(self, cur_furniture: Furniture):
        len_path = INF_POS

        return len_path



    
if __name__ == '__main__':
    grid = np.zeros((32, 32), dtype=int)