import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from config import *
from typing import Optional, Literal, List
from enum import Enum
import heapq
import copy
import math

register(
    id='furniture-rcm-v0',
    entry_point='furniture_rcm:FurnitureRcmEnv',
)

"""
0 ---------------------x
|
|
|
|
|
|
|
y

Tọa độ của phòng tuân thủ theo hệ trục Oxy đồ họa (hệ trục đồ họa) phù hợp cho GUI
Các tình toán vị trí đồ đạc trong phòng sử dụng đơn vị minimet (mm)
Chiều dài (N) và chiều rộng (M) được sử dụng đơn vị minimet(mm)
Render sử dụng đơn vị pixel (px)
Quy đổi 1px = PIXEL2M mm

"""

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def as_array(self):
        return np.array([self.x, self.y])
    
    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

class Door:
    def __init__(self, W, x, y):
        self.W = W
        self.x = x
        self.y = y


class classproperty(property):
    def __get__(self, obj, objtype=None):
        return self.fget(objtype)

class FurnitureType(Enum):
    OTHER = 0
    CARBINET = 1

    @classproperty
    def num_type(cls):
        return len(cls)


class Furniture:
    def __init__(
        self,
        code = 'DEFAULT',
        W = 0,
        D = 0,
        type = FurnitureType.OTHER,
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
    def area(self):
        return self.W * self.D
    

    @property
    def accessibility_area(self):
        D_ac = self.p_ac2.x - self.p_ac1.x
        W_ac = self.p_ac2.y - self.p_ac1.y
        return W_ac * D_ac - self.area


    @property
    def u_vector(self):
        if self.W < self.D:
            return self.normal_vec
        elif self.normal_vec[0] == 0:
            return np.array([1, 0])
        else:
            return np.array([0, 1])


    def set_position(self, x, y):
        self.center = Point(x, y)
        self.p1 = Point(x - self.W/2, y - self.D/2)  # top-left
        self.p2 = Point(x + self.W/2, y + self.D/2)  # bottom-right
        self.p_ac1 = Point(self.p1.x - self.clearances[2], self.p1.y - self.clearances[1])  # top-left after clearance   (point access area 1)
        self.p_ac2 = Point(self.p2.x + self.clearances[0], self.p2.y + self.clearances[3])  # bottom-right after clearance    (point access area 2)


    def reset(self):
        x, y = 0, 0
        self.set_position(x, y)
        self.normal_vec = np.array([1, 0])
    

    def rotate(self, k: Literal[0, 1, 2, 3] = 0):
        # k: số bước quay 90 độ theo chiều kim đồng hồ
        k2vector = {
            0: np.array([1, 0]),   # hướng phải
            1: np.array([0, 1]),   # hướng xuống
            2: np.array([-1, 0]),  # hướng trái
            3: np.array([0, -1])   # hướng lên
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

        # Ma trận quay trong hệ trục đồ họa
        R = np.array([
            [np.cos(theta),  np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        corners = [
            self.p1.as_array,                                # top-left
            np.array([self.p1.x, self.p2.y]),                  # bottom-left
            self.p2.as_array,                                # bottom-right
            np.array([self.p2.x, self.p1.y])                   # top-right
        ]

        access_points = [
            self.p_ac1.as_array,
            self.p_ac2.as_array
        ]

        # Dịch về gốc, xoay, dịch về lại tâm
        c = self.center.as_array
        rotated = [R @ (p - c) + c for p in corners]
        rotated_access = [R @ (p - c) + c for p in access_points]

        # Lấy lại bounding box sau khi xoay
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        self.p1 = Point(min(xs), min(ys))  # top-left sau xoay
        self.p2 = Point(max(xs), max(ys))  # bottom-right sau xoay

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


    def mapping_to_grid(self, furniture: Furniture):
        # TODO: Test lại xem logic hàm này có đúng chưa
        i1 = round(furniture.p1.y / self.m_sq)
        j1 = round(furniture.p1.x / self.n_sq)
        i2 = round(furniture.p2.y / self.m_sq)
        j2 = round(furniture.p2.x / self.n_sq)
        return i1, j1, i2, j2
        

    def add_furniture(self, furniture: Furniture):
        i1, j1, i2, j2 = self.mapping_to_grid(furniture)
        self.grid[i1:i2+1, j1:j2+1] = 1


    def reset(self):
        self.grid = np.zeros((self.g_size, self.g_size), dtype=int)
    

class FurnitureRcmEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, furnitures: List[Furniture], door: Door, render_mode='human', N=N_MIN, M=M_MIN, g_size=GRID_SIZE):
        self.render_mode = render_mode
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
        self.total_reachable = 0
        self.total_furtures_area = 0
        self.total_areaXcenter = 0
        self.total_scatter_matrix = 0
        self.total_furtures_area_C = 0
        self.total_cal_angle_C = 0

        # Config space
        max_room_sz = max(self.N, self.M)

        type_space = spaces.Discrete(len(FurnitureType))
        W_space = spaces.Box(low=0, high=W_MAX, shape=(), dtype=int)
        D_space = spaces.Box(low=0, high=D_MIN, shape=(), dtype=int)
        object_space = spaces.Tuple((type_space, W_space, D_space))
        map_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.g_size, self.g_size),
            dtype=np.int32
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Tuple((object_space, object_space, map_space))


    def is_out_of_room(self, furniture: Furniture):
        if (
            furniture.p1.x <= 0 or 
            furniture.p1.y <= 0 or 
            furniture.p2.x >= self.N or 
            furniture.p2.y >= self.M
        ):
            return True
        return False
    

    def is_overlapping(self, furniture: Furniture):
        for f in self.furnitures_inroom:
            if not(
                furniture.p2.x <= f.p1.x or  # A bên trái B
                furniture.p1.x >= f.p2.x or  # A bên phải B
                furniture.p2.y <= f.p1.y or  # A ở trên B
                furniture.p1.y >= f.p2.y     # A ở dưới B
            ):
                return True
        return False


    def is_valid_position(self, furniture: Furniture):
        return not self.is_out_of_room(furniture) and not self.is_overlapping(furniture)


    def _get_observation(self):
        cur_furniture = copy.deepcopy(self.furnitures[self.cur_index])
        next_furniture = Furniture()
        if self.cur_index + 1 < len(self.furnitures):
            next_furniture = copy.deepcopy(self.furnitures[self.cur_index + 1])

        cur_furniture_info = (cur_furniture.type.value, cur_furniture.W, cur_furniture.D)
        next_furniture_info = (next_furniture.type.value, next_furniture.W, next_furniture.D)
        return cur_furniture_info, next_furniture_info, self.occupancy_map.grid


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.furnitures_lobby = copy.deepcopy(self.furnitures)
        self.furnitures_inroom = []
        self.occupancy_map.reset()
        self.cur_index = 0

        self.total_violation_ratio = 0
        self.total_dot_nf_nw = 0
        self.total_reachable = 0
        self.total_furtures_area = 0
        self.total_areaXcenter = 0
        self.total_scatter_matrix = 0
        self.total_furtures_area_C = 0
        self.total_cal_angle_C = 0
        return self._get_observation(), {}


    def step(self, action):
        x = action[0] * self.N
        y = action[1] * self.M
        k = int(math.floor(np.clip(action[2], 0.0, 0.9999999) / 0.25))

        reward = 0
        terminated = False
        cur_furniture = copy.deepcopy(self.furnitures[self.cur_index])
        cur_furniture.set_position(x, y)
        cur_furniture.rotate(k)

        if self.is_valid_position(cur_furniture):
            self.occupancy_map.add_furniture(cur_furniture)
            # TODO: CHeck lại logic phần này
            reward = self.calculate_reward(cur_furniture)
            self.furnitures_inroom.append(cur_furniture)
            # Cho tới đây
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
        r_b = self.balance_reward(cur_furniture)

        if cur_furniture.type == FurnitureType.CARBINET:
            r_al = self.alignment_reward(cur_furniture)
            return (r_a + r_v + r_path + r_b + r_al) / 5 
        return (r_a + r_v + r_path + r_b) / 4 


    def accessibility_reward(self, cur_furniture: Furniture) -> float:
        violation_area = 0

        for f_inroom in self.furnitures_inroom:
            violation_area += self.violation_area(cur_furniture, f_inroom)
        violation_ratio = violation_area / cur_furniture.accessibility_area
        self.total_violation_ratio += violation_ratio
        F = len(self.furnitures_inroom) + 1 
        return 1 - 2 * self.total_violation_ratio / F


    def violation_area(f_current: Furniture, f_inroom: Furniture):
        """
            Hàm tính phần diện tích bị giao vào phần access của nội thất hiện tại
            đối với phần nội thất của nội thất trong phòng
        """
        w = min(f_current.p_ac2.x, f_inroom.p2.x) - max(f_current.p_ac1.x, f_inroom.p1.x)
        if w <= 0:
            return 0
        h = min(f_current.p_ac2.y, f_inroom.p2.y) - max(f_current.p_ac1.y, f_inroom.p1.y)
        if h <= 0:
            return 0
        return w * h
    

    def visibility_reward(self, cur_furniture: Furniture):
        n_wall, _ = self.near_wall_info(cur_furniture)
        n_f = cur_furniture.normal_vec
        dot_nf_nw = np.dot(n_f, n_wall)
        self.total_dot_nf_nw += dot_nf_nw
        F = len(self.furnitures_inroom) + 1
        return - self.total_dot_nf_nw / F


    def near_wall_info(self, cur_furniture: Furniture):
        d_left   = cur_furniture.center.x
        d_right  = self.N - cur_furniture.center.x
        d_bottom = self.M - cur_furniture.center.y
        d_top    = cur_furniture.center.y

        n_wall = np.array([1, 0])
        d_min = min(d_left, d_right, d_bottom, d_top)
        if d_min == d_left:
            n_wall = np.array([1, 0])
        elif d_min == d_right:
            n_wall = np.array([-1, 0])
        elif d_min == d_bottom:
            n_wall = np.array([0, -1])
        else:
            n_wall = np.array([0, 1])
        return n_wall, d_min
    

    def reachable_reward(self, cur_furniture: Furniture):
        d_door, I_f = self.find_shortest_path(cur_furniture)
        d_delta = math.sqrt(self.N**2 + self.M**2)
        if I_f == 1:
            e_Kf = math.exp(- ((d_door  / d_delta )** 2))
        else:
            e_Kf = 0
        self.total_reachable += ((1 - I_f) + e_Kf * I_f)
        F = len(self.furnitures_inroom) + 1
        return 1 - 2 * self.total_reachable / F

    
    def find_shortest_path(self, cur_furniture: Furniture):
        """
            A* algorithm
        """
        is_finded = 0
        door_i = math.floor(self.door.y / self.occupancy_map.m_sq)
        door_j = math.floor(self.door.x / self.occupancy_map.n_sq)

        if self.occupancy_map.grid[door_i][door_j] == 1:
            return INF_POS_NUM, is_finded

        fur_i = math.floor(cur_furniture.center.y / self.occupancy_map.m_sq)
        fur_j = math.floor(cur_furniture.center.x / self.occupancy_map.n_sq)

        fur_i1, fur_j1, fur_i2, fur_j2 = self.occupancy_map.mapping_to_grid(cur_furniture)

        start = (door_i , door_j)
        goal  = (fur_i  , fur_j )
        rows, cols = GRID_SIZE, GRID_SIZE
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), 0, start, [start])) # (f, g, node, path)
        visited = set()

        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            if current[0] >= fur_i1 and current[0] <= fur_i2 and current[1] >= fur_j1 and current[1] <= fur_j2:
                is_finded = 1
                return g, is_finded
            
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = current[0] + di, current[1] + dj
                if 0 <= ni < rows and 0 <= nj < cols and self.occupancy_map.grid[ni][nj] == 0:
                    neighbor = (ni, nj)
                    if neighbor not in visited:
                        new_g = g + 1
                        new_f = new_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))
        return INF_POS_NUM, is_finded


    def heuristic(self, a, b):
        """
            Manhattan
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def balance_reward(self, cur_furniture: Furniture):
        """
            Cân bằng các nội thất trong 1 căn phòng, chia đều nội thất xung quanh phòng
        """
        xf = cur_furniture.center.as_array
        self.total_furtures_area += cur_furniture.area
        self.total_areaXcenter += (cur_furniture * xf)
        xF_bar = self.total_areaXcenter / self.total_furtures_area

        self.total_scatter_matrix += cur_furniture * np.outer(xf - xF_bar, xf - xF_bar)
        Sigma_F = self.total_scatter_matrix / self.total_furtures_area
        sigma_E_sq = (self.N**2 + self.M**2) / 12
        I_matrix = np.eye(2)
        d_delta = math.sqrt(self.N**2 + self.M**2)

        o = np.array([self.N/2, self.M/2])
        term1 = np.exp(- np.sum((xF_bar - o)**2) / (d_delta ** 2))
        term2 = np.exp(- (np.linalg.norm(Sigma_F - sigma_E_sq * I_matrix, 'fro') ** 2) / (sigma_E_sq ** 2))

        return term1 + term2 - 1
    
    
    def alignment_reward(self, cur_furniture: Furniture):
        n_wall, d_fw = self.near_wall_info(cur_furniture)
        
        u_f = cur_furniture.u_vector
        if n_wall[0] == 0:
            u_wall = np.array([1, 0])
        else:
            u_wall = np.array([0, 1])
        
        dot_val = abs(np.dot(u_f, u_wall))
        theta = np.arccos(np.clip(dot_val, -1.0, 1.0))

        omega_f = d_fw / max(cur_furniture.W, cur_furniture.D)
        term = (np.cos(2*theta) ** 2) * (1 - np.tanh(omega_f) ** 2)
        
        self.total_cal_angle_C += cur_furniture.area * term
        self.total_furtures_area_C += cur_furniture.area

        return self.total_cal_angle_C / self.total_furtures_area_C 

    
if __name__ == '__main__':
    print(FurnitureType.num_type)
    # a = 0.9999999
    # print(math.floor(a / 0.25))
    # print(np.round(a * 3))
    """
    0.16  = 0 = 0.16
    0.17 - 0.49 = 1 = 0.33
    0.5 - 0.83 = 2 = 0.33
    0.84 - 1 = 3 = 0.16

    0 - 0.25 = 0
    0.25 - 0.5 = 1
    0.5 - 0.75 = 2
    0.75 - 1 = 3
     """