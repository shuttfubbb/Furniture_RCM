import gymnasium as gym
import numpy as np
import pygame

class RoomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, room_w=1000, room_d=500, render_mode=None):
        super().__init__()
        self.room_w = room_w
        self.room_d = room_d
        self.render_mode = render_mode

        # danh sách hình chữ nhật (x_center, y_center, w, h, color)
        self.rects = [
            (100, 150, 80, 40, (255, 0, 0)),
            (300, 250, 120, 60, (0, 0, 255)),
            (400, 100, 40, 100, (0, 255, 0)),
        ]

        # pygame init
        self.window = None
        self.clock = None

        # dummy space
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.array([0.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.array([0.0], dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.room_w, self.room_d))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.room_w, self.room_d))
        canvas.fill((255, 255, 255))  # nền trắng

        # vẽ các hình chữ nhật
        for (cx, cy, w, h, color) in self.rects:
            rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
            pygame.draw.rect(canvas, color, rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


env = RoomEnv(render_mode="human")
obs, _ = env.reset()
for _ in range(100):
    obs, r, done, trunc, info = env.step(0)
    env.render()
    input()
env.close()
