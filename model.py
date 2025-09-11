import torch
import torch.nn as nn
from torch.nn import functional as F
import gymnasium as gym
import numpy as np
from config import *
from furniture_rcm_env import FurnitureType
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class OIDFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Tuple,
        features_dim: int = 256,
        num_types: int = FurnitureType.num_type,
        W_min: int = W_MIN,
        W_max: int = W_MAX,
        D_min: int = D_MIN,
        D_max: int = D_MAX,
    ):
        """
        observation_space: Tuple((object_t, object_next, occupancy_map))
          - object = (type, W, D)
          - map = Box(g_size, g_size)256
        features_dim: dimension của vector output cuối cùng cho policy/critic
        num_types: số lượng loại nội thất (để one-hot)
        W_min, W_max, D_min, D_max: giá trị min/max để chuẩn hoá W, D
        """
        super().__init__(observation_space, features_dim)

        self.num_types = num_types
        self.W_min, self.W_max = W_min, W_max
        self.D_min, self.D_max = D_min, D_max

        # ----- Encoder cho furniture object -----
        # input dim = 2 (W,D normalized) + num_types (type one-hot)
        self.obj_encoder = nn.Sequential(
            nn.Linear(2 + num_types, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU()
        )

        # ----- CNN cho occupancy map -----
        map_shape = observation_space.spaces[2].shape  # (g_size, g_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Tính output dim của CNN
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *map_shape)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.cnn_fc = nn.Linear(cnn_out_dim, 128)

        # ----- Projection cuối -----
        # concat: 128 (obj_t) + 128 (obj_next) + 128 (map) = 384
        self.final_fc = nn.Linear(384, features_dim)

    def normalize_WD(self, W, D):
        W_norm = (W - self.W_min) / (self.W_max - self.W_min + 1e-8)
        D_norm = (D - self.D_min) / (self.D_max - self.D_min + 1e-8)
        return W_norm, D_norm

    def encode_object(self, obj):
        """
        obj = (type, W, D) dạng tensor
        """
        type_idx = obj[:, 0].long()
        W = obj[:, 1].float()
        D = obj[:, 2].float()

        # Normalize W, D
        W_norm, D_norm = self.normalize_WD(W, D)

        # One-hot cho type
        type_onehot = F.one_hot(type_idx, num_classes=self.num_types).float()

        x = torch.cat([W_norm.unsqueeze(1), D_norm.unsqueeze(1), type_onehot], dim=1)
        return self.obj_encoder(x)

    def forward(self, obs):
        """
        obs là tuple: (object_t, object_next, map)
        - object_t: (batch, 3) [type, W, D]
        - object_next: (batch, 3)
        - map: (batch, g_size, g_size)
        """
        obj_t, obj_next, occ_map = obs

        f_t = self.encode_object(obj_t)
        f_next = self.encode_object(obj_next)

        # Occupancy map qua CNN
        occ_map = occ_map.unsqueeze(1).float()  # thêm channel dim
        f_occ = self.cnn(occ_map)
        f_occ = self.cnn_fc(f_occ)

        # concat lại
        h = torch.cat([f_t, f_next, f_occ], dim=1)
        return self.final_fc(h)