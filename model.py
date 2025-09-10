import gymnasium as gym
import torch.nn as nn
from stable_baselines3.ppo import PPO

from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from stable_baselines3.common.policies import ActorCriticPolicy
# class OID_PPO(MultiInputActorCriticPolicy):