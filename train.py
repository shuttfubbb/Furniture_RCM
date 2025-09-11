import os
import gymnasium as gym
import numpy as np
import torch
import json
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

from config import *
from model import OIDFeatureExtractor
from furniture_rcm_env import FurnitureRcmEnv, FurnitureType, Door, Furniture


class EvalCallbackWithVecNorm(EvalCallback):
    def __init__(self, vec_env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        result = super()._on_step()
        # Sau khi lưu best_model thì ta lưu luôn vec_normalize
        if self.best_model_save_path is not None and self.n_calls % self.eval_freq == 0:
            vecnorm_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")
            self.vec_env.save(vecnorm_path)
        return result



# Hàm tạo environment cho training
def make_env(rank, seed=0):
    with open(DATA_DIR, "r", encoding="utf-8") as f:
        data = json.load(f)

    room = data["room"]
    N = room["N"]
    M = room["M"]
    door_json = data["door"]
    door = Door(
        W=door_json["W"],
        x=door_json["x"],
        y=door_json["y"]
    )
    furnitures_json = room["furnitures"]
    furnitures = []
    for f_json in furnitures_json:
        f = Furniture(
            code=f_json["code"],
            W=f_json["W"],
            D=f_json["D"],
            type=FurnitureType(f_json["type"]),
            clearances=f_json["clearances"]
        )
        num = f_json["num"]
        for i in range(num):
            furnitures.append(f)

    def _init():
        env = FurnitureRcmEnv(
            furnitures=furnitures,
            door=door,
            render_mode='human',
            N=N,
            M=M,
            g_size=GRID_SIZE
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train():
    # Tạo vectorized training env
    env_fns = [make_env(i, SEED) for i in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Eval env (chỉ dùng 1 env để đánh giá)
    eval_env = DummyVecEnv([make_env(0, SEED + 1000)])
    eval_env = VecMonitor(eval_env)

    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=OIDFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            num_types=FurnitureType.type,
            W_min=W_MIN,
            W_max=W_MAX,
            D_min=D_MIN,
            D_max=D_MAX,
        ),
    )

    # PPO model
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOG_FOLDER, f"OID_PPO_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=ACTOR_LR,
        n_steps=STEPS_PER_UPDATE,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        tensorboard_log=log_dir
    )

    # Eval callback
    save_dir = os.path.join(MODEL_FOLDER, f"OID_PPO_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    eval_callback = EvalCallbackWithVecNorm(
        vec_env,
        eval_env=eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Train
    total_timesteps = TOTAL_EPOCHS * STEPS_PER_UPDATE * NUM_ENVS 
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)


if __name__ == "__main__":
    train()
