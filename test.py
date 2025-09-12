import os
import json
import datetime
import torch

from stable_baselines3 import PPO

from config import *
from model import OIDFeatureExtractor
from furniture_rcm_env import FurnitureRcmEnv, FurnitureType, Door, Furniture


def make_env_from_data(seed=0, render_mode="human"):
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
        num = f_json["num"]
        for i in range(num):
            furnitures.append(
                Furniture(
                    code=f_json["code"],
                    W=f_json["W"],
                    D=f_json["D"],
                    type=FurnitureType(f_json["type"]),
                    clearances=f_json["clearances"]
                )
            )

    env = FurnitureRcmEnv(
        furnitures=furnitures,
        door=door,
        render_mode=render_mode,
        N=N,
        M=M,
        g_size=GRID_SIZE
    )
    env.reset(seed=seed)
    return env


def test(model_path):
    # Tạo env render
    env = make_env_from_data(seed=SEED, render_mode="human")

    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        env.render()

    print("Episode reward:", total_reward)
    env.close()


if __name__ == "__main__":
    run_id = ""  # thay bằng run_id thật
    model_path = os.path.join(MODEL_FOLDER, f"OID_PPO_{run_id}", "best_model.zip")
    test(model_path)
