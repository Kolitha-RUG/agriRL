import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.utils import get_device
from vine_env import VineEnv

import os

def check_environment():
    env = gym.make('VineEnv-v0', render_mode="terminal")
    env_checker.check_env(env, warn=True, skip_render_check=False)
    print(f"Using device: {get_device()}")

def train_agent():
    model_path = os.path.join("models", "ppo_vine_env")
    log_path = os.path.join("logs", "ppo_vine_env")

    if not os.path.exists("models"):
        os.makedirs("models")

    env = gym.make('VineEnv-v0', render_mode="terminal")

    model = PPO(policy = "MlpPolicy", env=env, verbose=1,device ='cpu', tensorboard_log=log_path)
    model.learn(total_timesteps=100000)

    model.save(model_path)

    env.close()

if __name__ == "__main__":
    # check_environment()
    train_agent()