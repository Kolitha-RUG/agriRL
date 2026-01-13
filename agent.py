import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.utils import get_device
from vine_env import VineEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold

import os

model_path = os.path.join("models", "ppo_vine_env")
log_path = os.path.join("logs", "ppo_vine_env")

if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists("logs"):   
    os.makedirs("logs")

def check_environment():
    env = gym.make('VineEnv-v0', render_mode="terminal")
    env_checker.check_env(env, warn=True, skip_render_check=False)
    print(f"Using device: {get_device()}")


def train():

    env = make_vec_env(VineEnv, n_envs=12, vec_env_cls=SubprocVecEnv)

    # Increase ent_coef to encourage exploration, this resulted in a better solution.
    model = PPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_path, ent_coef=0.05)

    eval_callback = EvalCallback(
        env,
        eval_freq=10_000,
        # callback_on_new_best = StopTrainingOnRewardThreshold(reward_threshold=???, verbose=1)
        # callback_after_eval  = StopTrainingOnNoModelImprovement(max_no_improvement_evals=???, min_evals=???, verbose=1)
        verbose=1,
        best_model_save_path=os.path.join(model_path, 'PPO'),
    )

    """
    total_timesteps: pass in a very large number to train (almost) indefinitely.
    callback: pass in reference to a callback fuction above
    """
    model.learn(total_timesteps=int(1e10), callback=eval_callback)

def test_agent():
    env = gym.make('VineEnv-v0', render_mode="terminal")

    model = PPO(policy = "MlpPolicy", env=env, verbose=1,device ='cpu', tensorboard_log=log_path)
    model.learn(total_timesteps=100000)

    model.save(model_path)

    env.close()

if __name__ == "__main__":
    # check_environment()
    # test_agent()
    train()