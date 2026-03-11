# evaluate_rl.py

import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from vine_env_multiagent import MultiAgentVineEnv
from evaluation_utils import run_episode

import ray
from ray.rllib.algorithms.ppo import PPO


# ==============================
# SETTINGS
# ==============================

CHECKPOINT_PATH = "ray_results_vine/checkpoint_path_here"

NUM_EPISODES = 20

LOG_DIR = "ray_results_vine/ppo_eval"


# ==============================


def run_rl():

    ray.init(ignore_reinit_error=True)

    algo = PPO.from_checkpoint(CHECKPOINT_PATH)

    env = MultiAgentVineEnv()

    writer = SummaryWriter(LOG_DIR)

    results = []

    for episode in range(NUM_EPISODES):

        obs, _ = env.reset()

        done = False

        while not done:

            actions = {}

            for agent_id, agent_obs in obs.items():

                actions[agent_id] = algo.compute_single_action(
                    agent_obs,
                    policy_id="human"
                )

            obs, rewards, terminated, truncated, infos = env.step(actions)

            done = terminated["__all__"] or truncated["__all__"]

        summary = infos["__common__"]["episode_summary"]

        results.append(summary)

        for k, v in summary.items():
            writer.add_scalar(k, v, episode)

    df = pd.DataFrame(results)

    df.to_csv("ppo_eval.csv", index=False)

    print(df.mean())


if __name__ == "__main__":

    run_rl()