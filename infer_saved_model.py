import os
from pathlib import Path

import ray
import numpy as np
import pandas as pd

from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

from vine_env import VineEnv
from tune import TorchActionMaskModel  # reuse the exact same model class
from env_config import get_env_config


# =========================
# SETTINGS
# =========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "ray_results_vine",
    "vineyard_ppo",
    # change this to your actual trial folder
    "PPO_MultiAgentVineAsync_dfdd0_00000_0_2026-04-17_07-11-22",
    # change this to the checkpoint you want
    "checkpoint_000000",
)

EPISODES = 10
RENDER = False

# optional overrides for evaluation
ENV_OVERRIDES = {
    # example:
    # "num_humans": 5,
    # "max_steps": 240,
}


# =========================
# REGISTRATION
# =========================
def env_creator(env_config):
    return VineEnv(**env_config)


def find_real_checkpoint_file(checkpoint_dir: str) -> str:
    """
    RLlib may want the file inside the checkpoint dir, e.g.
    checkpoint_000050/checkpoint-50
    """
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    base = os.path.basename(checkpoint_dir.rstrip("/\\"))
    num = base.split("_")[-1].lstrip("0") or "0"
    candidate = os.path.join(checkpoint_dir, f"checkpoint-{num}")

    if os.path.isfile(candidate):
        return candidate

    # fallback: return the dir itself
    return checkpoint_dir


def run_inference():
    env_config = get_env_config(**ENV_OVERRIDES)
    env_config["render_mode"] = "human" if RENDER else None

    register_env("MultiAgentVineAsync", env_creator)
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

    checkpoint_to_load = find_real_checkpoint_file(CHECKPOINT_PATH)

    algo = Algorithm.from_checkpoint(checkpoint_to_load)

    rows = []

    for ep in range(EPISODES):
        env = VineEnv(**env_config)
        obs, infos = env.reset(seed=ep)

        terminated = {"__all__": False}
        truncated = {"__all__": False}
        last_infos = None
        reward_sum = 0.0

        while not terminated["__all__"] and not truncated["__all__"]:
            action_dict = {}

            for agent_id in env.agents:
                action = algo.compute_single_action(
                    observation=obs[agent_id],
                    policy_id="shared_policy",
                    explore=False,
                )

                if isinstance(action, tuple):
                    action = action[0]

                action_dict[agent_id] = int(action)

            obs, rewards, terminated, truncated, infos = env.step(action_dict)
            last_infos = infos
            reward_sum += float(np.mean(list(rewards.values())))

            if RENDER:
                env.render()

        summary = None
        for agent_id in last_infos:
            s = last_infos[agent_id].get("episode_summary", None)
            if s:
                summary = dict(s)
                break

        if summary is None:
            summary = {}

        summary["episode"] = ep
        summary["rollout_mean_reward"] = reward_sum
        rows.append(summary)

        env.close()

    algo.stop()

    df = pd.DataFrame(rows)

    out_dir = Path(PROJECT_DIR) / "inference_results"
    out_dir.mkdir(exist_ok=True)

    out_csv = out_dir / "ppo_inference.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== PPO inference ===")
    for col in df.columns:
        if col == "episode":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"{col}: mean={df[col].mean():.3f}, std={df[col].std(ddof=0):.3f}")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    run_inference()