import os
from pathlib import Path
import numpy as np
import pandas as pd

from vine_env import VineEnv
from env_config import get_env_config

ACTION_HARVEST = 0
ACTION_TRANSPORT = 1
ACTION_ENQUEUE = 2
ACTION_REST = 3

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = Path(PROJECT_DIR) / "inference_results"
OUT_DIR.mkdir(exist_ok=True)

ENV_OVERRIDES = {}


def choose_action_always_manual(obs_dict):
    mask = obs_dict["action_mask"]
    if mask[ACTION_TRANSPORT] > 0.5:
        return ACTION_TRANSPORT
    if mask[ACTION_HARVEST] > 0.5:
        return ACTION_HARVEST
    if mask[ACTION_ENQUEUE] > 0.5:
        return ACTION_ENQUEUE
    return ACTION_REST


def choose_action_always_enqueue(obs_dict):
    mask = obs_dict["action_mask"]
    if mask[ACTION_ENQUEUE] > 0.5:
        return ACTION_ENQUEUE
    if mask[ACTION_HARVEST] > 0.5:
        return ACTION_HARVEST
    if mask[ACTION_TRANSPORT] > 0.5:
        return ACTION_TRANSPORT
    return ACTION_REST


def run_policy(policy_name, chooser, episodes=10):
    env_config = get_env_config(**ENV_OVERRIDES)
    rows = []

    for ep in range(episodes):
        env = VineEnv(**env_config)
        obs, infos = env.reset(seed=ep)

        terminated = {"__all__": False}
        truncated = {"__all__": False}
        last_infos = None
        reward_sum = 0.0

        while not terminated["__all__"] and not truncated["__all__"]:
            action_dict = {
                agent_id: chooser(obs[agent_id])
                for agent_id in env.agents
            }

            obs, rewards, terminated, truncated, infos = env.step(action_dict)
            last_infos = infos
            reward_sum += float(np.mean(list(rewards.values())))

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

    df = pd.DataFrame(rows)

    print(f"\n=== {policy_name} ===")
    for col in df.columns:
        if col == "episode":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"{col}: mean={df[col].mean():.3f}, std={df[col].std(ddof=0):.3f}")

    return df


if __name__ == "__main__":
    manual_df = run_policy("Always manual transport", choose_action_always_manual, episodes=10)
    enqueue_df = run_policy("Always enqueue when possible", choose_action_always_enqueue, episodes=10)

    manual_path = OUT_DIR / "baseline_manual.csv"
    enqueue_path = OUT_DIR / "baseline_enqueue.csv"

    manual_df.to_csv(manual_path, index=False)
    enqueue_df.to_csv(enqueue_path, index=False)

    print(f"\nSaved: {manual_path}")
    print(f"Saved: {enqueue_path}")