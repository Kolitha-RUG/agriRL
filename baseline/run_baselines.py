# run_baselines.py

import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from vine_env_multiagent import MultiAgentVineEnv
from heuristic_policies import POLICIES
from evaluation_utils import run_episode


# ==============================
# SETTINGS
# ==============================
NUM_EPISODES = 20
FATIGUE_THRESHOLD = 0.7

SELECTED_POLICIES = [
    "always_enqueue",
    "always_transport",
    "fatigue_threshold",
    "enqueue_when_high_fatigue",
]

LOG_DIR = "ray_results_vine/baselines"


def make_env():
    # If your environment needs config, add it here:
    # return MultiAgentVineEnv(env_config)
    return MultiAgentVineEnv()


def run_policy(name, policy_fn):
    print(f"\nRunning baseline: {name}")

    env = make_env()

    policy_log_dir = os.path.join(LOG_DIR, name)
    os.makedirs(policy_log_dir, exist_ok=True)

    writer = SummaryWriter(policy_log_dir)

    results = []

    for episode in range(NUM_EPISODES):
        summary = run_episode(env, policy_fn, FATIGUE_THRESHOLD)
        results.append(summary)

        print(f"Episode {episode + 1}/{NUM_EPISODES}: {summary}")

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(k, v, episode)

    writer.flush()
    writer.close()

    df = pd.DataFrame(results)

    csv_path = os.path.join(policy_log_dir, f"{name}_episodes.csv")
    df.to_csv(csv_path, index=False)

    print("\nMean results:")
    print(df.mean(numeric_only=True))

    return df


if __name__ == "__main__":
    all_means = []

    for name in SELECTED_POLICIES:
        df = run_policy(name, POLICIES[name])
        mean_row = df.mean(numeric_only=True).to_dict()
        mean_row["policy"] = name
        all_means.append(mean_row)

    summary_df = pd.DataFrame(all_means)
    os.makedirs(LOG_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(LOG_DIR, "baseline_summary.csv"), index=False)

    print("\nSaved combined summary to:")
    print(os.path.join(LOG_DIR, "baseline_summary.csv"))