# compare_two_saved_models.py

import os
from pathlib import Path

import ray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

from vine_env import VineEnv
from tune import TorchActionMaskModel
from env_config import get_env_config


# =========================
# SETTINGS
# =========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = Path(PROJECT_DIR) / "inference_results" / "model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = 20
RENDER = False

# Use exactly the same environment for both models
ENV_OVERRIDES = {
    # keep empty unless you want to force same test setting
    # "num_humans": 5,
    # "num_drones": 1,
    # "max_steps": 240,
}

# Change these two paths
CHECKPOINTS = {
    "Old reward": os.path.join(
        PROJECT_DIR,
        "ray_results_vine",
        "vineyard_ppo",
        "Reward_DF",
        "checkpoint_000000",
    ),

    "Simple reward": os.path.join(
        PROJECT_DIR,
        "ray_results_vine",
        "vineyard_ppo",
        "Reward_DFQ",
        "checkpoint_000000",
    ),
}

# Compare KPIs, not reward value
METRICS = [
    "kpi_delivered_total",
    "kpi_throughput_per_100_steps",
    "kpi_mean_fatigue",
    "episode_fatigue_increase_total",
    "kpi_rest_ratio",
    "kpi_mean_backlog",
    "kpi_peak_backlog",
    "kpi_completion_pct",
    "kpi_human_utilization",
    "kpi_drone_utilization",
]


# =========================
# HELPERS
# =========================
def env_creator(env_config):
    return VineEnv(**env_config)


def find_real_checkpoint_file(checkpoint_dir: str) -> str:
    """
    RLlib sometimes expects the internal checkpoint file:
    checkpoint_000020/checkpoint-20
    """
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    base = os.path.basename(checkpoint_dir.rstrip("/\\"))
    num = base.split("_")[-1].lstrip("0") or "0"
    candidate = os.path.join(checkpoint_dir, f"checkpoint-{num}")

    if os.path.isfile(candidate):
        return candidate

    return checkpoint_dir


def run_model_inference(model_name: str, checkpoint_path: str, episodes: int = 20) -> pd.DataFrame:
    env_config = get_env_config(**ENV_OVERRIDES)
    env_config["render_mode"] = "human" if RENDER else None

    checkpoint_to_load = find_real_checkpoint_file(checkpoint_path)
    algo = Algorithm.from_checkpoint(checkpoint_to_load)

    rows = []

    for ep in range(episodes):
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

            # Do not use this for comparing old vs new reward.
            # It is only stored for completeness.
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

        summary["model"] = model_name
        summary["episode"] = ep
        summary["rollout_mean_reward"] = reward_sum
        rows.append(summary)

        env.close()

    algo.stop()

    df = pd.DataFrame(rows)
    return df


def make_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, g in all_df.groupby("model"):
        for metric in METRICS:
            if metric not in g.columns:
                continue

            rows.append({
                "model": model_name,
                "metric": metric,
                "mean": g[metric].mean(),
                "std": g[metric].std(ddof=0),
                "min": g[metric].min(),
                "max": g[metric].max(),
            })

    return pd.DataFrame(rows)


def make_plots(summary_df: pd.DataFrame):
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    for metric in METRICS:
        sub = summary_df[summary_df["metric"] == metric]

        if sub.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.bar(sub["model"], sub["mean"], yerr=sub["std"], capsize=4)
        plt.title(metric)
        plt.ylabel("value")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}.png", dpi=200)
        plt.close()


def make_episode_plots(all_df: pd.DataFrame):
    plot_dir = OUT_DIR / "episode_plots"
    plot_dir.mkdir(exist_ok=True)

    for metric in METRICS:
        if metric not in all_df.columns:
            continue

        plt.figure(figsize=(7, 4))

        for model_name, g in all_df.groupby("model"):
            g = g.sort_values("episode")
            plt.plot(g["episode"], g[metric], marker="o", label=model_name)

        plt.title(metric)
        plt.xlabel("episode seed")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}_episodes.png", dpi=200)
        plt.close()


def make_paired_difference(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares both models using the same episode seeds.
    Positive difference means Simple reward is higher than Old reward.
    """
    if set(CHECKPOINTS.keys()) != {"Old reward", "Simple reward"}:
        return pd.DataFrame()

    rows = []

    old_df = all_df[all_df["model"] == "Old reward"].set_index("episode")
    simple_df = all_df[all_df["model"] == "Simple reward"].set_index("episode")

    common_eps = sorted(set(old_df.index).intersection(set(simple_df.index)))

    for metric in METRICS:
        if metric not in old_df.columns or metric not in simple_df.columns:
            continue

        diffs = simple_df.loc[common_eps, metric] - old_df.loc[common_eps, metric]

        rows.append({
            "metric": metric,
            "mean_difference_simple_minus_old": diffs.mean(),
            "std_difference": diffs.std(ddof=0),
            "min_difference": diffs.min(),
            "max_difference": diffs.max(),
        })

    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    register_env("MultiAgentVineAsync", env_creator)
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

    all_results = []

    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nRunning inference for: {model_name}")
        df = run_model_inference(model_name, checkpoint_path, EPISODES)

        out_csv = OUT_DIR / f"{model_name.replace(' ', '_').lower()}_inference.csv"
        df.to_csv(out_csv, index=False)

        print(f"Saved: {out_csv}")
        all_results.append(df)

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_model_inference_results.csv", index=False)

    summary_df = make_summary(all_df)
    summary_df.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)

    paired_df = make_paired_difference(all_df)
    paired_df.to_csv(OUT_DIR / "paired_difference_simple_minus_old.csv", index=False)

    make_plots(summary_df)
    make_episode_plots(all_df)

    print("\n=== Summary ===")
    print(summary_df)

    print("\n=== Paired difference: Simple reward minus Old reward ===")
    print(paired_df)

    print(f"\nSaved all results to: {OUT_DIR}")

    ray.shutdown()