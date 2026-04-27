# pareto_inference_compare.py

import os
from pathlib import Path
import re

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

OUT_DIR = Path(PROJECT_DIR) / "inference_results" / "pareto_weight_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = 20
RENDER = False

# Use same evaluation environment for all models
ENV_OVERRIDES = {
    # "num_humans": 5,
    # "num_drones": 1,
    # "max_steps": 240,
}

# Add as many models as you want here.
# Label convention: D,F,Q weights -> "111", "211", "121", etc.
MODELS = [
    {
        "label": "111",
        "checkpoint": os.path.join(
            PROJECT_DIR,
            "ray_results_vine",
            "vineyard_ppo",
            "Reward_DFQ_111",
            "checkpoint_000000",
        ),
    },
    {
        "label": "211",
        "checkpoint": os.path.join(
            PROJECT_DIR,
            "ray_results_vine",
            "vineyard_ppo",
            "Reward_DFQ_211",
            "checkpoint_000000",
        ),
    },
    {
        "label": "121",
        "checkpoint": os.path.join(
            PROJECT_DIR,
            "ray_results_vine",
            "vineyard_ppo",
            "Reward_DFQ_121",
            "checkpoint_000000",
        ),
    },
    {
        "label": "311",
        "checkpoint": os.path.join(
            PROJECT_DIR,
            "ray_results_vine",
            "vineyard_ppo",
            "Reward_DFQ_311",
            "checkpoint_000000",
        ),
    },
        {
        "label": "321",
        "checkpoint": os.path.join(
            PROJECT_DIR,
            "ray_results_vine",
            "vineyard_ppo",
            "Reward_DFQ_321",
            "checkpoint_000000",
        ),
    },
]

# Main Pareto objectives
DELIVERY_METRIC = "kpi_completion_pct"     # maximise
FATIGUE_METRIC = "kpi_mean_fatigue"        # minimise

# Extra metric only for interpretation
QUEUE_METRIC = "kpi_mean_backlog"


METRICS = [
    "kpi_delivered_total",
    "kpi_completion_pct",
    "kpi_delivered_pct",
    "kpi_throughput_per_100_steps",
    "kpi_mean_fatigue",
    "kpi_peak_fatigue",
    "episode_fatigue_increase_total",
    "kpi_mean_backlog",
    "kpi_peak_backlog",
    "kpi_rest_ratio",
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
    Handles both:
    checkpoint_000020/
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


def parse_weight_label(label: str):
    """
    Parses labels like:
    111 -> D=1, F=1, Q=1
    211 -> D=2, F=1, Q=1
    121 -> D=1, F=2, Q=1

    If parsing fails, returns NaN values.
    """
    digits = re.findall(r"\d", str(label))

    if len(digits) >= 3:
        return float(digits[0]), float(digits[1]), float(digits[2])

    return np.nan, np.nan, np.nan


def run_model_inference(model_label: str, checkpoint_path: str, episodes: int) -> pd.DataFrame:
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

        w_d, w_f, w_q = parse_weight_label(model_label)

        summary["model"] = model_label
        summary["episode"] = ep
        summary["w_delivery"] = w_d
        summary["w_fatigue"] = w_f
        summary["w_queue"] = w_q

        # Keep only for checking, not for comparing different reward weights.
        summary["rollout_reward_sum"] = reward_sum

        rows.append(summary)
        env.close()

    algo.stop()

    return pd.DataFrame(rows)


def make_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, g in all_df.groupby("model"):
        base = {
            "model": model_name,
            "episodes": len(g),
            "w_delivery": g["w_delivery"].iloc[0],
            "w_fatigue": g["w_fatigue"].iloc[0],
            "w_queue": g["w_queue"].iloc[0],
        }

        for metric in METRICS:
            if metric not in g.columns:
                continue

            base[f"{metric}_mean"] = g[metric].mean()
            base[f"{metric}_std"] = g[metric].std(ddof=0)
            base[f"{metric}_min"] = g[metric].min()
            base[f"{metric}_max"] = g[metric].max()

        rows.append(base)

    return pd.DataFrame(rows)


def add_pareto_flags(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pareto analysis using:
    maximise DELIVERY_METRIC
    minimise FATIGUE_METRIC
    """
    df = summary_df.copy()

    d_col = f"{DELIVERY_METRIC}_mean"
    f_col = f"{FATIGUE_METRIC}_mean"

    is_pareto = []
    dominated_by_list = []

    for i, row_i in df.iterrows():
        d_i = row_i[d_col]
        f_i = row_i[f_col]

        dominated = False
        dominated_by = []

        for j, row_j in df.iterrows():
            if i == j:
                continue

            d_j = row_j[d_col]
            f_j = row_j[f_col]

            better_or_equal_delivery = d_j >= d_i
            better_or_equal_fatigue = f_j <= f_i

            strictly_better = (d_j > d_i) or (f_j < f_i)

            if better_or_equal_delivery and better_or_equal_fatigue and strictly_better:
                dominated = True
                dominated_by.append(str(row_j["model"]))

        is_pareto.append(not dominated)
        dominated_by_list.append(", ".join(dominated_by))

    df["is_pareto_DF"] = is_pareto
    df["dominated_by_DF"] = dominated_by_list

    return df


def make_pareto_plot(summary_df: pd.DataFrame):
    d_col = f"{DELIVERY_METRIC}_mean"
    d_std_col = f"{DELIVERY_METRIC}_std"
    f_col = f"{FATIGUE_METRIC}_mean"
    f_std_col = f"{FATIGUE_METRIC}_std"
    q_col = f"{QUEUE_METRIC}_mean"

    plot_df = summary_df.copy()

    plt.figure(figsize=(7, 5))

    # Marker size uses backlog only as visual information.
    if q_col in plot_df.columns:
        q_values = plot_df[q_col].to_numpy(dtype=float)
        q_min = np.nanmin(q_values)
        q_max = np.nanmax(q_values)

        if np.isclose(q_min, q_max):
            sizes = np.full(len(plot_df), 120.0)
        else:
            sizes = 80.0 + 240.0 * (q_values - q_min) / (q_max - q_min)
    else:
        sizes = np.full(len(plot_df), 120.0)

    plt.scatter(
        plot_df[f_col],
        plot_df[d_col],
        s=sizes,
        alpha=0.8,
        label="Policies",
    )

    # Error bars
    if f_std_col in plot_df.columns and d_std_col in plot_df.columns:
        plt.errorbar(
            plot_df[f_col],
            plot_df[d_col],
            xerr=plot_df[f_std_col],
            yerr=plot_df[d_std_col],
            fmt="none",
            capsize=3,
            alpha=0.6,
        )

    # Labels
    for _, row in plot_df.iterrows():
        label = str(row["model"])
        star = " *" if row["is_pareto_DF"] else ""
        plt.annotate(
            label + star,
            (row[f_col], row[d_col]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    # Pareto front line
    front = plot_df[plot_df["is_pareto_DF"]].sort_values(f_col)

    if len(front) >= 2:
        plt.plot(
            front[f_col],
            front[d_col],
            marker="o",
            linewidth=1.5,
            label="Pareto front",
        )

    plt.xlabel("Mean fatigue, F (lower is better)")
    plt.ylabel("Completion %, D (higher is better)")
    plt.title("Pareto analysis: delivery vs fatigue")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "pareto_delivery_fatigue.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved Pareto plot: {out_path}")


def make_metric_bar_plots(summary_df: pd.DataFrame):
    plot_dir = OUT_DIR / "metric_bars"
    plot_dir.mkdir(exist_ok=True)

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in summary_df.columns:
            continue

        sub = summary_df.sort_values("model")

        plt.figure(figsize=(7, 4))
        plt.bar(
            sub["model"],
            sub[mean_col],
            yerr=sub[std_col] if std_col in sub.columns else None,
            capsize=4,
        )
        plt.title(metric)
        plt.ylabel("value")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}.png", dpi=200)
        plt.close()

    print(f"Saved metric bar plots: {plot_dir}")


def make_pairwise_differences(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise differences in D and F.
    Positive delta_D means row model has higher delivery than compared model.
    Negative delta_F means row model has lower fatigue than compared model.
    """
    d_col = f"{DELIVERY_METRIC}_mean"
    f_col = f"{FATIGUE_METRIC}_mean"
    q_col = f"{QUEUE_METRIC}_mean"

    rows = []

    for _, a in summary_df.iterrows():
        for _, b in summary_df.iterrows():
            if a["model"] == b["model"]:
                continue

            row = {
                "model_A": a["model"],
                "model_B": b["model"],
                "delta_D_A_minus_B": a[d_col] - b[d_col],
                "delta_F_A_minus_B": a[f_col] - b[f_col],
            }

            if q_col in summary_df.columns:
                row["delta_Q_A_minus_B"] = a[q_col] - b[q_col]

            rows.append(row)

    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    register_env("MultiAgentVineAsync", env_creator)
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

    all_results = []

    for model in MODELS:
        label = model["label"]
        checkpoint = model["checkpoint"]

        print(f"\nRunning inference for model: {label}")
        print(f"Checkpoint: {checkpoint}")

        df = run_model_inference(label, checkpoint, EPISODES)

        model_csv = OUT_DIR / f"inference_{label}.csv"
        df.to_csv(model_csv, index=False)
        print(f"Saved: {model_csv}")

        all_results.append(df)

    all_df = pd.concat(all_results, ignore_index=True)
    all_csv = OUT_DIR / "all_inference_results.csv"
    all_df.to_csv(all_csv, index=False)

    summary_df = make_summary(all_df)
    summary_df = add_pareto_flags(summary_df)

    summary_csv = OUT_DIR / "policy_summary_with_pareto.csv"
    summary_df.to_csv(summary_csv, index=False)

    pairwise_df = make_pairwise_differences(summary_df)
    pairwise_csv = OUT_DIR / "pairwise_differences_DFQ.csv"
    pairwise_df.to_csv(pairwise_csv, index=False)

    make_pareto_plot(summary_df)
    make_metric_bar_plots(summary_df)

    print("\n=== Pareto summary ===")
    display_cols = [
        "model",
        "w_delivery",
        "w_fatigue",
        "w_queue",
        f"{DELIVERY_METRIC}_mean",
        f"{FATIGUE_METRIC}_mean",
        f"{QUEUE_METRIC}_mean",
        "is_pareto_DF",
        "dominated_by_DF",
    ]

    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].sort_values(["is_pareto_DF", f"{DELIVERY_METRIC}_mean"], ascending=[False, False]))

    print(f"\nSaved all results to: {OUT_DIR}")
    print(f"Main summary: {summary_csv}")
    print(f"Pairwise differences: {pairwise_csv}")

    ray.shutdown()