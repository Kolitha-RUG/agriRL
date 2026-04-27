import os
from pathlib import Path
from itertools import combinations

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

OUT_DIR = Path(PROJECT_DIR) / "inference_results" / "constraint_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = 20
RENDER = False

ENV_OVERRIDES = {
    # Keep same evaluation setting for all models
    # "num_humans": 5,
    # "num_drones": 1,
    # "max_steps": 240,
}

# Add more models here dynamically
MODELS = {
    "511": os.path.join(
        PROJECT_DIR,
        "ray_results_vine",
        "vineyard_ppo",
        "Reward_DFQ_511",
        "checkpoint_000000",
    ),
    "211": os.path.join(
        PROJECT_DIR,
        "ray_results_vine",
        "vineyard_ppo",
        "Reward_DFQ_211",
        "checkpoint_000000",
    ),
    "111": os.path.join(
        PROJECT_DIR,
        "ray_results_vine",
        "vineyard_ppo",
        "Reward_DFQ_111",
        "checkpoint_000000",
    ),
    "311": os.path.join(
        PROJECT_DIR,
        "ray_results_vine",
        "vineyard_ppo",
        "Reward_DFQ_311",
        "checkpoint_000000",
    ),
    # "321": os.path.join(
    #     PROJECT_DIR,
    #     "ray_results_vine",
    #     "vineyard_ppo",
    #     "Reward_DFQ_321",
    #     "checkpoint_000000",
    # ),

}

ACTION_NAMES = {
    0: "HARVEST",
    1: "TRANSPORT",
    2: "ENQUEUE",
    3: "REST",
}

ACTION_HARVEST = 0
ACTION_TRANSPORT = 1
ACTION_ENQUEUE = 2
ACTION_REST = 3


# =========================
# HELPERS
# =========================
def env_creator(env_config):
    return VineEnv(**env_config)


def find_real_checkpoint_file(checkpoint_dir: str) -> str:
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    base = os.path.basename(checkpoint_dir.rstrip("/\\"))
    num = base.split("_")[-1].lstrip("0") or "0"
    candidate = os.path.join(checkpoint_dir, f"checkpoint-{num}")

    if os.path.isfile(candidate):
        return candidate

    return checkpoint_dir


def load_algorithms():
    algos = {}

    for label, checkpoint_path in MODELS.items():
        checkpoint_to_load = find_real_checkpoint_file(checkpoint_path)
        print(f"Loading {label}: {checkpoint_to_load}")
        algos[label] = Algorithm.from_checkpoint(checkpoint_to_load)

    return algos


def compute_action(algo, obs):
    action = algo.compute_single_action(
        observation=obs,
        policy_id="shared_policy",
        explore=False,
    )

    if isinstance(action, tuple):
        action = action[0]

    return int(action)


def mask_stats(mask):
    valid_actions = np.where(mask > 0.5)[0].astype(int).tolist()
    valid_count = len(valid_actions)

    productive_valid = [
        a for a in valid_actions
        if a in [ACTION_HARVEST, ACTION_TRANSPORT, ACTION_ENQUEUE]
    ]

    transport_enqueue_choice = (
        mask[ACTION_TRANSPORT] > 0.5
        and mask[ACTION_ENQUEUE] > 0.5
    )

    rest_tradeoff = (
        mask[ACTION_REST] > 0.5
        and len(productive_valid) > 0
    )

    return {
        "valid_count": valid_count,
        "forced_action": int(valid_count <= 1),
        "real_decision": int(valid_count > 1),
        "productive_valid_count": len(productive_valid),
        "transport_enqueue_choice": int(transport_enqueue_choice),
        "rest_tradeoff": int(rest_tradeoff),
        "valid_actions": ",".join(str(a) for a in valid_actions),
    }


def extract_episode_summary(last_infos):
    if last_infos is None:
        return {}

    for agent_id in last_infos:
        s = last_infos[agent_id].get("episode_summary", None)
        if s:
            return dict(s)

    return {}


def run_model_rollout(model_label, algo, episodes):
    env_config = get_env_config(**ENV_OVERRIDES)
    env_config["render_mode"] = "human" if RENDER else None

    kpi_rows = []
    mask_rows = []
    action_rows = []

    for ep in range(episodes):
        env = VineEnv(**env_config)
        obs, infos = env.reset(seed=ep)

        terminated = {"__all__": False}
        truncated = {"__all__": False}
        last_infos = None
        step = 0
        reward_sum = 0.0

        while not terminated["__all__"] and not truncated["__all__"]:
            action_dict = {}

            for agent_id in env.agents:
                agent_idx = env.agent_index[agent_id]
                h = env.humans[agent_idx]

                mask = obs[agent_id]["action_mask"]
                ms = mask_stats(mask)

                action = compute_action(algo, obs[agent_id])
                action_dict[agent_id] = action

                mask_rows.append({
                    "model": model_label,
                    "episode": ep,
                    "step": step,
                    "agent_id": agent_id,
                    "busy": int(h.busy),
                    "has_box": int(h.has_box),
                    "fatigue": float(h.fatigue),
                    "chosen_action": action,
                    "chosen_action_name": ACTION_NAMES[action],
                    **ms,
                })

                action_rows.append({
                    "model": model_label,
                    "episode": ep,
                    "step": step,
                    "agent_id": agent_id,
                    "chosen_action": action,
                    "chosen_action_name": ACTION_NAMES[action],
                })

            obs, rewards, terminated, truncated, infos = env.step(action_dict)
            last_infos = infos
            reward_sum += float(np.mean(list(rewards.values())))
            step += 1

            if RENDER:
                env.render()

        summary = extract_episode_summary(last_infos)
        summary["model"] = model_label
        summary["episode"] = ep
        summary["rollout_reward_sum"] = reward_sum
        kpi_rows.append(summary)

        env.close()

    return (
        pd.DataFrame(kpi_rows),
        pd.DataFrame(mask_rows),
        pd.DataFrame(action_rows),
    )


def summarise_masks(mask_df):
    rows = []

    for model, g in mask_df.groupby("model"):
        idle = g[g["busy"] == 0]
        busy = g[g["busy"] == 1]

        row = {
            "model": model,
            "n_agent_steps": len(g),

            # Including busy steps
            "valid_count_mean_all": g["valid_count"].mean(),
            "forced_action_rate_all": g["forced_action"].mean(),
            "real_decision_rate_all": g["real_decision"].mean(),
            "transport_enqueue_choice_rate_all": g["transport_enqueue_choice"].mean(),
            "rest_tradeoff_rate_all": g["rest_tradeoff"].mean(),

            # Only when agent is not busy
            "n_idle_agent_steps": len(idle),
            "valid_count_mean_idle": idle["valid_count"].mean() if len(idle) else np.nan,
            "forced_action_rate_idle": idle["forced_action"].mean() if len(idle) else np.nan,
            "real_decision_rate_idle": idle["real_decision"].mean() if len(idle) else np.nan,
            "transport_enqueue_choice_rate_idle": idle["transport_enqueue_choice"].mean() if len(idle) else np.nan,
            "rest_tradeoff_rate_idle": idle["rest_tradeoff"].mean() if len(idle) else np.nan,

            # Busy share
            "busy_agent_step_rate": len(busy) / max(1, len(g)),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def summarise_actions(action_df):
    rows = []

    for model, g in action_df.groupby("model"):
        total = len(g)

        row = {"model": model, "n_actions": total}

        for action_id, action_name in ACTION_NAMES.items():
            row[f"action_rate_{action_name}"] = float((g["chosen_action"] == action_id).mean())

        rows.append(row)

    return pd.DataFrame(rows)


def pairwise_action_similarity(action_df):
    rows = []

    models = sorted(action_df["model"].unique())

    for a, b in combinations(models, 2):
        da = action_df[action_df["model"] == a][
            ["episode", "step", "agent_id", "chosen_action"]
        ].rename(columns={"chosen_action": "action_A"})

        db = action_df[action_df["model"] == b][
            ["episode", "step", "agent_id", "chosen_action"]
        ].rename(columns={"chosen_action": "action_B"})

        merged = da.merge(db, on=["episode", "step", "agent_id"], how="inner")

        same_rate = float((merged["action_A"] == merged["action_B"]).mean()) if len(merged) else np.nan

        rows.append({
            "model_A": a,
            "model_B": b,
            "n_compared_agent_steps": len(merged),
            "same_action_rate_aligned_rollout": same_rate,
        })

    return pd.DataFrame(rows)


def same_state_policy_similarity(algos, episodes=5):
    """
    Uses the first model as a reference rollout.
    At each observed state, all models are asked what action they would choose.
    This checks whether policies differ on the same states.
    """
    labels = list(algos.keys())
    reference_label = labels[0]
    reference_algo = algos[reference_label]

    env_config = get_env_config(**ENV_OVERRIDES)
    rows = []

    for ep in range(episodes):
        env = VineEnv(**env_config)
        obs, infos = env.reset(seed=ep)

        terminated = {"__all__": False}
        truncated = {"__all__": False}
        step = 0

        while not terminated["__all__"] and not truncated["__all__"]:
            reference_action_dict = {}

            for agent_id in env.agents:
                actions = {}

                for label, algo in algos.items():
                    actions[label] = compute_action(algo, obs[agent_id])

                unique_actions = set(actions.values())

                row = {
                    "reference_model": reference_label,
                    "episode": ep,
                    "step": step,
                    "agent_id": agent_id,
                    "all_models_same_action": int(len(unique_actions) == 1),
                    "n_unique_actions": len(unique_actions),
                }

                for label in labels:
                    row[f"action_{label}"] = actions[label]

                rows.append(row)

                reference_action_dict[agent_id] = actions[reference_label]

            obs, rewards, terminated, truncated, infos = env.step(reference_action_dict)
            step += 1

        env.close()

    df = pd.DataFrame(rows)

    pair_rows = []
    for a, b in combinations(labels, 2):
        same = df[f"action_{a}"] == df[f"action_{b}"]
        pair_rows.append({
            "model_A": a,
            "model_B": b,
            "same_action_rate_same_states": float(same.mean()),
        })

    return df, pd.DataFrame(pair_rows)


def make_basic_plots(mask_summary, kpi_summary, action_summary):
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Constraint indicators
    cols = [
        "forced_action_rate_all",
        "real_decision_rate_all",
        "transport_enqueue_choice_rate_idle",
        "rest_tradeoff_rate_idle",
        "busy_agent_step_rate",
    ]

    for col in cols:
        if col not in mask_summary.columns:
            continue

        plt.figure(figsize=(7, 4))
        plt.bar(mask_summary["model"], mask_summary[col])
        plt.title(col)
        plt.ylabel("rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{col}.png", dpi=200)
        plt.close()

    # Delivery vs fatigue
    if "kpi_delivered_total_mean" in kpi_summary.columns and "kpi_mean_fatigue_mean" in kpi_summary.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(
            kpi_summary["kpi_mean_fatigue_mean"],
            kpi_summary["kpi_delivered_total_mean"],
            s=120,
        )

        for _, row in kpi_summary.iterrows():
            plt.annotate(
                str(row["model"]),
                (row["kpi_mean_fatigue_mean"], row["kpi_delivered_total_mean"]),
                textcoords="offset points",
                xytext=(6, 6),
            )

        plt.xlabel("Mean fatigue")
        plt.ylabel("Delivered boxes")
        plt.title("Delivery vs fatigue")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "delivery_vs_fatigue.png", dpi=200)
        plt.close()

    # Action rates
    action_rate_cols = [c for c in action_summary.columns if c.startswith("action_rate_")]

    for col in action_rate_cols:
        plt.figure(figsize=(7, 4))
        plt.bar(action_summary["model"], action_summary[col])
        plt.title(col)
        plt.ylabel("rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{col}.png", dpi=200)
        plt.close()

    print(f"Saved plots to: {plot_dir}")


def make_kpi_summary(kpi_df):
    numeric_cols = [
        c for c in kpi_df.columns
        if c not in ["model", "episode"]
        and pd.api.types.is_numeric_dtype(kpi_df[c])
    ]

    rows = []

    for model, g in kpi_df.groupby("model"):
        row = {"model": model, "episodes": len(g)}

        for c in numeric_cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=0)

        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    register_env("MultiAgentVineAsync", env_creator)
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

    all_kpis = []
    all_masks = []
    all_actions = []

    for label, checkpoint_path in MODELS.items():
        print(f"\nLoading model: {label}")
        checkpoint_to_load = find_real_checkpoint_file(checkpoint_path)

        algo = Algorithm.from_checkpoint(checkpoint_to_load)

        print(f"Running diagnostic rollout for {label}")
        kpi_df, mask_df, action_df = run_model_rollout(label, algo, EPISODES)

        # Stop this model before loading the next one
        algo.stop()
        del algo

        kpi_df.to_csv(OUT_DIR / f"kpis_{label}.csv", index=False)
        mask_df.to_csv(OUT_DIR / f"mask_trace_{label}.csv", index=False)
        action_df.to_csv(OUT_DIR / f"action_trace_{label}.csv", index=False)

        all_kpis.append(kpi_df)
        all_masks.append(mask_df)
        all_actions.append(action_df)

    kpi_df = pd.concat(all_kpis, ignore_index=True)
    mask_df = pd.concat(all_masks, ignore_index=True)
    action_df = pd.concat(all_actions, ignore_index=True)

    kpi_df.to_csv(OUT_DIR / "all_kpis.csv", index=False)
    mask_df.to_csv(OUT_DIR / "all_mask_traces.csv", index=False)
    action_df.to_csv(OUT_DIR / "all_action_traces.csv", index=False)

    kpi_summary = make_kpi_summary(kpi_df)
    mask_summary = summarise_masks(mask_df)
    action_summary = summarise_actions(action_df)
    aligned_similarity = pairwise_action_similarity(action_df)

    kpi_summary.to_csv(OUT_DIR / "kpi_summary.csv", index=False)
    mask_summary.to_csv(OUT_DIR / "constraint_mask_summary.csv", index=False)
    action_summary.to_csv(OUT_DIR / "action_distribution_summary.csv", index=False)
    aligned_similarity.to_csv(
        OUT_DIR / "pairwise_action_similarity_aligned_rollout.csv",
        index=False
    )

    make_basic_plots(mask_summary, kpi_summary, action_summary)

    print("\n=== Constraint mask summary ===")
    print(mask_summary)

    print("\n=== Action distribution summary ===")
    print(action_summary)

    print("\n=== Pairwise action similarity, aligned rollout ===")
    print(aligned_similarity)

    print("\n=== KPI summary ===")
    key_cols = [
        "model",
        "kpi_delivered_total_mean",
        "kpi_completion_pct_mean",
        "kpi_mean_fatigue_mean",
        "kpi_mean_backlog_mean",
        "kpi_rest_ratio_mean",
    ]
    key_cols = [c for c in key_cols if c in kpi_summary.columns]
    print(kpi_summary[key_cols])

    print(f"\nSaved diagnostics to: {OUT_DIR}")

    ray.shutdown()