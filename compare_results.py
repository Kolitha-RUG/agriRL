import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = Path(PROJECT_DIR) / "inference_results"

PPO_FILE = RESULT_DIR / "ppo_inference.csv"
MANUAL_FILE = RESULT_DIR / "baseline_manual.csv"
ENQUEUE_FILE = RESULT_DIR / "baseline_enqueue.csv"

METRICS = [
    "kpi_delivered_total",
    "kpi_throughput_per_100_steps",
    "kpi_mean_backlog",
    "kpi_mean_fatigue",
    "kpi_rest_ratio",
    "kpi_completion_pct",
    "r_total_per_step",
]

FILES = {
    "RL": PPO_FILE,
    "Manual": MANUAL_FILE,
    "Only Drone": ENQUEUE_FILE,
}


def load_summary():
    rows = []

    for label, path in FILES.items():
        df = pd.read_csv(path)

        for metric in METRICS:
            rows.append({
                "policy": label,
                "metric": metric,
                "mean": df[metric].mean(),
                "std": df[metric].std(ddof=0),
            })

    return pd.DataFrame(rows)


def make_plots(summary_df):
    plot_dir = RESULT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    for metric in METRICS:
        sub = summary_df[summary_df["metric"] == metric]

        plt.figure(figsize=(6, 4))
        plt.bar(sub["policy"], sub["mean"], yerr=sub["std"], capsize=4)
        plt.title(metric)
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    summary = load_summary()
    print(summary)

    summary_path = RESULT_DIR / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    make_plots(summary)

    print(f"\nSaved: {summary_path}")
    print(f"Saved plots to: {RESULT_DIR / 'plots'}")