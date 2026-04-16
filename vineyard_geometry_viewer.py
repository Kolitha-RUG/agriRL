import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Config
# =====================
FILE_PATH = "data/Vinha_Maria_Teresa_RL.xlsx"
OUTPUT_PATH = "vineyard_layout_clean.png"

FIGSIZE = (12, 9)
LINE_WIDTH = 1.4
HANDOVER_SIZE = 140
COLLECTION_SIZE = 220

SHOW_TITLE = False
SHOW_LEGEND = True

# Manually selected handover points
HANDOVER_POINTS = [
    (48.548, 59.740),
    (83.632, 76.301),
    (122.336, 81.086),
    (36.096, 165.033),
    (73.253, 171.914),
    (114.117, 173.284),
    (26.728, 260.673),
    (69.043, 271.251),
    (112.451, 259.898),
    (149.760, 280.885),
    (63.016, 354.571),
]


def load_vineyard(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = df.copy()
    df["lot"] = df["lot"].astype(str).str.strip()
    df["line"] = df["line"].astype(str).str.strip()

    for col in ["x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["lot", "line", "x", "y", "z"]).reset_index(drop=True)
    return df


def sort_line_points(g: pd.DataFrame) -> pd.DataFrame:
    """Sort points along dominant line direction."""
    if len(g) <= 2:
        return g.sort_values(["x", "y"]).reset_index(drop=True)

    pts = g[["x", "y"]].to_numpy(dtype=float)
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis

    return (
        g.assign(_proj=proj)
         .sort_values("_proj")
         .drop(columns="_proj")
         .reset_index(drop=True)
    )


def build_line_geometries(df: pd.DataFrame):
    line_segments = []

    for (lot, line), g in df.groupby(["lot", "line"], sort=True):
        gs = sort_line_points(g)
        xy = gs[["x", "y"]].to_numpy(dtype=float)

        if len(xy) >= 2:
            seg = xy
        else:
            seg = np.vstack([xy[0], xy[0]])

        line_segments.append((lot, line, seg))

    return line_segments


def main():
    df = load_vineyard(FILE_PATH)
    line_segments = build_line_geometries(df)

    # Collection point: same logic as env
    collection_point = np.array([df["x"].max(), df["y"].mean()], dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # -------------------------
    # Consistent color per lot
    # -------------------------
    lots = sorted(df["lot"].unique())
    cmap = plt.get_cmap("tab20", max(len(lots), 1))
    lot_to_color = {lot: cmap(i % cmap.N) for i, lot in enumerate(lots)}

    # -------------------------
    # Draw vineyard lines by lot
    # -------------------------
    used_lots = set()
    for lot, line, seg in line_segments:
        label = lot if lot not in used_lots else None
        used_lots.add(lot)

        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color=lot_to_color[lot],
            linewidth=LINE_WIDTH,
            alpha=0.95,
            zorder=1,
            label=label,
        )

    # -------------------------
    # Draw handover points
    # -------------------------
    handover_xy = np.array(HANDOVER_POINTS, dtype=float)
    ax.scatter(
        handover_xy[:, 0],
        handover_xy[:, 1],
        s=HANDOVER_SIZE,
        marker="o",
        facecolor="#F28E2B",
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
        label="Handover points",
    )

    # -------------------------
    # Draw collection point
    # -------------------------
    ax.scatter(
        collection_point[0],
        collection_point[1],
        s=COLLECTION_SIZE,
        marker="s",
        facecolor="#4E79A7",
        edgecolor="white",
        linewidth=1.8,
        zorder=5,

    )

    # ax.text(
    #     collection_point[0] + 4,
    #     collection_point[1],
    #     "Collection Point",
    #     fontsize=10,
    #     weight="bold",
    #     ha="left",
    #     va="center",
    #     color="#1F3A5F",
    #     zorder=6,
    # )

    # -------------------------
    # Styling
    # -------------------------
    if SHOW_TITLE:
        ax.set_title("Vineyard layout with handover and collection points", fontsize=14, pad=12)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    x_pad = (df["x"].max() - df["x"].min()) * 0.05
    y_pad = (df["y"].max() - df["y"].min()) * 0.05
    ax.set_xlim(df["x"].min() - x_pad, df["x"].max() + x_pad)
    ax.set_ylim(df["y"].min() - y_pad, df["y"].max() + y_pad)

    # if SHOW_LEGEND:
    #     ax.legend(
    #         loc="upper left",
    #         frameon=True,
    #         facecolor="white",
    #         edgecolor="lightgray",
    #         ncol=2,
    #         fontsize=9,
    #     )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved clean presentation image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()