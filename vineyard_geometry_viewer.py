import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================
# Config
# =====================
FILE_PATH = "data/Vinha_Maria_Teresa_RL.xlsx"   # change to your path if needed
SHOW_PLANT_POINTS = True
SHOW_LOT_LABELS = True
SHOW_LOT_HULLS = True
POINT_SIZE = 4
LINE_WIDTH = 0.9
FIGSIZE = (12, 10)

# Mouse usage:
# - Left click:  drop a red marker and show click coordinates + nearest plant + nearest line
# - Right click: drop an orange "candidate hover point" marker and print coordinates to terminal
# - Standard matplotlib toolbar still works for zoom/pan


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
    df["line_key"] = df["lot"] + "::" + df["line"]
    return df


def sort_line_points(g: pd.DataFrame) -> pd.DataFrame:
    """Sort plant points along the dominant direction of the line."""
    if len(g) <= 2:
        return g.sort_values(["x", "y"]).reset_index(drop=True)

    pts = g[["x", "y"]].to_numpy(dtype=float)
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]  # principal direction
    proj = centered @ axis

    return (
        g.assign(_proj=proj)
         .sort_values("_proj")
         .drop(columns="_proj")
         .reset_index(drop=True)
    )


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew monotonic-chain convex hull. No scipy needed."""
    pts = np.unique(points.astype(float), axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def build_line_geometries(df: pd.DataFrame):
    line_segments = []
    line_info = []

    for (lot, line), g in df.groupby(["lot", "line"], sort=True):
        gs = sort_line_points(g)
        xy = gs[["x", "y"]].to_numpy(dtype=float)

        if len(xy) >= 2:
            line_segments.append(xy)
        else:
            # single-point line edge case
            line_segments.append(np.vstack([xy[0], xy[0]]))

        centroid = xy.mean(axis=0)
        start = xy[0]
        end = xy[-1]
        line_info.append({
            "lot": lot,
            "line": line,
            "line_key": f"{lot}::{line}",
            "centroid": centroid,
            "start": start,
            "end": end,
            "n_plants": len(gs),
        })

    return line_segments, line_info


def main():
    df = load_vineyard(FILE_PATH)
    line_segments, line_info = build_line_geometries(df)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.canvas.manager.set_window_title("Vineyard geometry viewer")

    # consistent color per lot
    lots = sorted(df["lot"].unique())
    cmap = plt.get_cmap("tab20", max(len(lots), 1))
    lot_to_color = {lot: cmap(i % cmap.N) for i, lot in enumerate(lots)}

    # Draw lot hulls first (light and behind)
    if SHOW_LOT_HULLS:
        for lot, g in df.groupby("lot", sort=True):
            pts = g[["x", "y"]].to_numpy(dtype=float)
            hull = convex_hull(pts)
            if len(hull) >= 2:
                hull_closed = np.vstack([hull, hull[0]])
                ax.plot(
                    hull_closed[:, 0],
                    hull_closed[:, 1],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.35,
                    color=lot_to_color[lot],
                    zorder=1,
                )

    # Draw each line in its lot color
    for seg, info in zip(line_segments, line_info):
        ax.plot(
            seg[:, 0], seg[:, 1],
            color=lot_to_color[info["lot"]],
            linewidth=LINE_WIDTH,
            alpha=0.95,
            zorder=2,
        )

    # Draw plant points
    if SHOW_PLANT_POINTS:
        ax.scatter(
            df["x"], df["y"],
            s=POINT_SIZE,
            c=[lot_to_color[l] for l in df["lot"]],
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )

    # Lot labels
    if SHOW_LOT_LABELS:
        for lot, g in df.groupby("lot", sort=True):
            cx, cy = g[["x", "y"]].mean().to_numpy()
            ax.text(
                cx, cy, lot,
                fontsize=8,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.65, edgecolor="none"),
                zorder=4,
            )

    ax.set_title("Vineyard geometry by lot and line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    # Data arrays for nearest-neighbor click lookup
    plant_xy = df[["x", "y"]].to_numpy(dtype=float)
    line_centroids = np.vstack([li["centroid"] for li in line_info])

    click_annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95),
        arrowprops=dict(arrowstyle="->", alpha=0.7),
    )
    click_annotation.set_visible(False)

    hover_points = []

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        click_xy = np.array([event.xdata, event.ydata], dtype=float)

        # nearest plant
        d_plant = np.linalg.norm(plant_xy - click_xy, axis=1)
        plant_idx = int(np.argmin(d_plant))
        plant_row = df.iloc[plant_idx]

        # nearest line centroid
        d_line = np.linalg.norm(line_centroids - click_xy, axis=1)
        line_idx = int(np.argmin(d_line))
        line_row = line_info[line_idx]

        text = (
            f"click=({click_xy[0]:.3f}, {click_xy[1]:.3f})\n"
            f"nearest plant={plant_row['plant']}\n"
            f"lot={plant_row['lot']} line={plant_row['line']} z={plant_row['z']:.3f}\n"
            f"nearest line={line_row['line_key']}\n"
            f"line start=({line_row['start'][0]:.2f}, {line_row['start'][1]:.2f})\n"
            f"line end=({line_row['end'][0]:.2f}, {line_row['end'][1]:.2f})"
        )

        click_annotation.xy = (click_xy[0], click_xy[1])
        click_annotation.set_text(text)
        click_annotation.set_visible(True)

        if event.button == 1:
            # red marker for inspection point
            ax.plot(click_xy[0], click_xy[1], marker="x", color="red", markersize=8, zorder=10)
            print("[inspect]", text.replace("\n", " | "))

        elif event.button == 3:
            # orange star = candidate handover / hover point
            marker, = ax.plot(click_xy[0], click_xy[1], marker="*", color="orange", markersize=12, zorder=11)
            hover_points.append((click_xy[0], click_xy[1], marker))
            print(f"[candidate_handover] x={click_xy[0]:.3f}, y={click_xy[1]:.3f}")

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "c":
            # clear all manually added markers
            for artist in ax.lines[:]:
                # keep original line geometry; remove only user-added point markers by marker style
                if isinstance(artist, Line2D) and artist.get_marker() in {"x", "*"}:
                    artist.remove()
            hover_points.clear()
            click_annotation.set_visible(False)
            fig.canvas.draw_idle()
            print("[clear] removed all clicked markers")

        elif event.key == "p":
            # print all candidate handover points
            if not hover_points:
                print("[candidate_handover] none yet")
            else:
                print("[candidate_handover] saved points:")
                for i, (x, y, _) in enumerate(hover_points, 1):
                    print(f"  {i}: x={x:.3f}, y={y:.3f}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1, label="line geometry"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, color="gray", label="plants"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=7, color="red", label="inspect click"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=10, color="orange", label="candidate handover"),
    ]
    ax.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



# [candidate_handover] x=48.548, y=59.740 - C01,C02,D01,D02
# [candidate_handover] x=83.632, y=76.301 - E01,E02,F01,F02
# [candidate_handover] x=122.336, y=81.086 - G01,G02,H01,H02
# [candidate_handover] x=36.096, y=165.033 - C03,C04,D03,D04
# [candidate_handover] x=73.253, y=171.914 - E03,E04,F03,F04
# [candidate_handover] x=114.117, y=173.284 - G03,G04,H03,H04
# [candidate_handover] x=26.728, y=260.673 - A01,A02,B01,B02
# [candidate_handover] x=69.043, y=271.251 - C05,C06,D05,D06,E05,E06,E08,,E09
# [candidate_handover] x=112.451, y=259.898 - F05,F06,F08,F09,G05,G06,G08,G09,H05,H06,H07,H08
# [candidate_handover] x=149.760, y=280.885 - I01,I02,J01,J02,M01
# [candidate_handover] x=63.016, y=354.571 - C07,D07,E07,E10,F07,F10,G07