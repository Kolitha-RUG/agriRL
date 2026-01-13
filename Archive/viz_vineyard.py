# visualize_vineyard.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === READ FROM EXCEL ===
# Change this to your actual file path
excel_file = "Topo.xlsx"  # or "vineyard_data.xls" or .csv

# Read the excel file
df = pd.read_excel(excel_file)

print("=== Raw Data ===")
print(df.head(10))
print(f"\nColumns: {df.columns.tolist()}")

# Extract X and Y positions (adjust column names if different)
# Assuming columns are named "Position X", "Position Y", "Position Z"
x_col = "Position X"
y_col = "Position Y"

vines = df[[x_col, y_col]].values

# Normalize to start from (0, 0)
vines_normalized = vines - vines.min(axis=0)

print("\n=== Vineyard Statistics ===")
print(f"Number of vines: {len(vines)}")
print(f"Original X range: {vines[:, 0].min():.2f} to {vines[:, 0].max():.2f}")
print(f"Original Y range: {vines[:, 1].min():.2f} to {vines[:, 1].max():.2f}")
print(f"Field width (X): {vines[:, 0].max() - vines[:, 0].min():.2f} meters")
print(f"Field height (Y): {vines[:, 1].max() - vines[:, 1].min():.2f} meters")

# === PLOT ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original coordinates
ax1 = axes[0]
ax1.scatter(vines[:, 0], vines[:, 1], c='red', s=10, marker='o', label='Vines')
ax1.set_xlabel('X (meters)')
ax1.set_ylabel('Y (meters)')
ax1.set_title(f'Original Vineyard Layout ({len(vines)} vines)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Normalized coordinates
ax2 = axes[1]
ax2.scatter(vines_normalized[:, 0], vines_normalized[:, 1], c='red', s=10, marker='o', label='Vines')

# Suggested collection point (bottom center)
collection_point = np.array([
    vines_normalized[:, 0].max(), 
    vines_normalized[:, 1].mean() - 10
])
ax2.scatter(*collection_point, c='yellow', s=200, marker='s', 
            edgecolors='black', linewidths=2, label='Collection Point (suggested)')

ax2.set_xlabel('X (meters)')
ax2.set_ylabel('Y (meters)')
ax2.set_title('Normalized Vineyard Layout')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('vineyard_layout.png', dpi=150)
plt.show()

print(f"\nSuggested collection point: ({collection_point[0]:.2f}, {collection_point[1]:.2f})")
print(f"\nPlot saved to: vineyard_layout.png")