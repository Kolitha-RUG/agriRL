# test_vineyard.py
import torch
from vmas import make_env
from vineyard_scenario import VineyardScenario

# Create environment with REAL vineyard data
env = make_env(
    scenario=VineyardScenario(),
    num_envs=1,
    device="cpu",
    continuous_actions=True,
    # Real data parameters
    excel_file="Topo.xlsx",  # <-- Load from Excel!
    n_vines=100,             # Use 100 vines
    n_humans=3,
    n_drones=2,
    grapes_per_vine=1,
    collection_point=[0.95, 0.0],  # Right side of field
)

obs = env.reset()
print("Starting visualization with REAL vineyard...")

for step in range(2000):
    actions = env.get_random_actions()
    obs, rewards, dones, info = env.step(actions)
    
    env.render(mode="human")
    
    if rewards[0].item() > 0:
        print(f"Step {step}: Delivery!")
    
    if dones.any():
        print(f"Step {step}: Episode complete!")
        obs = env.reset()

env.close()