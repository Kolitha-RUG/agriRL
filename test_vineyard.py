# test_vineyard_render.py
import torch
from vmas import make_env
from vineyard_scenario import VineyardScenario

# Create environment (just 1 env for visualization)
env = make_env(
    scenario=VineyardScenario(),
    num_envs=1,  # Single env for rendering
    device="cpu",
    continuous_actions=True,
    n_humans=1,
    n_drones=1,
    n_vines=2,
    grapes_per_vine=3,
)

# Reset
obs = env.reset()
print("Starting visualization...")
print("Close the window to stop.")

# Run with rendering
for step in range(5000):
    actions = env.get_random_actions()
    obs, rewards, dones, info = env.step(actions)
    
    # Render
    env.render(mode="human")
    
    if rewards[0].item() > 0:
        print(f"Step {step}: Delivery!")
    
    if dones.any():
        print(f"Step {step}: Episode complete!")
        obs = env.reset()

