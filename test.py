# test_vineyard.py
import torch
from vmas import make_env
from vineyard_scenario import VineyardScenario

# Create environment
env = make_env(
    scenario=VineyardScenario(),
    num_envs=4,
    device="cpu",
    continuous_actions=False,  # Discrete actions!
    n_humans=1,
    n_drones=1,
    n_vines=2,
    grapes_per_vine=3,
)

# Reset
obs = env.reset()
print(f"Observation shape: {obs[0].shape}")

# Run some steps with random actions
for step in range(200):
    # Random actions: 0 or 1 for humans, 0 for drones
    actions = []
    for i, agent in enumerate(env.agents):
        if agent.name.startswith("human"):
            action = torch.randint(0, 2, (4,))  # 0=transport, 1=leave
        else:
            action = torch.zeros(4, dtype=torch.long)  # Drone: no choice
        actions.append(action)
    
    obs, rewards, dones, info = env.step(actions)
    
    if rewards[0].sum() > 0:
        print(f"Step {step}: Delivery! Reward = {rewards[0]}")
    
    if dones.any():
        print(f"Step {step}: Episode done!")
        break

print("Test complete!")