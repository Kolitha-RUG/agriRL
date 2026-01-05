# test_trained.py
import torch
from vmas import make_env
from vineyard_scenario import VineyardScenario
from train_vineyard_simple import ActorCritic

device = "cpu"  # Use CPU for visualization

# Load trained model
checkpoint = torch.load('best_model.pt', map_location=device)

human_net = ActorCritic(22, 2).to(device)
drone_net = ActorCritic(22, 2).to(device)
human_net.load_state_dict(checkpoint['human'])
drone_net.load_state_dict(checkpoint['drone'])
human_net.eval()
drone_net.eval()

# Create environment
env = make_env(
    scenario=VineyardScenario(),
    num_envs=1,
    device=device,
    continuous_actions=True,
    n_humans=1,
    n_drones=1,
    n_vines=2,
    grapes_per_vine=3,
)

print("Running trained policy...")
obs = env.reset()

total_reward = 0
for step in range(500):
    # Get observations
    human_obs = obs[0]  # Human observation
    drone_obs = obs[1]  # Drone observation
    
    # Get actions from trained networks
    with torch.no_grad():
        human_action = human_net.get_action(human_obs)
        drone_action = drone_net.get_action(drone_obs)
    
    actions = [human_action, drone_action]
    obs, rewards, dones, info = env.step(actions)
    
    total_reward += rewards[0].item() + rewards[1].item()
    
    env.render(mode="human")
    
    if dones.any():
        print(f"Episode done at step {step}! Total reward: {total_reward}")
        break

# env.close()
print(f"Final reward: {total_reward}")