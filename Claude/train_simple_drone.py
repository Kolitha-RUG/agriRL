"""
SIMPLE TRAINING for Simple Vineyard
====================================

This trains just the DRONE to:
1. Go to worker
2. Pick up box
3. Go to collection point
4. Deliver box

The worker doesn't need training (it just stands still and harvests).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from vmas import make_env
from simple_vineyard import SimpleVineyard
import numpy as np


# Simple neural network for the drone
class DronePolicy(nn.Module):
    """
    Simple network: observations -> actions
    """
    def __init__(self, obs_size=8, action_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            nn.Tanh(),  # Actions between -1 and 1
        )
    
    def forward(self, obs):
        return self.net(obs)


def train_simple(num_steps=10000, num_envs=8, device="cpu"):
    """
    Train the drone using simple policy gradient
    
    Args:
        num_steps: How many training steps
        num_envs: How many parallel environments
        device: "cpu" or "cuda"
    """
    
    print("=" * 60)
    print("SIMPLE TRAINING - Drone learns to deliver boxes")
    print("=" * 60)
    print(f"Training for {num_steps} steps")
    print(f"Using {num_envs} parallel environments")
    print(f"Device: {device}")
    print("=" * 60)
    print()
    
    # Create environment
    env = make_env(
        scenario=SimpleVineyard(),
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        max_steps=100,
    )
    
    print("Environment created!")
    print()
    
    # Create drone policy
    policy = DronePolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    # Training loop
    obs = env.reset()
    
    episode_rewards = []
    episode_boxes = []
    
    print("Starting training...")
    print()
    
    for step in range(num_steps):
        # Worker doesn't move
        worker_action = torch.zeros(num_envs, 2, device=device)
        
        # Drone action from policy
        drone_obs = obs[1]["obs"]  # Get drone observations
        drone_action = policy(drone_obs)
        
        # Step environment
        actions = [worker_action, drone_action]
        next_obs, rewards, dones, info = env.step(actions)
        
        # Simple training: move in direction that gave reward
        # This is called "REINFORCE" algorithm
        reward = rewards[0]  # Reward is same for all agents

        # Compute loss (negative reward = we want to maximize reward)
        # Reshape reward to [num_envs, 1] to match drone_action shape [num_envs, 2]
        loss = -(reward.unsqueeze(-1) * drone_action).mean()
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        obs = next_obs
        
        # Logging
        if step % 100 == 0:
            boxes = info[0]["boxes_delivered"].mean().item()
            avg_reward = reward.mean().item()
            
            episode_rewards.append(avg_reward)
            episode_boxes.append(boxes)
            
            if step % 1000 == 0:
                recent_boxes = np.mean(episode_boxes[-10:]) if len(episode_boxes) >= 10 else boxes
                print(f"Step {step:5d}: Boxes = {recent_boxes:.2f}, Reward = {avg_reward:.3f}")
    
    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    final_boxes = np.mean(episode_boxes[-20:])
    print(f"Final performance: {final_boxes:.2f} boxes delivered per episode")
    print()
    
    # Save model
    torch.save(policy.state_dict(), "simple_drone_policy.pt")
    print("Model saved as: simple_drone_policy.pt")
    print("=" * 60)
    
    return policy


def test_trained_policy(policy_path="simple_drone_policy.pt", num_episodes=5):
    """
    Test a trained policy
    """
    print("=" * 60)
    print("TESTING TRAINED DRONE")
    print("=" * 60)
    
    # Create environment
    env = make_env(
        scenario=SimpleVineyard(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        max_steps=100,
    )
    
    # Load policy
    policy = DronePolicy()
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    
    print("Policy loaded!")
    print(f"Running {num_episodes} test episodes...")
    print()
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_boxes = 0
        
        for step in range(100):
            worker_action = torch.zeros(1, 2)
            
            with torch.no_grad():
                drone_action = policy(obs[1]["obs"])
            
            actions = [worker_action, drone_action]
            obs, rewards, dones, info = env.step(actions)
            
            total_boxes = info[0]["boxes_delivered"].item()
        
        print(f"Episode {episode + 1}: {total_boxes:.0f} boxes delivered")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple drone training")
    parser.add_argument("--train", action="store_true", help="Train the drone")
    parser.add_argument("--test", action="store_true", help="Test trained drone")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--envs", type=int, default=8, help="Parallel environments")
    
    args = parser.parse_args()
    
    if args.test:
        test_trained_policy()
    else:
        # Default: train
        train_simple(num_steps=args.steps, num_envs=args.envs)
