"""
SIMPLE Vineyard Scenario - Step by Step Learning
==================================================

This is the SIMPLEST version:
- 1 Worker who harvests boxes
- 1 Drone who delivers boxes
- No fatigue, no battery (we'll add these later)
"""

import torch
from torch import Tensor
from typing import Dict
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class SimpleVineyard(BaseScenario):
    """
    Simplest vineyard scenario:
    - Worker stands still and harvests boxes
    - Drone learns to pick up boxes and deliver them
    """
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Create the world with 1 worker and 1 drone
        """
        # Simple parameters
        self.harvest_rate = 0.05  # How fast worker creates boxes
        self.agent_radius = 0.1
        self.pickup_distance = 0.2  # How close drone needs to be
        self.delivery_distance = 0.2
        
        # Create world
        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=2.0,  # Field is 4x4
            y_semidim=2.0,
        )
        
        # Add 1 Worker (blue circle)
        worker = Agent(
            name="worker",
            collide=False,
            color=Color.BLUE,
            shape=Sphere(radius=self.agent_radius),
            u_range=[0, 0],  # Can't move (stays in place)
            dynamics=Holonomic(),
        )
        
        # Worker state: how many boxes accumulated
        worker.boxes = torch.zeros(batch_dim, device=device)
        worker.harvest_progress = torch.zeros(batch_dim, device=device)
        
        world.add_agent(worker)
        
        # Add 1 Drone (red circle)
        drone = Agent(
            name="drone",
            collide=False,
            color=Color.RED,
            shape=Sphere(radius=self.agent_radius),
            u_range=[1, 1],  # Can move
            dynamics=Holonomic(),
        )
        
        # Drone state: is it carrying a box?
        drone.carrying = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        
        world.add_agent(drone)
        
        # Add collection point (black square)
        self.collection_point = Landmark(
            name="collection",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.BLACK,
        )
        world.add_landmark(self.collection_point)
        
        # Track total boxes delivered
        self.boxes_delivered = torch.zeros(batch_dim, device=device)
        
        return world
    
    def reset_world_at(self, env_index: int = None):
        """
        Reset positions
        """
        # Worker always at position (-1, 0) - left side of field
        worker = self.world.agents[0]
        if env_index is None:
            pos = torch.tensor([[-1.0, 0.0]], device=self.world.device).repeat(
                self.world.batch_dim, 1
            )
        else:
            pos = torch.tensor([[-1.0, 0.0]], device=self.world.device)
        worker.set_pos(pos, batch_index=env_index)
        
        # Reset worker state
        if env_index is None:
            worker.boxes[:] = 0
            worker.harvest_progress[:] = 0
        else:
            worker.boxes[env_index] = 0
            worker.harvest_progress[env_index] = 0
        
        # Drone starts at collection point (0, 0) - center
        drone = self.world.agents[1]
        if env_index is None:
            pos = torch.tensor([[0.0, 0.0]], device=self.world.device).repeat(
                self.world.batch_dim, 1
            )
        else:
            pos = torch.tensor([[0.0, 0.0]], device=self.world.device)
        drone.set_pos(pos, batch_index=env_index)
        
        # Reset drone state
        if env_index is None:
            drone.carrying[:] = False
        else:
            drone.carrying[env_index] = False
        
        # Collection point at center (0, 0)
        if env_index is None:
            pos = torch.tensor([[0.0, 0.0]], device=self.world.device).repeat(
                self.world.batch_dim, 1
            )
        else:
            pos = torch.tensor([[0.0, 0.0]], device=self.world.device)
        self.collection_point.set_pos(pos, batch_index=env_index)
        
        # Reset deliveries
        if env_index is None:
            self.boxes_delivered[:] = 0
        else:
            self.boxes_delivered[env_index] = 0
    
    def reward(self, agent: Agent):
        """
        Simple reward: +1 for each box delivered
        """
        worker = self.world.agents[0]
        drone = self.world.agents[1]
        
        # Worker harvests boxes
        worker.harvest_progress += self.harvest_rate
        new_box = (worker.harvest_progress >= 1.0).float()
        worker.boxes += new_box
        worker.harvest_progress[worker.harvest_progress >= 1.0] = 0
        
        reward = torch.zeros(self.world.batch_dim, device=self.world.device)
        
        # Check if drone picks up box from worker
        distance_to_worker = torch.linalg.vector_norm(
            drone.state.pos - worker.state.pos, dim=1
        )
        can_pickup = (distance_to_worker < self.pickup_distance) & (worker.boxes > 0) & (~drone.carrying)
        
        # Pickup happens
        drone.carrying = drone.carrying | can_pickup
        worker.boxes = torch.where(can_pickup, worker.boxes - 1, worker.boxes)
        
        # Check if drone delivers box at collection point
        distance_to_collection = torch.linalg.vector_norm(
            drone.state.pos - self.collection_point.state.pos, dim=1
        )
        can_deliver = (distance_to_collection < self.delivery_distance) & drone.carrying
        
        # Delivery happens
        delivered = can_deliver.float()
        self.boxes_delivered += delivered
        reward += delivered  # +1 reward for delivery
        drone.carrying = drone.carrying & (~can_deliver)
        
        return reward
    
    def observation(self, agent: Agent):
        """
        What each agent can see
        """
        worker = self.world.agents[0]
        drone = self.world.agents[1]
        
        if agent.name == "worker":
            # Worker observes:
            # - How many boxes it has
            # - Where the drone is
            obs = torch.cat([
                agent.state.pos,  # Worker position (2D)
                worker.boxes.unsqueeze(-1),  # Number of boxes (1D)
                drone.state.pos - agent.state.pos,  # Relative position of drone (2D)
            ], dim=-1)
            
        else:  # drone
            # Drone observes:
            # - Its own position
            # - Worker position (where boxes are)
            # - Collection point position (where to deliver)
            # - Is it carrying a box?
            obs = torch.cat([
                agent.state.pos,  # Drone position (2D)
                worker.state.pos - agent.state.pos,  # Relative position to worker (2D)
                self.collection_point.state.pos - agent.state.pos,  # Relative to collection (2D)
                drone.carrying.unsqueeze(-1).float(),  # Carrying? (1D)
                worker.boxes.unsqueeze(-1),  # How many boxes at worker (1D)
            ], dim=-1)
        
        return {"obs": obs}
    
    def done(self):
        """
        Episode never ends (will use max_steps)
        """
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)
    
    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        Extra info for logging
        """
        worker = self.world.agents[0]
        drone = self.world.agents[1]
        
        return {
            "boxes_delivered": self.boxes_delivered,
            "worker_boxes": worker.boxes,
            "drone_carrying": drone.carrying.float(),
        }


if __name__ == "__main__":
    """
    Test the simple scenario
    """
    from vmas import make_env
    
    print("=" * 60)
    print("TESTING SIMPLE VINEYARD SCENARIO")
    print("=" * 60)
    
    # Create environment
    env = make_env(
        scenario=SimpleVineyard(),
        num_envs=4,  # Run 4 parallel environments
        device="cpu",
        continuous_actions=True,
        max_steps=100,
    )
    
    print("Environment created!")
    print(f"Agents: {[agent.name for agent in env.agents]}")
    print()
    
    # Reset
    obs = env.reset()
    
    print("Worker observation shape:", obs[0]["obs"].shape)
    print("Drone observation shape:", obs[1]["obs"].shape)
    print()
    
    # Run with random actions for 50 steps
    print("Running 50 steps with random actions...")
    print()
    
    for step in range(50):
        # Worker doesn't move (action ignored)
        worker_action = torch.zeros(4, 2)  # 4 envs, 2D action
        
        # Drone takes random action
        drone_action = torch.randn(4, 2)  # Random movement
        
        actions = [worker_action, drone_action]
        
        obs, rewards, dones, info = env.step(actions)
        
        if step % 10 == 0:
            print(f"Step {step:2d}: "
                  f"Boxes delivered = {info[0]['boxes_delivered'].mean():.1f}, "
                  f"Worker boxes = {info[0]['worker_boxes'].mean():.1f}, "
                  f"Reward = {rewards[0].mean():.2f}")
    
    print()
    print("=" * 60)
    print("Test complete!")
    print("The worker is harvesting boxes.")
    print("The drone is moving randomly (not trained yet).")
    print("Once trained, drone should learn to:")
    print("  1. Go to worker and pick up boxes")
    print("  2. Go to collection point and deliver")
    print("=" * 60)
