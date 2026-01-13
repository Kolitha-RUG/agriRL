# vineyard_scenario.py

import torch
import numpy as np
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import World, Agent, Landmark, Sphere
from vmas.simulator.utils import Color

# === MODE CONSTANTS ===
HUMAN_IDLE = 0
HUMAN_GOING_TO_VINE = 1
HUMAN_HARVESTING = 2
HUMAN_DECIDING = 3
HUMAN_TRANSPORTING = 4
HUMAN_RETURNING = 5

DRONE_IDLE = 0
DRONE_GOING_TO_PICKUP = 1
DRONE_DELIVERING = 2


class VineyardScenario(BaseScenario):
    
    def make_world(self, batch_dim, device, **kwargs):
        """Create the vineyard world."""
        
        # === LOAD REAL DATA OR USE DEFAULT ===
        excel_file = kwargs.get("excel_file", None)
        
        if excel_file is not None:
            # Load real vine positions from Excel
            import pandas as pd
            df = pd.read_excel(excel_file)
            vine_positions_raw = df[["Position X", "Position Y"]].values
            
            # Subsample if requested
            n_vines_requested = kwargs.get("n_vines", 100)
            if n_vines_requested < len(vine_positions_raw):
                indices = np.linspace(0, len(vine_positions_raw) - 1, n_vines_requested, dtype=int)
                vine_positions_raw = vine_positions_raw[indices]
            
            # Normalize to [0, 1] then scale to [-0.9, 0.9]
            vine_positions_raw = vine_positions_raw - vine_positions_raw.min(axis=0)
            max_dim = vine_positions_raw.max()
            self.vine_positions_np = (vine_positions_raw / max_dim) * 1.8 - 0.9
            self.n_vines = len(self.vine_positions_np)
            self.use_real_positions = True
            print(f"Loaded {self.n_vines} vines from {excel_file}")
        else:
            # Use default generated positions
            self.n_vines = kwargs.get("n_vines", 2)
            self.vine_positions_np = None
            self.use_real_positions = False
        
        # === PARAMETERS ===
        self.n_humans = kwargs.get("n_humans", 1)
        self.n_drones = kwargs.get("n_drones", 1)
        self.grapes_per_vine = kwargs.get("grapes_per_vine", 3)
        self.harvest_time = kwargs.get("harvest_time", 10)
        self.human_speed = kwargs.get("human_speed", 0.03)
        self.drone_speed = kwargs.get("drone_speed", 0.08)
        self.fatigue_rate = kwargs.get("fatigue_rate", 0.005)
        self.fatigue_speed_penalty = kwargs.get("fatigue_speed_penalty", 0.5)
        self.interact_dist = kwargs.get("interact_dist", 0.1)
        self.collection_point_pos = kwargs.get("collection_point", [0.95, 0.0])
        
        self.batch_dim = batch_dim
        self.device = device
        self.n_agents = self.n_humans + self.n_drones
        
        # === CREATE WORLD ===
        world = World(batch_dim=batch_dim, device=device)
        
        # Humans (blue)
        for i in range(self.n_humans):
            agent = Agent(
                name=f"human_{i}",
                shape=Sphere(radius=0.04),
                color=Color.BLUE,
            )
            world.add_agent(agent)
        
        # Drones (green)
        for i in range(self.n_drones):
            agent = Agent(
                name=f"drone_{i}",
                shape=Sphere(radius=0.03),
                color=Color.GREEN,
            )
            world.add_agent(agent)
        
        # Vines (red) - smaller if many vines
        vine_radius = 0.08 if self.n_vines <= 10 else 0.02
        self.vines = []
        for i in range(self.n_vines):
            vine = Landmark(
                name=f"vine_{i}",
                shape=Sphere(radius=vine_radius),
                color=Color.RED,
                collide=False,
            )
            world.add_landmark(vine)
            self.vines.append(vine)
        
        # Collection point (yellow)
        self.collection_point = Landmark(
            name="collection_point",
            shape=Sphere(radius=0.08),
            color=Color.YELLOW,
            collide=False,
        )
        world.add_landmark(self.collection_point)
        
        # === STATE VARIABLES ===
        self.vine_grapes = torch.full(
            (batch_dim, self.n_vines), self.grapes_per_vine,
            dtype=torch.int, device=device
        )
        self.boxes_waiting = torch.zeros(
            batch_dim, self.n_vines, dtype=torch.int, device=device
        )
        self.agent_has_box = torch.zeros(
            batch_dim, self.n_agents, dtype=torch.bool, device=device
        )
        self.agent_mode = torch.zeros(
            batch_dim, self.n_agents, dtype=torch.int, device=device
        )
        self.agent_timer = torch.zeros(
            batch_dim, self.n_agents, dtype=torch.int, device=device
        )
        self.agent_target_vine = torch.zeros(
            batch_dim, self.n_agents, dtype=torch.int, device=device
        )
        self.fatigue = torch.zeros(
            batch_dim, self.n_humans, dtype=torch.float, device=device
        )
        self.deliveries_this_step = torch.zeros(batch_dim, device=device)
        
        return world
    
    
    def reset_world_at(self, env_index):
        """Reset positions and state."""
        
        # === SET VINE POSITIONS ===
        if self.use_real_positions:
            # Use real positions from Excel
            for i, vine in enumerate(self.vines):
                pos = torch.tensor(self.vine_positions_np[i], dtype=torch.float32, device=self.device)
                vine.set_pos(pos, batch_index=env_index)
        else:
            # Generate positions (original behavior)
            for i, vine in enumerate(self.vines):
                x = -0.5 + i * (1.0 / max(1, self.n_vines - 1)) if self.n_vines > 1 else 0.0
                vine.set_pos(
                    torch.tensor([x, 0.5], device=self.device),
                    batch_index=env_index,
                )
        
        # === SET COLLECTION POINT ===
        if self.use_real_positions:
            # Collection point on the right side
            self.collection_point.set_pos(
                torch.tensor(self.collection_point_pos, dtype=torch.float32, device=self.device),
                batch_index=env_index,
            )
        else:
            # Original position (bottom-left)
            self.collection_point.set_pos(
                torch.tensor([-0.8, -0.8], device=self.device),
                batch_index=env_index,
            )
        
        # === POSITION AGENTS NEAR COLLECTION POINT ===
        cp = self.collection_point_pos if self.use_real_positions else [-0.8, -0.8]
        
        for i in range(self.n_humans):
            self.world.agents[i].set_pos(
                torch.tensor([cp[0] - 0.1, cp[1] - 0.1 + i * 0.05], dtype=torch.float32, device=self.device),
                batch_index=env_index,
            )
        
        for i in range(self.n_drones):
            self.world.agents[self.n_humans + i].set_pos(
                torch.tensor([cp[0] - 0.05, cp[1] + 0.1 + i * 0.05], dtype=torch.float32, device=self.device),
                batch_index=env_index,
            )
        
        # === RESET STATE ===
        if env_index is None:
            self.vine_grapes[:] = self.grapes_per_vine
            self.boxes_waiting[:] = 0
            self.agent_has_box[:] = False
            self.agent_mode[:] = 0
            self.agent_timer[:] = 0
            self.agent_target_vine[:] = 0
            self.fatigue[:] = 0.0
            self.deliveries_this_step[:] = 0.0
        else:
            self.vine_grapes[env_index] = self.grapes_per_vine
            self.boxes_waiting[env_index] = 0
            self.agent_has_box[env_index] = False
            self.agent_mode[env_index] = 0
            self.agent_timer[env_index] = 0
            self.agent_target_vine[env_index] = 0
            self.fatigue[env_index] = 0.0
            self.deliveries_this_step[env_index] = 0.0
    
    
    # === REST OF THE FILE STAYS THE SAME ===
    # (observation, reward, done, process_action, process_step, etc.)
    # ... copy everything from line 164 onwards from your original file
    
    
    def observation(self, agent):
        """Full state observation."""
        obs = []
        
        # Own position
        obs.append(agent.state.pos)
        
        # Collection point
        obs.append(self.collection_point.state.pos)
        
        # Vines: position, grapes, boxes waiting
        for i, vine in enumerate(self.vines):
            obs.append(vine.state.pos)
            obs.append(self.vine_grapes[:, i:i+1].float() / self.grapes_per_vine)
            obs.append(self.boxes_waiting[:, i:i+1].float() / self.grapes_per_vine)
        
        # All agents: position, has_box, mode
        for i, other in enumerate(self.world.agents):
            obs.append(other.state.pos)
            obs.append(self.agent_has_box[:, i:i+1].float())
            obs.append(self.agent_mode[:, i:i+1].float() / 5.0)
        
        # Fatigue
        for i in range(self.n_humans):
            obs.append(self.fatigue[:, i:i+1])
        
        # Agent type
        is_human = 1.0 if agent.name.startswith("human") else 0.0
        obs.append(torch.full((self.batch_dim, 1), is_human, device=self.device))
        
        return torch.cat(obs, dim=-1)
    
    
    def reward(self, agent):
        """Common reward. Also runs state machine."""
        
        # Run state machine once per step
        if agent == self.world.agents[0]:
            actions = []
            for a in self.world.agents:
                if hasattr(a, 'action') and a.action.u is not None:
                    actions.append(a.action.u)
                else:
                    actions.append(None)
            self.process_step(actions)
        
        return self.deliveries_this_step.clone()
    
    
    def done(self):
        """Episode done when all boxes delivered."""
        total_grapes = self.vine_grapes.sum(dim=-1)
        total_waiting = self.boxes_waiting.sum(dim=-1)
        total_carrying = self.agent_has_box.sum(dim=-1)
        
        return (total_grapes == 0) & (total_waiting == 0) & (total_carrying == 0)
    
    
    def process_action(self, agent):
        """Override default action processing - we handle movement ourselves."""
        agent.state.force = torch.zeros(self.batch_dim, 2, device=self.device)
        agent.state.vel = torch.zeros(self.batch_dim, 2, device=self.device)
        
    
    
    def process_step(self, actions):
        """Main state machine logic."""
        
        self.deliveries_this_step[:] = 0
        
        # Process humans
        for i in range(self.n_humans):
            action = actions[i] if actions and actions[i] is not None else None
            self._process_human(i, action)
        
        # Process drones
        for i in range(self.n_drones):
            self._process_drone(self.n_humans + i)
    
    
    # ===== HUMAN STATE MACHINE =====
    
    def _process_human(self, idx, action):
        """Process human agent state machine."""
        
        mode = self.agent_mode[:, idx]
        
        # IDLE → find vine
        idle_mask = (mode == HUMAN_IDLE)
        if idle_mask.any():
            self._human_find_vine(idx, idle_mask)
        
        # GOING_TO_VINE → move
        going_mask = (mode == HUMAN_GOING_TO_VINE)
        if going_mask.any():
            self._human_move_to_vine(idx, going_mask)
        
        # HARVESTING → wait
        harvest_mask = (mode == HUMAN_HARVESTING)
        if harvest_mask.any():
            self._human_harvest(idx, harvest_mask)
        
        # DECIDING → apply action
        decide_mask = (mode == HUMAN_DECIDING)
        if decide_mask.any() and action is not None:
            self._human_decide(idx, decide_mask, action)
        
        # TRANSPORTING → move to collection
        transport_mask = (mode == HUMAN_TRANSPORTING)
        if transport_mask.any():
            self._human_transport(idx, transport_mask)
        
        # RETURNING → back to idle
        return_mask = (mode == HUMAN_RETURNING)
        if return_mask.any():
            self.agent_mode[:, idx] = torch.where(
                return_mask, HUMAN_IDLE, self.agent_mode[:, idx]
            )
    
    
    def _human_find_vine(self, idx, mask):
        """Find vine with grapes."""
        for v in range(self.n_vines):
            has_grapes = self.vine_grapes[:, v] > 0
            should_go = mask & has_grapes
            if should_go.any():
                self.agent_target_vine[:, idx] = torch.where(
                    should_go, v, self.agent_target_vine[:, idx]
                )
                self.agent_mode[:, idx] = torch.where(
                    should_go, HUMAN_GOING_TO_VINE, self.agent_mode[:, idx]
                )
                mask = mask & ~should_go
    
    
    def _human_move_to_vine(self, idx, mask):
        """Move toward target vine."""
        agent = self.world.agents[idx]
        target_vine_idx = self.agent_target_vine[:, idx]
        
        # Get target positions
        target_pos = torch.zeros(self.batch_dim, 2, device=self.device)
        for v in range(self.n_vines):
            vine_mask = (target_vine_idx == v) & mask
            target_pos = torch.where(
                vine_mask.unsqueeze(-1), self.vines[v].state.pos, target_pos
            )
        
        # Move
        self._move_agent(agent, target_pos, self.human_speed, 
                         self.fatigue[:, idx], mask)
        
        # Check arrived
        dist = torch.linalg.norm(agent.state.pos - target_pos, dim=-1)
        arrived = (dist < self.interact_dist) & mask
        if arrived.any():
            self.agent_mode[:, idx] = torch.where(
                arrived, HUMAN_HARVESTING, self.agent_mode[:, idx]
            )
            self.agent_timer[:, idx] = torch.where(
                arrived, self.harvest_time, self.agent_timer[:, idx]
            )
    
    
    def _human_harvest(self, idx, mask):
        """Harvest grapes."""
        # Decrease timer
        self.agent_timer[:, idx] = torch.where(
            mask, self.agent_timer[:, idx] - 1, self.agent_timer[:, idx]
        )
        
        # Done harvesting?
        done = (self.agent_timer[:, idx] <= 0) & mask
        if done.any():
            # Get box
            self.agent_has_box[:, idx] = done | self.agent_has_box[:, idx]
            
            # Decrease grapes
            target_v = self.agent_target_vine[:, idx]
            for v in range(self.n_vines):
                vine_mask = (target_v == v) & done
                self.vine_grapes[:, v] = torch.where(
                    vine_mask, self.vine_grapes[:, v] - 1, self.vine_grapes[:, v]
                )
            
            # Enter deciding mode
            self.agent_mode[:, idx] = torch.where(
                done, HUMAN_DECIDING, self.agent_mode[:, idx]
            )
    
    
    def _human_decide(self, idx, mask, action):
        """Apply human decision based on continuous action.
        
        action[:, 0] > 0 means "leave for drone"
        action[:, 0] <= 0 means "transport"
        """
        
        if action is None:
            return
        
        # Get first dimension of action as decision
        # action shape: [batch_dim, 2] for continuous
        if action.dim() == 1:
            decision = action
        else:
            decision = action[:, 0]  # First dimension is our decision
        
        # Transport (decision <= 0)
        transport = (decision <= 0) & mask
        self.agent_mode[:, idx] = torch.where(
            transport, HUMAN_TRANSPORTING, self.agent_mode[:, idx]
        )
        
        # Leave for drone (decision > 0)
        leave = (decision > 0) & mask
        if leave.any():
            # Drop box
            self.agent_has_box[:, idx] = torch.where(
                leave, False, self.agent_has_box[:, idx]
            )
            
            # Add to waiting
            target_v = self.agent_target_vine[:, idx]
            for v in range(self.n_vines):
                vine_mask = (target_v == v) & leave
                self.boxes_waiting[:, v] = torch.where(
                    vine_mask, self.boxes_waiting[:, v] + 1, self.boxes_waiting[:, v]
                )
            
            # Return to find next vine
            self.agent_mode[:, idx] = torch.where(
                leave, HUMAN_RETURNING, self.agent_mode[:, idx]
            )
    def _human_transport(self, idx, mask):
        """Transport box to collection point."""
        agent = self.world.agents[idx]
        target_pos = self.collection_point.state.pos
        
        # Move (with fatigue)
        self._move_agent(agent, target_pos, self.human_speed,
                         self.fatigue[:, idx], mask)
        
        # Add fatigue
        self.fatigue[:, idx] = torch.where(
            mask, 
            torch.clamp(self.fatigue[:, idx] + self.fatigue_rate, 0, 1),
            self.fatigue[:, idx]
        )
        
        # Check arrived
        dist = torch.linalg.norm(agent.state.pos - target_pos, dim=-1)
        arrived = (dist < self.interact_dist) & mask
        if arrived.any():
            # Deliver
            self.agent_has_box[:, idx] = torch.where(
                arrived, False, self.agent_has_box[:, idx]
            )
            self.deliveries_this_step += arrived.float()
            self.agent_mode[:, idx] = torch.where(
                arrived, HUMAN_IDLE, self.agent_mode[:, idx]
            )
    
    
    # ===== DRONE STATE MACHINE =====
    
    def _process_drone(self, idx):
        """Process drone agent state machine."""
        
        mode = self.agent_mode[:, idx]
        
        # IDLE → find box
        idle_mask = (mode == DRONE_IDLE)
        if idle_mask.any():
            self._drone_find_box(idx, idle_mask)
        
        # GOING_TO_PICKUP → move
        pickup_mask = (mode == DRONE_GOING_TO_PICKUP)
        if pickup_mask.any():
            self._drone_pickup(idx, pickup_mask)
        
        # DELIVERING → move to collection
        deliver_mask = (mode == DRONE_DELIVERING)
        if deliver_mask.any():
            self._drone_deliver(idx, deliver_mask)
    
    
    def _drone_find_box(self, idx, mask):
        """Find vine with waiting boxes."""
        for v in range(self.n_vines):
            has_boxes = self.boxes_waiting[:, v] > 0
            should_go = mask & has_boxes
            if should_go.any():
                self.agent_target_vine[:, idx] = torch.where(
                    should_go, v, self.agent_target_vine[:, idx]
                )
                self.agent_mode[:, idx] = torch.where(
                    should_go, DRONE_GOING_TO_PICKUP, self.agent_mode[:, idx]
                )
                mask = mask & ~should_go
    
    
    def _drone_pickup(self, idx, mask):
        """Move to vine and pickup box."""
        agent = self.world.agents[idx]
        target_vine_idx = self.agent_target_vine[:, idx]
        
        # Get target
        target_pos = torch.zeros(self.batch_dim, 2, device=self.device)
        for v in range(self.n_vines):
            vine_mask = (target_vine_idx == v) & mask
            target_pos = torch.where(
                vine_mask.unsqueeze(-1), self.vines[v].state.pos, target_pos
            )
        
        # Move
        self._move_agent(agent, target_pos, self.drone_speed, None, mask)
        
        # Check arrived
        dist = torch.linalg.norm(agent.state.pos - target_pos, dim=-1)
        arrived = (dist < self.interact_dist) & mask
        if arrived.any():
            # Pickup
            self.agent_has_box[:, idx] = arrived | self.agent_has_box[:, idx]
            
            # Remove from waiting
            for v in range(self.n_vines):
                vine_mask = (target_vine_idx == v) & arrived
                self.boxes_waiting[:, v] = torch.where(
                    vine_mask, self.boxes_waiting[:, v] - 1, self.boxes_waiting[:, v]
                )
            
            self.agent_mode[:, idx] = torch.where(
                arrived, DRONE_DELIVERING, self.agent_mode[:, idx]
            )
    
    
    def _drone_deliver(self, idx, mask):
        """Deliver box to collection point."""
        agent = self.world.agents[idx]
        target_pos = self.collection_point.state.pos
        
        # Move
        self._move_agent(agent, target_pos, self.drone_speed, None, mask)
        
        # Check arrived
        dist = torch.linalg.norm(agent.state.pos - target_pos, dim=-1)
        arrived = (dist < self.interact_dist) & mask
        if arrived.any():
            self.agent_has_box[:, idx] = torch.where(
                arrived, False, self.agent_has_box[:, idx]
            )
            self.deliveries_this_step += arrived.float()
            self.agent_mode[:, idx] = torch.where(
                arrived, DRONE_IDLE, self.agent_mode[:, idx]
            )
    
    
    # ===== HELPER =====
    
    def _move_agent(self, agent, target_pos, base_speed, fatigue, mask):
        """Move agent toward target position."""
        
        direction = target_pos - agent.state.pos
        dist = torch.linalg.norm(direction, dim=-1, keepdim=True)
        direction_norm = direction / (dist + 1e-6)
        
        # Apply fatigue penalty if provided
        if fatigue is not None:
            speed = base_speed * (1 - self.fatigue_speed_penalty * fatigue.unsqueeze(-1))
        else:
            speed = base_speed
        
        # Move
        movement = direction_norm * speed
        movement = movement * mask.unsqueeze(-1).float()
        
        new_pos = agent.state.pos + movement
        agent.set_pos(new_pos, batch_index=None)