import torch
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import World, Agent, Landmark, Sphere
from vmas.simulator.utils import Color

# Mode constants
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
        
        # === PARAMETERS ===
        self.n_humans = kwargs.get("n_humans", 1)
        self.n_drones = kwargs.get("n_drones", 1)
        self.n_vines = kwargs.get("n_vines", 2)
        self.grapes_per_vine = kwargs.get("grapes_per_vine", 3)
        
        # Timing (in steps)
        self.harvest_time = kwargs.get("harvest_time", 10)
        
        # Speeds
        self.human_speed = kwargs.get("human_speed", 0.05)
        self.drone_speed = kwargs.get("drone_speed", 0.15)
        
        # Fatigue
        self.fatigue_rate = kwargs.get("fatigue_rate", 0.01)  # Per step while transporting
        self.fatigue_speed_penalty = kwargs.get("fatigue_speed_penalty", 0.5)  # Max speed reduction
        
        # Distances
        self.interact_dist = kwargs.get("interact_dist", 0.1)
        
        # Store for later
        self.batch_dim = batch_dim
        self.device = device
        self.n_agents = self.n_humans + self.n_drones
        
        # === CREATE WORLD ===
        world = World(batch_dim=batch_dim, device=device)
        
        # === HUMANS (blue) ===
        for i in range(self.n_humans):
            agent = Agent(
                name=f"human_{i}",
                shape=Sphere(radius=0.06),
                color=Color.BLUE,
                u_range=1.0,  # Action range (we'll control movement manually)
            )
            world.add_agent(agent)
        
        # === DRONES (green) ===
        for i in range(self.n_drones):
            agent = Agent(
                name=f"drone_{i}",
                shape=Sphere(radius=0.04),
                color=Color.GREEN,
                u_range=1.0,
            )
            world.add_agent(agent)
        
        # === VINES (red) ===
        self.vines = []
        for i in range(self.n_vines):
            vine = Landmark(
                name=f"vine_{i}",
                shape=Sphere(radius=0.08),
                color=Color.RED,
                collide=False,
            )
            world.add_landmark(vine)
            self.vines.append(vine)
        
        # === COLLECTION POINT (yellow) ===
        self.collection_point = Landmark(
            name="collection_point",
            shape=Sphere(radius=0.1),
            color=Color.YELLOW,
            collide=False,
        )
        world.add_landmark(self.collection_point)
        
        # === STATE VARIABLES ===
        # Grapes remaining at each vine
        self.vine_grapes = torch.full(
            (batch_dim, self.n_vines), self.grapes_per_vine,
            dtype=torch.int, device=device
        )
        
        # Boxes waiting at each vine (for drone pickup)
        self.boxes_waiting = torch.zeros(
            batch_dim, self.n_vines,
            dtype=torch.int, device=device
        )
        
        # Agent states
        self.agent_has_box = torch.zeros(
            batch_dim, self.n_agents,
            dtype=torch.bool, device=device
        )
        
        self.agent_mode = torch.zeros(
            batch_dim, self.n_agents,
            dtype=torch.int, device=device
        )
        
        self.agent_timer = torch.zeros(
            batch_dim, self.n_agents,
            dtype=torch.int, device=device
        )
        
        self.agent_target_vine = torch.zeros(
            batch_dim, self.n_agents,
            dtype=torch.int, device=device
        )
        
        # Fatigue (humans only)
        self.fatigue = torch.zeros(
            batch_dim, self.n_humans,
            dtype=torch.float, device=device
        )
        
        # Reward tracking
        self.deliveries_this_step = torch.zeros(batch_dim, device=device)
        
        return world
    
    def reset_world_at(self, env_index):
        """Reset positions and state when environment restarts."""
        
        # === POSITION COLLECTION POINT (bottom left) ===
        self.collection_point.set_pos(
            torch.tensor([-0.8, -0.8], device=self.device),
            batch_index=env_index,
        )
        
        # === POSITION VINES (spread across field) ===
        for i, vine in enumerate(self.vines):
            # Spread vines horizontally in upper area
            x = -0.5 + i * (1.0 / max(1, self.n_vines - 1)) if self.n_vines > 1 else 0.0
            y = 0.5
            vine.set_pos(
                torch.tensor([x, y], device=self.device),
                batch_index=env_index,
            )
        
        # === POSITION HUMANS (near collection point) ===
        for i in range(self.n_humans):
            self.world.agents[i].set_pos(
                torch.tensor([-0.6 + i * 0.2, -0.6], device=self.device),
                batch_index=env_index,
            )
        
        # === POSITION DRONES (near collection point) ===
        for i in range(self.n_drones):
            agent_idx = self.n_humans + i
            self.world.agents[agent_idx].set_pos(
                torch.tensor([-0.4 + i * 0.2, -0.8], device=self.device),
                batch_index=env_index,
            )
        
        # === RESET STATE VARIABLES ===
        if env_index is None:
            # Reset ALL environments
            self.vine_grapes[:] = self.grapes_per_vine
            self.boxes_waiting[:] = 0
            self.agent_has_box[:] = False
            self.agent_mode[:] = 0  # All IDLE
            self.agent_timer[:] = 0
            self.agent_target_vine[:] = 0
            self.fatigue[:] = 0.0
            self.deliveries_this_step[:] = 0.0
        else:
            # Reset single environment
            self.vine_grapes[env_index] = self.grapes_per_vine
            self.boxes_waiting[env_index] = 0
            self.agent_has_box[env_index] = False
            self.agent_mode[env_index] = 0
            self.agent_timer[env_index] = 0
            self.agent_target_vine[env_index] = 0
            self.fatigue[env_index] = 0.0
            self.deliveries_this_step[env_index] = 0.0

    def observation(self, agent):
        """
        Full state observation for all agents.
        
        Returns tensor of shape [batch_dim, obs_size]
        """
        obs = []
        
        # === 1. Agent's own position ===
        obs.append(agent.state.pos)  # [batch_dim, 2]
        
        # === 2. Collection point position ===
        obs.append(self.collection_point.state.pos)  # [batch_dim, 2]
        
        # === 3. For each vine: position, grapes remaining, boxes waiting ===
        for i, vine in enumerate(self.vines):
            obs.append(vine.state.pos)  # [batch_dim, 2]
            
            # Normalize grapes remaining (0 to 1)
            grapes_norm = self.vine_grapes[:, i:i+1].float() / self.grapes_per_vine
            obs.append(grapes_norm)  # [batch_dim, 1]
            
            # Normalize boxes waiting (0 to 1, cap at grapes_per_vine)
            boxes_norm = self.boxes_waiting[:, i:i+1].float() / self.grapes_per_vine
            obs.append(boxes_norm)  # [batch_dim, 1]
        
        # === 4. For each agent: position, has_box, mode ===
        for i, other_agent in enumerate(self.world.agents):
            obs.append(other_agent.state.pos)  # [batch_dim, 2]
            
            # Has box (0 or 1)
            has_box = self.agent_has_box[:, i:i+1].float()
            obs.append(has_box)  # [batch_dim, 1]
            
            # Mode (normalized)
            mode_norm = self.agent_mode[:, i:i+1].float() / 5.0  # Max mode is ~5
            obs.append(mode_norm)  # [batch_dim, 1]
        
        # === 5. Fatigue for all humans ===
        for i in range(self.n_humans):
            obs.append(self.fatigue[:, i:i+1])  # [batch_dim, 1]
        
        # === 6. Is this agent a human? (agent type indicator) ===
        is_human = 1.0 if agent.name.startswith("human") else 0.0
        is_human_tensor = torch.full(
            (self.batch_dim, 1), is_human, device=self.device
        )
        obs.append(is_human_tensor)  # [batch_dim, 1]
        
        # Concatenate all
        return torch.cat(obs, dim=-1)
    
    def reward(self, agent):
        """
        Common reward with optional shaping.
        """
        reward = self.deliveries_this_step.clone()
        
        # Optional: small penalty for high fatigue (encourages balance)
        # fatigue_penalty = -0.01 * self.fatigue.sum(dim=-1)
        # reward += fatigue_penalty
        
        # Optional: small penalty per step (encourages efficiency)
        # reward -= 0.001
        
        return reward
    
    def done(self):
        """
        Episode ends when:
        - All grapes harvested (vine_grapes = 0)
        - All boxes delivered (none waiting, none being carried)
        """
        total_grapes = self.vine_grapes.sum(dim=-1)        # Grapes left
        total_waiting = self.boxes_waiting.sum(dim=-1)     # Boxes at vines
        total_carrying = self.agent_has_box.sum(dim=-1)    # Boxes being carried
        
        all_done = (total_grapes == 0) & (total_waiting == 0) & (total_carrying == 0)
        
        return all_done
    
    def process_step(self, actions):
        """
        Main state machine. Called every step.
        
        actions: list of action tensors, one per agent
                For humans in DECIDING mode: 0=transport, 1=leave for drone
                For others: ignored (automatic behavior)
        """
        # Reset delivery counter
        self.deliveries_this_step[:] = 0
        
        # Process each agent
        for i in range(self.n_humans):
            self._process_human(i, actions[i] if actions else None)
        
        for i in range(self.n_drones):
            agent_idx = self.n_humans + i
            self._process_drone(agent_idx)

    def _process_human(self, idx, action):
        """Process one human agent."""
        
        agent = self.world.agents[idx]
        mode = self.agent_mode[:, idx]
        
        # === IDLE: Find a vine to go to ===
        idle_mask = (mode == HUMAN_IDLE)
        if idle_mask.any():
            self._human_start_going_to_vine(idx, idle_mask)
        
        # === GOING TO VINE: Move toward target vine ===
        going_mask = (mode == HUMAN_GOING_TO_VINE)
        if going_mask.any():
            self._human_move_to_vine(idx, going_mask)
        
        # === HARVESTING: Wait for timer ===
        harvest_mask = (mode == HUMAN_HARVESTING)
        if harvest_mask.any():
            self._human_harvest(idx, harvest_mask)
        
        # === DECIDING: Apply action from policy ===
        decide_mask = (mode == HUMAN_DECIDING)
        if decide_mask.any() and action is not None:
            self._human_decide(idx, decide_mask, action)
        
        # === TRANSPORTING: Move to collection point ===
        transport_mask = (mode == HUMAN_TRANSPORTING)
        if transport_mask.any():
            self._human_transport(idx, transport_mask)
        
        # === RETURNING: Go back to find next vine ===
        return_mask = (mode == HUMAN_RETURNING)
        if return_mask.any():
            self._human_return(idx, return_mask)

    def _human_start_going_to_vine(self, idx, mask):
        """Find nearest vine with grapes and go there."""
        
        agent = self.world.agents[idx]
        
        # Find vine with grapes remaining
        for v in range(self.n_vines):
            has_grapes = self.vine_grapes[:, v] > 0
            should_go = mask & has_grapes
            
            if should_go.any():
                # Set target vine
                self.agent_target_vine[:, idx] = torch.where(
                    should_go, 
                    torch.tensor(v, device=self.device), 
                    self.agent_target_vine[:, idx]
                )
                # Change mode
                self.agent_mode[:, idx] = torch.where(
                    should_go,
                    torch.tensor(HUMAN_GOING_TO_VINE, device=self.device),
                    self.agent_mode[:, idx]
                )
                # Update mask (handled agents)
                mask = mask & ~should_go
        
        # If no grapes anywhere, stay idle (episode should end soon)

    def _human_move_to_vine(self, idx, mask):
        """Move toward target vine. Start harvesting when arrived."""
        
        agent = self.world.agents[idx]
        target_vine_idx = self.agent_target_vine[:, idx]
        
        # Get target position for each env in batch
        target_pos = torch.zeros(self.batch_dim, 2, device=self.device)
        for v in range(self.n_vines):
            vine_mask = (target_vine_idx == v) & mask
            if vine_mask.any():
                target_pos[vine_mask] = self.vines[v].state.pos[vine_mask]
        
        # Calculate direction
        direction = target_pos - agent.state.pos
        dist = torch.linalg.norm(direction, dim=-1, keepdim=True)
        
        # Normalize direction
        direction_norm = direction / (dist + 1e-6)
        
        # Speed (affected by fatigue)
        speed = self.human_speed * (1 - self.fatigue_speed_penalty * self.fatigue[:, idx:idx+1])
        
        # Move (only for agents in this mode)
        movement = direction_norm * speed
        movement = movement * mask.unsqueeze(-1)  # Zero out non-masked
        
        new_pos = agent.state.pos + movement
        agent.set_pos(new_pos, batch_index=None)
        
        # Check if arrived
        arrived = (dist.squeeze(-1) < self.interact_dist) & mask
        if arrived.any():
            # Start harvesting
            self.agent_mode[:, idx] = torch.where(
                arrived,
                torch.tensor(HUMAN_HARVESTING, device=self.device),
                self.agent_mode[:, idx]
            )
            # Set harvest timer
            self.agent_timer[:, idx] = torch.where(
                arrived,
                torch.tensor(self.harvest_time, device=self.device),
                self.agent_timer[:, idx]
            )

    def _human_harvest(self, idx, mask):
        """Wait for harvest to complete, then enter DECIDING mode."""
        
        # Decrease timer
        self.agent_timer[:, idx] = torch.where(
            mask,
            self.agent_timer[:, idx] - 1,
            self.agent_timer[:, idx]
        )
        
        # Check if done harvesting
        done_harvest = (self.agent_timer[:, idx] <= 0) & mask
        
        if done_harvest.any():
            # Get box
            self.agent_has_box[:, idx] = torch.where(
                done_harvest,
                torch.tensor(True, device=self.device),
                self.agent_has_box[:, idx]
            )
            
            # Decrease grapes at vine
            target_vine_idx = self.agent_target_vine[:, idx]
            for v in range(self.n_vines):
                vine_mask = (target_vine_idx == v) & done_harvest
                self.vine_grapes[:, v] = torch.where(
                    vine_mask,
                    self.vine_grapes[:, v] - 1,
                    self.vine_grapes[:, v]
                )
            
            # Enter DECIDING mode
            self.agent_mode[:, idx] = torch.where(
                done_harvest,
                torch.tensor(HUMAN_DECIDING, device=self.device),
                self.agent_mode[:, idx]
            )

    def _human_decide(self, idx, mask, action):
        """
        Apply human's decision.
        action = 0: transport myself
        action = 1: leave box for drone
        """
        
        # Action 0: Transport
        transport = (action == 0).squeeze(-1) & mask
        if transport.any():
            self.agent_mode[:, idx] = torch.where(
                transport,
                torch.tensor(HUMAN_TRANSPORTING, device=self.device),
                self.agent_mode[:, idx]
            )
        
        # Action 1: Leave for drone
        leave = (action == 1).squeeze(-1) & mask
        if leave.any():
            # Drop box at vine
            self.agent_has_box[:, idx] = torch.where(
                leave,
                torch.tensor(False, device=self.device),
                self.agent_has_box[:, idx]
            )
            
            # Add box to waiting pile
            target_vine_idx = self.agent_target_vine[:, idx]
            for v in range(self.n_vines):
                vine_mask = (target_vine_idx == v) & leave
                self.boxes_waiting[:, v] = torch.where(
                    vine_mask,
                    self.boxes_waiting[:, v] + 1,
                    self.boxes_waiting[:, v]
                )
            
            # Go to next vine (or stay if more grapes here)
            self.agent_mode[:, idx] = torch.where(
                leave,
                torch.tensor(HUMAN_RETURNING, device=self.device),
                self.agent_mode[:, idx]
            )

    def _human_transport(self, idx, mask):
        """Move to collection point. Deliver and add fatigue."""
        
        agent = self.world.agents[idx]
        target_pos = self.collection_point.state.pos
        
        # Direction
        direction = target_pos - agent.state.pos
        dist = torch.linalg.norm(direction, dim=-1, keepdim=True)
        direction_norm = direction / (dist + 1e-6)
        
        # Speed (affected by fatigue)
        speed = self.human_speed * (1 - self.fatigue_speed_penalty * self.fatigue[:, idx:idx+1])
        
        # Move
        movement = direction_norm * speed * mask.unsqueeze(-1)
        new_pos = agent.state.pos + movement
        agent.set_pos(new_pos, batch_index=None)
        
        # Add fatigue while transporting
        self.fatigue[:, idx] = torch.where(
            mask,
            torch.clamp(self.fatigue[:, idx] + self.fatigue_rate, 0, 1),
            self.fatigue[:, idx]
        )
        
        # Check if arrived
        arrived = (dist.squeeze(-1) < self.interact_dist) & mask
        if arrived.any():
            # Deliver box
            self.agent_has_box[:, idx] = torch.where(
                arrived, False, self.agent_has_box[:, idx]
            )
            
            # Count delivery
            self.deliveries_this_step += arrived.float()
            
            # Back to IDLE
            self.agent_mode[:, idx] = torch.where(
                arrived,
                torch.tensor(HUMAN_IDLE, device=self.device),
                self.agent_mode[:, idx]
            )

    def _human_return(self, idx, mask):
        """After leaving box, go back to IDLE to find next vine."""
        
        # Simple: just go back to IDLE immediately
        # The IDLE handler will find the next vine
        self.agent_mode[:, idx] = torch.where(
            mask,
            torch.tensor(HUMAN_IDLE, device=self.device),
            self.agent_mode[:, idx]
        )

    def _process_drone(self, idx):
        """Process one drone agent."""
        
        mode = self.agent_mode[:, idx]
        
        # === IDLE: Find box to pickup ===
        idle_mask = (mode == DRONE_IDLE)
        if idle_mask.any():
            self._drone_find_box(idx, idle_mask)
        
        # === GOING TO PICKUP ===
        pickup_mask = (mode == DRONE_GOING_TO_PICKUP)
        if pickup_mask.any():
            self._drone_move_to_pickup(idx, pickup_mask)
        
        # === DELIVERING ===
        deliver_mask = (mode == DRONE_DELIVERING)
        if deliver_mask.any():
            self._drone_deliver(idx, deliver_mask)

    def _drone_find_box(self, idx, mask):
        """Find nearest vine with waiting boxes."""
        
        agent = self.world.agents[idx]
        
        # Check each vine for waiting boxes
        for v in range(self.n_vines):
            has_boxes = self.boxes_waiting[:, v] > 0
            should_go = mask & has_boxes
            
            if should_go.any():
                self.agent_target_vine[:, idx] = torch.where(
                    should_go, v, self.agent_target_vine[:, idx]
                )
                self.agent_mode[:, idx] = torch.where(
                    should_go,
                    torch.tensor(DRONE_GOING_TO_PICKUP, device=self.device),
                    self.agent_mode[:, idx]
                )
                mask = mask & ~should_go
        
        # If no boxes waiting, stay idle

    def _drone_move_to_pickup(self, idx, mask):
        """Move to vine and pickup box."""
        
        agent = self.world.agents[idx]
        target_vine_idx = self.agent_target_vine[:, idx]
        
        # Get target position
        target_pos = torch.zeros(self.batch_dim, 2, device=self.device)
        for v in range(self.n_vines):
            vine_mask = (target_vine_idx == v) & mask
            if vine_mask.any():
                target_pos[vine_mask] = self.vines[v].state.pos[vine_mask]
        
        # Move
        direction = target_pos - agent.state.pos
        dist = torch.linalg.norm(direction, dim=-1, keepdim=True)
        direction_norm = direction / (dist + 1e-6)
        
        movement = direction_norm * self.drone_speed * mask.unsqueeze(-1)
        new_pos = agent.state.pos + movement
        agent.set_pos(new_pos, batch_index=None)
        
        # Check if arrived
        arrived = (dist.squeeze(-1) < self.interact_dist) & mask
        if arrived.any():
            # Pickup box
            self.agent_has_box[:, idx] = torch.where(
                arrived, True, self.agent_has_box[:, idx]
            )
            
            # Remove from waiting
            for v in range(self.n_vines):
                vine_mask = (target_vine_idx == v) & arrived
                self.boxes_waiting[:, v] = torch.where(
                    vine_mask,
                    self.boxes_waiting[:, v] - 1,
                    self.boxes_waiting[:, v]
                )
            
            # Go to deliver
            self.agent_mode[:, idx] = torch.where(
                arrived,
                torch.tensor(DRONE_DELIVERING, device=self.device),
                self.agent_mode[:, idx]
            )

    def _drone_deliver(self, idx, mask):
        """Move to collection point and deliver."""
        
        agent = self.world.agents[idx]
        target_pos = self.collection_point.state.pos
        
        # Move
        direction = target_pos - agent.state.pos
        dist = torch.linalg.norm(direction, dim=-1, keepdim=True)
        direction_norm = direction / (dist + 1e-6)
        
        movement = direction_norm * self.drone_speed * mask.unsqueeze(-1)
        new_pos = agent.state.pos + movement
        agent.set_pos(new_pos, batch_index=None)
        
        # Check if arrived
        arrived = (dist.squeeze(-1) < self.interact_dist) & mask
        if arrived.any():
            # Deliver
            self.agent_has_box[:, idx] = torch.where(
                arrived, False, self.agent_has_box[:, idx]
            )
            
            # Count delivery
            self.deliveries_this_step += arrived.float()
            
            # Back to IDLE
            self.agent_mode[:, idx] = torch.where(
                arrived,
                torch.tensor(DRONE_IDLE, device=self.device),
                self.agent_mode[:, idx]
            )


