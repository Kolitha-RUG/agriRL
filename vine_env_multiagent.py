import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import pygame
from typing import Dict, List, Optional, Tuple, Any

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# === Human actions ===
ACTION_HARVEST = 0
ACTION_TRANSPORT = 1
ACTION_ENQUEUE = 2

# === Drone status ===
DRONE_IDLE = 0
DRONE_GO_TO_VINE = 1
DRONE_DELIVER = 2


def compute_num_vines(topology_mode: str, vineyard_file: str) -> int:
    """Compute number of vines based on topology mode."""
    df = pd.read_excel(vineyard_file)
    if topology_mode == "full":
        return len(df)
    elif topology_mode == "row":
        return df["lot"].nunique()
    else:
        raise ValueError(f"Unknown topology mode: {topology_mode}")


def one_hot(idx: int, size: int) -> np.ndarray:
    """Create one-hot encoded vector."""
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v


class Vine:
    """Represents a vine or row of vines in the vineyard."""
    
    def __init__(self, position: Tuple[float, float], max_boxes: int):
        self.position = np.array(position, dtype=np.float32)
        self.total_boxes = int(max_boxes)
        self.boxes_remaining = int(max_boxes)
        self.queued_boxes = 0

    def harvest_box(self) -> bool:
        """Remove 1 available box from the vine."""
        if self.boxes_remaining > 0:
            self.boxes_remaining -= 1
            return True
        return False


class Human:
    """Represents a human worker agent."""
    
    def __init__(self, position: Tuple[float, float], assigned_vine: int):
        self.position = np.array(position, dtype=np.float32)
        self.assigned_vine = int(assigned_vine)
        self.fatigue = 0.0
        self.has_box = False
        self.current_action = ACTION_HARVEST
        self.busy = False
        self.time_left = 0.0


class Drone:
    """Represents a drone agent."""
    
    def __init__(self, position: Tuple[float, float]):
        self.position = np.array(position, dtype=np.float32)
        self.status = DRONE_IDLE
        self.has_box = False
        self.busy = False
        self.time_left = 0.0
        self.target_vine: Optional[int] = None


class MultiAgentVineEnv(MultiAgentEnv):
    """
    Multi-Agent Vineyard Harvesting Environment.
    
    Each human worker is an independent learning agent. Agents must coordinate
    to efficiently harvest grapes, queue boxes for drone pickup, or transport
    boxes manually to the collection point.
    
    Agents:
        - human_0, human_1, ..., human_N: Human worker agents
        
    Actions (per agent):
        - 0: HARVEST - Harvest a box from assigned vine
        - 1: TRANSPORT - Transport held box to collection point
        - 2: ENQUEUE - Queue held box for drone pickup
        
    Observations (per agent):
        Each agent receives a personalized observation including:
        - All vine positions and states (shared)
        - Collection point position (shared)
        - Own state (position, fatigue, action, has_box, assigned_vine)
        - Other agents' observable states
        - Drone states (shared)
        
    Rewards:
        - Individual rewards based on agent's contribution to deliveries
        - Penalties for backlog and fatigue
    """
    
    metadata = {"render_modes": ["terminal", "human"], "render_fps": 100}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Parse config with defaults
        config = config or {}
        self.render_mode = config.get("render_mode", "terminal")
        self.topology_mode = config.get("topology_mode", "row")
        self.num_humans = config.get("num_humans", 2)
        self.num_drones = config.get("num_drones", 1)
        self.max_boxes_per_vine = config.get("max_boxes_per_vine", 10)
        self.max_backlog = config.get("max_backlog", 10)
        self.max_steps = config.get("max_steps", 200)
        self.dt = float(config.get("dt", 1.0))
        self.harvest_time = float(config.get("harvest_time", 8.0))
        self.human_speed = float(config.get("human_speed", 1.0))
        self.drone_speed = float(config.get("drone_speed", 2.0))
        self.vineyard_file = config.get("vineyard_file", "data/Vinha_Maria_Teresa_RL.xlsx")
        
        # Reward shaping parameters
        self.reward_delivery = config.get("reward_delivery", 1.0)
        self.reward_backlog_penalty = config.get("reward_backlog_penalty", 0.01)
        self.reward_fatigue_penalty = config.get("reward_fatigue_penalty", 0.001)
        
        # Compute environment dimensions
        self.num_vines = compute_num_vines(self.topology_mode, self.vineyard_file)
        self.num_actions = 3
        self.num_drone_status = 3
        
        # Define agent IDs
        self.possible_agents = [f"human_{i}" for i in range(self.num_humans)]
        self.agents = self.possible_agents.copy()
        
        # Calculate observation dimension per agent
        # Each agent observes:
        # - Vine positions: num_vines * 2
        # - Collection point: 2
        # - Boxes remaining per vine: num_vines (normalized)
        # - Queued boxes per vine: num_vines (normalized)
        # - Own position: 2
        # - Own fatigue: 1
        # - Own current action (one-hot): num_actions
        # - Own has_box: 1
        # - Own assigned_vine (one-hot): num_vines
        # - Other humans' positions: (num_humans - 1) * 2
        # - Other humans' has_box: (num_humans - 1)
        # - Drone positions: num_drones * 2
        # - Drone status (one-hot): num_drones * num_drone_status
        # - Drone has_box: num_drones
        
        self.obs_dim = (
            self.num_vines * 2  # vine positions
            + 2  # collection point
            + self.num_vines  # boxes remaining
            + self.num_vines  # queued boxes
            + 2  # own position
            + 1  # own fatigue
            + self.num_actions  # own action one-hot
            + 1  # own has_box
            + self.num_vines  # own assigned_vine one-hot
            + (self.num_humans - 1) * 2  # other humans' positions
            + (self.num_humans - 1)  # other humans' has_box
            + self.num_drones * 2  # drone positions
            + self.num_drones * self.num_drone_status  # drone status
            + self.num_drones  # drone has_box
        )
        
        # Define observation and action spaces for each agent
        self.observation_spaces = {
            agent_id: spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent_id in self.possible_agents
        }
        
        self.action_spaces = {
            agent_id: spaces.Discrete(self.num_actions)
            for agent_id in self.possible_agents
        }
        
        # Initialize state variables (will be set in reset)
        self.vines: List[Vine] = []
        self.humans: List[Human] = []
        self.drones: List[Drone] = []
        self.collection_point = np.array([0.5, 0.5], dtype=np.float32)
        self.field_size = np.array([1.0, 1.0], dtype=np.float32)
        self.x_min = 0.0
        self.y_min = 0.0
        self.steps = 0
        self.delivered = 0
        
        # Pygame rendering
        self._pygame_initialized = False
        self._screen = None
        self._clock = None

    def _load_vineyard(self, file_path: str) -> pd.DataFrame:
        """Load vineyard data from Excel file."""
        df = pd.read_excel(file_path)
        df["x"] = df["x"] / 1000.0
        df["y"] = df["y"] / 1000.0
        df["z"] = df["z"] / 1000.0
        df["x"] -= df["x"].min()
        df["y"] -= df["y"].min()
        return df

    def _build_full_vines(self, df: pd.DataFrame) -> List[Vine]:
        """Build vines in full topology mode (one vine per data point)."""
        vines = []
        for _, r in df.iterrows():
            v = Vine(position=(r.x, r.y), max_boxes=self.max_boxes_per_vine)
            v.line = r.line
            v.lot = r.lot
            v.z = r.z
            vines.append(v)
        return vines

    def _build_row_vines(self, df: pd.DataFrame) -> List[Vine]:
        """Build vines in row topology mode (one vine per lot/row)."""
        vines = []
        grouped = df.groupby("lot")
        
        for lot_id, g in grouped:
            x_mean = g["x"].mean()
            y_mean = g["y"].mean()
            z_mean = g["z"].mean()
            total_boxes = len(g) * self.max_boxes_per_vine
            
            v = Vine(position=(x_mean, y_mean), max_boxes=total_boxes)
            v.lot = lot_id
            v.z = z_mean
            v.n_vines = len(g)
            vines.append(v)
            
        return vines

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Returns:
            observations: Dict mapping agent_id to observation array
            infos: Dict mapping agent_id to info dict
        """
        super().reset(seed=seed)
        
        # Load vineyard data
        df = self._load_vineyard(self.vineyard_file)
        
        # Build vines based on topology mode
        if self.topology_mode == "full":
            self.vines = self._build_full_vines(df)
        elif self.topology_mode == "row":
            self.vines = self._build_row_vines(df)
        else:
            raise ValueError(f"Unknown topology_mode: {self.topology_mode}")
        
        # Calculate field dimensions
        xs = np.array([v.position[0] for v in self.vines])
        ys = np.array([v.position[1] for v in self.vines])
        self.x_min = xs.min()
        self.y_min = ys.min()
        self.field_size = np.array([xs.max() - self.x_min, ys.max() - self.y_min], dtype=np.float32)
        
        # Collection point at right edge, center height
        self.collection_point = np.array([xs.max(), np.mean(ys)], dtype=np.float32)
        
        # Initialize counters
        self.steps = 0
        self.delivered = 0
        
        # Initialize humans at their assigned vines
        self.humans = []
        for h in range(self.num_humans):
            assigned = h % self.num_vines
            pos = self.vines[assigned].position.copy()
            self.humans.append(Human(pos, assigned))
        
        # Initialize drones at collection point
        self.drones = [Drone(self.collection_point.copy()) for _ in range(self.num_drones)]
        
        # Reset active agents
        self.agents = self.possible_agents.copy()
        
        # Build observations for all agents
        observations = {agent_id: self._get_obs_for_agent(i) for i, agent_id in enumerate(self.agents)}
        infos = {agent_id: {} for agent_id in self.agents}
        
        return observations, infos

    def step(self, action_dict: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one environment step.
        
        Args:
            action_dict: Dict mapping agent_id to action (0, 1, or 2)
            
        Returns:
            observations: Dict mapping agent_id to observation
            rewards: Dict mapping agent_id to reward
            terminateds: Dict with agent terminations and "__all__" key
            truncateds: Dict with agent truncations and "__all__" key
            infos: Dict mapping agent_id to info dict
        """
        self.steps += 1
        delivered_before = self.delivered
        
        # Track individual contributions for reward attribution
        individual_deliveries = [0] * self.num_humans
        
        # 1) Progress ongoing actions (timers) for humans
        for i, h in enumerate(self.humans):
            if h.busy:
                h.time_left -= self.dt
                h.fatigue = float(np.clip(h.fatigue + 0.002 * self.dt, 0.0, 1.0))
                
                if h.time_left <= 0.0:
                    h.busy = False
                    h.time_left = 0.0
                    
                    if h.current_action == ACTION_HARVEST:
                        h.has_box = True
                    elif h.current_action == ACTION_TRANSPORT:
                        if h.has_box:
                            h.has_box = False
                            self.delivered += 1
                            individual_deliveries[i] = 1
                        h.position = self.collection_point.copy()
        
        # 2) Apply new decisions for humans that are free
        for i, agent_id in enumerate(self.agents):
            h = self.humans[i]
            
            if h.busy:
                continue
            
            # Get action from action_dict (default to harvest if not provided)
            a = action_dict.get(agent_id, ACTION_HARVEST)
            h.current_action = a
            vine = self.vines[h.assigned_vine]
            
            if a == ACTION_HARVEST:
                if (not h.has_box) and vine.boxes_remaining > 0:
                    ok = vine.harvest_box()
                    if ok:
                        h.busy = True
                        h.time_left = self.harvest_time
                        h.position = vine.position.copy()
                        
            elif a == ACTION_ENQUEUE:
                if h.has_box and vine.queued_boxes < self.max_backlog:
                    vine.queued_boxes += 1
                    h.has_box = False
                    h.position = vine.position.copy()
                    
            elif a == ACTION_TRANSPORT:
                if h.has_box:
                    dist = float(np.linalg.norm(h.position - self.collection_point))
                    travel_time = dist / max(self.human_speed, 1e-6)
                    h.busy = True
                    h.time_left = travel_time
        
        # 3) Progress drone timers
        for d in self.drones:
            if d.busy:
                d.time_left -= self.dt
                if d.time_left <= 0.0:
                    d.busy = False
                    d.time_left = 0.0
                    
                    if d.status == DRONE_GO_TO_VINE:
                        v = self.vines[d.target_vine]
                        d.position = v.position.copy()
                        if v.queued_boxes > 0:
                            v.queued_boxes -= 1
                            d.has_box = True
                            d.status = DRONE_DELIVER
                            dist = float(np.linalg.norm(d.position - self.collection_point))
                            d.busy = True
                            d.time_left = dist / max(self.drone_speed, 1e-6)
                        else:
                            d.status = DRONE_IDLE
                            d.target_vine = None
                            
                    elif d.status == DRONE_DELIVER:
                        d.position = self.collection_point.copy()
                        if d.has_box:
                            d.has_box = False
                            self.delivered += 1
                        d.status = DRONE_IDLE
                        d.target_vine = None
        
        # 4) Assign idle drones to nearest queued vine
        for d in self.drones:
            if (not d.busy) and d.status == DRONE_IDLE:
                candidates = [idx for idx, v in enumerate(self.vines) if v.queued_boxes > 0]
                if candidates:
                    dists = [np.linalg.norm(self.vines[idx].position - d.position) for idx in candidates]
                    target = candidates[int(np.argmin(dists))]
                    d.target_vine = target
                    d.status = DRONE_GO_TO_VINE
                    dist = float(np.linalg.norm(self.vines[target].position - d.position))
                    d.busy = True
                    d.time_left = dist / max(self.drone_speed, 1e-6)
        
        # Calculate rewards
        delivered_delta = self.delivered - delivered_before
        backlog_total = sum(v.queued_boxes for v in self.vines)
        
        # Build per-agent rewards
        rewards = {}
        for i, agent_id in enumerate(self.agents):
            h = self.humans[i]
            # Base reward from team deliveries (shared)
            team_reward = self.reward_delivery * delivered_delta / self.num_humans
            # Bonus for individual direct deliveries
            individual_bonus = self.reward_delivery * individual_deliveries[i]
            # Penalties
            backlog_penalty = self.reward_backlog_penalty * backlog_total / self.num_humans
            fatigue_penalty = self.reward_fatigue_penalty * h.fatigue
            
            rewards[agent_id] = team_reward + individual_bonus - backlog_penalty - fatigue_penalty
        
        # Check termination conditions
        all_harvested = all(v.boxes_remaining == 0 for v in self.vines)
        no_queue = all(v.queued_boxes == 0 for v in self.vines)
        no_carry = all(not h.has_box for h in self.humans) and all(not d.has_box for d in self.drones)
        all_idle = all(not h.busy for h in self.humans) and all(not d.busy for d in self.drones)
        
        terminated = bool(all_harvested and no_queue and no_carry and all_idle)
        truncated = bool(self.steps >= self.max_steps)
        
        # Build termination/truncation dicts
        terminateds = {agent_id: terminated for agent_id in self.agents}
        terminateds["__all__"] = terminated
        
        truncateds = {agent_id: truncated for agent_id in self.agents}
        truncateds["__all__"] = truncated
        
        # Build observations
        observations = {agent_id: self._get_obs_for_agent(i) for i, agent_id in enumerate(self.agents)}
        
        # Build infos
        infos = {
            agent_id: {
                "delivered": self.delivered,
                "backlog_total": backlog_total,
                "individual_delivery": individual_deliveries[i],
            }
            for i, agent_id in enumerate(self.agents)
        }
        
        return observations, rewards, terminateds, truncateds, infos

    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Normalize a position to [0, 1] range."""
        return np.array([
            (pos[0] - self.x_min) / max(self.field_size[0], 1e-6),
            (pos[1] - self.y_min) / max(self.field_size[1], 1e-6),
        ], dtype=np.float32)

    def _get_obs_for_agent(self, agent_idx: int) -> np.ndarray:
        """
        Build observation for a specific agent.
        
        The observation includes:
        - Shared environment state (vines, collection point, drones)
        - Agent's own state (privileged information)
        - Other agents' observable states
        """
        obs = []
        h = self.humans[agent_idx]
        
        # === SHARED STATE ===
        
        # Vine positions (normalized)
        for v in self.vines:
            obs.extend(self._normalize_position(v.position))
        
        # Collection point (normalized)
        obs.extend(self._normalize_position(self.collection_point))
        
        # Boxes remaining per vine (normalized)
        max_boxes_actual = max(v.total_boxes for v in self.vines) if self.vines else 1
        for v in self.vines:
            obs.append(v.boxes_remaining / max(max_boxes_actual, 1))
        
        # Queued boxes per vine (normalized)
        for v in self.vines:
            obs.append(v.queued_boxes / max(self.max_backlog, 1))
        
        # === OWN STATE ===
        
        # Own position
        obs.extend(self._normalize_position(h.position))
        
        # Own fatigue
        obs.append(h.fatigue)
        
        # Own current action (one-hot)
        obs.extend(one_hot(h.current_action, self.num_actions))
        
        # Own has_box
        obs.append(1.0 if h.has_box else 0.0)
        
        # Own assigned_vine (one-hot)
        obs.extend(one_hot(h.assigned_vine, self.num_vines))
        
        # === OTHER AGENTS' STATES ===
        
        for other_idx, other_h in enumerate(self.humans):
            if other_idx != agent_idx:
                # Other's position
                obs.extend(self._normalize_position(other_h.position))
        
        for other_idx, other_h in enumerate(self.humans):
            if other_idx != agent_idx:
                # Other's has_box
                obs.append(1.0 if other_h.has_box else 0.0)
        
        # === DRONE STATES ===
        
        for d in self.drones:
            obs.extend(self._normalize_position(d.position))
        
        for d in self.drones:
            obs.extend(one_hot(d.status, self.num_drone_status))
        
        for d in self.drones:
            obs.append(1.0 if d.has_box else 0.0)
        
        return np.array(obs, dtype=np.float32)

    def render(self):
        """Render the environment."""
        if self.render_mode == "terminal":
            self._render_terminal()
        elif self.render_mode == "human":
            self._render_pygame()

    def _render_terminal(self):
        """Text-based rendering to terminal."""
        print(f"Step {self.steps} | delivered={self.delivered}")
        for i, v in enumerate(self.vines):
            print(f"  Vine {i}: rem={v.boxes_remaining} queued={v.queued_boxes}")
        for i, h in enumerate(self.humans):
            agent_id = f"human_{i}"
            print(f"  {agent_id}: vine={h.assigned_vine} busy={h.busy} "
                  f"t={h.time_left:.1f} has_box={h.has_box} fat={h.fatigue:.2f}")
        for i, d in enumerate(self.drones):
            print(f"  Drone {i}: status={d.status} busy={d.busy} "
                  f"t={d.time_left:.1f} has_box={d.has_box}")
        print("=" * 60)

    def _render_pygame(self):
        """Pygame visual rendering."""
        SCREEN_W, SCREEN_H = 800, 800
        PADDING = 40
        
        if not self._pygame_initialized:
            pygame.init()
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Multi-Agent Vine Environment")
            self._clock = pygame.time.Clock()
            self._pygame_initialized = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_initialized = False
                return
        
        self._screen.fill((240, 240, 240))
        
        fx = max(self.field_size[0], 1e-6)
        fy = max(self.field_size[1], 1e-6)
        scale = min((SCREEN_W - 2 * PADDING) / fx, (SCREEN_H - 2 * PADDING) / fy)
        
        def world_to_screen(pos):
            x = (pos[0] - self.x_min) * scale + PADDING
            y = (pos[1] - self.y_min) * scale + PADDING
            x = max(0, min(SCREEN_W - 1, x))
            y = max(0, min(SCREEN_H - 1, y))
            return int(x), int(y)
        
        # Collection point (gold)
        cp = world_to_screen(self.collection_point)
        pygame.draw.circle(self._screen, (255, 215, 0), cp, 10)
        
        # Vines (green)
        for v in self.vines:
            x, y = world_to_screen(v.position)
            pygame.draw.rect(self._screen, (34, 139, 34), (x, y, 10, 10))
        
        # Humans (blue with different shades for each agent)
        colors = [(30, 144, 255), (65, 105, 225), (0, 0, 205), (25, 25, 112)]
        for i, h in enumerate(self.humans):
            x, y = world_to_screen(h.position)
            color = colors[i % len(colors)]
            pygame.draw.circle(self._screen, color, (x, y), 8)
        
        # Drones (red triangles)
        for d in self.drones:
            x, y = world_to_screen(d.position)
            pygame.draw.polygon(
                self._screen,
                (220, 20, 60),
                [(x, y - 8), (x - 8, y + 8), (x + 8, y + 8)],
            )
        
        pygame.display.flip()
        self._clock.tick(10)

    def close(self):
        """Clean up resources."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False


# ============================================================================
# Training Example
# ============================================================================

def example_training_config():
    """
    Example RLlib configuration for training with the multi-agent environment.
    
    This demonstrates how to set up PPO training with multiple policies.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.tune.registry import register_env
    
    # Register the environment
    register_env(
        "MultiAgentVineEnv",
        lambda config: MultiAgentVineEnv(config)
    )
    
    # Environment config
    env_config = {
        "topology_mode": "row",
        "num_humans": 2,
        "num_drones": 1,
        "max_boxes_per_vine": 10,
        "max_steps": 200,
        "render_mode": "terminal",
    }
    
    # Create a temporary env to get agent IDs
    temp_env = MultiAgentVineEnv(env_config)
    agent_ids = temp_env.possible_agents
    temp_env.close()
    
    # Option 1: Shared policy (all agents use the same policy)
    def shared_policy_mapping(agent_id, episode, **kwargs):
        return "shared_policy"
    
    config_shared = (
        PPOConfig()
        .environment(env="MultiAgentVineEnv", env_config=env_config)
        .multi_agent(
            policy_mapping_fn=shared_policy_mapping,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(),
                }
            ),
        )
    )
    
    # Option 2: Independent policies (each agent has its own policy)
    def independent_policy_mapping(agent_id, episode, **kwargs):
        return agent_id  # Each agent uses its own policy
    
    config_independent = (
        PPOConfig()
        .environment(env="MultiAgentVineEnv", env_config=env_config)
        .multi_agent(
            policy_mapping_fn=independent_policy_mapping,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    agent_id: RLModuleSpec() for agent_id in agent_ids
                }
            ),
        )
    )
    
    return config_shared, config_independent


def test_environment():
    """Test the multi-agent environment with random actions."""
    print("Testing MultiAgentVineEnv...")
    
    config = {
        "topology_mode": "row",
        "num_humans": 2,
        "num_drones": 1,
        "max_boxes_per_vine": 5,
        "max_steps": 5000,
        "render_mode": "human",
    }
    
    env = MultiAgentVineEnv(config)
    
    print(f"\nPossible agents: {env.possible_agents}")
    print(f"Observation space: {env.observation_spaces}")
    print(f"Action space: {env.action_spaces}")
    
    obs, info = env.reset()
    print(f"\nInitial observations keys: {obs.keys()}")
    print(f"Observation shape for human_0: {obs['human_0'].shape}")
    
    total_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
    done = False
    step = 0
    
    while not done:
        # Sample random actions for all agents
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in env.agents
        }
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward
        
        if step % 10 == 0:
            env.render()
        
        done = terminateds["__all__"] or truncateds["__all__"]
        step += 1
    
    print(f"\nEpisode finished after {step} steps")
    print(f"Total rewards: {total_rewards}")
    print(f"Total delivered: {env.delivered}")
    
    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_environment()
