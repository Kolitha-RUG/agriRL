import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import pygame
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# === Human actions ===
ACTION_HARVEST = 0
ACTION_TRANSPORT = 1
ACTION_ENQUEUE = 2
ACTION_REST = 3
# === Drone status ===
DRONE_IDLE = 0
DRONE_GO_TO_VINE = 1
DRONE_DELIVER = 2
DRONE_GO_TO_CHARGE = 3
DRONE_CHARGE = 4


DEFAULT_HANDOVER_POINTS_XY = [
    (48.548, 59.740),
    (83.632, 76.301),
    (122.336, 81.086),
    (36.096, 165.033),
    (73.253, 171.914),
    (114.117, 173.284),
    (26.728, 260.673),
    (69.043, 271.251),
    (112.451, 259.898),
    (149.760, 280.885),
    (63.016, 354.571),
]

def compute_num_vines(topology_mode: str, vineyard_file: str) -> int:
    """
    Compute number of work units based on topology mode.

    full  -> one plant per point
    line  -> one work unit per (lot, line)
    row   -> legacy alias for line
    """
    df = pd.read_excel(vineyard_file)

    if topology_mode == "full":
        return len(df)
    elif topology_mode in ("line", "row"):
        df["lot"] = df["lot"].astype(str).str.strip()
        df["line"] = df["line"].astype(str).str.strip()
        return int(df[["lot", "line"]].drop_duplicates().shape[0])
    else:
        raise ValueError(f"Unknown topology mode: {topology_mode}")


def one_hot(idx: int, size: int) -> np.ndarray:
    """Create one-hot encoded vector."""
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v


class Vine:
    """
    Represents a LINE work unit.

    For now we keep the class name `Vine` to avoid a large refactor.
    Semantically this now stores line-level harvest flow in kg.
    """

    def __init__(self, 
                 
        position: Tuple[float, float], total_kg: float, box_capacity_kg: float):
        self.position = np.array(position, dtype=np.float32)

        # Harvest stock
        self.total_kg = float(total_kg)
        self.kg_remaining = float(total_kg)

        # Harvested material not yet sent out
        self.kg_buffer = 0.0                  # harvested kg not yet converted into service units
        self.box_capacity_kg = float(box_capacity_kg)

        # Service units waiting at the line
        self.boxes_ready = 0                  # full boxes ready for carry/enqueue
        self.queued_boxes = 0                 # queued for drone pickup
        self.queue_contributors = deque()

        # Final partial-box simplification
        self.final_partial_released = False
        self.final_partial_box_kg = 0.0

        # Useful for normalization / KPI summaries
        self.total_boxes_equivalent = int(
            np.ceil(self.total_kg / max(self.box_capacity_kg, 1e-6))
        )

    def add_harvested_kg(self, amount_kg: float) -> int:
        """
        Add harvested kg into the line buffer and convert it into ready boxes.

        Returns:
            number of newly created ready boxes
        """
        amount = float(np.clip(amount_kg, 0.0, self.kg_remaining))
        self.kg_remaining -= amount
        self.kg_buffer += amount

        created = 0

        # Full boxes
        while self.kg_buffer + 1e-9 >= self.box_capacity_kg:
            self.kg_buffer -= self.box_capacity_kg
            self.boxes_ready += 1
            created += 1

        # Final partial box at line completion (Phase 4 simplification)
        if (
            self.kg_remaining <= 1e-9
            and (not self.final_partial_released)
            and self.kg_buffer > 1e-9
        ):
            self.boxes_ready += 1
            self.final_partial_released = True
            self.final_partial_box_kg = self.kg_buffer
            self.kg_buffer = 0.0
            created += 1

        return created

    def take_ready_box(self) -> bool:
        """Worker takes one ready box from the line."""
        if self.boxes_ready > 0:
            self.boxes_ready -= 1
            return True
        return False


class Human:
    """Represents a human worker agent."""
    
    def __init__(self, position: Tuple[float, float], assigned_vine: int):
        self.position = np.array(position, dtype=np.float32)
        self.assigned_vine = int(assigned_vine)
        self.fatigue = 0.0

        # Still keep this name for minimal refactor:
        self.has_box = False
        self.carried_box_kg = 0.0

        self.current_action = ACTION_HARVEST
        self.busy = False
        self.time_left = 0.0
        self.delivered_count = 0
        self.transport_fatigue_multiplier = 1.0

        # Phase 3: kg produced by the currently running harvest action
        self.pending_harvest_kg = 0.0
class Drone:
    """Represents a drone agent."""
    
    def __init__(self, position: Tuple[float, float]):
        self.position = np.array(position, dtype=np.float32)
        self.status = DRONE_IDLE
        self.has_box = False
        self.busy = False
        self.time_left = 0.0
        self.target_vine: Optional[int] = None
        self.delivered_count = 0
        self.battery = 100.0
        self.last_contributor: Optional[int] = None



class MultiAgentVineEnvAsync(MultiAgentEnv):
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
        - 3: REST - Take a rest to reduce fatigue
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

    def __init__(
        self,
        handover_points_xy: Optional[List[Tuple[float, float]]] = None,
        render_mode: str = "terminal",
        topology_mode: str = "line",
        num_humans: int = 2,
        num_drones: int = 1,
        yield_per_plant_kg: float = 0.6,
        box_capacity_kg: float = 8.0,
        harvest_rate_kg_s: float = 0.004,
        max_backlog: int = 10,
        max_steps: int = 2000,
        dt: float = 5.0,
        harvest_time: float = 300.0,
        enqueue_time: float = 1.0,
        rest_time: float = 5.0,
        human_speed: float = 1.0,
        drone_speed: float = 5.0,
        vineyard_file: str = "data/Vinha_Maria_Teresa_RL.xlsx",
        local_vine_k: int = 6,
        reward_delivery: float = 1.0,
        reward_backlog_penalty: float = 0.5,
        reward_fatigue_inc_penalty: float = 0.8,
        reward_harvest: float = 0.08,
        reward_enqueue: float = 0.10,
        reward_drone_credit: float = 1.0,
        reward_fatigue_level_penalty: float = 2.0


    ):
        super().__init__()
        
        # Direct assignments (no dict)
        self.render_mode = render_mode
        self.topology_mode = topology_mode
        self.num_humans = num_humans
        self.num_drones = num_drones
        self.yield_per_plant_kg = float(yield_per_plant_kg)
        self.box_capacity_kg = float(box_capacity_kg)
        self.harvest_rate_kg_s = float(harvest_rate_kg_s)
        self.max_backlog = max_backlog
        self.max_steps = max_steps
        self.dt = float(dt) #seconds
        self.harvest_time = float(harvest_time)
        self.enqueue_time = float(enqueue_time)
        self.rest_time = float(rest_time)
        self.human_speed = float(human_speed) #m/s
        self.drone_speed = float(drone_speed) #m/s
        self.vineyard_file = vineyard_file
        self._base_df = self._load_vineyard(self.vineyard_file)
        self.local_vine_k = int(local_vine_k)

        self.handover_points_xy = (
                list(handover_points_xy)
                if handover_points_xy is not None
                else list(DEFAULT_HANDOVER_POINTS_XY)
            )

        self.rest_fatigue_threshold = 0.6

        self.drone_flight_time_full = 16.0  # seconds (960)
        self.drone_batt_drain_rate = 100.0 / self.drone_flight_time_full  # % per second
        self.drone_charge_time_full = 16.0  # or use real charge time if you have it
        self.drone_batt_charge_rate = 100.0 / self.drone_charge_time_full  # % per second

        self.reward_delivery = float(reward_delivery)
        self.reward_backlog_penalty = float(reward_backlog_penalty)
        self.reward_fatigue_inc_penalty = float(reward_fatigue_inc_penalty)
        self.reward_harvest = float(reward_harvest)
        self.reward_enqueue = float(reward_enqueue)
        self.reward_drone_credit = float(reward_drone_credit)
        self.reward_fatigue_level_penalty = float(reward_fatigue_level_penalty)


        if self.topology_mode == "full":
            self.num_vines = len(self._base_df)
        elif self.topology_mode in ("line", "row"):
            self.num_vines = int(self._base_df[["lot", "line"]].drop_duplicates().shape[0])
        else:
            raise ValueError(f"Unknown topology mode: {self.topology_mode}")
        self.num_actions = 4
        self.num_drone_status = 5

        self.possible_agents = [f"human_{i}" for i in range(self.num_humans)]
        self.agents = self.possible_agents.copy()
        self.agent_index = {agent_id: i for i, agent_id in enumerate(self.possible_agents)}
        self.queue_contributors = deque()

        self.obs_dim = (
                    self.local_vine_k * 2              # local vine x,y positions
                    + self.local_vine_k                # local vine z
                    + 2                                # collection point x,y
                    + 1                                # collection point z
                    + 2                                # charging point x,y
                    + 1                                # charging point z
                    + self.local_vine_k                # local kg_remaining
                    + self.local_vine_k                # local service units (ready + queued)
                    + 2                                # own x,y position
                    + 1                                # own z position
                    + 1                                # fatigue
                    + self.num_actions                 # current action one-hot
                    + 1                                # own has_box
                    + 1                                # assigned_vine as scalar
                    + (self.num_humans - 1) * 2        # other humans x,y
                    + (self.num_humans - 1)            # other humans z
                    + (self.num_humans - 1)            # other humans has_box
                    + self.num_drones * 2              # drone x,y positions
                    + self.num_drones                  # drone z positions
                    + self.num_drones * self.num_drone_status
                    + self.num_drones                  # drone has_box
                    + self.num_drones                  # drone battery
                )
        # Flat obs: [obs_vector, action_mask]
        self.observation_spaces = {
            agent_id: spaces.Dict({
                "obs": spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(self.num_actions,), dtype=np.float32),
            })
            for agent_id in self.possible_agents
        }

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0.0, high=1.0, shape=(self.num_actions,), dtype=np.float32),
        })

        self.action_spaces = {
            agent_id: spaces.Discrete(self.num_actions)
            for agent_id in self.possible_agents
        }

        self.action_space = spaces.Discrete(self.num_actions)

        # init state vars (unchanged)
        self.vines = []
        self.humans = []
        self.drones = []
        self.collection_point = np.array([0.5, 0.5], dtype=np.float32)
        self.charging_point = np.array([0.0, 0.0], dtype=np.float32) # Will be updated in reset
        self.field_size = np.array([1.0, 1.0], dtype=np.float32)
        self.x_min = 0.0
        self.y_min = 0.0
        self.collection_z = 0.0
        self.charging_z = 0.0
        self.z_min = 0.0
        self.z_range = 1.0

        # slope / fatigue tuning
        self.human_transport_fatigue_rate = 0.4   # base fatigue rate during transport
        self.slope_time_factor = 1.5              # increase travel time with slope grade
        self.slope_fatigue_factor = 2.0           # increase fatigue with uphill slope
        self.steps = 0
        self.delivered = 0

        self._pygame_initialized = False
        self._screen = None
        self._clock = None

    def _load_vineyard(self, file_path: str) -> pd.DataFrame:
        """
        Load vineyard data from Excel file.

        Assumptions for Phase 2:
        - x, y, z are already in meters
        - keep real metric scale
        - shift x and y so the map starts at (0, 0)
        - keep z in real elevation meters
        """
        df = pd.read_excel(file_path).copy()

        required_cols = {"lot", "line", "x", "y", "z"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in vineyard file: {missing}")

        df["lot"] = df["lot"].astype(str).str.strip()
        df["line"] = df["line"].astype(str).str.strip()

        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df["z"] = pd.to_numeric(df["z"], errors="coerce")

        df = df.dropna(subset=["lot", "line", "x", "y", "z"]).reset_index(drop=True)

        # IMPORTANT: keep metric units, do NOT divide by 1000
        df["x"] = df["x"] - df["x"].min()
        df["y"] = df["y"] - df["y"].min()

        # Repeated line IDs only make sense inside a lot
        df["line_key"] = df["lot"] + "::" + df["line"]

        return df

    def _build_full_vines(self, df: pd.DataFrame) -> List[Vine]:
        """Build plant-level work units in full topology mode."""
        vines = []
        for _, r in df.iterrows():
            total_kg = float(self.yield_per_plant_kg)

            v = Vine(
                position=(r.x, r.y),
                total_kg=total_kg,
                box_capacity_kg=self.box_capacity_kg,
            )
            v.line = r.line
            v.lot = r.lot
            v.z = r.z
            vines.append(v)

        return vines

    def _build_line_vines(self, df: pd.DataFrame) -> List[Vine]:
        """
        Build work units in line topology mode:
        one unit per unique (lot, line).

        NOTE:
        We keep the existing variable name `vines` for now to avoid a huge refactor.
        Semantically, each `Vine` object below is actually a LINE work unit.
        """
        vines = []

        grouped = df.groupby(["lot", "line"], sort=True)

        for (lot_id, line_id), g in grouped:
            g = g.sort_values(["y", "x"]).reset_index(drop=True)

            xy = g[["x", "y"]].to_numpy(dtype=np.float32)
            start_xy = xy[0]
            end_xy = xy[-1]
            centroid_xy = xy.mean(axis=0)

            z_mean = float(g["z"].mean())
            line_length_m = float(np.linalg.norm(end_xy - start_xy))
            n_plants = int(len(g))

            line_total_kg = float(n_plants * self.yield_per_plant_kg)

            v = Vine(
                position=(float(centroid_xy[0]), float(centroid_xy[1])),
                total_kg=line_total_kg,
                box_capacity_kg=self.box_capacity_kg,
            )
            v.lot = lot_id
            v.line = line_id
            v.line_key = f"{lot_id}::{line_id}"

            v.z = z_mean
            v.n_plants = n_plants
            v.line_length_m = line_length_m
            v.start_position = np.array(start_xy, dtype=np.float32)
            v.end_position = np.array(end_xy, dtype=np.float32)

            vines.append(v)

        return vines
    
    def _build_handover_points(self) -> None:
        """
        Build designated handover points from manually selected XY coordinates.

        For each handover point:
        - keep the given XY
        - estimate Z from the nearest line centroid
        - initialize queue state

        For each line:
        - assign nearest handover point automatically
        """
        self.handover_points = []

        if len(self.vines) == 0:
            return

        line_xy = np.array([v.position for v in self.vines], dtype=np.float32)
        line_z = np.array([v.z for v in self.vines], dtype=np.float32)

        # 1) Create handover nodes
        for hid, (x, y) in enumerate(self.handover_points_xy):
            hp_xy = np.array([x, y], dtype=np.float32)

            # nearest line -> use its z as handover z
            dists = np.linalg.norm(line_xy - hp_xy, axis=1)
            nearest_idx = int(np.argmin(dists))
            hp_z = float(line_z[nearest_idx])

            self.handover_points.append({
                "id": hid,
                "position": hp_xy,
                "z": hp_z,
                "queued_boxes": 0,
                "queue_contributors": deque(),
                "line_indices": [],
            })

    # 2) Assign each line to the nearest handover node
    for line_idx, v in enumerate(self.vines):
        hp_positions = np.array([hp["position"] for hp in self.handover_points], dtype=np.float32)
        dists = np.linalg.norm(hp_positions - v.position, axis=1)
        nearest_hid = int(np.argmin(dists))

        v.handover_id = nearest_hid
        self.handover_points[nearest_hid]["line_indices"].append(line_idx)

    def _line_has_open_harvest(self, v: Vine) -> bool:
        return v.kg_remaining > 1e-9

    def _line_has_ready_box_work(self, v: Vine) -> bool:
        return v.boxes_ready > 0

    def _line_has_any_work(self, v: Vine) -> bool:
        return (
            v.kg_remaining > 1e-9
            or v.kg_buffer > 1e-9
            or v.boxes_ready > 0
            or v.queued_boxes > 0
        )
    
    def _find_next_vine(self, from_pos: np.ndarray, exclude_vines: Optional[set] = None) -> Optional[int]:
        if exclude_vines is None:
            exclude_vines = set()

        candidates = [
            i for i, v in enumerate(self.vines)
            if self._line_has_open_harvest(v) and i not in exclude_vines
        ]

        if not candidates:
            return None

        dists = [np.linalg.norm(self.vines[i].position - from_pos) for i in candidates]
        return candidates[int(np.argmin(dists))]

    def _ready_agent_ids(self) -> List[str]:
        return [
            agent_id
            for agent_id, i in self.agent_index.items()
            if not self.humans[i].busy
        ]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Returns:
            observations: Dict mapping agent_id to observation array
            infos: Dict mapping agent_id to info dict
        """
        super().reset(seed=seed)
        
        # Load vineyard data
        # Reuse cached vineyard data loaded once in __init__
        df = self._base_df
        
        # Build vines based on topology mode
        if self.topology_mode == "full":
            self.vines = self._build_full_vines(df)
        elif self.topology_mode in ("line", "row"):
            self.vines = self._build_line_vines(df)
        else:
            raise ValueError(f"Unknown topology_mode: {self.topology_mode}")
        

        # Build designated handover points and assign each line to the nearest one
        self._build_handover_points()
        
        xs = np.array([v.position[0] for v in self.vines], dtype=np.float32)
        ys = np.array([v.position[1] for v in self.vines], dtype=np.float32)
        zs = np.array([v.z for v in self.vines], dtype=np.float32)
        # after vines/humans/drones are initialized
        
        self.x_min = float(xs.min())
        self.y_min = float(ys.min())
        self.z_min = float(zs.min())

        self.field_size = np.array(
            [xs.max() - self.x_min, ys.max() - self.y_min],
            dtype=np.float32
        )
        self.z_range = float(max(zs.max() - self.z_min, 1e-6))
        
        # Collection point at right edge, center height
        self.collection_point = np.array([xs.max(), np.mean(ys)], dtype=np.float32)
        self.charging_point = np.array([self.x_min, self.y_min], dtype=np.float32)

        # choose realistic z levels for non-vine locations
        self.collection_z = float(np.min(zs))
        self.charging_z = float(zs.max())
        

        # --- static per-episode caches for faster observations ---
        self.vine_xy_norm = np.array(
            [self._normalize_position(v.position) for v in self.vines],
            dtype=np.float32
        )

        self.vine_z_norm = np.array(
            [self._normalize_z(v.z) for v in self.vines],
            dtype=np.float32
        )

        self.collection_xy_norm = self._normalize_position(self.collection_point).astype(np.float32)
        self.collection_z_norm = np.float32(self._normalize_z(self.collection_z))

        self.charging_xy_norm = self._normalize_position(self.charging_point).astype(np.float32)
        self.charging_z_norm = np.float32(self._normalize_z(self.charging_z))

        self.max_line_kg_actual = max(v.total_kg for v in self.vines) if self.vines else 1.0
        self.max_boxes_equivalent_actual = max(v.total_boxes_equivalent for v in self.vines) if self.vines else 1

        # Keep the old variable name to avoid touching too many KPI lines later.
        # It now means total initial box-equivalent service units, not literal boxes.
        self.initial_total_boxes = int(sum(v.total_boxes_equivalent for v in self.vines))
        self.initial_total_kg = float(sum(v.total_kg for v in self.vines))
        # Initialize counters
        self.steps = 0
        self.delivered = 0
        # --- KPI accumulators (steps-based) ---
        self.ep_step_count = 0
        self.ep_delivered_total = 0  # will accumulate delivered_delta
        self.ep_backlog_sum = 0
        self.ep_backlog_peak = 0

        self.ep_human_rest_steps = 0
        self.ep_human_busy_steps = 0

        self.ep_drone_flying_steps = 0
        # --- episode counters (for TensorBoard) ---
        self.ep_human_action_steps = np.zeros((self.num_humans, self.num_actions), dtype=np.int32)
        self.ep_human_action_seconds = np.zeros((self.num_humans, self.num_actions), dtype=np.float32)

        self.ep_drone_status_steps = np.zeros((self.num_drones, self.num_drone_status), dtype=np.int32)
        self.ep_drone_status_seconds = np.zeros((self.num_drones, self.num_drone_status), dtype=np.float32)

        self.ep_fatigue_increase_total = 0.0
        self.ep_delivered_delta_total = 0

        # --- fatigue KPIs (steps-based) ---
        self.ep_fatigue_sum = 0.0          # sum over steps of mean fatigue across humans
        self.ep_fatigue_peak = 0.0         # max fatigue seen (any human, any time)
        self.ep_fatigue_hi_steps = 0        # optional: count of human-steps above threshold
        self.fatigue_hi_threshold = 0.7     # choose threshold

        # --- reward-term breakdown (episode accumulators) ---
        self.ep_r_delivery_sum = 0.0
        self.ep_r_fatigue_inc_sum = 0.0
        self.ep_r_backlog_sum = 0.0
        self.ep_r_fatigue_level_sum = 0.0
        self.ep_reward_sum = 0.0
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
        self.pending_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
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
        # print(self.steps)
        # Track individual contributions for reward attribution
        individual_deliveries = [0] * self.num_humans
        harvest_events = [0] * self.num_humans
        enqueue_events = [0] * self.num_humans
        drone_credit_deliveries = [0] * self.num_humans
        fatigue_before = [h.fatigue for h in self.humans]
        # --- count what humans are doing this timestep ---
        # --- count only actual human decisions this timestep ---
        for agent_id, a in action_dict.items():
            i = self.agent_index[agent_id]
            a = int(a)
            self.ep_human_action_steps[i, a] += 1
            self.ep_human_action_seconds[i, a] += self.dt
        # 1) Progress ongoing actions (timers) for humans
        for i, h in enumerate(self.humans):
            if h.busy:
                h.time_left -= self.dt
                # Fatigue accumulates/recover while action is being executed
                if h.current_action == ACTION_HARVEST:
                    h.fatigue = float(np.clip(h.fatigue + 0.2 * self.dt, 0.0, 1.0))
                elif h.current_action == ACTION_TRANSPORT:
                    mult = getattr(h, "transport_fatigue_multiplier", 1.0)
                    fatigue_rate = self.human_transport_fatigue_rate * mult
                    h.fatigue = float(np.clip(h.fatigue + fatigue_rate * self.dt, 0.0, 1.0))
                elif h.current_action == ACTION_REST:
                    h.fatigue = float(np.clip(h.fatigue - 0.2 * self.dt, 0.0, 1.0))
                
                if h.time_left <= 0.0:
                    h.busy = False
                    h.time_left = 0.0

                    line = self.vines[h.assigned_vine]

                    if h.current_action == ACTION_HARVEST:
                        # Harvest action completes by adding kg to the line buffer
                        line.add_harvested_kg(h.pending_harvest_kg)
                        h.pending_harvest_kg = 0.0

                    elif h.current_action == ACTION_TRANSPORT:
                        if h.has_box:
                            h.has_box = False
                            h.carried_box_kg = 0.0
                            self.delivered += 1
                            h.delivered_count += 1
                            individual_deliveries[i] = 1

                        h.position = self.collection_point.copy()
                        h.transport_fatigue_multiplier = 1.0

            if (not h.busy) and (not h.has_box):
                cur_v = self.vines[h.assigned_vine]

                # Only leave if the current line has no harvest left and no ready box left
                if (not self._line_has_open_harvest(cur_v)) and (cur_v.boxes_ready <= 0):
                    assigned_vines = {hh.assigned_vine for hh in self.humans}

                    nxt = self._find_next_vine(cur_v.position, exclude_vines=assigned_vines)

                    if nxt is None:
                        nxt = self._find_next_vine(from_pos=cur_v.position)

                    if nxt is not None:
                        h.assigned_vine = nxt
                        h.position = self.vines[nxt].position.copy()
        
        # 2) Apply new decisions only for agents that actually acted
        for agent_id, a in action_dict.items():
            i = self.agent_index[agent_id]
            h = self.humans[i]

            if h.busy:
                continue

            line = self.vines[h.assigned_vine]

            a = int(a)
            h.current_action = a

            can_harvest = (not h.has_box) and (line.kg_remaining > 1e-9)
            can_take_ready_box = (not h.has_box) and (line.boxes_ready > 0)
            can_transport = h.has_box or can_take_ready_box
            can_enqueue = (h.has_box or can_take_ready_box) and (line.queued_boxes < self.max_backlog)

            can_do_productive = can_harvest or can_transport or can_enqueue
            can_rest = (h.fatigue >= self.rest_fatigue_threshold) or (not can_do_productive)

            if a == ACTION_REST and not can_rest:
                continue

            if a != ACTION_REST and not can_do_productive:
                continue

            if a == ACTION_HARVEST:
                if can_harvest:
                    harvest_events[i] += 1
                    h.busy = True

                    fatigue_slowdown = 1.0 + 0.8 * h.fatigue
                    h.time_left = self.harvest_time * fatigue_slowdown

                    # Amount of kg this harvest action contributes
                    h.pending_harvest_kg = min(
                        line.kg_remaining,
                        self.harvest_rate_kg_s * self.harvest_time,
                    )

                    h.position = line.position.copy()

            elif a == ACTION_ENQUEUE:
                if can_enqueue:
                    # If the worker is not already holding a box, pick one ready box from the line
                    if not h.has_box:
                        ok = line.take_ready_box()
                        if not ok:
                            continue
                        h.has_box = True
                        h.carried_box_kg = self.box_capacity_kg

                    line.queued_boxes += 1
                    line.queue_contributors.append(i)
                    enqueue_events[i] += 1

                    h.busy = True
                    h.time_left = self.enqueue_time
                    h.has_box = False
                    h.carried_box_kg = 0.0
                    h.position = line.position.copy()

            elif a == ACTION_TRANSPORT:
                if can_transport:
                    # If not already carrying, pick one ready box from the line first
                    if not h.has_box:
                        ok = line.take_ready_box()
                        if not ok:
                            continue
                        h.has_box = True
                        h.carried_box_kg = self.box_capacity_kg

                    start_xyz = self._get_human_xyz(h)
                    end_xyz = self._get_collection_xyz()
                    travel_time, fatigue_multiplier = self._human_transport_costs(start_xyz, end_xyz)

                    h.busy = True
                    h.time_left = travel_time
                    h.transport_fatigue_multiplier = fatigue_multiplier

            elif a == ACTION_REST:
                h.busy = True
                h.time_left = self.rest_time

        fatigue_after = [h.fatigue for h in self.humans]
        fatigue_increase = [max(0.0, fa - fb) for fa, fb in zip(fatigue_after, fatigue_before)]
        # 3) Progress drone timers
        for d in self.drones:
            if d.busy:
                d.time_left -= self.dt
                # Drain battery during flight (time-based)
                if d.status in (DRONE_GO_TO_VINE, DRONE_DELIVER, DRONE_GO_TO_CHARGE):
                    d.battery = max(0.0, d.battery - self.drone_batt_drain_rate * self.dt)
                if d.time_left <= 0.0:
                    d.busy = False
                    d.time_left = 0.0
                    
                    if d.status == DRONE_GO_TO_VINE:
                        v = self.vines[d.target_vine]
                        d.position = v.position.copy()
                        if v.queued_boxes > 0:
                            v.queued_boxes -= 1
                            contributor = v.queue_contributors.popleft() if len(v.queue_contributors) > 0 else None
                            d.last_contributor = contributor
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
                            d.delivered_count += 1
                            if d.last_contributor is not None:
                                drone_credit_deliveries[d.last_contributor] += 1
                            d.last_contributor = None
                        
                        d.status = DRONE_IDLE
                        d.target_vine = None
                    
                    elif d.status == DRONE_GO_TO_CHARGE:
                        d.position = self.charging_point.copy()
                        d.status = DRONE_CHARGE
                        # Wait time for charging? Let's say it takes 1 step to start charging, 
                        # or we can just apply charging in the IDLE/CHARGE logic below.
                        # For now, let's keep it busy for a moment or just switch state.
                        d.busy = False 
                        
        
        # 3.5) Charging logic (for drones at charger)
        for d in self.drones:
            if d.status == DRONE_CHARGE and not d.busy:
                # Charge up
                d.battery = min(100.0, d.battery + self.drone_batt_charge_rate * self.dt) # Charge speed
                if d.battery >= 100.0:
                    d.battery = 100.0
                    d.status = DRONE_IDLE
                else:
                    # Still charging, consume a step
                    pass
        
        # 4) Assign idle drones
        for d in self.drones:
            if (not d.busy) and d.status == DRONE_IDLE:
                
                # Check battery first
                if d.battery <= 20.0:
                    d.status = DRONE_GO_TO_CHARGE
                    dist = float(np.linalg.norm(d.position - self.charging_point))
                    d.busy = True
                    d.time_left = dist / max(self.drone_speed, 1e-6)
                    continue

                candidates = [idx for idx, v in enumerate(self.vines) if v.queued_boxes > 0]
                if candidates:
                    dists = [np.linalg.norm(self.vines[idx].position - d.position) for idx in candidates]
                    target = candidates[int(np.argmin(dists))]

                    # --- NEW: time-based battery feasibility check (go-to-vine + deliver-to-CP) ---
                    dist_to_vine = float(np.linalg.norm(self.vines[target].position - d.position))
                    t_go = dist_to_vine / max(self.drone_speed, 1e-6)

                    dist_to_cp = float(np.linalg.norm(self.vines[target].position - self.collection_point))
                    t_deliver = dist_to_cp / max(self.drone_speed, 1e-6)

                    battery_needed = self.drone_batt_drain_rate * (t_go + t_deliver)  # % needed
                    safety_margin = 5.0  # % (tune)

                    if d.battery <= battery_needed + safety_margin:
                        # Not enough battery to complete safely -> go charge instead
                        d.status = DRONE_GO_TO_CHARGE
                        dist = float(np.linalg.norm(d.position - self.charging_point))
                        d.busy = True
                        d.time_left = dist / max(self.drone_speed, 1e-6)
                        continue
                    # --- END NEW ---

                    # Proceed with mission
                    d.target_vine = target
                    d.status = DRONE_GO_TO_VINE
                    d.busy = True
                    d.time_left = t_go
                        
        # --- count drone status time this timestep ---
        for j, d in enumerate(self.drones):
            s = int(d.status)
            self.ep_drone_status_steps[j, s] += 1
            self.ep_drone_status_seconds[j, s] += self.dt

        self.ep_fatigue_increase_total += float(sum(fatigue_increase))
        
        # --- fatigue KPI update (per step) ---
        mean_fatigue_t = float(np.mean([h.fatigue for h in self.humans]))
        max_fatigue_t = float(np.max([h.fatigue for h in self.humans]))

        self.ep_fatigue_sum += mean_fatigue_t
        if max_fatigue_t > self.ep_fatigue_peak:
            self.ep_fatigue_peak = max_fatigue_t

        # optional: count human-steps above threshold
        self.ep_fatigue_hi_steps += int(sum(1 for h in self.humans if h.fatigue >= self.fatigue_hi_threshold))



        # Calculate rewards
        delivered_delta = self.delivered - delivered_before
        self.ep_delivered_delta_total += int(delivered_delta)
        backlog_total = sum(v.queued_boxes for v in self.vines)

        # --- update KPI accumulators ---
        self.ep_step_count += 1
        self.ep_delivered_total += int(delivered_delta)

        self.ep_backlog_sum += int(backlog_total)
        if backlog_total > self.ep_backlog_peak:
            self.ep_backlog_peak = int(backlog_total)

        # Humans: count REST + BUSY
        for h in self.humans:
            if h.busy and h.current_action == ACTION_REST:
                self.ep_human_rest_steps += 1

            if h.busy and h.current_action in (ACTION_HARVEST, ACTION_TRANSPORT, ACTION_ENQUEUE):
                self.ep_human_busy_steps += 1

        # Drones: utilization = flying steps
        for d in self.drones:
            if d.busy and d.status in (DRONE_GO_TO_VINE, DRONE_DELIVER):
                self.ep_drone_flying_steps += 1

        fatigue_total = float(sum(fatigue_increase))

        # -----------------------------
        # TEAM REWARD
        # -----------------------------
        backlog_norm = float(backlog_total) / max(1, self.num_vines * self.max_backlog)

        fatigue_excess = float(np.mean([max(0.0, h.fatigue - 0.6) for h in self.humans]))

        r_delivery = self.reward_delivery * float(delivered_delta)
        r_fat_inc = -self.reward_fatigue_inc_penalty * fatigue_total
        r_backlog = -self.reward_backlog_penalty * backlog_norm
        r_fat_lvl = -self.reward_fatigue_level_penalty * fatigue_excess

        team_reward = r_delivery + r_fat_inc + r_backlog + r_fat_lvl

        self.ep_r_delivery_sum += float(r_delivery)
        self.ep_r_fatigue_inc_sum += float(r_fat_inc)
        self.ep_r_backlog_sum += float(r_backlog)
        self.ep_r_fatigue_level_sum += float(r_fat_lvl)

        lambda_team = 0.60
        step_rewards = {}

        for i, agent_id in enumerate(self.possible_agents):
            fatigue_excess_i = max(0.0, self.humans[i].fatigue - 0.6)

            local_reward = (
                self.reward_delivery * float(individual_deliveries[i])
                + self.reward_harvest * float(harvest_events[i])
                + self.reward_enqueue * float(enqueue_events[i])
                + self.reward_drone_credit * float(drone_credit_deliveries[i])
                - self.reward_fatigue_inc_penalty * float(fatigue_increase[i])
                - self.reward_fatigue_level_penalty * float(fatigue_excess_i)
            )

            agent_reward = lambda_team * team_reward + (1.0 - lambda_team) * local_reward
            step_rewards[agent_id] = float(agent_reward)
            self.pending_rewards[agent_id] += float(agent_reward)

        # for episode-level logging
        self.ep_reward_sum += float(np.mean(list(step_rewards.values())))

        remaining_harvest_kg = float(sum(v.kg_remaining for v in self.vines))
        remaining_ready_boxes = int(sum(v.boxes_ready for v in self.vines))
        remaining_queue = int(sum(v.queued_boxes for v in self.vines))

        all_harvested = all((v.kg_remaining <= 1e-9) and (v.boxes_ready == 0) for v in self.vines)
        no_queue = all(v.queued_boxes == 0 for v in self.vines)
        no_human_boxes = all(not h.has_box for h in self.humans)
        no_drone_boxes = all(not d.has_box for d in self.drones)

        terminated = bool(all_harvested and no_queue and no_human_boxes and no_drone_boxes)
        truncated = bool(self.steps >= self.max_steps)

        summary = {}
        if terminated or truncated:
            human_steps = self.ep_human_action_steps.sum(axis=0)
            drone_steps = self.ep_drone_status_steps.sum(axis=0)

            steps = max(1, int(self.ep_step_count))

            mean_backlog = self.ep_backlog_sum / steps
            peak_backlog = int(self.ep_backlog_peak)

            delivered_total = int(self.ep_delivered_total)
            throughput_per_100 = 100.0 * delivered_total / steps

            remaining_human_carried = int(sum(1 for h in self.humans if h.has_box))
            remaining_drone_carried = int(sum(1 for d in self.drones if d.has_box))

            remaining_unharvested_box_equiv = int(
                np.ceil(remaining_harvest_kg / max(self.box_capacity_kg, 1e-6))
            )

            total_initial_boxes = max(1, int(self.initial_total_boxes))
            delivered_pct = 100.0 * delivered_total / total_initial_boxes

            remaining_work_pct = 100.0 * (
                remaining_unharvested_box_equiv
                + remaining_ready_boxes
                + remaining_queue
                + remaining_human_carried
                + remaining_drone_carried
            ) / total_initial_boxes

            completion_pct = 100.0 - remaining_work_pct

            human_total_steps = max(1, self.num_humans * steps)
            rest_ratio = self.ep_human_rest_steps / human_total_steps
            human_util = self.ep_human_busy_steps / human_total_steps

            drone_total_steps = max(1, self.num_drones * steps)
            drone_util = self.ep_drone_flying_steps / drone_total_steps

            summary = {
                "kpi_delivered_total": delivered_total,
                "kpi_throughput_per_100_steps": throughput_per_100,
                "kpi_mean_backlog": float(mean_backlog),
                "kpi_peak_backlog": peak_backlog,
                "kpi_rest_ratio": float(rest_ratio),
                "kpi_delivered_pct": float(delivered_pct),
                "kpi_remaining_work_pct": float(remaining_work_pct),
                "kpi_completion_pct": float(completion_pct),
                "kpi_human_utilization": float(human_util),
                "kpi_drone_utilization": float(drone_util),
                "episode_fatigue_increase_total": float(self.ep_fatigue_increase_total),
                "episode_delivered_delta_total": int(self.ep_delivered_delta_total),
                "kpi_mean_fatigue": float(self.ep_fatigue_sum / steps),
                "kpi_peak_fatigue": float(self.ep_fatigue_peak),
                "kpi_fatigue_hi_ratio": float(self.ep_fatigue_hi_steps / max(1, self.num_humans * steps)),
                "r_delivery_per_step": float(self.ep_r_delivery_sum / steps),
                "r_fatigue_inc_per_step": float(self.ep_r_fatigue_inc_sum / steps),
                "r_backlog_per_step": float(self.ep_r_backlog_sum / steps),
                "r_fatigue_level_per_step": float(self.ep_r_fatigue_level_sum / steps),
                "r_total_per_step": float(self.ep_reward_sum / steps),
            }

        # In async mode:
        # - if episode continues, only return decision-ready agents
        # - if episode ends, flush all agents one last time
        if terminated or truncated:
            ready_agents = self.possible_agents.copy()
        else:
            ready_agents = self._ready_agent_ids()

        self.agents = ready_agents.copy()

        observations = {}
        rewards = {}
        infos = {}

        for agent_id in ready_agents:
            i = self.agent_index[agent_id]

            observations[agent_id] = self._get_obs_for_agent(i)
            rewards[agent_id] = float(self.pending_rewards[agent_id])
            self.pending_rewards[agent_id] = 0.0

            infos[agent_id] = {
                "delivered": self.delivered,
                "delivered_delta": delivered_delta,
                "fatigue_total_increase": fatigue_total,
                "backlog_total": backlog_total,
                "individual_delivery": individual_deliveries[i],
                "harvest": harvest_events[i],
                "enqueue": enqueue_events[i],
                "drone_credit_delivery": drone_credit_deliveries[i],
                "episode_summary": summary,
            }

        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}

        return observations, rewards, terminateds, truncateds, infos

    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Normalize a 2D position to [0, 1] range."""
        x = (pos[0] - self.x_min) / max(self.field_size[0], 1e-6)
        y = (pos[1] - self.y_min) / max(self.field_size[1], 1e-6)
        return np.clip(np.array([x, y], dtype=np.float32), 0.0, 1.0)

    def _normalize_z(self, z: float) -> float:
        """Normalize z to [0, 1]."""
        zn = (float(z) - self.z_min) / max(self.z_range, 1e-6)
        return float(np.clip(zn, 0.0, 1.0))

    def get_vine_xyz(self, vine_idx: int) -> np.ndarray:
        """Return vine location as xyz."""
        v = self.vines[vine_idx]
        return np.array([v.position[0], v.position[1], v.z], dtype=np.float32)

    def _get_collection_xyz(self) -> np.ndarray:
        return np.array(
            [self.collection_point[0], self.collection_point[1], self.collection_z],
            dtype=np.float32
        )

    def _get_charging_xyz(self) -> np.ndarray:
        return np.array(
            [self.charging_point[0], self.charging_point[1], self.charging_z],
            dtype=np.float32
        )

    def _get_human_xyz(self, human: Human) -> np.ndarray:
        """
        Approximate current human xyz.
        Humans mostly operate either at their assigned vine or at collection point.
        """
        if np.allclose(human.position, self.collection_point):
            z = self.collection_z
        else:
            z = self.vines[human.assigned_vine].z
        return np.array([human.position[0], human.position[1], z], dtype=np.float32)

    def _get_drone_xyz(self, drone: Drone) -> np.ndarray:
        """
        Approximate current drone xyz from its current 2D position.
        """
        if np.allclose(drone.position, self.collection_point):
            z = self.collection_z
        elif np.allclose(drone.position, self.charging_point):
            z = self.charging_z
        elif drone.target_vine is not None:
            z = self.vines[drone.target_vine].z
        else:
            # fallback: nearest vine altitude
            dists = [np.linalg.norm(v.position - drone.position) for v in self.vines]
            nearest = int(np.argmin(dists))
            z = self.vines[nearest].z
        return np.array([drone.position[0], drone.position[1], z], dtype=np.float32)

    def _grade_between_xyz(self, start_xyz: np.ndarray, end_xyz: np.ndarray) -> float:
        """
        Return slope grade = |dz| / horizontal_distance.
        """
        dxy = float(np.linalg.norm(end_xyz[:2] - start_xyz[:2]))
        dz = float(end_xyz[2] - start_xyz[2])
        return abs(dz) / max(dxy, 1e-6)
    
    def _get_handover_xyz(self, handover_id: int) -> np.ndarray:
        hp = self.handover_points[handover_id]
        return np.array([hp["position"][0], hp["position"][1], hp["z"]], dtype=np.float32)

    def _human_transport_costs(self, start_xyz: np.ndarray, end_xyz: np.ndarray) -> Tuple[float, float]:
        """
        Compute slope-adjusted human transport travel time and fatigue multiplier.
        """
        dxy = float(np.linalg.norm(end_xyz[:2] - start_xyz[:2]))
        grade = self._grade_between_xyz(start_xyz, end_xyz)

        base_time = dxy / max(self.human_speed, 1e-6)
        travel_time = base_time * (1.0 + self.slope_time_factor * grade)

        dz = float(end_xyz[2] - start_xyz[2])
        uphill_ratio = max(0.0, dz) / max(abs(dz), 1e-6) if abs(dz) > 1e-9 else 0.0
        fatigue_multiplier = 1.0 + self.slope_fatigue_factor * grade * (0.5 + 0.5 * uphill_ratio)

        return float(travel_time), float(fatigue_multiplier)

    def _get_local_vine_indices(self, human: Human) -> np.ndarray:
        """
        Return indices of the K nearest vines to this human.
        """
        if self.num_vines == 0:
            return np.array([], dtype=np.int32)

        dists = np.array(
            [np.linalg.norm(v.position - human.position) for v in self.vines],
            dtype=np.float32,
        )
        k = min(self.local_vine_k, self.num_vines)
        return np.argsort(dists)[:k].astype(np.int32)

    def _get_obs_for_agent(self, agent_idx: int) -> np.ndarray:
        """
        Build observation for a specific agent using only the nearest K vines.
        """
        obs = []
        h = self.humans[agent_idx]
        local_indices = self._get_local_vine_indices(h)

        # === LOCAL VINE STATE ===

        # local vine positions
        for slot in range(self.local_vine_k):
            if slot < len(local_indices):
                idx = int(local_indices[slot])
                obs.extend(self.vine_xy_norm[idx])
            else:
                obs.extend([0.0, 0.0])

        # local vine z
        for slot in range(self.local_vine_k):
            if slot < len(local_indices):
                idx = int(local_indices[slot])
                obs.append(float(self.vine_z_norm[idx]))
            else:
                obs.append(0.0)

        # collection point
        obs.extend(self.collection_xy_norm)
        obs.append(float(self.collection_z_norm))

        # charging point
        obs.extend(self.charging_xy_norm)
        obs.append(float(self.charging_z_norm))

        # local kg remaining
        for slot in range(self.local_vine_k):
            if slot < len(local_indices):
                idx = int(local_indices[slot])
                v = self.vines[idx]
                obs.append(v.kg_remaining / max(self.max_line_kg_actual, 1e-6))
            else:
                obs.append(0.0)

        # local service units = ready + queued
        for slot in range(self.local_vine_k):
            if slot < len(local_indices):
                idx = int(local_indices[slot])
                v = self.vines[idx]
                service_units = v.boxes_ready + v.queued_boxes
                obs.append(service_units / max(self.max_boxes_equivalent_actual, 1))
            else:
                obs.append(0.0)

        # === OWN STATE ===

        h_xyz = self._get_human_xyz(h)
        obs.extend(self._normalize_position(h.position))
        obs.append(self._normalize_z(h_xyz[2]))

        obs.append(h.fatigue)

        obs.extend(one_hot(h.current_action, self.num_actions))

        obs.append(1.0 if h.has_box else 0.0)

        # scalar instead of one-hot
        obs.append(h.assigned_vine / max(self.num_vines - 1, 1))

        # === OTHER HUMANS ===

        for other_idx, other_h in enumerate(self.humans):
            if other_idx != agent_idx:
                other_xyz = self._get_human_xyz(other_h)
                obs.extend(self._normalize_position(other_h.position))
                obs.append(self._normalize_z(other_xyz[2]))

        for other_idx, other_h in enumerate(self.humans):
            if other_idx != agent_idx:
                obs.append(1.0 if other_h.has_box else 0.0)

        # === DRONES ===

        for d in self.drones:
            d_xyz = self._get_drone_xyz(d)
            obs.extend(self._normalize_position(d.position))
            obs.append(self._normalize_z(d_xyz[2]))

        for d in self.drones:
            obs.extend(one_hot(d.status, self.num_drone_status))

        for d in self.drones:
            obs.append(1.0 if d.has_box else 0.0)

        for d in self.drones:
            obs.append(d.battery / 100.0)

        obs = np.array(obs, dtype=np.float32)
        mask = np.ones(self.num_actions, dtype=np.float32)

        line = self.vines[h.assigned_vine]

        can_harvest = (not h.has_box) and (line.kg_remaining > 1e-9)
        can_take_ready_box = (not h.has_box) and (line.boxes_ready > 0)

        if not can_harvest:
            mask[ACTION_HARVEST] = 0.0

        if not (h.has_box or can_take_ready_box):
            mask[ACTION_TRANSPORT] = 0.0
            mask[ACTION_ENQUEUE] = 0.0

        if (h.has_box or can_take_ready_box) and line.queued_boxes >= self.max_backlog:
            mask[ACTION_ENQUEUE] = 0.0

        can_do_productive = bool(
            (mask[ACTION_HARVEST] > 0.0)
            or (mask[ACTION_TRANSPORT] > 0.0)
            or (mask[ACTION_ENQUEUE] > 0.0)
        )

        if h.fatigue >= self.rest_fatigue_threshold or not can_do_productive:
            mask[ACTION_REST] = 1.0
        else:
            mask[ACTION_REST] = 0.0

        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        obs = np.clip(obs, 0.0, 1.0).astype(np.float32)
        mask = mask.astype(np.float32)

        return {
            "obs": obs,
            "action_mask": mask,
        }

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
            print(
                f"  Line {i}: lot={getattr(v, 'lot', '?')} "
                f"line={getattr(v, 'line', '?')} "
                f"kg_rem={v.kg_remaining:.2f} "
                f"ready={v.boxes_ready} "
                f"queued={v.queued_boxes}"
            )

        print("Handover points:")
        for hp in self.handover_points:
            print(
                f"  HP {hp['id']}: "
                f"xy=({hp['position'][0]:.2f}, {hp['position'][1]:.2f}) "
                f"z={hp['z']:.2f} "
                f"queued={hp['queued_boxes']} "
                f"n_lines={len(hp['line_indices'])}"
            )
        for i, h in enumerate(self.humans):
            agent_id = f"human_{i}"
            print(f"  {agent_id}: vine={h.assigned_vine} busy={h.busy} "
                  f"t={h.time_left:.1f} has_box={h.has_box} fat={h.fatigue:.2f}")
        for i, d in enumerate(self.drones):
            print(f"  Drone {i}: status={d.status} busy={d.busy} "
                  f"t={d.time_left:.1f} bat={d.battery:.1f}")
        for i, v in enumerate(self.vines):
            print(
                f"  Line {i}: lot={getattr(v, 'lot', '?')} "
                f"line={getattr(v, 'line', '?')} "
                f"hp={getattr(v, 'handover_id', -1)} "
                f"kg_rem={v.kg_remaining:.2f} "
                f"ready={v.boxes_ready} "
                f"queued={v.queued_boxes}"
            )
        print("=" * 60)

    def _render_pygame(self):
        
        """Pygame visual rendering."""
        SCREEN_W, SCREEN_H = 800, 800
        PADDING = 40
        TEXT_COLOR = (0, 0, 0)

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
        pygame.draw.rect(self._screen, (255, 215, 0), (cp[0]-10, cp[1]-10, 20, 20))
        
        # Charging point (Purple)
        chp = world_to_screen(self.charging_point)
        pygame.draw.rect(self._screen, (138, 43, 226), (chp[0]-8, chp[1]-8, 16, 16))
        
        # Vines (green)
        # Lines (green)
        font = pygame.font.SysFont(None, 18)

        for v in self.vines:
            x, y = world_to_screen(v.position)

            # Optional visual cue: darker if finished
            if v.kg_remaining <= 1e-9 and v.boxes_ready == 0 and v.queued_boxes == 0:
                color = (120, 120, 120)
            else:
                color = (34, 139, 34)

            pygame.draw.rect(self._screen, color, (x, y, 10, 10))

            # Show new Phase 3–4 state
            text_str = f"{v.kg_remaining:.1f}kg | r{v.boxes_ready} | q{v.queued_boxes}"
            text_surf = font.render(text_str, True, TEXT_COLOR)
            self._screen.blit(text_surf, (x + 12, y - 10))

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
        
            # --- DRAW STATS UPPER RIGHT ---
        font = pygame.font.SysFont(None, 18)
        line_y = 10

        # Humans stats
        for i, h in enumerate(self.humans):
            action_name = ["HARVEST", "TRANSPORT", "ENQUEUE", "REST"][h.current_action]
            stats = f"H{i}: {action_name} | Fatigue: {h.fatigue:.2f} | Delivered: {h.delivered_count}"
            text_surf = font.render(stats, True, TEXT_COLOR)
            self._screen.blit(text_surf, (SCREEN_W - 380, line_y))
            line_y += 24

        # Drone stats
        for i, d in enumerate(self.drones):
            status_name = ["IDLE","GO_TO_VINE","DELIVER", "GO_TO_CHRG", "CHARGING"][d.status]
            stats = f"Drone {i}: {status_name} | Bat: {d.battery:.0f} | Del: {d.delivered_count}"
            text_surf = font.render(stats, True, TEXT_COLOR)
            self._screen.blit(text_surf, (SCREEN_W - 380, line_y))
            line_y += 24

        pygame.display.flip()
        self._clock.tick(10)

    def close(self):
        """Clean up resources."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False

def test_environment():
    """Test the Multi-Agent Vine Environment."""
    env = MultiAgentVineEnvAsync(
        render_mode="human",
        topology_mode="line",
        num_humans=5,
        num_drones=1,
        yield_per_plant_kg=0.6,     # placeholder scenario value
        box_capacity_kg=8.0,        # placeholder scenario value
        harvest_rate_kg_s=0.004,    # placeholder; calibrate later from productivity
        max_backlog=5,
        max_steps=5000,
        dt=5.0,
        harvest_time=300.0,         # 5 min harvest cycle
        human_speed=1.0,
        drone_speed=5.0,
        vineyard_file="data/Vinha_Maria_Teresa_RL.xlsx",
    )

    observations, infos = env.reset()
    done = {"__all__": False}

    while not done["__all__"]:
        action_dict = {}
        for agent_id in env.agents:
            action_dict[agent_id] = env.action_space.sample()
        
        observations, rewards, terminateds, truncateds, infos = env.step(action_dict)
        done = {**terminateds}
        # print(truncateds)
        env.render()

    env.close()

if __name__ == "__main__":
    test_environment()
