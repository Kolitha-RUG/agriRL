import os
from copy import deepcopy

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_ENV_CONFIG = {
    "render_mode": None,
    "topology_mode": "line",
    "vineyard_file": os.path.join(PROJECT_DIR, "data", "Vinha_Maria_Teresa_RL.xlsx"),
    "local_vine_k": 6,
    "num_humans": 5,
    "num_drones": 1,

    "yield_per_plant_kg": 0.6,
    "box_capacity_kg": 8.0,

    # 1 step = 1 minute
    "dt": 1.0,
    "harvest_rate_kg_s": 0.24,
    "harvest_time": 5.0,
    "enqueue_time": 1.0,
    "rest_time": 5.0,
    "rest_fatigue_threshold": 0.5,
    
    "human_speed": 30.0,
    "drone_speed": 100.0,

    "human_harvest_fatigue_rate": 0.0035,
    "human_transport_fatigue_rate": 0.01,
    "human_rest_recovery_rate": 0.0030,

    "drone_endurance_loaded_s": 18.0,
    "drone_endurance_unloaded_s": 29.0,
    "drone_charge_time_full_s": 36.6,

    "drone_handover_service_time": 2.0,
    "drone_dropoff_service_time": 2.0,

    "max_steps": 500,
    "max_backlog": 10,

    "reward_backlog_penalty": 1,
    "reward_fatigue_inc_penalty": 1.5,
    "reward_delivery": 1,
    "reward_fatigue_level_penalty": 1,
}

def get_env_config(**overrides):
    cfg = deepcopy(BASE_ENV_CONFIG)
    cfg.update(overrides)
    return cfg