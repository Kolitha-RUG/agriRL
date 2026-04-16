import os
import ray
import gymnasium as gym
import numpy as np

from ray import tune
from ray.tune import Tuner, RunConfig
from ray.tune.registry import register_env

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN

import torch
import torch.nn as nn

from vine_env import VineEnv
import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

class VineyardKPI(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):

        summary = None
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if isinstance(info, dict):
                summary = info.get("episode_summary", None)
                if summary:  
                    break

        if not summary:
            return


        for k, v in summary.items():

            episode.custom_metrics[k] = float(v)

class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_actions = int(action_space.n)

        orig_space = getattr(obs_space, "original_space", obs_space)

        if not isinstance(orig_space, gym.spaces.Dict):
            raise ValueError(f"Expected Dict observation space, got {type(orig_space)}: {orig_space}")

        base_obs_space = orig_space["obs"]

        self.internal_model = FullyConnectedNetwork(
            base_obs_space, action_space, num_outputs, model_config, name + "_internal"
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["obs"].float()
        action_mask = input_dict["obs"]["action_mask"].float()

        logits, _ = self.internal_model({"obs": obs})

        masked_logits = torch.where(
            action_mask > 0.5,
            logits,
            torch.full_like(logits, -1e9),
        )
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


def env_creator(env_config):
    env = VineEnv(**env_config)

    return env


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"


if __name__ == "__main__":
    os.environ["RAY_DISABLE_METRICS"] = "1"

    ray.init(ignore_reinit_error=True)

    register_env("MultiAgentVineAsync", env_creator)
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

    env_config = dict(
        render_mode=None,
        topology_mode="line",
        vineyard_file=os.path.join(PROJECT_DIR, "data", "Vinha_Maria_Teresa_RL.xlsx"),
        local_vine_k=6,
        num_humans=10,
        num_drones=1,

        yield_per_plant_kg=0.6,
        box_capacity_kg=8.0,

        dt=1.0,                     # 1 step = 1 minute

        harvest_rate_kg_s=0.24,     # actually kg/step
        harvest_time=5.0,           # 5 min
        enqueue_time=1.0,           # 1 min
        rest_time=5.0,              # 5 min

        human_speed=48.0,           # m/step
        drone_speed=300.0,          # m/step

        human_harvest_fatigue_rate=0.002,
        human_transport_fatigue_rate=0.003,
        human_rest_recovery_rate=0.004,

        drone_endurance_loaded_s=18.0,     # steps
        drone_endurance_unloaded_s=29.0,   # steps
        drone_charge_time_full_s=36.6,     # steps

        drone_handover_service_time=1.0,  # 1 min at handover
        drone_dropoff_service_time=1.0,   # 1 min at collection

        max_steps=480,              # 8 hours
        max_backlog=10,

        reward_backlog_penalty=0.05,
        reward_fatigue_inc_penalty=1.5,
        reward_delivery=3,
        reward_fatigue_level_penalty=2,
    )
    config = (
        PPOConfig()

        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="MultiAgentVineAsync", env_config=env_config)
        .env_runners(num_env_runners=10, num_envs_per_env_runner=1)
        .training(
            gamma=0.99,
            lr=1e-3,
            train_batch_size=8000,
            entropy_coeff=0.01,
            model={
                "custom_model": "torch_action_mask_model",
                "fcnet_hiddens": [32, 32],
                "fcnet_activation": "relu",
            },
        )
        .multi_agent(
            policies=["shared_policy"],
            policy_mapping_fn=policy_mapping_fn,
        )
        .callbacks(VineyardKPI)
        .resources(
            num_gpus=1,
        )
    )

    tuner = Tuner(
        "PPO",
        param_space=config,
        run_config=RunConfig(
            name="vineyard_ppo",
            storage_path=os.path.join(PROJECT_DIR, "ray_results_vine"),
            stop={"training_iteration": 200},
        ),
    )

    results = tuner.fit()
    print("Done. TensorBoard logdir: ray_results_vine")

#  tensorboard --logdir=ray_results_vine/vineyard_ppo

