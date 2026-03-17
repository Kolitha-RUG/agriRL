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

from vine_env_multiagent_async import MultiAgentVineEnvAsync
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
        # print("MODEL obs_space:", obs_space)
        # print("MODEL original_space:", getattr(obs_space, "original_space", None))

        self.num_actions = int(action_space.n)

        orig_space = getattr(obs_space, "original_space", obs_space)

        if not isinstance(orig_space, gym.spaces.Dict):
            raise ValueError(f"Expected Dict observation space, got {type(orig_space)}: {orig_space}")

        base_obs_space = orig_space["obs"]

        self.internal_model = FullyConnectedNetwork(
            base_obs_space, action_space, num_outputs, model_config, name + "_internal"
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["obs"]
        action_mask = input_dict["obs"]["action_mask"]

        logits, _ = self.internal_model({"obs": obs})
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()


def env_creator(env_config):
    env = MultiAgentVineEnvAsync(**env_config)
    # print("ENV FILE:", MultiAgentVineEnvAsync.__module__)
    # print("OBS SPACE TYPE:", type(env.observation_space))
    # print("OBS SPACE:", env.observation_space)
    return env


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"


if __name__ == "__main__":
    os.environ["RAY_DISABLE_METRICS"] = "1"

    ray.init(ignore_reinit_error=True)

    register_env("MultiAgentVine", env_creator)
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

    env_config = dict(
        render_mode=None,
        topology_mode="row",
        vineyard_file=os.path.join(PROJECT_DIR, "data", "Vinha_Maria_Teresa_RL.xlsx"),
        local_vine_k=6,
        num_humans=5,
        num_drones=2,
        max_boxes_per_vine=10,
        max_steps=200,
        max_backlog=10,
        dt=1.0,
        harvest_time=10.0,
        human_speed=0.2,
        drone_speed=1.0,
        reward_backlog_penalty= 0.05,
        reward_fatigue_inc_penalty = 1.5,
        reward_delivery=3,
        reward_fatigue_level_penalty= 0.5
    )

# delivery weight: 0.2 / 0.12 = 1.67

# fatigue increase weight: 0.2 / 0.08 = 2.5

# backlog weight: 0.2 / 3.0 = 0.067

# fatigue level weight: 0.2 / 0.5 = 0.4
    config = (
        PPOConfig()

        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="MultiAgentVineAsync", env_config=env_config)
        .env_runners(num_env_runners=10, num_envs_per_env_runner=5)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=2000,
            model={
                "custom_model": "torch_action_mask_model",
                "fcnet_hiddens": [128, 64],
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
            stop={"training_iteration": 50},
        ),
    )

    results = tuner.fit()
    print("Done. TensorBoard logdir: ray_results_vine")

#  tensorboard --logdir=ray_results_vine/vineyard_ppo

# r_delivery_per_step ≈ +0.18

# r_backlog_per_step ≈ −0.031

# r_fatigue_inc_per_step ≈ −0.098

# r_fatigue_level_per_step ≈ −0.37

# r_total_per_step ≈ −0.32

# Then:

# delivery weight: 0.2 / 0.12 = 1.67

# fatigue increase weight: 0.2 / 0.08 = 2.5

# backlog weight: 0.2 / 3.0 = 0.067

# fatigue level weight: 0.2 / 0.5 = 0.4