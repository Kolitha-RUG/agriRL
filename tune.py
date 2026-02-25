import os
import ray
import gym
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

from vine_env_multiagent import MultiAgentVineEnv
import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

class VineyardKPI(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):
        # Grab one agent's info at the end (all agents carry the same episode_summary dict)
        summary = None
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if isinstance(info, dict):
                summary = info.get("episode_summary", None)
                if summary:  # non-empty dict
                    break

        if not summary:
            return

        # Write each summary field into TensorBoard custom_metrics
        for k, v in summary.items():
            # ensure TB-friendly scalars
            episode.custom_metrics[k] = float(v)


class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_actions = int(action_space.n)
        self.base_obs_dim = int(obs_space.shape[0] - self.num_actions)

        base_obs_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.base_obs_dim,), dtype=np.float32
        )
        self.internal_model = FullyConnectedNetwork(
            base_obs_space, action_space, num_outputs, model_config, name + "_internal"
        )

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]  # [B, base_obs_dim + num_actions]
        obs = x[:, : self.base_obs_dim]
        action_mask = x[:, self.base_obs_dim :]

        logits, _ = self.internal_model({"obs": obs})
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()


def env_creator(env_config):
    return MultiAgentVineEnv(**env_config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"


if __name__ == "__main__":
    os.environ["RAY_DISABLE_METRICS"] = "1"

    ray.init(ignore_reinit_error=True)

    register_env("MultiAgentVine", env_creator)
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

    env_config = dict(
        render_mode="terminal",
        topology_mode="row",
        vineyard_file=os.path.join(PROJECT_DIR, "data", "Vinha_Maria_Teresa_RL.xlsx"),
        num_humans=5,
        num_drones=1,
        max_boxes_per_vine=10,
        max_steps=1000,
        max_backlog=10,
        dt=1.0,
        harvest_time=5.0,
        human_speed=0.5,
        drone_speed=1.0,
    )

    config = (
        PPOConfig()

        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="MultiAgentVine", env_config=env_config)
        .env_runners(num_env_runners=0)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=4000,
            model={
                "custom_model": "torch_action_mask_model",
                "fcnet_hiddens": [256, 256],
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