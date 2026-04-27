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
from env_config import get_env_config


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

    env_config = get_env_config()
    config = (
        PPOConfig()

        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="MultiAgentVineAsync", env_config=env_config,
                    #  disable_env_checking=True,
                     )
        .env_runners(num_env_runners=5, num_envs_per_env_runner=1)
        .training(
            gamma=0.99,
            lr=1e-3,
            train_batch_size=4000,
            minibatch_size=512,
            num_epochs=6,
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
            num_gpus=0,
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

