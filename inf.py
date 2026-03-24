import os
import gymnasium as gym
import torch
import torch.nn as nn
import ray

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.tune.registry import register_env
from tune import env_creator
from vine_env_multiagent_async import MultiAgentVineEnvAsync


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
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


def make_env():
    env_config = dict(
        render_mode="human",
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
        reward_backlog_penalty=0.05,
        reward_fatigue_inc_penalty=1.5,
        reward_delivery=3,
        reward_fatigue_level_penalty=0.5,
    )
    return MultiAgentVineEnvAsync(**env_config)


def run_async_inference(checkpoint_path: str, episodes: int = 1, explore: bool = False):
    ray.init(ignore_reinit_error=True)
    register_env("MultiAgentVineAsync", env_creator)
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)
    algo = Algorithm.from_checkpoint(checkpoint_path)
    env = make_env()

    for ep in range(episodes):
        obs, infos = env.reset()
        terminated = {"__all__": False}
        truncated = {"__all__": False}

        while not terminated["__all__"] and not truncated["__all__"]:
            action_dict = {}

            # Only ready agents are in obs
            for agent_id, agent_obs in obs.items():
                action = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id="shared_policy",
                    explore=explore,
                )
                action_dict[agent_id] = int(action)

            obs, rewards, terminated, truncated, infos = env.step(action_dict)

            if env.render_mode == "human":
                env.render()

        if infos:
            any_agent = next(iter(infos))
            summary = infos[any_agent].get("episode_summary", {})
            print(summary)

    env.close()
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    checkpoint_path = os.path.join(
        PROJECT_DIR,
        "ray_results_vine",
        "vineyard_ppo",
        # example:
        "PPO_MultiAgentVineAsync_71be6_00000_0_2026-03-18_12-56-05",
        "checkpoint_000000"
    )

    run_async_inference(checkpoint_path=checkpoint_path, episodes=1, explore=False)