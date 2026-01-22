"""
Simple training script to test MultiAgentVineEnv trainability with RLlib.
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from vine_env_multiagent import MultiAgentVineEnv


def env_creator(env_config):
    return MultiAgentVineEnv(**env_config)


if __name__ == "__main__":
    ray.init()
    
    register_env("MultiAgentVine", env_creator)
    
    config = (
        PPOConfig()
        .environment(
            env="MultiAgentVine",
            env_config={
                "render_mode": "terminal",
                "topology_mode": "row",
                "num_humans": 2,
                "num_drones": 2,
                "max_boxes_per_vine": 0.01,
                "max_steps": 1000,
            },
        )
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=1,
        )
        .training(
            train_batch_size=4000,
            gamma=0.99,
            lr=3e-4,
        )
        .multi_agent(
            policies=["shared_policy"],
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
    )
    
    algo = config.build_algo()
    
    print("Starting training...")
    for i in range(50):
        result = algo.train()
        
        er = result['env_runners']
        reward = er['episode_return_mean']
        episodes = er['num_episodes']
        ep_len = er['episode_len_mean']
        
        print(f"Iter {i+1:3d} | Return: {reward:8.2f} | Ep Len: {ep_len:6.1f} | Episodes: {episodes}")
    
    print("\nTraining complete!")
    algo.stop()
    ray.shutdown()