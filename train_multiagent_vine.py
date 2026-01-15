import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig

from vine_env_multiagent import MultiAgentVineEnv


def env_creator(env_config):
    return MultiAgentVineEnv(**env_config)


def main():
    ray.init(ignore_reinit_error=True)

    ENV_NAME = "MultiAgentVineEnv-v0"
    register_env(ENV_NAME, env_creator)

    env_config = dict(
        render_mode="terminal",
        topology_mode="row",
        num_humans=3,
        num_drones=1,
        max_boxes_per_vine=1,
        max_backlog=5,
        max_steps=2000,
        dt=1.0,
        harvest_time=5.0,
        human_speed=0.5,
        drone_speed=1.0,
        vineyard_file="data/Vinha_Maria_Teresa_RL.xlsx",
        reward_delivery=1.0,
        reward_backlog_penalty=0.5,
        reward_fatigue_penalty=0.5,
    )

    dummy = MultiAgentVineEnv(**env_config)
    obs_space = dummy.observation_spaces["human_0"]
    act_space = dummy.action_spaces["human_0"]
    dummy.close()

    policies = {
        "shared_policy": PolicySpec(
            policy_class=None,
            observation_space=obs_space,
            action_space=act_space,
            config={},
        )
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_config)
        .framework("torch")
        .env_runners(
            num_env_runners=2,
            rollout_fragment_length=200,
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size_per_learner=4000,
            num_epochs=10,              # <-- FIX (was num_sgd_iter)
            minibatch_size=256,     # keep this
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=10.0,
        )
        .resources(num_gpus=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    storage_path = os.path.join(os.getcwd(), "rllib_results")  # <-- FIX

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            stop={"training_iteration": 10},
            storage_path=storage_path,   # <-- FIX (was local_dir)
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            verbose=1,
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric="episode_reward_mean", mode="max")
    print("Best mean reward:", best.metrics.get("episode_reward_mean"))
    print("Best checkpoint:", best.checkpoint)

    ray.shutdown()


if __name__ == "__main__":
    main()
