import ray
from ray.rllib.algorithms.ppo import PPOConfig
from vine_env_multiagent import MultiAgentVineEnv

ray.init(ignore_reinit_error=True)

def policy_mapping_fn(agent_id, episode, **kwargs):
    return "shared_policy"


config = (
    PPOConfig()
    # --- ENV ---
    .environment(
        env=MultiAgentVineEnv,
        env_config={"max_steps": 200},
        disable_env_checking=True,
    )

    # --- ENV RUNNERS (replaces rollouts) ---
    .env_runners(
        num_env_runners=0,   # local sampling only (debug-safe)
    )

    # --- TRAINING ---
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size_per_learner=2000,
        minibatch_size=256,
        num_epochs=5,
    )

    # --- MULTI-AGENT (MANDATORY) ---
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=policy_mapping_fn,
        count_steps_by="agent_steps",
    )

    # --- FRAMEWORK ---
    .framework("torch")

    # --- DEBUGGING ---
    .debugging(log_level="INFO")
)

algo = config.build()

for i in range(5):
    result = algo.train()
    print(
        f"\n Iter {i} | "
        f"episode_reward_mean = {result['sampler_results']['episode_return_mean']} \n"
    )

algo.stop()
ray.shutdown()
