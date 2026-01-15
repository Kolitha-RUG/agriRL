"""
Training Script for Multi-Agent Vine Environment with RLlib

This script provides different training configurations:
1. Shared Policy: All agents share a single policy (parameter sharing)
2. Independent Policies: Each agent has its own policy
3. Heterogeneous Policies: Different agent types use different policies

Usage:
    python train_multiagent_vine.py --mode shared --num-iterations 100
    python train_multiagent_vine.py --mode independent --num-iterations 100
"""

import argparse
import os
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

# Import the multi-agent environment
from vine_env_multiagent import MultiAgentVineEnv


def register_environment():
    """Register the multi-agent vine environment with Ray."""
    register_env(
        "MultiAgentVineEnv-v0",
        lambda config: MultiAgentVineEnv(config)
    )


def get_env_config(
    topology_mode: str = "row",
    num_humans: int = 2,
    num_drones: int = 1,
) -> Dict[str, Any]:
    """Get default environment configuration."""
    return {
        "topology_mode": topology_mode,
        "num_humans": num_humans,
        "num_drones": num_drones,
        "max_boxes_per_vine": 10,
        "max_backlog": 10,
        "max_steps": 200,
        "dt": 1.0,
        "harvest_time": 8.0,
        "human_speed": 1.0,
        "drone_speed": 2.0,
        "render_mode": "terminal",
        # Reward shaping
        "reward_delivery": 1.0,
        "reward_backlog_penalty": 0.1,
        "reward_fatigue_penalty": 0.1,
    }


def create_shared_policy_config(
    env_config: Dict,
    num_workers: int = 2,
    train_batch_size: int = 4000,
) -> PPOConfig:
    """
    Create configuration where all agents share a single policy.
    
    This is often a good starting point as it:
    - Reduces the number of parameters to learn
    - Allows agents to learn from each other's experiences
    - Works well when agents have similar roles
    """
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_policy"
    
    config = (
        PPOConfig()
        .environment(env="MultiAgentVineEnv-v0", env_config=env_config)
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=train_batch_size,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            num_sgd_iter=10,
            minibatch_size=128,
        )
        .multi_agent(
            policy_mapping_fn=policy_mapping_fn,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(),
                }
            ),
        )
    )
    
    return config


def create_independent_policy_config(
    env_config: Dict,
    num_workers: int = 2,
    train_batch_size: int = 4000,
) -> PPOConfig:
    """
    Create configuration where each agent has its own policy.
    
    This allows:
    - Agents to develop specialized behaviors
    - Different learning rates per agent
    - Heterogeneous strategies
    """
    # Get agent IDs
    temp_env = MultiAgentVineEnv(env_config)
    agent_ids = temp_env.possible_agents
    temp_env.close()
    
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id
    
    config = (
        PPOConfig()
        .environment(env="MultiAgentVineEnv-v0", env_config=env_config)
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=train_batch_size,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            num_sgd_iter=10,
            minibatch_size=128,
        )
        .multi_agent(
            policy_mapping_fn=policy_mapping_fn,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    agent_id: RLModuleSpec() for agent_id in agent_ids
                }
            ),
        )
    )
    
    return config


def create_centralized_critic_config(
    env_config: Dict,
    num_workers: int = 2,
    train_batch_size: int = 4000,
) -> PPOConfig:
    """
    Create configuration with centralized training and decentralized execution (CTDE).
    
    In this setup:
    - Each agent has its own policy (actor)
    - Critics can access additional information during training
    - Useful for cooperative scenarios where coordination is important
    
    Note: This is a simplified version. For full CTDE implementations,
    consider using algorithms like MAPPO, QMIX, or VDN.
    """
    temp_env = MultiAgentVineEnv(env_config)
    agent_ids = temp_env.possible_agents
    temp_env.close()
    
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id
    
    config = (
        PPOConfig()
        .environment(env="MultiAgentVineEnv-v0", env_config=env_config)
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=train_batch_size,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            num_sgd_iter=10,
            minibatch_size=128,
        )
        .multi_agent(
            policy_mapping_fn=policy_mapping_fn,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    agent_id: RLModuleSpec() for agent_id in agent_ids
                }
            ),
        )
    )
    
    return config


def train(
    mode: str = "shared",
    num_iterations: int = 100,
    num_workers: int = 2,
    checkpoint_freq: int = 10,
    results_dir: str = "./results",
    env_config: Dict = None,
):
    """
    Run training with the specified configuration.
    
    Args:
        mode: One of "shared", "independent", or "centralized"
        num_iterations: Number of training iterations
        num_workers: Number of parallel workers
        checkpoint_freq: Save checkpoint every N iterations
        results_dir: Directory to save results
        env_config: Environment configuration dict
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register environment
    register_environment()
    
    # Get environment config
    if env_config is None:
        env_config = get_env_config()
    
    # Create algorithm config based on mode
    if mode == "shared":
        config = create_shared_policy_config(env_config, num_workers)
        print("Training with SHARED policy (all agents share one policy)")
    elif mode == "independent":
        config = create_independent_policy_config(env_config, num_workers)
        print("Training with INDEPENDENT policies (each agent has own policy)")
    elif mode == "centralized":
        config = create_centralized_critic_config(env_config, num_workers)
        print("Training with CENTRALIZED critic configuration")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Build algorithm
    algo = config.build()
    
    print(f"\nStarting training for {num_iterations} iterations...")
    print(f"Results will be saved to: {results_dir}")
    print("-" * 60)
    
    best_reward = float("-inf")
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Extract relevant metrics
        episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
        episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", 0)
        
        # Track best reward
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
        
        # Print progress
        print(f"Iteration {i + 1:4d} | "
              f"Reward: {episode_reward_mean:8.2f} | "
              f"Episode Length: {episode_len_mean:6.1f} | "
              f"Best: {best_reward:8.2f}")
        
        # Save checkpoint
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_path = algo.save(results_dir)
            print(f"  -> Checkpoint saved: {checkpoint_path}")
    
    # Final checkpoint
    final_checkpoint = algo.save(results_dir)
    print(f"\nTraining complete! Final checkpoint: {final_checkpoint}")
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    return final_checkpoint


def evaluate(checkpoint_path: str, num_episodes: int = 5, render: bool = True):
    """
    Evaluate a trained model.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    from ray.rllib.algorithms.algorithm import Algorithm
    
    ray.init(ignore_reinit_error=True)
    register_environment()
    
    # Load the trained algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Create evaluation environment
    env_config = get_env_config()
    env_config["render_mode"] = "human" if render else "terminal"
    env = MultiAgentVineEnv(env_config)
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    
    total_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            # Get actions from all policies
            actions = {}
            for agent_id in env.agents:
                policy_id = algo.config.multi_agent_config["policy_mapping_fn"](
                    agent_id, None
                )
                action = algo.compute_single_action(
                    obs[agent_id], policy_id=policy_id
                )
                actions[agent_id] = action
            
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            
            if render:
                env.render()
            
            done = terminateds["__all__"] or truncateds["__all__"]
            step += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, "
              f"Steps = {step}, Delivered = {env.delivered}")
    
    print(f"\nMean Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    
    env.close()
    algo.stop()
    ray.shutdown()


def hyperparameter_search(
    num_samples: int = 10,
    max_iterations: int = 50,
    results_dir: str = "./tune_results",
):
    """
    Run hyperparameter search using Ray Tune.
    
    This searches over:
    - Learning rate
    - Entropy coefficient
    - GAE lambda
    - Number of SGD iterations
    """
    ray.init(ignore_reinit_error=True)
    register_environment()
    
    env_config = get_env_config()
    
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_policy"
    
    # Define search space
    config = (
        PPOConfig()
        .environment(env="MultiAgentVineEnv-v0", env_config=env_config)
        .env_runners(num_env_runners=2)
        .training(
            lr=tune.loguniform(1e-5, 1e-3),
            entropy_coeff=tune.uniform(0.0, 0.1),
            lambda_=tune.uniform(0.9, 1.0),
            num_sgd_iter=tune.choice([5, 10, 20]),
            train_batch_size=4000,
        )
        .multi_agent(policy_mapping_fn=policy_mapping_fn)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={"shared_policy": RLModuleSpec()}
            )
        )
    )
    
    # Run tune
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            stop={"training_iteration": max_iterations},
            storage_path=results_dir,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                num_to_keep=3,
            ),
        ),
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_reward_mean",
            mode="max",
            num_samples=num_samples,
        ),
    )
    
    results = tuner.fit()
    
    # Get best result
    best_result = results.get_best_result(
        metric="env_runners/episode_reward_mean",
        mode="max"
    )
    
    print(f"\nBest hyperparameters: {best_result.config}")
    print(f"Best reward: {best_result.metrics['env_runners']['episode_reward_mean']}")
    
    ray.shutdown()
    return best_result


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Agent Vine Environment")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--mode", type=str, default="shared",
        choices=["shared", "independent", "centralized"],
        help="Policy mode: shared (one policy for all), independent (one per agent)"
    )
    train_parser.add_argument(
        "--num-iterations", type=int, default=100,
        help="Number of training iterations"
    )
    train_parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Number of parallel workers"
    )
    train_parser.add_argument(
        "--num-humans", type=int, default=2,
        help="Number of human agents"
    )
    train_parser.add_argument(
        "--topology", type=str, default="row",
        choices=["row", "full"],
        help="Vineyard topology mode"
    )
    train_parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Directory to save results"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "checkpoint", type=str,
        help="Path to checkpoint to evaluate"
    )
    eval_parser.add_argument(
        "--num-episodes", type=int, default=5,
        help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--no-render", action="store_true",
        help="Disable rendering"
    )
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter search")
    tune_parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of hyperparameter combinations to try"
    )
    tune_parser.add_argument(
        "--max-iterations", type=int, default=50,
        help="Max iterations per trial"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test the environment")
    
    args = parser.parse_args()
    
    if args.command == "train":
        env_config = get_env_config(
            topology_mode=args.topology,
            num_humans=args.num_humans,
        )
        train(
            mode=args.mode,
            num_iterations=args.num_iterations,
            num_workers=args.num_workers,
            results_dir=args.results_dir,
            env_config=env_config,
        )
    
    elif args.command == "evaluate":
        evaluate(
            checkpoint_path=args.checkpoint,
            num_episodes=args.num_episodes,
            render=not args.no_render,
        )
    
    elif args.command == "tune":
        hyperparameter_search(
            num_samples=args.num_samples,
            max_iterations=args.max_iterations,
        )
    
    elif args.command == "test":
        from vine_env_multiagent import test_environment
        test_environment()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
