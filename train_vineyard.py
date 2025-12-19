# train_vineyard.py
import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.envs.utils import check_env_specs

from vineyard_scenario import VineyardScenario

# === CONFIG ===
device = "cuda" if torch.cuda.is_available() else "cpu"
num_envs = 32
max_steps = 500
frames_per_batch = 2000
total_frames = 100_000
num_epochs = 10
clip_epsilon = 0.2
gamma = 0.99
lr = 3e-4
max_grad_norm = 1.0

print(f"Using device: {device}")

# === CREATE ENVIRONMENT ===
env = VmasEnv(
    scenario=VineyardScenario(),
    num_envs=num_envs,
    device=device,
    continuous_actions=True,
    max_steps=max_steps,
    n_humans=1,
    n_drones=1,
    n_vines=2,
    grapes_per_vine=3,
)

print("Environment created!")
check_env_specs(env)
print("Environment check passed!")

# === GET SPECS ===
print("\nAgent groups:", list(env.observation_spec.keys()))

obs_size = 22  # From observation spec
action_size = 2  # From action spec

print(f"Obs size: {obs_size}, Action size: {action_size}")


# === HELPER: Create actor for an agent group ===
def make_actor(group_name):
    """Create actor (policy) for one agent group."""
    
    obs_key = (group_name, "observation")
    action_key = (group_name, "action")
    
    # Policy network: obs -> [loc, scale]
    policy_net = nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 2 * action_size),
    ).to(device)
    
    # TensorDict module for policy
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[obs_key],
        out_keys=[(group_name, "params")],
    )
    
    # Split into loc and scale
    split_module = TensorDictModule(
        NormalParamExtractor(),
        in_keys=[(group_name, "params")],
        out_keys=[(group_name, "loc"), (group_name, "scale")],
    )
    
    # Combine into sequence
    policy_seq = TensorDictSequential(policy_module, split_module)
    
    # Probabilistic actor
    actor = ProbabilisticActor(
        module=policy_seq,
        in_keys=[(group_name, "loc"), (group_name, "scale")],
        out_keys=[action_key],
        distribution_class=TanhNormal,
        return_log_prob=True,
        log_prob_key=(group_name, "log_prob"),
    )
    
    return actor, policy_net


def make_critic(group_name):
    """Create critic (value function) for one agent group."""
    
    obs_key = (group_name, "observation")
    value_key = (group_name, "state_value")
    
    value_net = nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    ).to(device)
    
    critic = ValueOperator(
        module=value_net,
        in_keys=[obs_key],
        out_keys=[value_key],
    )
    
    return critic, value_net


# === CREATE ACTORS AND CRITICS ===
human_actor, human_policy_net = make_actor("human")
human_critic, human_value_net = make_critic("human")

drone_actor, drone_policy_net = make_actor("drone")
drone_critic, drone_value_net = make_critic("drone")

# Combined actor for data collection
combined_actor = TensorDictSequential(human_actor, drone_actor)

print("Networks created!")

# === DATA COLLECTOR ===
collector = SyncDataCollector(
    env,
    combined_actor,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device=device,
)

print("Collector created!")

# === REPLAY BUFFER ===
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
    batch_size=256,
)

# === LOSS FUNCTIONS ===
human_loss = ClipPPOLoss(
    actor_network=human_actor,
    critic_network=human_critic,
    clip_epsilon=clip_epsilon,
    entropy_bonus=True,
    entropy_coef=0.01,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)
human_loss.set_keys(
    reward=("human", "reward"),
    action=("human", "action"),
    done=("human", "done"),
    terminated=("human", "terminated"),
    value=("human", "state_value"),
    sample_log_prob=("human", "log_prob"),
)
human_loss.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=0.95)

drone_loss = ClipPPOLoss(
    actor_network=drone_actor,
    critic_network=drone_critic,
    clip_epsilon=clip_epsilon,
    entropy_bonus=True,
    entropy_coef=0.01,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)
drone_loss.set_keys(
    reward=("drone", "reward"),
    action=("drone", "action"),
    done=("drone", "done"),
    terminated=("drone", "terminated"),
    value=("drone", "state_value"),
    sample_log_prob=("drone", "log_prob"),
)
drone_loss.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=0.95)

# === OPTIMIZERS ===
human_optimizer = torch.optim.Adam(
    list(human_policy_net.parameters()) + list(human_value_net.parameters()), 
    lr=lr
)
drone_optimizer = torch.optim.Adam(
    list(drone_policy_net.parameters()) + list(drone_value_net.parameters()), 
    lr=lr
)

# === TRAINING LOOP ===
print("\n=== Starting Training ===\n")

for i, batch in enumerate(collector):
    
    # Get reward info
    human_reward = batch[("next", "human", "reward")].sum().item()
    drone_reward = batch[("next", "drone", "reward")].sum().item()
    
    # Store in replay buffer
    replay_buffer.extend(batch.reshape(-1))
    
    # PPO update
    for epoch in range(num_epochs):
        for minibatch in replay_buffer:
            minibatch = minibatch.to(device)
            
            # Human update
            try:
                human_loss_vals = human_loss(minibatch)
                h_loss = (
                    human_loss_vals["loss_objective"]
                    + human_loss_vals["loss_critic"]
                    + human_loss_vals["loss_entropy"]
                )
                human_optimizer.zero_grad()
                h_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(human_policy_net.parameters()) + list(human_value_net.parameters()), 
                    max_grad_norm
                )
                human_optimizer.step()
            except Exception as e:
                print(f"Human loss error: {e}")
            
            # Drone update
            try:
                drone_loss_vals = drone_loss(minibatch)
                d_loss = (
                    drone_loss_vals["loss_objective"]
                    + drone_loss_vals["loss_critic"]
                    + drone_loss_vals["loss_entropy"]
                )
                drone_optimizer.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(drone_policy_net.parameters()) + list(drone_value_net.parameters()), 
                    max_grad_norm
                )
                drone_optimizer.step()
            except Exception as e:
                print(f"Drone loss error: {e}")
    
    # Clear buffer
    replay_buffer.empty()
    
    # Logging
    frames_done = (i + 1) * frames_per_batch
    print(f"Batch {i+1} | Frames: {frames_done}/{total_frames} | "
          f"Human R: {human_reward:.1f} | Drone R: {drone_reward:.1f}")

print("\n=== Training Complete ===")

# Save model
torch.save({
    'human_policy': human_policy_net.state_dict(),
    'human_value': human_value_net.state_dict(),
    'drone_policy': drone_policy_net.state_dict(),
    'drone_value': drone_value_net.state_dict(),
}, 'vineyard_model.pt')
print("Model saved to vineyard_model.pt")

collector.shutdown()
env.close()