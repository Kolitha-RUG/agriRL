# train_vineyard_simple.py
import torch
from torch import nn
from torch.distributions import Normal
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

from vineyard_scenario import VineyardScenario

# === CONFIG ===
device = "cuda" if torch.cuda.is_available() else "cpu"
num_envs = 32
max_steps = 500
frames_per_batch = 2016  # Divisible by 32
total_frames = 200_000
num_epochs = 10
mini_batch_size = 256
clip_epsilon = 0.2
gamma = 0.99
gae_lambda = 0.95
lr = 3e-4
max_grad_norm = 1.0
entropy_coef = 0.01
value_coef = 0.5

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

obs_size = 22
action_size = 2


# === NETWORKS ===
class ActorCritic(nn.Module):
    """Combined actor-critic network."""
    
    def __init__(self, obs_size, action_size, hidden_size=64):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        self.critic = nn.Linear(hidden_size, 1)
    
    def get_action_and_value(self, obs, action=None):
        shared = self.shared(obs)
        
        mean = self.actor_mean(shared)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(shared)
        
        return action, log_prob, entropy, value
    
    def get_action(self, obs):
        """Get action only (for data collection)."""
        shared = self.shared(obs)
        mean = self.actor_mean(shared)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        return torch.tanh(action)  # Bound to [-1, 1]


# Create networks
human_net = ActorCritic(obs_size, action_size).to(device)
drone_net = ActorCritic(obs_size, action_size).to(device)

print("Networks created!")


# === POLICY WRAPPER FOR COLLECTOR ===
class PolicyWrapper:
    """Wraps networks for use with collector."""
    
    def __init__(self, human_net, drone_net):
        self.human_net = human_net
        self.drone_net = drone_net
    
    def __call__(self, td):
        with torch.no_grad():
            human_obs = td[("human", "observation")]
            drone_obs = td[("drone", "observation")]
            
            human_action = self.human_net.get_action(human_obs)
            drone_action = self.drone_net.get_action(drone_obs)
            
            td[("human", "action")] = human_action
            td[("drone", "action")] = drone_action
        
        return td


policy = PolicyWrapper(human_net, drone_net)


# === DATA COLLECTOR ===
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device=device,
)

print("Collector created!")


# === COMPUTE GAE ===
def compute_gae(rewards, values, dones, gamma, lam):
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t].float()) * last_gae
    
    returns = advantages + values
    return advantages, returns


# === OPTIMIZERS ===
human_optimizer = torch.optim.Adam(human_net.parameters(), lr=lr)
drone_optimizer = torch.optim.Adam(drone_net.parameters(), lr=lr)


# === TRAINING LOOP ===
print("\n=== Starting Training ===\n")

best_reward = -float('inf')

for batch_idx, batch in enumerate(collector):
    
    # === EXTRACT DATA ===
    human_obs = batch[("human", "observation")].reshape(-1, obs_size)
    human_action = batch[("human", "action")].reshape(-1, action_size)
    human_reward = batch[("next", "human", "reward")].reshape(-1)
    human_done = batch[("next", "human", "done")].reshape(-1)
    
    drone_obs = batch[("drone", "observation")].reshape(-1, obs_size)
    drone_action = batch[("drone", "action")].reshape(-1, action_size)
    drone_reward = batch[("next", "drone", "reward")].reshape(-1)
    drone_done = batch[("next", "drone", "done")].reshape(-1)
    
    # === COMPUTE ADVANTAGES ===
    with torch.no_grad():
        _, _, _, human_values = human_net.get_action_and_value(human_obs)
        _, _, _, drone_values = drone_net.get_action_and_value(drone_obs)
        
        human_values = human_values.squeeze(-1)
        drone_values = drone_values.squeeze(-1)
        
        human_adv, human_returns = compute_gae(human_reward, human_values, human_done, gamma, gae_lambda)
        drone_adv, drone_returns = compute_gae(drone_reward, drone_values, drone_done, gamma, gae_lambda)
        
        human_adv = (human_adv - human_adv.mean()) / (human_adv.std() + 1e-8)
        drone_adv = (drone_adv - drone_adv.mean()) / (drone_adv.std() + 1e-8)
        
        _, human_old_logprob, _, _ = human_net.get_action_and_value(human_obs, human_action)
        _, drone_old_logprob, _, _ = drone_net.get_action_and_value(drone_obs, drone_action)
    
    # === PPO UPDATE ===
    num_samples = human_obs.shape[0]
    
    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples, device=device)
        
        for start in range(0, num_samples, mini_batch_size):
            end = min(start + mini_batch_size, num_samples)
            mb_idx = perm[start:end]
            
            # Human update
            _, new_logprob, entropy, new_value = human_net.get_action_and_value(
                human_obs[mb_idx], human_action[mb_idx]
            )
            new_value = new_value.squeeze(-1)
            
            ratio = (new_logprob - human_old_logprob[mb_idx]).exp()
            pg_loss1 = -human_adv[mb_idx] * ratio
            pg_loss2 = -human_adv[mb_idx] * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((new_value - human_returns[mb_idx]) ** 2).mean()
            ent_loss = -entropy.mean()
            loss = pg_loss + value_coef * v_loss + entropy_coef * ent_loss
            
            human_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(human_net.parameters(), max_grad_norm)
            human_optimizer.step()
            
            # Drone update
            _, new_logprob, entropy, new_value = drone_net.get_action_and_value(
                drone_obs[mb_idx], drone_action[mb_idx]
            )
            new_value = new_value.squeeze(-1)
            
            ratio = (new_logprob - drone_old_logprob[mb_idx]).exp()
            pg_loss1 = -drone_adv[mb_idx] * ratio
            pg_loss2 = -drone_adv[mb_idx] * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((new_value - drone_returns[mb_idx]) ** 2).mean()
            ent_loss = -entropy.mean()
            loss = pg_loss + value_coef * v_loss + entropy_coef * ent_loss
            
            drone_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drone_net.parameters(), max_grad_norm)
            drone_optimizer.step()
    
    # === LOGGING ===
    total_reward = human_reward.sum().item() + drone_reward.sum().item()
    frames_done = (batch_idx + 1) * frames_per_batch
    
    print(f"Batch {batch_idx+1:3d} | Frames: {frames_done:6d}/{total_frames} | "
          f"Reward: {total_reward:6.1f}")
    
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save({
            'human_net': human_net.state_dict(),
            'drone_net': drone_net.state_dict(),
        }, 'vineyard_best.pt')

print("\n=== Training Complete ===")
print(f"Best reward: {best_reward:.1f}")

torch.save({
    'human_net': human_net.state_dict(),
    'drone_net': drone_net.state_dict(),
}, 'vineyard_final.pt')
print("Models saved!")

collector.shutdown()
env.close()