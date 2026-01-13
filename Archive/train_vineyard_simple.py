# train_vineyard_simple.py
import torch
from torch import nn
from torch.distributions import Normal
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

from vineyard_scenario import VineyardScenario


# === ACTOR-CRITIC NETWORK (can be imported) ===
class ActorCritic(nn.Module):
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
        shared = self.shared(obs)
        mean = self.actor_mean(shared)
        std = self.actor_log_std.exp().expand_as(mean)
        action = Normal(mean, std).sample()
        return torch.tanh(action)


# === GAE COMPUTATION (can be imported) ===
def compute_gae(rewards, values, dones, gamma, lam):
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_value = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t].float()) * last_gae
    return advantages, advantages + values


# === TRAINING CODE (only runs when executed directly) ===
if __name__ == "__main__":
    
    # === CONFIG ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 32
    max_steps = 500
    frames_per_batch = 2016
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

    # Create networks
    human_net = ActorCritic(obs_size, action_size).to(device)
    drone_net = ActorCritic(obs_size, action_size).to(device)
    human_optimizer = torch.optim.Adam(human_net.parameters(), lr=lr)
    drone_optimizer = torch.optim.Adam(drone_net.parameters(), lr=lr)

    print("Networks created!")

    # === POLICY FOR COLLECTOR ===
    class Policy:
        def __init__(self, human_net, drone_net):
            self.human_net = human_net
            self.drone_net = drone_net
        
        def __call__(self, td):
            with torch.no_grad():
                td[("human", "action")] = self.human_net.get_action(td[("human", "observation")])
                td[("drone", "action")] = self.drone_net.get_action(td[("drone", "observation")])
            return td

    policy = Policy(human_net, drone_net)

    # === DATA COLLECTOR ===
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )
    print("Collector created!")

    # === TRAINING LOOP ===
    print("\n=== Starting Training ===\n")
    best_reward = -float('inf')

    for batch_idx, batch in enumerate(collector):
        
        # Extract data
        h_obs = batch[("human", "observation")].reshape(-1, obs_size)
        h_act = batch[("human", "action")].reshape(-1, action_size)
        h_rew = batch[("next", "human", "reward")].reshape(-1)
        
        d_obs = batch[("drone", "observation")].reshape(-1, obs_size)
        d_act = batch[("drone", "action")].reshape(-1, action_size)
        d_rew = batch[("next", "drone", "reward")].reshape(-1)
        
        # Get done flags
        try:
            h_done = batch[("next", "done")].reshape(-1)
            d_done = batch[("next", "done")].reshape(-1)
        except KeyError:
            h_done = torch.zeros_like(h_rew, dtype=torch.bool)
            d_done = torch.zeros_like(d_rew, dtype=torch.bool)
        
        # Compute advantages
        with torch.no_grad():
            _, _, _, h_val = human_net.get_action_and_value(h_obs)
            _, _, _, d_val = drone_net.get_action_and_value(d_obs)
            h_val, d_val = h_val.squeeze(-1), d_val.squeeze(-1)
            
            h_adv, h_ret = compute_gae(h_rew, h_val, h_done, gamma, gae_lambda)
            d_adv, d_ret = compute_gae(d_rew, d_val, d_done, gamma, gae_lambda)
            
            h_adv = (h_adv - h_adv.mean()) / (h_adv.std() + 1e-8)
            d_adv = (d_adv - d_adv.mean()) / (d_adv.std() + 1e-8)
            
            _, h_old_lp, _, _ = human_net.get_action_and_value(h_obs, h_act)
            _, d_old_lp, _, _ = drone_net.get_action_and_value(d_obs, d_act)
        
        # PPO updates
        n = h_obs.shape[0]
        for _ in range(num_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                
                # Human
                _, lp, ent, val = human_net.get_action_and_value(h_obs[idx], h_act[idx])
                ratio = (lp - h_old_lp[idx]).exp()
                loss = torch.max(-h_adv[idx] * ratio, 
                               -h_adv[idx] * ratio.clamp(1-clip_epsilon, 1+clip_epsilon)).mean()
                loss += value_coef * ((val.squeeze() - h_ret[idx])**2).mean()
                loss -= entropy_coef * ent.mean()
                
                human_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(human_net.parameters(), max_grad_norm)
                human_optimizer.step()
                
                # Drone
                _, lp, ent, val = drone_net.get_action_and_value(d_obs[idx], d_act[idx])
                ratio = (lp - d_old_lp[idx]).exp()
                loss = torch.max(-d_adv[idx] * ratio,
                               -d_adv[idx] * ratio.clamp(1-clip_epsilon, 1+clip_epsilon)).mean()
                loss += value_coef * ((val.squeeze() - d_ret[idx])**2).mean()
                loss -= entropy_coef * ent.mean()
                
                drone_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(drone_net.parameters(), max_grad_norm)
                drone_optimizer.step()
        
        # Log
        total_reward = h_rew.sum().item() + d_rew.sum().item()
        frames = (batch_idx + 1) * frames_per_batch
        print(f"Batch {batch_idx+1:3d} | Frames: {frames:6d}/{total_frames} | Reward: {total_reward:.1f}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({'human': human_net.state_dict(), 'drone': drone_net.state_dict()}, 'best_model.pt')

    print(f"\n=== Done! Best reward: {best_reward:.1f} ===")
    torch.save({'human': human_net.state_dict(), 'drone': drone_net.state_dict()}, 'final_model.pt')
    collector.shutdown()
    env.close()