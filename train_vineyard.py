# train_vineyard.py
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from vineyard_scenario import VineyardScenario

# === Step 1: Override VMAS task to use our scenario ===
# BenchMARL expects registered tasks, so we trick it
from benchmarl.environments.vmas import VmasTask
from torchrl.envs import VmasEnv

# Store original method
_original_get_env_fun = VmasTask.get_env_fun

def custom_get_env_fun(self, num_envs, continuous_actions, seed, device):
    """Override to use our VineyardScenario."""
    
    # If task name contains "vineyard", use our scenario
    if "navigation" in self.name.lower():  # We'll use NAVIGATION as placeholder
        return lambda: VmasEnv(
            scenario=VineyardScenario(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            n_humans=1,
            n_drones=1,
            n_vines=2,
            grapes_per_vine=3,
        )
    else:
        return _original_get_env_fun(self, num_envs, continuous_actions, seed, device)

# Apply override
VmasTask.get_env_fun = custom_get_env_fun


# === Step 2: Configure experiment ===
experiment_config = ExperimentConfig.get_from_yaml()

# Adjust for quick testing
experiment_config.max_n_frames = 100_000          # Total training frames
experiment_config.on_policy_collected_frames_per_batch = 1000  # Frames per iteration
experiment_config.on_policy_n_envs_per_worker = 10            # Parallel envs
experiment_config.evaluation = True
experiment_config.render = False                   # Set True to see during training
experiment_config.evaluation_interval = 10_000
experiment_config.evaluation_episodes = 10


# === Step 3: Configure algorithm (MAPPO) ===
algorithm_config = MappoConfig.get_from_yaml()


# === Step 4: Configure neural network ===
model_config = MlpConfig.get_from_yaml()
model_config.num_cells = [64, 64]  # Two hidden layers with 64 neurons


# === Step 5: Choose task ===
task = VmasTask.NAVIGATION.get_from_yaml()  # We override this to use VineyardScenario


# === Step 6: Create and run experiment ===
experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    seed=0,
    config=experiment_config,
)

# Run training!
experiment.run()