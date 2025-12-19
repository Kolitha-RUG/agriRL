# AgriRL - Agricultural Multi-Agent Reinforcement Learning

A multi-agent reinforcement learning project focused on agricultural scenarios using the VMAS (Vectorized Multi-Agent System) simulator. This project simulates cooperative tasks in vineyard environments with heterogeneous agents including drones and human workers.

## Overview

AgriRL leverages the VMAS framework to create custom agricultural scenarios where multiple agents with different dynamics and capabilities must cooperate to complete tasks such as grape harvesting and transportation in a vineyard setting.

### Key Features

- **Multi-Agent Simulation**: Support for heterogeneous agents with different dynamics:
  - **Drones**: Holonomic dynamics for aerial mobility
  - **Humans**: Differential drive dynamics for ground movement
  - **Vehicles**: Kinematic bicycle model (extensible)

- **Sensor Integration**: LIDAR sensors for agent perception and obstacle detection

- **Custom Scenarios**: Vineyard-specific scenarios with:
  - Grape harvesting mechanics
  - Delivery/transport tasks
  - Obstacle avoidance
  - Goal-based navigation

- **Vectorized Simulation**: Efficient parallel simulation across multiple environments using VMAS

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster training)

### Setup

1. **Install PyTorch** (adjust CUDA version as needed):
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

2. **Install PyTorch Geometric dependencies**:
```bash
# Get PyTorch version
python -c "import torch; print(torch.__version__.split('+')[0])"

# Install torch-cluster (adjust versions accordingly)
pip install torch-cluster -f https://data.pyg.org/whl/torch-<VERSION>+<CUDA>.html
pip install torch-geometric
```

3. **Install VMAS**:
```bash
pip install vmas
```

### Dependencies

- `torch >= 2.8.0`
- `vmas >= 1.5.2`
- `torch-geometric >= 2.7.0`
- `torch-cluster >= 1.6.3`
- `gym >= 0.26.2`
- `numpy`
- `pyglet <= 1.5.27`

## Project Structure

```
agriRL/
├── vineyard_scenario.py          # Custom vineyard scenario implementation
├── test.py                        # Test script for vineyard scenario
├── agry_try.ipynb                # Experiment notebook
├── exp.ipynb                      # Additional experiments
├── VMAS tutorial.ipynb           # VMAS framework tutorial
└── Simulation_and_training_in_VMAS_and_BenchMARL copy.ipynb
```

## Usage

### Basic Example

```python
import torch
from vmas import make_env
from vineyard_scenario import VineyardScenario

# Create environment
env = make_env(
    scenario=VineyardScenario(),
    num_envs=4,                    # Number of parallel environments
    device="cpu",                  # or "cuda"
    continuous_actions=False,       # Discrete actions
    n_humans=1,                    # Number of human agents
    n_drones=1,                    # Number of drone agents
    n_vines=2,                     # Number of vine plants
    grapes_per_vine=3,             # Grapes per vine
)

# Reset environment
obs = env.reset()
print(f"Observation shape: {obs[0].shape}")

# Run simulation
for step in range(200):
    # Generate actions for each agent
    actions = []
    for agent in env.agents:
        if agent.name.startswith("human"):
            action = torch.randint(0, 2, (4,))  # 0=transport, 1=leave
        else:
            action = torch.zeros(4, dtype=torch.long)  # Drone actions
        actions.append(action)

    # Step environment
    obs, rewards, dones, info = env.step(actions)

    # Check for delivery rewards
    if rewards[0].sum() > 0:
        print(f"Step {step}: Delivery! Reward = {rewards[0]}")

    if dones.any():
        print(f"Step {step}: Episode done!")
        break
```

### Creating Custom Scenarios

Extend the `BaseScenario` class from VMAS:

```python
from vmas.simulator.scenario import BaseScenario

class MyScenario(BaseScenario):
    def make_world(self, batch_dim, device, **kwargs):
        # Initialize world with agents, landmarks, obstacles
        pass

    def reset_world_at(self, env_index):
        # Reset specific environment
        pass

    def observation(self, agent):
        # Define agent observations
        pass

    def reward(self, agent):
        # Define reward function
        pass
```

## Agent Dynamics

### Holonomic (Drones)
- 2 actions: force_x, force_y
- Full omnidirectional movement
- Ideal for aerial agents

### Differential Drive (Humans)
- 2 actions: forward velocity, angular velocity
- Realistic ground movement
- Cannot move sideways

### Kinematic Bicycle (Vehicles)
- 2 actions: forward velocity, steering angle
- Car-like dynamics
- Suitable for agricultural vehicles

## Configuration Parameters

### World Parameters
- `batch_dim`: Number of parallel environments
- `substeps`: Physics simulation substeps (higher = more accurate)
- `dt`: Simulation timestep
- `collision_force`: Collision response strength

### Scenario Parameters
- `n_agents_drone`: Number of drone agents
- `n_agents_human`: Number of human agents
- `n_obstacles`: Number of obstacles
- `lidar_range`: LIDAR sensor range
- `n_lidar_rays`: Number of LIDAR rays per agent
- `shared_rew`: Shared vs individual rewards
- `agent_collision_penalty`: Penalty for collisions

## Rendering

VMAS provides built-in rendering capabilities. To visualize:

```python
# Enable rendering during environment creation
env = make_env(scenario=scenario, num_envs=1, render=True)

# During simulation
env.render(mode='human')
```

## Notebooks

- `VMAS tutorial.ipynb`: Introduction to VMAS framework
- `agry_try.ipynb`: Agricultural scenario experiments
- `Simulation_and_training_in_VMAS_and_BenchMARL copy.ipynb`: Training with BenchMARL

## Contributing

Feel free to extend this project with:
- New agricultural scenarios
- Additional agent types
- Advanced reward functions
- Integration with RL training frameworks (e.g., BenchMARL, RLlib)

## License

[Specify your license here]

## References

- [VMAS Documentation](https://github.com/proroklab/VectorizedMultiAgentSimulator)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [BenchMARL](https://github.com/facebookresearch/BenchMARL)

## Contact

[Your contact information]
