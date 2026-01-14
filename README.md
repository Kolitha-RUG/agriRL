# AgriRL - Agricultural Reinforcement Learning for Vineyard Harvest Coordination

A reinforcement learning framework for optimizing human-drone coordination in vineyard harvest operations using Gymnasium and Stable Baselines3.

## Overview

AgriRL simulates a realistic vineyard environment where human workers and autonomous drones must coordinate to efficiently harvest and transport grape boxes. The system uses real vineyard topology data and implements a Gymnasium-compatible environment for training RL agents with PPO (Proximal Policy Optimization).

### Key Features

- **Realistic Vineyard Simulation**: Uses actual vineyard layout data from Excel files
- **Multi-Agent Coordination**: Centralized scheduler controls multiple human workers with scripted drone assistance
- **Gymnasium Environment**: Full Gymnasium API compatibility for standard RL workflows
- **PPO Training**: Optimized training with Stable Baselines3 and parallel environment support
- **Flexible Topology Modes**:
  - `full`: Individual vine plants
  - `row`: Aggregated vine rows for faster simulation
- **Real-Time Visualization**: PyGame-based rendering for monitoring training progress

## Installation

### Prerequisites

- Python 3.8+
- CPU or CUDA-capable GPU (training uses CPU by default but can be configured)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd agriRL
```

2. **Install dependencies**:
```bash
pip install gymnasium numpy pandas matplotlib pygame
pip install stable-baselines3[extra]
pip install openpyxl  # For Excel file reading
```

### Dependencies

- `gymnasium >= 0.29.0`
- `stable-baselines3 >= 2.0.0`
- `numpy`
- `pandas`
- `matplotlib`
- `pygame`
- `openpyxl`

## Project Structure

```
agriRL/
├── vine_env.py              # Custom Gymnasium environment for vineyard simulation
├── agent.py                 # PPO training script with parallel environments
├── viz.py                   # PyGame visualization script
├── data/
│   └── Vinha_Maria_Teresa_RL.xlsx  # Real vineyard topology data
├── models/
│   └── ppo_vine_env/        # Saved model checkpoints
├── logs/
│   └── ppo_vine_env/        # TensorBoard training logs
├── Archive/                 # Previous VMAS-based implementation
└── vineyard_layout.png      # Vineyard layout visualization
```

## Environment Details

### VineEnv-v0

A custom Gymnasium environment simulating vineyard harvest operations.

#### State Space

The environment provides normalized observations (0-1 range) including:
- **Vines**: Positions, remaining boxes, queued boxes
- **Collection Point**: Central delivery location
- **Humans**: Positions, fatigue levels, current actions, box carrying status, assigned vines
- **Drones**: Positions, status (idle/traveling/delivering), box carrying status

#### Action Space

`MultiDiscrete([3, 3, ..., 3])` - One discrete action per human worker:
- `0`: **HARVEST** - Pick a box from assigned vine (8 seconds)
- `1`: **TRANSPORT** - Carry box to collection point
- `2`: **ENQUEUE** - Queue box for drone pickup

#### Reward Function

```python
reward = delivered_boxes - 0.01 * backlog_penalty - 0.001 * fatigue_penalty
```

Optimizes for:
- Maximizing delivered boxes
- Minimizing queued boxes waiting for drones
- Minimizing worker fatigue

#### Configuration Parameters

```python
env = gym.make('VineEnv-v0',
    render_mode="terminal",      # "terminal", "human", or None
    topology_mode="row",         # "full" or "row"
    num_humans=2,                # Number of human workers
    num_drones=1,                # Number of autonomous drones
    max_boxes_per_vine=10,       # Boxes per vine/row
    max_backlog=10,              # Maximum queued boxes
    max_steps=200,               # Episode length
    dt=1.0,                      # Seconds per step
    harvest_time=8.0,            # Seconds to harvest one box
    human_speed=1.0,             # Human movement speed (units/sec)
    drone_speed=2.0,             # Drone movement speed (units/sec)
)
```

## Usage

### Training an Agent

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from vine_env import VineEnv

# Create parallel environments
env = make_vec_env(VineEnv, n_envs=12)

# Initialize PPO model with exploration bonus
model = PPO('MlpPolicy', env,
            verbose=1,
            device='cpu',
            ent_coef=0.05,  # Encourage exploration
            tensorboard_log='./logs/ppo_vine_env')

# Train the model
model.learn(total_timesteps=1_000_000)

# Save the model
model.save('./models/ppo_vine_env')
```

### Running the Training Script

```bash
python agent.py
```

This will:
- Create 12 parallel environments for efficient training
- Train using PPO with entropy coefficient 0.05
- Evaluate every 10,000 steps
- Save best models to `models/ppo_vine_env/PPO/`
- Log metrics to TensorBoard

### Visualizing Training Progress

```bash
tensorboard --logdir=./logs/ppo_vine_env
```

### Testing the Environment

```python
import gymnasium as gym
from vine_env import VineEnv

# Create environment
env = gym.make('VineEnv-v0', render_mode="human", topology_mode="row")
obs, info = env.reset()

# Run random policy
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

    print(f"Reward: {reward:.2f}, Delivered: {info['delivered']}")
```

### Running the Visualization

```bash
python viz.py
```

This launches a PyGame window showing:
- Vines (green squares) with remaining/queued box counts
- Human workers (blue circles) labeled H0, H1, ...
- Drones (red triangles) labeled D0, D1, ...
- Collection point (gold circle)

## Agent Behavior

### Human Workers
- Assigned to specific vines
- Can harvest boxes (8 seconds per box)
- Transport boxes to collection point
- Queue boxes for drone pickup
- Experience fatigue over time

### Drones (Scripted)
- Automatically navigate to vines with queued boxes
- Pick up queued boxes
- Deliver to collection point
- Always select nearest queued vine

### Coordination Strategy

The RL agent learns to coordinate human actions by:
1. Deciding when workers should harvest vs. transport
2. Balancing direct transport with drone-assisted delivery
3. Managing worker fatigue
4. Optimizing queue utilization for drone efficiency

## Training Tips

- **Exploration**: Higher `ent_coef` (0.05-0.1) helps discover drone coordination strategies
- **Parallel Envs**: Use 8-16 environments for faster training
- **Topology Mode**: Start with `"row"` mode for faster iteration, then scale to `"full"`
- **Episode Length**: Adjust `max_steps` based on vineyard size and worker count

## Data Format

The vineyard Excel file should contain:
- `x`, `y`, `z`: Vine positions (converted from mm to meters)
- `line`: Vine row identifier
- `lot`: Lot/block identifier

Used in `row` mode to aggregate vines per row.

## Performance Monitoring

Key metrics to track:
- **Total Delivered**: Cumulative boxes delivered
- **Backlog**: Boxes waiting for drone pickup
- **Episode Length**: Steps until completion
- **Average Reward**: Per-step reward signal

## Rendering Modes

- `"terminal"`: Text-based output with detailed state information
- `"human"`: PyGame visualization (best for debugging)
- `None`: No rendering (fastest for training)

## License

[Specify your license here]

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

## Contact

[Your contact information]
