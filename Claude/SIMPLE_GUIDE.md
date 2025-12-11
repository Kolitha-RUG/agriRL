# Simple Vineyard - Step by Step Guide

Let's start with the SIMPLEST version and learn step by step!

## What You Have

**Two files:**
1. `simple_vineyard.py` - The environment (1 worker + 1 drone)
2. `train_simple_drone.py` - Training script for the drone

---

## Step 1: Test the Environment

First, let's make sure the environment works:

```bash
python simple_vineyard.py
```

**What happens:**
- Creates 4 parallel environments
- Worker stands still at position (-1, 0) and harvests boxes
- Drone moves RANDOMLY 
- You'll see boxes slowly accumulating at the worker
- Since drone is random, it won't deliver many boxes

**Expected output:**
```
Step  0: Boxes delivered = 0.0, Worker boxes = 0.0, Reward = 0.00
Step 10: Boxes delivered = 0.2, Worker boxes = 1.5, Reward = 0.05
Step 20: Boxes delivered = 0.5, Worker boxes = 2.0, Reward = 0.02
...
```

---

## Step 2: Train the Drone

Now let's train the drone to actually pick up and deliver boxes:

```bash
python train_simple_drone.py --train --steps 10000
```

**What happens:**
- Drone starts with random movements
- Gradually learns: "When I go to worker and then to collection point, I get reward!"
- After ~5000 steps, should start delivering boxes consistently
- Saves trained model as `simple_drone_policy.pt`

**Expected progress:**
```
Step     0: Boxes = 0.10, Reward = 0.002
Step  1000: Boxes = 0.50, Reward = 0.010
Step  2000: Boxes = 1.20, Reward = 0.025
Step  5000: Boxes = 2.50, Reward = 0.050
Step 10000: Boxes = 3.80, Reward = 0.075
```

**Training time:** ~2-5 minutes on CPU

---

## Step 3: Test the Trained Drone

See how well it learned:

```bash
python train_simple_drone.py --test
```

**What happens:**
- Loads the trained model
- Runs 5 test episodes
- Shows how many boxes were delivered

**Expected output:**
```
Episode 1: 4 boxes delivered
Episode 2: 5 boxes delivered
Episode 3: 4 boxes delivered
Episode 4: 5 boxes delivered
Episode 5: 4 boxes delivered
```

---

## Understanding the Code

### simple_vineyard.py - The Environment

```python
# Worker (Blue circle)
# - Stands at position (-1, 0)
# - Harvests boxes automatically
# - Accumulates boxes waiting for pickup

# Drone (Red circle)  
# - Starts at collection point (0, 0)
# - Can move anywhere
# - Picks up boxes when close to worker
# - Delivers boxes when close to collection point

# Collection Point (Black square)
# - At position (0, 0)
# - Where drone delivers boxes
```

### Observations (What agents see)

**Worker sees:**
- Its own position (2 numbers)
- How many boxes it has (1 number)
- Where drone is relative to it (2 numbers)
- **Total: 5 numbers**

**Drone sees:**
- Its own position (2 numbers)
- Where worker is relative to it (2 numbers)
- Where collection point is relative to it (2 numbers)
- Is it carrying a box? (1 number: 0 or 1)
- How many boxes at worker (1 number)
- **Total: 8 numbers**

### Reward

Simple: **+1** every time drone delivers a box to collection point

### What the Drone Learns

The drone learns a **policy** (a function): 
```
Observations → Actions
```

Through training, it learns:
1. "When I see worker has boxes, move toward worker"
2. "When I'm near worker, box gets attached to me"
3. "When I'm carrying box, move toward collection point"
4. "When I reach collection point, get +1 reward!"

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
You need to install PyTorch:
```bash
pip install torch vmas
```

### "ModuleNotFoundError: No module named 'vmas'"
Install VMAS:
```bash
pip install vmas
```

### Drone not learning well
Try training longer:
```bash
python train_simple_drone.py --train --steps 20000
```

Or use more parallel environments (faster learning):
```bash
python train_simple_drone.py --train --steps 10000 --envs 16
```

---

## Next Steps (After This Works)

Once you understand this simple version, we can add:

1. ✅ **Multiple workers** (drone has to choose which one)
2. ✅ **Fatigue** (workers get tired)
3. ✅ **Battery** (drone needs to recharge)
4. ✅ **Multiple drones** (coordination needed)
5. ✅ **Workers can move** (dynamic positions)

But let's start with this simple version first!

---

## Quick Commands Cheat Sheet

```bash
# Test environment
python simple_vineyard.py

# Train drone (quick)
python train_simple_drone.py --train --steps 5000

# Train drone (better)
python train_simple_drone.py --train --steps 20000 --envs 16

# Test trained drone
python train_simple_drone.py --test

# Train on GPU (faster)
python train_simple_drone.py --train --steps 10000 --device cuda
```

---

## Understanding the Output

When training, you'll see:
```
Step  5000: Boxes = 2.50, Reward = 0.050
```

- **Step 5000**: Training iteration number
- **Boxes = 2.50**: Average boxes delivered (over recent episodes)
- **Reward = 0.050**: Average reward per step

**Good performance:** Boxes > 3.0 after training
**Great performance:** Boxes > 4.0 after training

---

## What's Next?

Try it out! Run these commands:

```bash
# 1. Test it works
python simple_vineyard.py

# 2. Train the drone
python train_simple_drone.py --train --steps 10000

# 3. See how it does
python train_simple_drone.py --test
```

Once this works, tell me and we'll add the next feature! 🚀
