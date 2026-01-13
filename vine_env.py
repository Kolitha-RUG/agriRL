import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt

# Register once (ensure the module path matches your filename)
register(
    id="VineEnv-v0",
    entry_point="vine_env:VineEnv",  # file vine_env.py contains VineEnv
)

# === Human actions ===
ACTION_HARVEST   = 0
ACTION_TRANSPORT = 1
ACTION_ENQUEUE   = 2

# === Drone status (scripted) ===
DRONE_IDLE       = 0
DRONE_GO_TO_VINE = 1
DRONE_DELIVER    = 2


def one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v


class Vine:
    def __init__(self, position, max_boxes: int):
        self.position = np.array(position, dtype=np.float32)
        self.total_boxes = int(max_boxes)
        self.boxes_remaining = int(max_boxes)
        self.queued_boxes = 0

    def harvest_box(self) -> bool:
        """Remove 1 available box from the vine."""
        if self.boxes_remaining > 0:
            self.boxes_remaining -= 1
            return True
        return False


class Human:
    def __init__(self, position, assigned_vine: int):
        self.position = np.array(position, dtype=np.float32)
        self.assigned_vine = int(assigned_vine)

        self.fatigue = 0.0
        self.has_box = False

        self.current_action = ACTION_HARVEST
        self.busy = False
        self.time_left = 0.0  # seconds


class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)

        self.status = DRONE_IDLE
        self.has_box = False

        self.busy = False
        self.time_left = 0.0  # seconds
        self.target_vine = None  # int index or None


class VineEnv(gym.Env):
    metadata = {"render_modes": ["terminal","human"], "render_fps": 1}

    def __init__(
        self,
        render_mode="terminal",
        num_vines=5,
        num_humans=2,
        num_drones=1,
        max_boxes_per_vine=10,
        max_backlog=10,
        field_size=(100.0, 100.0),
        max_steps=200,
        # time model
        dt=1.0,                 # seconds per env step
        harvest_time=8.0,        # seconds to fill/produce 1 box
        human_speed=1.0,         # units/sec (for transport)
        drone_speed=2.0,         # units/sec
    ):
        super().__init__()
        self.render_mode = render_mode 
        self.num_vines = num_vines
        self.num_humans = num_humans
        self.num_drones = num_drones
        self.max_boxes_per_vine = max_boxes_per_vine
        self.max_backlog = max_backlog
        self.field_size = np.array(field_size, dtype=np.float32)
        self.max_steps = max_steps

        self.dt = float(dt)
        self.harvest_time = float(harvest_time)
        self.human_speed = float(human_speed)
        self.drone_speed = float(drone_speed)

        self.num_actions = 3
        self.num_drone_status = 3

        # obs: vine pos (2V) + collection (2) + boxes_rem (V) + queued (V)
        #   + humans: pos (2H) + fatigue (H) + act onehot (3H) + has_box (H) + assigned_vine onehot (V*H)
        #   + drones: pos (2D) + status onehot (3D) + has_box (D)
        obs_dim = (
            self.num_vines * 2
            + 2
            + self.num_vines
            + self.num_vines
            + self.num_humans * 2
            + self.num_humans
            + self.num_humans * self.num_actions
            + self.num_humans
            + self.num_humans * self.num_vines
            + self.num_drones * 2
            + self.num_drones * self.num_drone_status
            + self.num_drones
        )

        # Since we normalize everything into [0,1], Box(0,1) is consistent.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Central scheduler outputs one action per human each step
        self.action_space = spaces.MultiDiscrete([self.num_actions] * self.num_humans)

        self.collection_point = (self.field_size * 0.5).astype(np.float32)

        # Will be created in reset()
        self.vines = []
        self.humans = []
        self.drones = []
        self.steps = 0
        self.delivered = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.delivered = 0

        # Example: random vine positions
        vine_positions = self.np_random.random((self.num_vines, 2), dtype=np.float32) * self.field_size

        self.vines = [Vine(vine_positions[i], self.max_boxes_per_vine) for i in range(self.num_vines)]

        # Humans start at their assigned vine position (simple first version)
        self.humans = []
        for h in range(self.num_humans):
            assigned = h % self.num_vines
            pos = self.vines[assigned].position.copy()
            self.humans.append(Human(pos, assigned))

        # Drones start at collection point
        self.drones = [Drone(self.collection_point.copy()) for _ in range(self.num_drones)]

        return self._get_obs(), {}

    def step(self, actions):
        self.steps += 1
        actions = np.asarray(actions, dtype=np.int64)

        delivered_before = self.delivered

        # 1) Progress ongoing actions (timers) for humans
        for h in self.humans:
            if h.busy:
                h.time_left -= self.dt
                # fatigue while working
                h.fatigue = float(np.clip(h.fatigue + 0.002 * self.dt, 0.0, 1.0))
                if h.time_left <= 0.0:
                    h.busy = False
                    h.time_left = 0.0
                    # finalize action
                    if h.current_action == ACTION_HARVEST:
                        # harvest completes => human now holds a box
                        h.has_box = True
                    elif h.current_action == ACTION_TRANSPORT:
                        # transport completes => delivered
                        if h.has_box:
                            h.has_box = False
                            self.delivered += 1
                        h.position = self.collection_point.copy()

        # 2) Apply new decisions for humans that are free
        for i, h in enumerate(self.humans):
            if h.busy:
                continue  # ignore scheduler action while busy

            a = int(actions[i])
            h.current_action = a

            vine = self.vines[h.assigned_vine]

            if a == ACTION_HARVEST:
                # start harvesting only if not already holding a box and vine has boxes
                if (not h.has_box) and vine.boxes_remaining > 0:
                    ok = vine.harvest_box()
                    if ok:
                        h.busy = True
                        h.time_left = self.harvest_time
                        # keep human at vine position
                        h.position = vine.position.copy()

            elif a == ACTION_ENQUEUE:
                # enqueue is instant
                if h.has_box and vine.queued_boxes < self.max_backlog:
                    vine.queued_boxes += 1
                    h.has_box = False
                    h.position = vine.position.copy()

            elif a == ACTION_TRANSPORT:
                # start transport if holding a box
                if h.has_box:
                    dist = float(np.linalg.norm(h.position - self.collection_point))
                    travel_time = dist / max(self.human_speed, 1e-6)
                    h.busy = True
                    h.time_left = travel_time

        # 3) Scripted drones: progress timers
        for d in self.drones:
            if d.busy:
                d.time_left -= self.dt
                if d.time_left <= 0.0:
                    d.busy = False
                    d.time_left = 0.0

                    if d.status == DRONE_GO_TO_VINE:
                        v = self.vines[d.target_vine]
                        d.position = v.position.copy()
                        if v.queued_boxes > 0:
                            v.queued_boxes -= 1
                            d.has_box = True
                            # now go deliver
                            d.status = DRONE_DELIVER
                            dist = float(np.linalg.norm(d.position - self.collection_point))
                            d.busy = True
                            d.time_left = dist / max(self.drone_speed, 1e-6)
                        else:
                            d.status = DRONE_IDLE
                            d.target_vine = None

                    elif d.status == DRONE_DELIVER:
                        d.position = self.collection_point.copy()
                        if d.has_box:
                            d.has_box = False
                            self.delivered += 1
                        d.status = DRONE_IDLE
                        d.target_vine = None

        # 4) If drone is idle, assign it to nearest queued vine
        for d in self.drones:
            if (not d.busy) and d.status == DRONE_IDLE:
                candidates = [idx for idx, v in enumerate(self.vines) if v.queued_boxes > 0]
                if candidates:
                    dists = [np.linalg.norm(self.vines[idx].position - d.position) for idx in candidates]
                    target = candidates[int(np.argmin(dists))]
                    d.target_vine = target
                    d.status = DRONE_GO_TO_VINE
                    dist = float(np.linalg.norm(self.vines[target].position - d.position))
                    d.busy = True
                    d.time_left = dist / max(self.drone_speed, 1e-6)

        # Reward: delivered boxes this step - backlog penalty - fatigue penalty
        delivered_delta = self.delivered - delivered_before
        backlog_total = sum(v.queued_boxes for v in self.vines)
        fatigue_sum = sum(h.fatigue for h in self.humans)

        reward = float(delivered_delta) - 0.01 * backlog_total - 0.001 * fatigue_sum

        # Termination: everything harvested AND no queue AND nobody carrying AND drones idle
        all_harvested = all(v.boxes_remaining == 0 for v in self.vines)
        no_queue = all(v.queued_boxes == 0 for v in self.vines)
        no_carry = all(not h.has_box for h in self.humans) and all(not d.has_box for d in self.drones)
        all_idle = all((not h.busy) for h in self.humans) and all((not d.busy) for d in self.drones)

        terminated = bool(all_harvested and no_queue and no_carry and all_idle)
        truncated = bool(self.steps >= self.max_steps)

        info = {
            "delivered": self.delivered,
            "backlog_total": backlog_total,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = []

        # vines: pos + remaining + queued
        for v in self.vines:
            obs.extend((v.position / self.field_size).tolist())  # (x,y) -> [0,1]
        obs.extend((self.collection_point / self.field_size).tolist())

        for v in self.vines:
            obs.append(v.boxes_remaining / max(self.max_boxes_per_vine, 1))
        for v in self.vines:
            obs.append(v.queued_boxes / max(self.max_backlog, 1))

        # humans
        for h in self.humans:
            obs.extend((h.position / self.field_size).tolist())
        for h in self.humans:
            obs.append(h.fatigue)
        for h in self.humans:
            obs.extend(one_hot(h.current_action, self.num_actions).tolist())
        for h in self.humans:
            obs.append(1.0 if h.has_box else 0.0)
        for h in self.humans:
            obs.extend(one_hot(h.assigned_vine, self.num_vines).tolist())

        # drones
        for d in self.drones:
            obs.extend((d.position / self.field_size).tolist())
        for d in self.drones:
            obs.extend(one_hot(d.status, self.num_drone_status).tolist())
        for d in self.drones:
            obs.append(1.0 if d.has_box else 0.0)

        return np.asarray(obs, dtype=np.float32)

    def render(self):
        if self.render_mode != "terminal":
            print(f"Step {self.steps} | delivered={self.delivered}")

        for i, v in enumerate(self.vines):
            print(f"  Vine {i}: rem={v.boxes_remaining} queued={v.queued_boxes} pos={v.position}")
        for i, h in enumerate(self.humans):
            print(f"  Human {i}: vine={h.assigned_vine} busy={h.busy} t={h.time_left:.1f} has_box={h.has_box} fat={h.fatigue:.2f}")
        for i, d in enumerate(self.drones):
            print(f"  Drone {i}: status={d.status} busy={d.busy} t={d.time_left:.1f} has_box={d.has_box} pos={d.position}")
        print("====================================================")

        if self.render_mode != "human":
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"Step {self.steps}, Delivered {self.delivered}")
        ax.set_xlim(0, self.field_size[0])
        ax.set_ylim(0, self.field_size[1])

        # draw collection point
        cp = self.collection_point
        ax.scatter(cp[0], cp[1], c="gold", marker="*", s=200, label="Collection Point")

        # draw vines
        for i, v in enumerate(self.vines):
            ax.scatter(v.position[0], v.position[1], c="green", marker="s", s=100)
            ax.text(
                v.position[0],
                v.position[1] + 0.3,
                f"R:{v.boxes_remaining}\nQ:{v.queued_boxes}",
                fontsize=8,
                ha="center",
            )

        # draw humans
        for i, h in enumerate(self.humans):
            ax.scatter(h.position[0], h.position[1], c="blue", marker="o", s=100)
            ax.text(
                h.position[0],
                h.position[1] - 0.3,
                f"H{i}",
                fontsize=8,
                ha="center",
            )

        # draw drones
        for i, d in enumerate(self.drones):
            ax.scatter(d.position[0], d.position[1], c="red", marker="^", s=100)
            ax.text(
                d.position[0],
                d.position[1] - 0.3,
                f"D{i}",
                fontsize=8,
                ha="center",
            )

        ax.legend(loc="upper right")
        plt.show()       

def my_check_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make('VineEnv-v0')
    check_env(env.unwrapped)
    print("Environment passed all checks!")


if __name__ == "__main__":
    # my_check_env()
    # env = gym.make('VineEnv-v0', render_mode="human")
    # obs, info = env.reset()
    # done = False
    plt.ion()

    env = gym.make("VineEnv-v0", render_mode="human")
    obs, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated