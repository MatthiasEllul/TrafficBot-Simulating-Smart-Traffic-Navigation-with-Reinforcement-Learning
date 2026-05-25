"""
train_agent_ppo.py - PPO training using the same CARLA environment and reward
function as train_agent_ddqn.py, allowing a direct algorithmic comparison.

The CARLA environment, reward function, map, spawn configuration, and all
infrastructure are identical to the DQN script. The only differences are:
  - Algorithm: PPO (via Stable Baselines 3) instead of Double DQN
  - Action space: continuous steering/throttle instead of 7 discrete bins
  - Exploration: handled by PPO internally (entropy bonus) - no manual epsilon
  - Training: on-policy rollout buffer instead of experience replay
  - Model: SB3 CnnPolicy with speed scalar injected via observations

Install dependencies (run once):
    pip install stable-baselines3==1.6.2 torch --break-system-packages

Usage:
    py -3.7 train_agent_ppo.py
"""

import gc
import glob
import os
import sys
import random
import time
import threading
import subprocess
import numpy as np
import cv2
import math
import gym
from gym import spaces

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

# Hyperparameters
SHOW_PREVIEW        = False
IM_WIDTH            = 160
IM_HEIGHT           = 120
MODEL_IM_CHANNELS   = 1             # grayscale

# Episode settings - identical to DQN script
SECONDS_PER_EPISODE     = 30
SECONDS_PER_EPISODE_MAX = 60
EP_LEN_INCREASE_WINDOW  = 3

MODEL_NAME              = "PPO_CNN"
EPISODES                = 10_000    # total training episodes
AGGREGATE_STATS_EVERY   = 50        # TensorBoard log interval
CHECKPOINT_EVERY        = 200       # save checkpoint every N episodes
MIN_REWARD              = 50        # threshold save condition
NPC_VEHICLES            = 0

# PPO-specific hyperparameters
# n_steps: rollout buffer collects this many steps before each policy update
# At ~50 steps/episode (fixed spawn, short episodes) a buffer of 512 covers
# roughly 10 episodes - enough for a stable gradient estimate
PPO_N_STEPS      = 512
PPO_BATCH_SIZE   = 64       # minibatch size for each gradient update
PPO_N_EPOCHS     = 4        # number of update passes over each rollout buffer
PPO_GAMMA        = 0.99     # discount - identical to DQN DISCOUNT
PPO_GAE_LAMBDA   = 0.95     # GAE smoothing factor
PPO_CLIP_RANGE   = 0.2      # PPO clip ratio - prevents large policy updates
PPO_ENT_COEF     = 0.01     # entropy bonus - encourages exploration
PPO_LR           = 3e-4     # learning rate
PPO_VF_COEF      = 0.5      # value function loss weight

# Lane invasion - identical to DQN script
LANE_INVASION_PENALTY   = -20
LANE_GRACE_SECONDS      = 5.0

# Road following - identical to DDQN script
OFFROAD_TERMINATE_DIST  = 3.5
OFFROAD_TERMINAL_REWARD = -200

# Fixed spawn - identical to DDQN script
USE_FIXED_SPAWN     = True
FIXED_SPAWN_INDEX   = 4
SPAWN_YAW_JITTER    = 5.0

# CARLA
CARLA_EXE_PATH     = r"C:\Users\matth\Downloads\CARLA_0.9.8\WindowsNoEditor\CarlaUE4.exe"
CARLA_LAUNCH_FLAGS = [
    "-quality-level=Low",
    "-RenderOffScreen",
    "-benchmark",
    "-fps=20",
]
CARLA_STARTUP_WAIT    = 65
CARLA_MAP             = 'Town02'
MAX_RECOVERY_ATTEMPTS = 5

BLUEPRINT_BLACKLIST = {
    'vehicle.bh.crossbike', 'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets', 'vehicle.harley-davidson.low_rider',
    'vehicle.kawasaki.ninja', 'vehicle.yamaha.yzf', 'vehicle.vespa.zx125',
}

# --- Resume training ---
# Set to the path of a previous PPO checkpoint zip file to resume training
# from that point. SB3 restores all weights and optimizer state
# Set to None to start from scratch
# Note: do NOT include the .zip extension - SB3 adds it automatically
RESUME_FROM = r"models\checkpoints\checkpoint_PPO_CNN_ep6000"
EARLY_STOPPING_WINDOW           = 5
PLATEAU_MIN_DELTA               = 0.5
EARLY_STOP_COLLISION_THRESHOLD  = 0.1
EARLY_STOP_MIN_EPISODES         = 1_000


# Custom CNN feature extractor
# SB3 expects a single observation tensor. We stack the grayscale image and
# speed scalar together as a (MODEL_IM_CHANNELS + 1, H, W) tensor, with the
# speed value broadcast across an extra channel. This lets the standard SB3
# CnnPolicy be used while still injecting the speed input, matching the DQN
# architecture's dual-input design

class DualInputCNNExtractor(BaseFeaturesExtractor):
    """
    CNN matching the DQN architecture:
      Conv2D(32) → Pool → Conv2D(64) → Pool → Conv2D(64) → Pool → Pool → Flatten
    The speed value is concatenated after flattening, identical to the DQN model
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # observation_space.shape = (2, H, W) - channel 0 is image, channel 1 is speed
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, IM_HEIGHT, IM_WIDTH)
            cnn_out = self.cnn(dummy).shape[1]

        # +1 for speed scalar concatenated after CNN
        self.linear = nn.Sequential(
            nn.Linear(cnn_out + 1, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, 2, H, W)
        img   = observations[:, 0:1, :, :]         # (batch, 1, H, W)
        # Index with 1 (not 1:2) to get shape (batch,), then unsqueeze to (batch, 1)
        speed = observations[:, 1, 0, 0].unsqueeze(1)  # (batch, 1)

        cnn_out = self.cnn(img)                    # (batch, cnn_features)
        merged  = torch.cat([cnn_out, speed], dim=1)  # (batch, cnn_features + 1)
        return self.linear(merged)


# CARLA Gym Environment
class CarlaEnvPPO(gym.Env):
    """
    OpenAI Gym-compatible wrapper around the CARLA environment
    Observation: (2, H, W) float32 array - channel 0: normalised grayscale,
                                            channel 1: normalised speed (broadcast)
    Action: Discrete(7) - same 7-action map as DQN script
    Reward: identical reward function to DQN script
    """

    metadata = {'render.modes': ['human']}

    # 7 discrete actions - identical to DQN ACTION_MAP
    _ACTIONS = [
        {"throttle": 1.0, "steer": -1.0, "brake": 0.0},
        {"throttle": 1.0, "steer": -0.5, "brake": 0.0},
        {"throttle": 1.0, "steer":  0.0, "brake": 0.0},
        {"throttle": 1.0, "steer":  0.5, "brake": 0.0},
        {"throttle": 1.0, "steer":  1.0, "brake": 0.0},
        {"throttle": 0.5, "steer":  0.0, "brake": 0.0},
        {"throttle": 0.0, "steer":  0.0, "brake": 1.0},
    ]

    def __init__(self):
        super().__init__()

        # Observation: 2-channel (image + speed) at model resolution
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(2, IM_HEIGHT, IM_WIDTH),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self._ACTIONS))

        # CARLA connection
        self.client            = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world             = self.client.load_world(CARLA_MAP)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.carla_map         = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3           = self.blueprint_library.filter("model3")[0]

        self._lock           = threading.Lock()
        self.front_camera    = None
        self.collision_hist  = []
        self.lane_inv_hist   = []
        self.actor_list      = []
        self.sensor_list     = []
        self.vehicle         = None
        self.episode_start   = 0.0
        self.next_wp         = None
        self.total_lane_inv  = 0
        self.npc_list        = []

        # Stats tracked for TensorBoard callback
        self.episode_count     = 0
        self.episode_rewards   = []
        self.collision_flags   = []
        self.offroad_flags     = []
        self.lane_inv_flags    = []
        self.steps_per_ep      = []
        self._ep_reward        = 0.0
        self._ep_steps         = 0

        # Adaptive episode length
        self.seconds_per_ep        = SECONDS_PER_EPISODE
        self._ep_len_window_count  = 0

    def _get_obs(self, kmh):
        with self._lock:
            img = self.front_camera.copy() if self.front_camera is not None \
                  else np.zeros((IM_HEIGHT, IM_WIDTH, 1), dtype=np.uint8)
        img_f     = img[:, :, 0].astype(np.float32) / 255.0   # (H, W)
        speed_f   = float(min(kmh / 120.0, 1.0))
        speed_ch  = np.full((IM_HEIGHT, IM_WIDTH), speed_f, dtype=np.float32)  # broadcast
        return np.stack([img_f, speed_ch], axis=0)  # (2, H, W)

    def _process_img(self, image):
        i    = np.array(image.raw_data)
        i2   = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        gray = np.mean(i2[:, :, :3], axis=2, keepdims=True).astype(np.uint8)
        with self._lock:
            self.front_camera = gray

    def reset(self):
        # Destroy previous episode actors
        for s in self.sensor_list:
            try: s.stop()
            except Exception: pass
        for a in self.actor_list:
            try: a.destroy()
            except Exception: pass
        self.actor_list   = []
        self.sensor_list  = []
        self.vehicle      = None

        self.collision_hist = []
        self.lane_inv_hist  = []
        self.total_lane_inv = 0

        # Spawn
        spawn_points = self.world.get_map().get_spawn_points()
        if USE_FIXED_SPAWN:
            sp_base = spawn_points[FIXED_SPAWN_INDEX]
            yaw_off = random.uniform(-SPAWN_YAW_JITTER, SPAWN_YAW_JITTER)
            sp = carla.Transform(
                sp_base.location,
                carla.Rotation(pitch=sp_base.rotation.pitch,
                               yaw=sp_base.rotation.yaw + yaw_off,
                               roll=sp_base.rotation.roll)
            )
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, sp)
            except RuntimeError:
                sp2 = carla.Transform(
                    carla.Location(x=sp_base.location.x, y=sp_base.location.y,
                                   z=sp_base.location.z + 0.5), sp.rotation)
                self.vehicle = self.world.spawn_actor(self.model_3, sp2)
        else:
            random.shuffle(spawn_points)
            for sp in spawn_points:
                try:
                    self.vehicle = self.world.spawn_actor(self.model_3, sp); break
                except RuntimeError as e:
                    if "collision" in str(e).lower(): continue
                    raise
        if self.vehicle is None:
            raise RuntimeError("reset(): all spawn points occupied.")
        self.actor_list.append(self.vehicle)

        sensor_spawn = carla.Transform(carla.Location(x=2.5, z=0.7))

        rgb_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute("image_size_x", str(IM_WIDTH))
        rgb_bp.set_attribute("image_size_y", str(IM_HEIGHT))
        rgb_bp.set_attribute("fov", "110")
        cam = self.world.spawn_actor(rgb_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(cam); self.sensor_list.append(cam)
        cam.listen(lambda d: self._process_img(d))

        col_bp = self.blueprint_library.find("sensor.other.collision")
        col = self.world.spawn_actor(col_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(col); self.sensor_list.append(col)
        col.listen(lambda e: self.collision_hist.append(e))

        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        lane = self.world.spawn_actor(lane_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(lane); self.sensor_list.append(lane)
        lane.listen(lambda e: self.lane_inv_hist.append(e))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(1)

        with self._lock:
            self.front_camera = None
        while True:
            with self._lock:
                if self.front_camera is not None: break
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Waypoint progress tracking
        loc = self.vehicle.get_location()
        wp  = self.carla_map.get_waypoint(loc, project_to_road=True,
                                           lane_type=carla.LaneType.Driving)
        nxt = wp.next(3.0)
        self.next_wp = nxt[0] if nxt else wp

        self._ep_reward = 0.0
        self._ep_steps  = 0
        return self._get_obs(0)

    def step(self, action):
        # Guard: vehicle can be None immediately after CARLA crash recovery
        # because reconnect() clears the actor state but SB3 may call step()
        # before reset() is triggered. Return done=True to force SB3 to reset
        if self.vehicle is None:
            dummy_obs = np.zeros((2, IM_HEIGHT, IM_WIDTH), dtype=np.float32)
            return dummy_obs, 0.0, True, {}

        ctrl = self._ACTIONS[action]
        self.vehicle.apply_control(carla.VehicleControl(**ctrl))

        v   = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Collision terminal
        if len(self.collision_hist) > 0:
            self._record_episode(collided=True, offroad=False)
            return self._get_obs(kmh), -200.0, True, {}

        # Road following
        location = self.vehicle.get_location()
        waypoint = self.carla_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving)
        road_dev = location.distance(waypoint.transform.location)

        if road_dev > OFFROAD_TERMINATE_DIST:
            self._record_episode(collided=False, offroad=True)
            return self._get_obs(kmh), float(OFFROAD_TERMINAL_REWARD), True, {}

        road_reward = max(0.0, 1.0 - road_dev / OFFROAD_TERMINATE_DIST)

        vehicle_yaw  = math.radians(self.vehicle.get_transform().rotation.yaw)
        waypoint_yaw = math.radians(waypoint.transform.rotation.yaw)
        heading_reward = max(0.0, math.cos(vehicle_yaw - waypoint_yaw))

        # Waypoint progress
        dist_to_next = location.distance(self.next_wp.transform.location)
        progress_reward = 0.0
        if dist_to_next < 3.0:
            progress_reward = 5.0
            nxt = self.next_wp.next(3.0)
            if nxt: self.next_wp = nxt[0]

        # Speed multiplier (identical to DDQN)
        speed_multiplier = min(kmh / 15.0, 1.0)

        if kmh < 5:
            speed_reward = -2.0
        elif kmh < 30:
            speed_reward = 0.5 * (kmh / 30.0)
        elif kmh <= 60:
            speed_reward = 0.5
        else:
            speed_reward = max(0.0, 0.5 - (kmh - 60) / 80.0)

        # Lane invasion
        elapsed     = time.time() - self.episode_start
        new_inv     = len(self.lane_inv_hist)
        self.lane_inv_hist.clear()
        lane_penalty = 0.0
        if elapsed > LANE_GRACE_SECONDS and new_inv > 0:
            self.total_lane_inv += 1
            lane_penalty = LANE_INVASION_PENALTY

        # Combined reward (identical to DDQN) 
        positive = (road_reward * 2.0) + heading_reward + progress_reward
        reward   = (positive * speed_multiplier) + speed_reward + lane_penalty

        self._ep_reward += reward
        self._ep_steps  += 1

        done = (time.time() - self.episode_start) >= self.seconds_per_ep
        if done:
            self._record_episode(collided=False, offroad=False)

        return self._get_obs(kmh), reward, done, {}

    def _record_episode(self, collided, offroad):
        self.episode_count += 1
        self.episode_rewards.append(self._ep_reward)
        self.collision_flags.append(1 if collided else 0)
        self.offroad_flags.append(1 if offroad else 0)
        self.lane_inv_flags.append(1 if self.total_lane_inv > 0 else 0)
        self.steps_per_ep.append(self._ep_steps)

    def reconnect(self):
        """Re-establish CARLA connection after a crash and restart."""
        self.client            = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world             = self.client.load_world(CARLA_MAP)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.carla_map         = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3           = self.blueprint_library.filter("model3")[0]
        self.actor_list        = []
        self.sensor_list       = []
        self.vehicle           = None
        print(f"[CarEnv] Reconnected to simulator on {CARLA_MAP}.")

    def render(self, mode='human'):
        pass

    def close(self):
        for s in self.sensor_list:
            try: s.stop()
            except Exception: pass
        for a in self.actor_list:
            try: a.destroy()
            except Exception: pass


# TensorBoard + Stats Callback
class TrainingCallback(BaseCallback):
    """
    Logs the same metrics as the DQN TensorBoard to allow direct comparison:
    reward_avg, reward_min, reward_max, collision_rate, offroad_rate,
    lane_invasion_rate, avg_steps, episode_length_cap.
    Also handles: checkpointing, best-model saving, early stopping,
    and adaptive episode length
    """

    def __init__(self, env: CarlaEnvPPO, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.env      = env
        self.log_dir  = log_dir
        os.makedirs(log_dir, exist_ok=True)

        import tensorflow as tf
        self.writer = tf.summary.FileWriter(log_dir)

        self.last_ep_logged     = 0
        self.best_avg_reward    = -float('inf')
        self.best_es_reward     = -float('inf')
        self.plateau_counter    = 0
        self.ep_len_counter     = 0
        self.stop_training      = False

    def _write(self, tag, value, step):
        import tensorflow as tf
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=float(value))])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def _on_step(self) -> bool:
        ep = self.env.episode_count
        if ep == self.last_ep_logged or ep % AGGREGATE_STATS_EVERY != 0:
            return not self.stop_training
        self.last_ep_logged = ep

        n  = AGGREGATE_STATS_EVERY
        r  = self.env.episode_rewards[-n:]
        cr = self.env.collision_flags[-n:]
        or_ = self.env.offroad_flags[-n:]
        li = self.env.lane_inv_flags[-n:]
        st = self.env.steps_per_ep[-n:]

        avg_r    = sum(r)  / len(r)
        min_r    = min(r)
        max_r    = max(r)
        col_rate = sum(cr) / len(cr)
        off_rate = sum(or_) / len(or_)
        li_rate  = sum(li) / len(li)
        avg_st   = sum(st) / len(st)

        self._write("reward_avg",          avg_r,    ep)
        self._write("reward_min",          min_r,    ep)
        self._write("reward_max",          max_r,    ep)
        self._write("collision_rate",      col_rate, ep)
        self._write("offroad_rate",        off_rate, ep)
        self._write("lane_invasion_rate",  li_rate,  ep)
        self._write("avg_steps",           avg_st,   ep)
        self._write("episode_length_cap",  self.env.seconds_per_ep, ep)

        print(f"\n[ep {ep:>5}]  avg={avg_r:>8.1f}  min={min_r:>8.1f}  "
              f"max={max_r:>8.1f}  col={col_rate:.2f}  "
              f"off={off_rate:.2f}  lane={li_rate:.2f}  "
              f"steps={avg_st:.0f}  cap={self.env.seconds_per_ep}s")

        # Adaptive episode length
        if col_rate < 0.3:
            self.ep_len_counter += 1
        else:
            self.ep_len_counter = 0
        if (self.ep_len_counter >= EP_LEN_INCREASE_WINDOW
                and self.env.seconds_per_ep < SECONDS_PER_EPISODE_MAX):
            self.env.seconds_per_ep = min(
                self.env.seconds_per_ep + 5, SECONDS_PER_EPISODE_MAX)
            self.ep_len_counter = 0
            print(f"[Training] Episode cap → {self.env.seconds_per_ep}s")

        # Best model save
        if avg_r > self.best_avg_reward:
            self.best_avg_reward = avg_r
            path = (f"models/best_{MODEL_NAME}"
                    f"__{max_r:.2f}max_{avg_r:.2f}avg_{min_r:.2f}min"
                    f"__ep{ep}__{int(time.time())}")
            self.model.save(path)
            print(f"[Best] Saved: {path}")

        # Threshold save
        if min_r >= MIN_REWARD:
            path = (f"models/{MODEL_NAME}"
                    f"__{max_r:.2f}max_{avg_r:.2f}avg_{min_r:.2f}min"
                    f"__{int(time.time())}")
            self.model.save(path)

        # Checkpoint save
        if ep % CHECKPOINT_EVERY == 0:
            path = f"models/checkpoints/checkpoint_{MODEL_NAME}_ep{ep}"
            self.model.save(path)
            print(f"[Checkpoint] ep {ep} saved.")

        # Early stopping
        if ep >= EARLY_STOP_MIN_EPISODES:
            genuinely_good = (col_rate <= EARLY_STOP_COLLISION_THRESHOLD
                              and off_rate <= EARLY_STOP_COLLISION_THRESHOLD)
            if avg_r - self.best_es_reward < PLATEAU_MIN_DELTA and genuinely_good:
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0
            if avg_r > self.best_es_reward:
                self.best_es_reward = avg_r
            if self.plateau_counter >= EARLY_STOPPING_WINDOW:
                print(f"\n[Early Stop] ep {ep} - reward plateaued, "
                      f"collision={col_rate:.2f}, offroad={off_rate:.2f}")
                self.stop_training = True
                return False

        return True


# CARLA process helpers (identical to DDQN script)

def launch_carla():
    print("[CARLA] Launching simulator...")
    proc = subprocess.Popen(
        [CARLA_EXE_PATH] + CARLA_LAUNCH_FLAGS,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[CARLA] PID {proc.pid} - waiting {CARLA_STARTUP_WAIT}s...")
    time.sleep(CARLA_STARTUP_WAIT)
    return proc


def kill_carla_process(proc):
    pid = getattr(proc, 'pid', None)
    if pid:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print(f"[CARLA] Process tree {pid} terminated.")
        except Exception as e:
            print(f"[CARLA] taskkill failed ({e}), trying terminate().")
            try: proc.terminate()
            except Exception: pass
    gc.collect()
    time.sleep(40)


def wait_for_carla(retries=20, delay=5.0):
    for attempt in range(1, retries + 1):
        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(10.0)
            client.get_world()
            print(f"[CARLA] Connected on attempt {attempt}.")
            return
        except RuntimeError:
            print(f"[CARLA] Not ready (attempt {attempt}/{retries})...")
            time.sleep(delay)
    raise RuntimeError("[CARLA] Could not connect.")


# Main
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    os.makedirs("models",             exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/recovery",    exist_ok=True)

    log_dir  = f"logs/{MODEL_NAME}-{int(time.time())}"
    carla_proc = launch_carla()
    wait_for_carla()

    env = CarlaEnvPPO()

    # PPO policy kwargs - point SB3 to our custom CNN extractor
    policy_kwargs = dict(
        features_extractor_class  = DualInputCNNExtractor,
        features_extractor_kwargs = dict(features_dim=256),
        net_arch                  = [],   # no additional MLP layers after extractor
    )

    if RESUME_FROM:
        print(f"[PPO] Resuming from checkpoint: {RESUME_FROM}")
        model = PPO.load(
            RESUME_FROM,
            env           = env,
            tensorboard_log = log_dir,
            verbose       = 0,
        )
        print("[PPO] Weights and optimizer state restored.")
    else:
        model = PPO(
            policy          = "CnnPolicy",
            env             = env,
            learning_rate   = PPO_LR,
            n_steps         = PPO_N_STEPS,
            batch_size      = PPO_BATCH_SIZE,
            n_epochs        = PPO_N_EPOCHS,
            gamma           = PPO_GAMMA,
            gae_lambda      = PPO_GAE_LAMBDA,
            clip_range      = PPO_CLIP_RANGE,
            ent_coef        = PPO_ENT_COEF,
            vf_coef         = PPO_VF_COEF,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = log_dir,
            verbose         = 0,
        )

    callback = TrainingCallback(env=env, log_dir=log_dir)

    # Total timesteps - roughly EPISODES × average steps/ep
    # At 30s episodes and ~50 steps/ep on the fixed spawn: 10,000 × 50 = 500,000
    # Adjust if avg_steps turns out to be higher.
    TOTAL_TIMESTEPS = EPISODES * 200

    print(f"\n[PPO] Starting training - {TOTAL_TIMESTEPS:,} total timesteps")
    print(f"[PPO] Map: {CARLA_MAP}  Spawn: index {FIXED_SPAWN_INDEX}"
          f"  {'(fixed)' if USE_FIXED_SPAWN else '(random)'}")
    print(f"[PPO] Logs: {log_dir}\n")

    timesteps_done    = 0
    recovery_attempts = 0
    first_run         = True

    try:
        while timesteps_done < TOTAL_TIMESTEPS:
            remaining = TOTAL_TIMESTEPS - timesteps_done
            try:
                model.learn(
                    total_timesteps     = remaining,
                    callback            = callback,
                    reset_num_timesteps = first_run,  # False on recovery - keeps counter
                )
                # learn() returned normally - training finished
                timesteps_done    = TOTAL_TIMESTEPS
                recovery_attempts = 0

            except (RuntimeError, Exception) as e:
                if "time-out" in str(e).lower() or "simulator" in str(e).lower():
                    # CARLA crash - attempt recovery
                    recovery_attempts += 1
                    print(f"\n[Recovery] CARLA crash (attempt {recovery_attempts}/"
                          f"{MAX_RECOVERY_ATTEMPTS}): {e}")

                    if recovery_attempts > MAX_RECOVERY_ATTEMPTS:
                        print("[Recovery] Max attempts exceeded. Stopping.")
                        break

                    # Save emergency checkpoint before recovery
                    try:
                        ep_now = env.episode_count
                        path   = (f"models/recovery/recovery_PPO"
                                  f"_ep{ep_now}_{int(time.time())}")
                        model.save(path)
                        print(f"[Recovery] Emergency checkpoint saved: {path}")
                    except Exception as save_err:
                        print(f"[Recovery] Could not save checkpoint: {save_err}")

                    # Record how many timesteps were done before the crash
                    timesteps_done = model.num_timesteps
                    print(f"[Recovery] {timesteps_done:,} / {TOTAL_TIMESTEPS:,} "
                          f"timesteps completed before crash.")

                    # Kill CARLA, wait for memory to clear, relaunch
                    kill_carla_process(carla_proc)
                    carla_proc = launch_carla()
                    wait_for_carla()
                    env.reconnect()

                    first_run = False  # resume, don't reset SB3 timestep counter
                    print(f"[Recovery] Resuming from timestep {timesteps_done:,}.")
                else:
                    # Non-CARLA exception - re-raise
                    raise

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving before exit...")

    finally:
        # Final model save
        try:
            final_path = (f"models/{MODEL_NAME}_FINAL_{int(time.time())}")
            model.save(final_path)
            print(f"[Done] Final model saved: {final_path}")
        except Exception as e:
            print(f"[Warning] Final save failed: {e}")

        env.close()
        kill_carla_process(carla_proc)
        print("[Done] Training complete.")