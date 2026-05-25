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
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from collections import deque
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from threading import Thread
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Hyperparameters
SHOW_PREVIEW        = False
IM_WIDTH            = 160
IM_HEIGHT           = 120
MODEL_IM_WIDTH      = 160
MODEL_IM_HEIGHT     = 120
MODEL_IM_CHANNELS   = 1             # grayscale

SECONDS_PER_EPISODE     = 30
SECONDS_PER_EPISODE_MAX = 60
EP_LEN_INCREASE_WINDOW  = 3

REPLAY_MEMORY_SIZE      = 2_000
MIN_REPLAY_MEMORY_SIZE  = 500
MINIBATCH_SIZE          = 8
PREDICTION_BATCH_SIZE   = 1
TRAINING_BATCH_SIZE     = 1
UPDATE_TARGET_EVERY     = 5
MODEL_NAME              = "DQN_CNN"
MEMORY_FRACTION         = 0.6
MIN_REWARD              = 50
EPISODES                = 10_000
DISCOUNT                = 0.99

# Starting from scratch on a fixed spawn - full exploration needed
# Slow decay keeps meaningful exploration until around last 1000 episodes
epsilon         = 0.5
EPSILON_DECAY   = 0.9999995
MIN_EPSILON     = 0.001

AGGREGATE_STATS_EVERY   = 50
NPC_VEHICLES            = 0
CHECKPOINT_EVERY        = 250

# Lane invasion
# Binary flat penalty - -20 per step where a crossing is detected (1 per step regardless of how many CARLA sensor events fired)
LANE_INVASION_PENALTY   = -20
LANE_GRACE_SECONDS      = 5.0

# Road following / off-road 
# Maximum distance from the nearest road waypoint before the episode terminates
# This directly prevents the straight-off-road exploit seen in testing where
# agents drove off the map and survived because CARLA does not register
# off-road driving as a collision
OFFROAD_TERMINATE_DIST  = 3.5
OFFROAD_TERMINAL_REWARD = -200      # same as collision - all failures equally bad

# Vehicle blueprints that crash CARLA 0.9.8 on Windows (TireConfig bug)
BLUEPRINT_BLACKLIST = {
    'vehicle.bh.crossbike', 'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets', 'vehicle.harley-davidson.low_rider',
    'vehicle.kawasaki.ninja', 'vehicle.yamaha.yzf', 'vehicle.vespa.zx125',
}

# Early stopping
EARLY_STOPPING_WINDOW           = 5
PLATEAU_MIN_DELTA               = 0.5
EARLY_STOP_COLLISION_THRESHOLD  = 0.1
EARLY_STOP_MIN_EPISODES         = 1_000

# CARLA
# *** UPDATE THIS PATH to match your installation ***
CARLA_EXE_PATH     = r"C:\Users\matth\Downloads\CARLA_0.9.8\WindowsNoEditor\CarlaUE4.exe"
CARLA_LAUNCH_FLAGS = [
    "-quality-level=Low",
    "-RenderOffScreen",
    "-benchmark",
    "-fps=20",
]
CARLA_STARTUP_WAIT    = 65
CARLA_MAP             = 'Town02'  # simple grid - shorter straights than Town01
MAX_RECOVERY_ATTEMPTS = 5

# --- Fixed spawn point ---
# Training on a fixed spawn eliminates the variability that slowed learning -
# some spawns had long straights (reinforcing straight-driving) while others
# had immediate curves. Spawn 29 (x=21.42, y=109.40, yaw=0) faces east on
# Town02's bottom horizontal road with the inner vertical road junction
# approximately 20 metres ahead, giving a clean straight lead-in followed by
# a forced turn decision.
USE_FIXED_SPAWN     = True
FIXED_SPAWN_INDEX   = 4   # change after visual verification if needed
SPAWN_YAW_JITTER    = 5.0  # ± degrees of random yaw offset each episode
                            # prevents the agent memorising the exact start frame

# Weight transfer
# Set to None for a completely fresh start with a fixed spawn point
# The agent will learn from scratch without any prior policy biases
# If transferring, use TRANSFER_CONV_ONLY=True to keep visual features only 
# and discard the dense layers that encoded the old Q-values and policy
TRANSFER_WEIGHTS_FROM = None
TRANSFER_CONV_ONLY    = True

# Action map
ACTION_MAP = {
    0: {"throttle": 1.0, "steer": -1.0, "brake": 0.0},   # hard left
    1: {"throttle": 1.0, "steer": -0.5, "brake": 0.0},   # gentle left
    2: {"throttle": 1.0, "steer":  0.0, "brake": 0.0},   # straight
    3: {"throttle": 1.0, "steer":  0.5, "brake": 0.0},   # gentle right
    4: {"throttle": 1.0, "steer":  1.0, "brake": 0.0},   # hard right
    5: {"throttle": 0.5, "steer":  0.0, "brake": 0.0},   # slow straight
    6: {"throttle": 0.0, "steer":  0.0, "brake": 1.0},   # full brake
}
NUM_ACTIONS = len(ACTION_MAP)


# Modified TensorBoard
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step   = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    def set_model(self, model):              pass
    def on_epoch_end(self, epoch, logs=None): self.update_stats(**logs)
    def on_batch_end(self, batch, logs=None): pass
    def on_train_end(self, _):               pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Prioritized Experience Replay
class PrioritizedReplayMemory:
    def __init__(self, maxlen, alpha=0.6):
        self.memory     = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha      = alpha
        self._lock      = threading.Lock()

    def append(self, transition, error=1.0):
        priority = (abs(error) + 1e-5) ** self.alpha
        with self._lock:
            self.memory.append(transition)
            self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        with self._lock:
            probs   = np.array(self.priorities, dtype=np.float32)
            mem_len = len(self.memory)
            probs  /= probs.sum()
            indices = np.random.choice(mem_len, batch_size, p=probs, replace=False)
            samples = [self.memory[i] for i in indices]
        weights = (mem_len * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, weights, indices

    def update_priorities(self, indices, errors):
        with self._lock:
            for i, e in zip(indices, errors):
                self.priorities[i] = (abs(e) + 1e-5) ** self.alpha

    def __len__(self):
        with self._lock:
            return len(self.memory)


# CARLA Environment
class CarEnv:
    SHOW_CAM  = SHOW_PREVIEW
    im_width  = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self):
        self.client            = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world             = self.client.load_world(CARLA_MAP)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)  # max visibility
        self.carla_map         = self.world.get_map()   # for waypoint queries in step()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3           = self.blueprint_library.filter("model3")[0]
        self._lock             = threading.Lock()
        self.front_camera      = None
        self.npc_list          = []

    def spawn_npcs(self):
        if NPC_VEHICLES == 0:
            print("[Environment] NPC_VEHICLES=0, skipping NPC spawn.")
            return
        vehicle_bps  = [bp for bp in self.blueprint_library.filter('vehicle.*')
                        if bp.id not in BLUEPRINT_BLACKLIST]
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for sp in spawn_points[:NPC_VEHICLES]:
            npc = self.world.try_spawn_actor(random.choice(vehicle_bps), sp)
            if npc:
                npc.set_autopilot(True)
                self.npc_list.append(npc)
        print(f"[Environment] Spawned {len(self.npc_list)} NPC vehicles.")

    def check_and_respawn_npcs(self):
        if NPC_VEHICLES == 0:
            return
        vehicle_bps  = [bp for bp in self.blueprint_library.filter('vehicle.*')
                        if bp.id not in BLUEPRINT_BLACKLIST]
        spawn_points = self.world.get_map().get_spawn_points()
        alive = []
        for npc in self.npc_list:
            try:
                if npc.is_alive: alive.append(npc)
            except Exception: pass
        self.npc_list = alive
        needed = NPC_VEHICLES - len(self.npc_list)
        if needed > 0:
            random.shuffle(spawn_points)
            for sp in spawn_points[:needed * 2]:
                if len(self.npc_list) >= NPC_VEHICLES: break
                npc = self.world.try_spawn_actor(random.choice(vehicle_bps), sp)
                if npc:
                    npc.set_autopilot(True)
                    self.npc_list.append(npc)

    def refresh_world(self):
        try:
            # Use get_world() here - we only load the map once at startup and
            # after crash recovery. Calling load_world() every episode is slow
            self.world     = self.client.get_world()
            self.carla_map = self.world.get_map()
        except Exception as e:
            print(f"[Warning] Could not refresh world reference: {e}")

    def reconnect(self):
        self.client            = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world             = self.client.load_world(CARLA_MAP)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.carla_map         = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3           = self.blueprint_library.filter("model3")[0]
        self.npc_list          = []
        print(f"[CarEnv] Reconnected to simulator on {CARLA_MAP}.")

    def reset(self):
        self.collision_hist       = []
        self.lane_invasion_hist   = []
        self.actor_list           = []
        self.sensor_list          = []
        self.total_lane_invasions = 0

        # Spawn ego vehicle
        spawn_points = self.world.get_map().get_spawn_points()

        if USE_FIXED_SPAWN:
            # Use the designated spawn point with a small random yaw offset
            # each episode to prevent the agent memorising the exact start frame
            sp_base = spawn_points[FIXED_SPAWN_INDEX]
            yaw_offset = random.uniform(-SPAWN_YAW_JITTER, SPAWN_YAW_JITTER)
            sp = carla.Transform(
                sp_base.location,
                carla.Rotation(
                    pitch=sp_base.rotation.pitch,
                    yaw=sp_base.rotation.yaw + yaw_offset,
                    roll=sp_base.rotation.roll
                )
            )
            try:
                self.vehicle   = self.world.spawn_actor(self.model_3, sp)
                self.transform = sp
            except RuntimeError:
                # If the fixed point is occupied, shift slightly upward and retry
                sp_shifted = carla.Transform(
                    carla.Location(
                        x=sp_base.location.x,
                        y=sp_base.location.y,
                        z=sp_base.location.z + 0.5
                    ),
                    sp.rotation
                )
                self.vehicle   = self.world.spawn_actor(self.model_3, sp_shifted)
                self.transform = sp_shifted
        else:
            # Random spawn - shuffle and try each point in sequence
            random.shuffle(spawn_points)
            self.vehicle = None
            for sp in spawn_points:
                try:
                    self.vehicle   = self.world.spawn_actor(self.model_3, sp)
                    self.transform = sp
                    break
                except RuntimeError as e:
                    if "collision" in str(e).lower(): continue
                    raise
            if self.vehicle is None:
                raise RuntimeError("reset(): all spawn points occupied.")

        self.actor_list.append(self.vehicle)

        sensor_spawn = carla.Transform(carla.Location(x=2.5, z=0.7))

        rgb_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute("image_size_x", f"{self.im_width}")
        rgb_bp.set_attribute("image_size_y", f"{self.im_height}")
        rgb_bp.set_attribute("fov", "110")
        self.sensor = self.world.spawn_actor(rgb_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.sensor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.collision_hist.append(event))

        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.sensor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda e: self.lane_invasion_hist.append(e))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(1)

        with self._lock:
            self.front_camera = None
        while True:
            with self._lock:
                if self.front_camera is not None:
                    break
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Initialise waypoint progress tracking - target is 3m ahead along road
        loc = self.vehicle.get_location()
        self.current_wp = self.carla_map.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        nxt = self.current_wp.next(3.0)
        self.next_wp = nxt[0] if nxt else self.current_wp

        with self._lock:
            img = self.front_camera.copy()
        return img, np.array([0.0], dtype=np.float32)

    def process_img(self, image):
        i    = np.array(image.raw_data)
        i2   = i.reshape((self.im_height, self.im_width, 4))
        gray = np.mean(i2[:, :, :3], axis=2, keepdims=True).astype(np.uint8)
        if self.SHOW_CAM:
            cv2.imshow("Front Camera", gray[:, :, 0])
            cv2.waitKey(1)
        with self._lock:
            self.front_camera = gray

    def step(self, action):
        global epsilon, SECONDS_PER_EPISODE

        self.vehicle.apply_control(carla.VehicleControl(**ACTION_MAP[action]))

        v   = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Collision terminal
        if len(self.collision_hist) != 0:
            with self._lock:
                img = self.front_camera.copy()
            return (img, np.array([min(kmh / 120.0, 1.0)], dtype=np.float32)), -200, True, None

        # Road following reward
        # Query the nearest waypoint on the drivable road surface
        location = self.vehicle.get_location()
        waypoint = self.carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        road_deviation = location.distance(waypoint.transform.location)

        # Off-road terminal - prevents the straight-off-road exploit seen
        # in testing. When the vehicle wanders more than OFFROAD_TERMINATE_DIST
        # metres from the road centre the episode ends immediately
        if road_deviation > OFFROAD_TERMINATE_DIST:
            with self._lock:
                img = self.front_camera.copy()
            speed_obs = np.array([min(kmh / 120.0, 1.0)], dtype=np.float32)
            return (img, speed_obs), OFFROAD_TERMINAL_REWARD, True, None

        # Road centre reward: 1.0 directly on centre, 0.0 at OFFROAD_TERMINATE_DIST
        road_reward = max(0.0, 1.0 - road_deviation / OFFROAD_TERMINATE_DIST)

        # Heading alignment reward
        vehicle_yaw  = math.radians(self.vehicle.get_transform().rotation.yaw)
        waypoint_yaw = math.radians(waypoint.transform.rotation.yaw)
        heading_reward = max(0.0, math.cos(vehicle_yaw - waypoint_yaw))

        # Waypoint progress reward
        # Fires when the vehicle reaches the next target waypoint 3m ahead.
        # Spinning, braking, and driving straight into a curve all fail to
        # trigger this - it is the only reward that requires actual navigation
        location = self.vehicle.get_location()
        dist_to_next = location.distance(self.next_wp.transform.location)
        progress_reward = 0.0
        if dist_to_next < 3.0:
            progress_reward = 5.0
            nxt = self.next_wp.next(3.0)
            if nxt:
                self.next_wp = nxt[0]

        # Speed multiplier + reward
        # All positive rewards (road, heading, progress) are gated behind a
        # speed multiplier that rises from 0 at standstill to 1.0 at 15 km/h.
        # A stationary agent earns zero positive reward regardless of its road
        # position, making braking always net-negative (only speed_penalty fires).
        # This replaces the idle terminal - no terminal needed because braking
        # is structurally unprofitable under this formulation.
        speed_multiplier = min(kmh / 15.0, 1.0)

        if kmh < 5:
            speed_reward = -2.0
        elif kmh < 30:
            speed_reward = 0.5 * (kmh / 30.0)
        elif kmh <= 60:
            speed_reward = 0.5
        else:
            speed_reward = max(0.0, 0.5 - (kmh - 60) / 80.0)

        # Lane invasion - binary flat penalty
        elapsed       = time.time() - self.episode_start
        new_invasions = len(self.lane_invasion_hist)
        self.lane_invasion_hist.clear()

        lane_penalty = 0.0
        if elapsed > LANE_GRACE_SECONDS and new_invasions > 0:
            self.total_lane_invasions += 1
            lane_penalty = LANE_INVASION_PENALTY

        # Combined reward
        # Positive rewards only fire when moving (speed_multiplier)
        # speed_reward and lane_penalty fire unconditionally
        positive_reward = (road_reward * 2.0) + heading_reward + progress_reward
        reward = (positive_reward * speed_multiplier) + speed_reward + lane_penalty

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon  = max(MIN_EPSILON, epsilon)

        done = self.episode_start + SECONDS_PER_EPISODE < time.time()

        with self._lock:
            img = self.front_camera.copy()
        speed_obs = np.array([min(kmh / 120.0, 1.0)], dtype=np.float32)
        return (img, speed_obs), reward, done, None

    def cleanup(self):
        for sensor in getattr(self, 'sensor_list', []):
            try: sensor.stop()
            except Exception: pass
        for actor in getattr(self, 'actor_list', []):
            try: actor.destroy()
            except Exception: pass
        for npc in self.npc_list:
            try: npc.destroy()
            except Exception: pass


# DQN Agent
class DQNAgent:
    def __init__(self):
        self.model        = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory         = PrioritizedReplayMemory(REPLAY_MEMORY_SIZE)
        self.tensorboard           = ModifiedTensorBoard(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph                 = tf.get_default_graph()
        self.terminate             = False
        self.last_logged_episode   = 0
        self.training_initialized  = False
        self._last_mean_td_error   = 0.0
        self._last_mean_q_value    = 0.0

    def create_model(self):
        img_input = Input(
            shape=(MODEL_IM_HEIGHT, MODEL_IM_WIDTH, MODEL_IM_CHANNELS),
            name="img_input"
        )
        x = Conv2D(32, (3, 3), activation='relu')(img_input)
        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        x = MaxPooling2D(2, 2)(x)
        x = Flatten()(x)

        speed_input = Input(shape=(1,), name="speed_input")
        merged = Concatenate()([x, speed_input])
        merged = Dense(256, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        output = Dense(NUM_ACTIONS, activation='linear')(merged)

        model = Model(inputs=[img_input, speed_input], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch, _, indices = self.replay_memory.sample(MINIBATCH_SIZE)

        cur_imgs   = np.array([t[0][0] for t in minibatch], dtype=np.float32) / 255.0
        cur_speeds = np.array([t[0][1] for t in minibatch], dtype=np.float32)
        new_imgs   = np.array([t[3][0] for t in minibatch], dtype=np.float32) / 255.0
        new_speeds = np.array([t[3][1] for t in minibatch], dtype=np.float32)

        with self.graph.as_default():
            cur_qs = self.model.predict([cur_imgs, cur_speeds], PREDICTION_BATCH_SIZE)
            on_qs  = self.model.predict([new_imgs, new_speeds], PREDICTION_BATCH_SIZE)
            tgt_qs = self.target_model.predict([new_imgs, new_speeds], PREDICTION_BATCH_SIZE)

        X_img = []; X_spd = []; y = []; errors = []

        for idx, (state, action, reward, new_state, done) in enumerate(minibatch):
            new_q = reward if done else reward + DISCOUNT * tgt_qs[idx][np.argmax(on_qs[idx])]
            q     = cur_qs[idx].copy()
            errors.append(abs(new_q - q[action]))
            q[action] = new_q
            X_img.append(state[0])
            X_spd.append(state[1])
            y.append(q)

        self.replay_memory.update_priorities(indices, errors)
        self._last_mean_td_error = float(np.mean(errors))
        self._last_mean_q_value  = float(np.mean(cur_qs))

        log_now = self.tensorboard.step > self.last_logged_episode
        if log_now:
            self.last_logged_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(
                [np.array(X_img) / 255.0, np.array(X_spd)],
                np.array(y),
                batch_size=TRAINING_BATCH_SIZE,
                verbose=0, shuffle=False,
                callbacks=[self.tensorboard] if log_now else None,
            )

        if log_now:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        img, speed = state
        img_in   = np.array(img, dtype=np.float32).reshape(
            1, MODEL_IM_HEIGHT, MODEL_IM_WIDTH, MODEL_IM_CHANNELS) / 255.0
        speed_in = np.array(speed, dtype=np.float32).reshape(1, 1)
        return self.model.predict([img_in, speed_in])[0]

    def train_in_loop(self):
        X_img = np.random.uniform(
            size=(1, MODEL_IM_HEIGHT, MODEL_IM_WIDTH, MODEL_IM_CHANNELS)).astype(np.float32)
        X_spd = np.random.uniform(size=(1, 1)).astype(np.float32)
        y     = np.random.uniform(size=(1, NUM_ACTIONS)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit([X_img, X_spd], y, verbose=False, batch_size=1)
        self.training_initialized = True

        oom_backoff = 0
        while not self.terminate:
            try:
                self.train()
                oom_backoff = 0
                time.sleep(0.01)
            except tf.errors.ResourceExhaustedError:
                oom_backoff = min(oom_backoff + 10, 60)
                print(f"\n[Training] OOM - pausing {oom_backoff}s...")
                time.sleep(oom_backoff)
            except Exception as e:
                print(f"\n[Training] Unexpected error (continuing): {e}")
                time.sleep(1)

    def load_transfer_weights(self, checkpoint_path, conv_only=False):
        """
        Transfer compatible layer weights from a previous checkpoint via
        direct HDF5 reading. Skips the first Conv2D (RGB→grayscale channel
        change). All other layers transfer by name then shape match

        conv_only=True: transfer conv layers only, skip all dense layers.
        Use this when retraining on a new map/reward function where the
        Q-value layers (dense) encoded a bad policy that needs discarding,
        but the visual feature layers (conv) are still useful
        """
        import h5py

        print(f"[Transfer] Loading from: {checkpoint_path}")
        try:
            f = h5py.File(checkpoint_path, 'r')
        except Exception as e:
            print(f"[Transfer] Could not open file: {e} - starting from scratch.")
            return

        if 'model_weights' not in f:
            print("[Transfer] No model_weights found - starting from scratch.")
            f.close()
            return

        wg = f['model_weights']
        ckpt_names = set(wg.keys())

        def get_layer_weights(name):
            if name not in wg: return None
            grp = wg[name]
            if name in grp: grp = grp[name]
            datasets = {k: np.array(grp[k]) for k in grp.keys()
                        if isinstance(grp[k], h5py.Dataset)}
            if not datasets: return None
            def key(n):
                nl = n.lower()
                if 'kernel' in nl or '_w' in nl: return 0
                if 'bias'   in nl or '_b' in nl: return 1
                return 2
            return [datasets[k] for k in sorted(datasets.keys(), key=key)]

        cache = {n: get_layer_weights(n) for n in ckpt_names if get_layer_weights(n)}

        transferred = []; skipped = []
        first_conv  = False; used = set()

        for layer in self.model.layers:
            nw = layer.get_weights()
            if not nw: continue

            if not first_conv and 'conv2d' in layer.name:
                first_conv = True
                skipped.append(f"{layer.name} (first conv - channels 3→1)")
                continue

            # conv_only mode: skip dense layers so Q-values retrain from scratch
            if conv_only and 'dense' in layer.name:
                skipped.append(f"{layer.name} (skipped - conv_only mode)")
                continue

            ow = get_layer_weights(layer.name)
            src = layer.name if ow else None

            if ow is None:
                lt = layer.name.rstrip('_0123456789')
                for cn, cw in cache.items():
                    if cn in used or lt not in cn: continue
                    if (len(cw) == len(nw) and
                            all(o.shape == n.shape for o, n in zip(cw, nw))):
                        ow = cw; src = cn
                        print(f"[Transfer] Shape match: {layer.name} ← {cn}")
                        break

            partial_cand = partial_src = None
            if ow is None and 'dense' in layer.name:
                for cn, cw in cache.items():
                    if cn in used or 'dense' not in cn: continue
                    if len(cw) == 2:
                        partial_cand = cw; partial_src = cn; break

            if ow is None and partial_cand is None:
                skipped.append(f"{layer.name} (not found)")
                continue

            if ow is not None:
                if (len(ow) == len(nw) and
                        all(o.shape == n.shape for o, n in zip(ow, nw))):
                    layer.set_weights(ow)
                    transferred.append(layer.name)
                    if src: used.add(src)
                    continue

            cw_use = ow if ow is not None else partial_cand
            cs_use = src if ow is not None else partial_src

            if 'dense' in layer.name and cw_use is not None and len(cw_use) == 2:
                ok, ob = cw_use
                nks, nbs = nw[0].shape, nw[1].shape
                if (ok.ndim == 2 and nks[1] == ok.shape[1] and
                        nks[0] == ok.shape[0] + 1 and ob.shape == nbs):
                    pk = layer.get_weights()[0].copy()
                    pk[:ok.shape[0], :] = ok
                    layer.set_weights([pk, ob])
                    transferred.append(f"{layer.name} (partial - speed row random)")
                    if cs_use: used.add(cs_use)
                    continue

            skipped.append(f"{layer.name} (shape mismatch)")

        f.close()
        self.target_model.set_weights(self.model.get_weights())
        print(f"[Transfer] Transferred {len(transferred)}: {', '.join(transferred) or 'none'}")
        print(f"[Transfer] Skipped    {len(skipped)}: {', '.join(skipped) or 'none'}")


# CARLA process helpers

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
            print(f"[CARLA] Not ready (attempt {attempt}/{retries}), retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"[CARLA] Could not connect after {retries} attempts.")


# Main Training Loop
if __name__ == "__main__":
    FPS               = 20
    ep_rewards        = [-200]
    steps_per_episode = []
    collisions        = []
    lane_invasions    = []
    offroads          = []

    best_avg_reward         = -float('inf')
    plateau_counter         = 0
    best_es_reward          = -float('inf')
    ep_len_increase_counter = 0

    average_reward = -200.0
    min_reward     = -200.0
    max_reward     = -200.0
    collision_rate = 1.0
    offroad_rate   = 0.0
    lane_invasion_rate = 0.0

    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    os.makedirs("models",             exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/recovery",    exist_ok=True)

    agent = DQNAgent()

    if TRANSFER_WEIGHTS_FROM:
        agent.load_transfer_weights(TRANSFER_WEIGHTS_FROM, conv_only=TRANSFER_CONV_ONLY)

    carla_process = launch_carla()
    wait_for_carla()

    env = CarEnv()
    env.spawn_npcs()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    dummy = np.ones((MODEL_IM_HEIGHT, MODEL_IM_WIDTH, MODEL_IM_CHANNELS), dtype=np.float32)
    agent.get_qs((dummy, np.array([0.0], dtype=np.float32)))

    episode           = 1
    recovery_attempts = 0

    try:
        with tqdm(total=EPISODES, ascii=True, unit='episodes') as pbar:
            while episode <= EPISODES:
                try:
                    env.refresh_world()
                    env.check_and_respawn_npcs()

                    env.collision_hist     = []
                    env.lane_invasion_hist = []
                    agent.tensorboard.step = episode
                    episode_reward         = 0
                    step                   = 1
                    had_collision          = 0
                    went_offroad           = 0
                    current_state          = env.reset()

                    while True:
                        if np.random.random() > epsilon:
                            action = np.argmax(agent.get_qs(current_state))
                        else:
                            action = np.random.randint(0, NUM_ACTIONS)
                            time.sleep(1 / FPS)

                        new_state, reward, done, _ = env.step(action)
                        episode_reward += reward

                        agent.update_replay_memory(
                            (current_state, action, reward, new_state, done))
                        current_state = new_state
                        step         += 1

                        if done:
                            had_collision = 1 if len(env.collision_hist) > 0 else 0
                            # Detect off-road terminal by reward value
                            went_offroad  = 1 if reward == OFFROAD_TERMINAL_REWARD else 0
                            break

                    for sensor in getattr(env, 'sensor_list', []):
                        try: sensor.stop()
                        except Exception: pass
                    for actor in env.actor_list:
                        try: actor.destroy()
                        except Exception: pass

                    ep_rewards.append(episode_reward)
                    steps_per_episode.append(step)
                    collisions.append(had_collision)
                    lane_invasions.append(1 if env.total_lane_invasions > 0 else 0)
                    offroads.append(went_offroad)
                    recovery_attempts = 0

                    # Adaptive episode length
                    if (episode >= AGGREGATE_STATS_EVERY
                            and episode % AGGREGATE_STATS_EVERY == 0):
                        recent_cr = (
                            sum(collisions[-AGGREGATE_STATS_EVERY:])
                            / AGGREGATE_STATS_EVERY
                        )
                        if recent_cr < 0.3:
                            ep_len_increase_counter += 1
                        else:
                            ep_len_increase_counter = 0

                        if (ep_len_increase_counter >= EP_LEN_INCREASE_WINDOW
                                and SECONDS_PER_EPISODE < SECONDS_PER_EPISODE_MAX):
                            SECONDS_PER_EPISODE = min(
                                SECONDS_PER_EPISODE + 5, SECONDS_PER_EPISODE_MAX)
                            ep_len_increase_counter = 0
                            print(f"\n[Training] Episode cap → {SECONDS_PER_EPISODE}s "
                                  f"at ep {episode} (collision rate: {recent_cr:.2f})")

                    # Periodic stats
                    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                        recent         = ep_rewards[-AGGREGATE_STATS_EVERY:]
                        average_reward = sum(recent) / len(recent)
                        min_reward     = min(recent)
                        max_reward     = max(recent)

                        avg_steps = (
                            sum(steps_per_episode[-AGGREGATE_STATS_EVERY:])
                            / AGGREGATE_STATS_EVERY
                        )
                        collision_rate = (
                            sum(collisions[-AGGREGATE_STATS_EVERY:])
                            / AGGREGATE_STATS_EVERY
                        )
                        lane_invasion_rate = (
                            sum(lane_invasions[-AGGREGATE_STATS_EVERY:])
                            / AGGREGATE_STATS_EVERY
                        )
                        offroad_rate = (
                            sum(offroads[-AGGREGATE_STATS_EVERY:])
                            / AGGREGATE_STATS_EVERY
                        )

                        agent.tensorboard.update_stats(
                            reward_avg         = average_reward,
                            reward_min         = min_reward,
                            reward_max         = max_reward,
                            epsilon            = epsilon,
                            avg_steps          = avg_steps,
                            episode_length_cap = SECONDS_PER_EPISODE,
                            collision_rate     = collision_rate,
                            lane_invasion_rate = lane_invasion_rate,
                            offroad_rate       = offroad_rate,
                            mean_td_error      = agent._last_mean_td_error,
                            mean_q_value       = agent._last_mean_q_value,
                        )

                        if average_reward > best_avg_reward:
                            best_avg_reward = average_reward
                            agent.model.save(
                                f'models/best_{MODEL_NAME}'
                                f'__{max_reward:_>7.2f}max'
                                f'_{average_reward:_>7.2f}avg'
                                f'_{min_reward:_>7.2f}min'
                                f'__ep{episode}'
                                f'__{int(time.time())}.model'
                            )

                        if min_reward >= MIN_REWARD:
                            agent.model.save(
                                f'models/{MODEL_NAME}'
                                f'__{max_reward:_>7.2f}max'
                                f'_{average_reward:_>7.2f}avg'
                                f'_{min_reward:_>7.2f}min'
                                f'__{int(time.time())}.model'
                            )

                        if episode >= EARLY_STOP_MIN_EPISODES:
                            # Require both collision_rate and offroad_rate to be
                            # low before considering early stopping. Previously
                            # the agent triggered early stop by going off-road
                            # (avoiding collisions) while never learning to turn
                            genuinely_good = (
                                collision_rate <= EARLY_STOP_COLLISION_THRESHOLD
                                and offroad_rate <= EARLY_STOP_COLLISION_THRESHOLD
                            )
                            if (average_reward - best_es_reward < PLATEAU_MIN_DELTA
                                    and genuinely_good):
                                plateau_counter += 1
                            else:
                                plateau_counter = 0
                            if average_reward > best_es_reward:
                                best_es_reward = average_reward
                            if plateau_counter >= EARLY_STOPPING_WINDOW:
                                print(
                                    f"\n[Early Stop] ep {episode} - "
                                    f"reward_avg={average_reward:.2f} plateaued, "
                                    f"collision_rate={collision_rate:.2f}, "
                                    f"offroad_rate={offroad_rate:.2f}."
                                )
                                raise KeyboardInterrupt

                    # Rolling checkpoint
                    if episode % CHECKPOINT_EVERY == 0:
                        agent.model.save(
                            f'models/checkpoints/checkpoint_{MODEL_NAME}'
                            f'_ep{episode}.model'
                        )
                        print(f"\n[Checkpoint] ep {episode} - "
                              f"avg: {average_reward:.1f}, "
                              f"collision: {collision_rate:.2f}, "
                              f"offroad: {offroad_rate:.2f}, "
                              f"lane: {lane_invasion_rate:.2f}, "
                              f"ε: {epsilon:.5f}, "
                              f"cap: {SECONDS_PER_EPISODE}s")

                    episode += 1
                    pbar.update(1)

                except KeyboardInterrupt:
                    raise

                except Exception as e:
                    recovery_attempts += 1
                    print(f"\n[Recovery] Crash at ep {episode} "
                          f"(attempt {recovery_attempts}/{MAX_RECOVERY_ATTEMPTS}): {e}")

                    if recovery_attempts > MAX_RECOVERY_ATTEMPTS:
                        print("[Recovery] Max attempts exceeded. Stopping.")
                        raise KeyboardInterrupt

                    try:
                        agent.model.save(
                            f'models/recovery/recovery_ep{episode}_{int(time.time())}.model')
                        print("[Recovery] Emergency checkpoint saved.")
                    except Exception:
                        print("[Recovery] Could not save emergency checkpoint.")

                    kill_carla_process(carla_process)
                    carla_process = launch_carla()
                    wait_for_carla()
                    env.reconnect()
                    env.spawn_npcs()
                    print(f"[Recovery] Simulator restarted. Resuming from ep {episode}.")

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving before exit...")

    finally:
        agent.terminate = True
        trainer_thread.join()

        try:
            agent.model.save(
                f'models/{MODEL_NAME}'
                f'__{max_reward:_>7.2f}max'
                f'_{average_reward:_>7.2f}avg'
                f'_{min_reward:_>7.2f}min'
                f'__FINAL_{int(time.time())}.model'
            )
        except Exception as e:
            print(f"[Warning] Final model save failed: {e}")

        try:
            agent.save_for_deployment("models/deployment")
        except Exception as e:
            print(f"[Warning] Deployment save failed: {e}")

        env.cleanup()
        kill_carla_process(carla_process)
        print("[Done] Training complete.")