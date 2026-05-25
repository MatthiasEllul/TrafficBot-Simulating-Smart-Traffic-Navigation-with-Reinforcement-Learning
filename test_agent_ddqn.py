"""
test_agent_ddqn.py - Batch evaluation of multiple DQN checkpoint models in CARLA 0.9.8

Runs each model in MODELS_TO_TEST through NUM_EPISODES evaluation episodes,
then prints a ranked comparison table so you can identify the best checkpoint
empirically rather than from training metrics alone.

Usage:
    py -3.7 test_agent_ddqn.py

Configuration:
    1. Edit MODELS_TO_TEST with the paths and labels for each checkpoint
    2. Set CARLA_EXE_PATH to your CarlaUE4.exe location
    3. Set NUM_EPISODES and EPISODE_SECONDS as needed
    4. Set SHOW_PREVIEW = True to watch the agent drive (slows evaluation)
"""

import glob
import os
import sys
import random
import time
import threading
import subprocess
import math
import gc
import numpy as np
import cv2
import tensorflow as tf
import keras.backend.tensorflow_backend as backend

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Models to evaluate
# Add or remove entries as needed. 'label' is shown in the results table
# 'note' is optional context shown in the summary (e.g. whether the model may have used a reward exploit during training)
MODELS_TO_TEST = [
    # {
    #     "label": "Mar-28 ep4500",
    #     "path":  r"models\best_DQN_CNN__1333.97max__473.98avg_-411.90min__ep4500__1774729779.model",
    #     "note":  "reward-hacking run - lane_invasion_rate 0.94 in training",
    # },
    # {
    #     "label": "Apr-2 ep5300",
    #     "path":  r"models\best_DQN_CNN___747.90max__349.86avg_-365.13min__ep5300__1775161499.model",
    #     "note":  "honest penalty run - best reward_avg",
    # },
    # {
    #     "label": "Apr-8 ep4250",
    #     "path":  r"models\best_DQN_CNN___711.50max__294.40avg_-870.73min__ep4250__1775637817.model",
    #     "note":  "honest penalty run - best combined metrics",
    # },
    # {
    #     "label": "Apr-20 ep3250",
    #     "path":  r"models\best_DQN_CNN__3028.78max_1552.89avg____4.94min__ep3250__1776661047.model",
    #     "note":  "waypoint run - early peak",
    # },
    # {
    #     "label": "Apr-20 ep8400",
    #     "path":  r"models\best_DQN_CNN__4053.60max_1870.03avg____8.07min__ep8400__1776717205.model",
    #     "note":  "waypoint run - balanced metrics",
    # },
    # {
    #     "label": "Apr-20 ep8450",
    #     "path":  r"models\best_DQN_CNN__4017.40max_2250.63avg__415.63min__ep8450__1776718340.model",
    #     "note":  "waypoint run - highest ever reward average",
    # },
    # {
    #     "label": "Apr-20 ep8750",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep8750.model",
    #     "note":  "waypoint run - policy leading up to best results ever recorded",
    # },
    # {
    #     "label": "Apr-23 ep2850",
    #     "path":  r"models\best_DQN_CNN__1879.00max_1058.71avg_-142.99min__ep2850__1776898342.model",
    #     "note":  "waypoint + progress run - all-time lowest ever collision rate recorded in training (0.18)",
    # },
    # {
    #     "label": "Apr-23 ep4250",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep4250.model",
    #     "note":  "waypoint + progress run - equally the lowest ever collision rate recorded in training (0.18)",
    # },
    # {
    #     "label": "Apr-23 ep5300",
    #     "path":  r"models\best_DQN_CNN__2509.44max_1484.00avg__-52.97min__ep5300__1776957944.model",
    #     "note":  "waypoint + progress run - mid-training run policy demo (1/4)",
    # },
    # {
    #     "label": "Apr-23 ep6950",
    #     "path":  r"models\best_DQN_CNN__2995.93max_1656.98avg__-59.27min__ep6950__1776998584.model",
    #     "note":  "waypoint + progress run - mid-training run policy demo (2/4)",
    # },
    # {
    #     "label": "Apr-23 ep7000",
    #     "path":  r"models\best_DQN_CNN__3036.92max_1678.07avg_-108.72min__ep7000__1777000157.model",
    #     "note":  "waypoint + progress run - mid-training run policy demo (3/4)",
    # },
    # {
    #     "label": "Apr-23 ep7100",
    #     "path":  r"models\best_DQN_CNN__3006.51max_1790.51avg__-14.86min__ep7100__1777003091.model",
    #     "note":  "waypoint + progress run - mid-training run policy demo (4/4)",
    # },
    # {
    #     "label": "Apr-23 ep8000",
    #     "path":  r"models\best_DQN_CNN__3068.65max_1880.74avg__-32.50min__ep8000__1777028449.model",
    #     "note":  "waypoint + progress run - all-time highest ever reward average recorded in training (1880.74)",
    # },
    # {
    #     "label": "Apr-23 9850",
    #     "path":  r"models\recovery\recovery_ep9879_1777082738.model",
    #     "note":  "waypoint + progress run - lowest lane invasion rate recorded for this run",
    # },
    # {
    #     "label": "Apr-26 5400",
    #     "path":  r"models\best_DQN_CNN__3057.61max_1055.25avg_-200.00min__ep5400__1777213341.model",
    #     "note":  "town03 run - best reward average",
    # },
    # {
    #     "label": "Apr-26 6500",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep6500.model",
    #     "note":  "town03 run - collision, lane, and offroad all almost 0",
    # },
    # {
    #     "label": "Apr-26 6700",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep6700.model",
    #     "note":  "town03 run - collision, lane, and offroad all almost 0",
    # },
    # {
    #     "label": "Apr-26 8750",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep8750.model",
    #     "note":  "town03 run - offroad rate dropped to 0 and remained for the rest of the run",
    # },
    # {
    #     "label": "Apr-26 9000",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep9000.model",
    #     "note":  "town03 run - offroad rate dropped to 0 and remained for the rest of the run",
    # },
    # {
    #     "label": "Apr-26 9750",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep9750.model",
    #     "note":  "town03 run - weights at end of run",
    # },
    # {
    #     "label": "Apr-26 10000",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep10000.model",
    #     "note":  "town03 run - weights at end of run",
    # },
    # {
    #     "label": "Apr-28 100",
    #     "path":  r"models\best_DQN_CNN___867.13max__242.64avg___19.29min__ep100__1777291531.model",
    #     "note":  "town01 run",
    # },
    # {
    #     "label": "Apr-28 200",
    #     "path":  r"models\best_DQN_CNN___736.08max__260.82avg_-101.45min__ep200__1777292394.model",
    #     "note":  "town01 run",
    # },
    # {
    #     "label": "Apr-28 300",
    #     "path":  r"models\best_DQN_CNN___888.06max__271.56avg__-48.63min__ep300__1777293243.model",
    #     "note":  "town01 run",
    # },
    # {
    #     "label": "Apr-28 550",
    #     "path":  r"models\best_DQN_CNN__1285.25max__300.68avg__-42.44min__ep550__1777295661.model",
    #     "note":  "town01 run",
    # },
    # {
    #     "label": "Apr-28 1050",
    #     "path":  r"models\best_DQN_CNN__1681.14max__317.39avg__-84.61min__ep1050__1777299845.model",
    #     "note":  "town01 run",
    # },
    # {
    #     "label": "Apr-28 1700",
    #     "path":  r"models\best_DQN_CNN__1295.29max__412.74avg_-138.65min__ep1700__1777305307.model",
    #     "note":  "town01 run",
    # },
    # {
    #     "label": "Apr-28 3550",
    #     "path":  r"models\best_DQN_CNN__2158.35max__547.55avg___15.94min__ep3550__1777322088.model",
    #     "note":  "town01 run - lowest collision rate (0.32)",
    # },
    # {
    #     "label": "Apr-28 3700",
    #     "path":  r"models\best_DQN_CNN__2493.41max__659.84avg__-99.52min__ep3700__1777323616.model",
    #     "note":  "town01 run - lowest collision rate (0.32)",
    # },
    # {
    #     "label": "Apr-28 6700",
    #     "path":  r"models\best_DQN_CNN__3294.70max_1038.25avg___-7.87min__ep6700__1777351494.model",
    #     "note":  "town01 run - balanced mid-run metrics",
    # },
    # {
    #     "label": "Apr-28 8350",
    #     "path":  r"models\best_DQN_CNN__3437.87max_1045.72avg__-20.95min__ep8350__1777366014.model",
    #     "note":  "town01 run",
    # },
    # {
    #     "label": "Apr-28 8800",
    #     "path":  r"models\best_DQN_CNN__3640.30max_1382.93avg__-99.92min__ep8800__1777372195.model",
    #     "note":  "town01 run",
    # },
    # {
    #     "label": "Apr-28 9750",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep9750.model",
    #     "note":  "town01 run - highest reward average",
    # },
    # {
    #     "label": "Apr-29 1",
    #     "path":  r"models\best_DQN_CNN___311.13max___55.56avg_-200.00min__ep1__1777465663.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 550",
    #     "path":  r"models\best_DQN_CNN___764.13max__204.69avg__-85.95min__ep550__1777470074.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 1750",
    #     "path":  r"models\best_DQN_CNN__1437.96max__301.55avg_-200.00min__ep1750__1777480771.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 3600",
    #     "path":  r"models\best_DQN_CNN__1965.23max__420.61avg__-31.01min__ep3600__1777495809.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 5000",
    #     "path":  r"models\best_DQN_CNN__1579.28max__436.06avg__-38.99min__ep5000__1777508178.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 6150",
    #     "path":  r"models\best_DQN_CNN__2497.89max__656.20avg__-58.16min__ep6150__1777519046.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 7750",
    #     "path":  r"models\best_DQN_CNN__1801.31max__656.63avg____8.10min__ep7750__1777532740.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 8250",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep8250.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 8500",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep8500.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 9250",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep9250.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 9500",
    #     "path":  r"models\checkpoints\checkpoint_DQN_CNN_ep9500.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "Apr-29 9800",
    #     "path":  r"models\DQN_CNN__2737.07max__780.02avg___57.12min__1777550079.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-2 10,000",
    #     "path":  r"models\DQN_CNN___100.10max___48.71avg__-23.17min__FINAL_1777744537.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-3 2100",
    #     "path":  r"models\best_DQN_CNN___771.93max__136.48avg___-0.02min__ep2100__1777767087.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-3 2350",
    #     "path":  r"models\DQN_CNN___194.24max__123.55avg__-15.71min__FINAL_1777768923.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-4 150",
    #     "path":  r"models\best_DQN_CNN___320.75max___83.78avg_-164.47min__ep150__1777877733.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-4 200",
    #     "path":  r"models\best_DQN_CNN___320.44max___92.69avg___66.83min__ep200__1777878147.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-4 700",
    #     "path":  r"models\DQN_CNN___317.87max___80.74avg___66.89min__1777881021.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-4 750",
    #     "path":  r"models\DQN_CNN___282.90max___85.51avg___66.99min__1777881419.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-4 800",
    #     "path":  r"models\DQN_CNN___426.94max___81.30avg___66.92min__1777881806.model",
    #     "note":  "town02 run",
    # },
    # {
    #     "label": "May-4 900",
    #     "path":  r"models\DQN_CNN___375.29max___79.80avg___66.85min__1777882580.model",
    #     "note":  "town02 run",
    # },
]

# ─── Evaluation settings ──────────────────────────────────────────────────────
NUM_EPISODES    = 20        # episodes per model - more = more reliable estimate
EPISODE_SECONDS = 60        # longer than training for a fairer assessment
SHOW_PREVIEW    = True     # set True to watch the agent drive (slows testing)

# Spawn point control
# USE_FIXED_SPAWN = False  →  random spawn each episode (original behaviour)
# USE_FIXED_SPAWN = True   →  always spawn at FIXED_SPAWN_INDEX
# Set FIXED_SPAWN_INDEX to match the index used during training (e.g. 4 for
# the Town02 fixed-spawn runs)
USE_FIXED_SPAWN   = True
FIXED_SPAWN_INDEX = 4     # only used when USE_FIXED_SPAWN = True

# CARLA settings
CARLA_EXE_PATH = r"C:\Users\matth\Downloads\CARLA_0.9.8\WindowsNoEditor\CarlaUE4.exe"
CARLA_LAUNCH_FLAGS = [
    "-quality-level=Low",
    "-RenderOffScreen",
    "-benchmark",
    "-fps=20",
]
CARLA_STARTUP_WAIT = 65 

# Model / environment constants (must match training)
IM_WIDTH  = 160
IM_HEIGHT = 120
ACTION_MAP = {
    0: {"throttle": 1.0, "steer": -1.0, "brake": 0.0},
    1: {"throttle": 1.0, "steer": -0.5, "brake": 0.0},
    2: {"throttle": 1.0, "steer":  0.0, "brake": 0.0},
    3: {"throttle": 1.0, "steer":  0.5, "brake": 0.0},
    4: {"throttle": 1.0, "steer":  1.0, "brake": 0.0},
    5: {"throttle": 0.5, "steer":  0.0, "brake": 0.0},
    6: {"throttle": 0.0, "steer":  0.0, "brake": 1.0},
}
NUM_ACTIONS = len(ACTION_MAP)

ACTION_NAMES = {
    0: "hard left",   1: "gentle left", 2: "straight",
    3: "gentle right", 4: "hard right", 5: "slow", 6: "brake",
}

BLUEPRINT_BLACKLIST = {
    'vehicle.bh.crossbike', 'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets', 'vehicle.harley-davidson.low_rider',
    'vehicle.kawasaki.ninja', 'vehicle.yamaha.yzf', 'vehicle.vespa.zx125',
}


# Helpers

def create_model():
    """
    Build the exact same CNN architecture used during training
    Must match train_agent_ddqn.py create_model() precisely
    """
    from keras.models import Model
    from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                               Dense, Dropout, Concatenate)
    from keras.optimizers import Adam

    img_input = Input(shape=(IM_HEIGHT, IM_WIDTH, 1), name="img_input")
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


def load_checkpoint(path):
    """
    Load weights from a .model checkpoint file directly via HDF5,
    bypassing keras.models.load_model entirely to avoid the h5py 3.x
    incompatibility that causes 'truth value of array is ambiguous'.
    Builds a fresh model then copies weights layer by layer by shape match
    """
    import h5py

    model = create_model()

    with h5py.File(path, 'r') as f:
        if 'model_weights' not in f:
            raise ValueError(f"No model_weights group found in {path}")
        wg = f['model_weights']

        def get_weights(layer_name):
            if layer_name not in wg:
                return None
            grp = wg[layer_name]
            if layer_name in grp:
                grp = grp[layer_name]
            datasets = {k: np.array(grp[k]) for k in grp.keys()
                        if isinstance(grp[k], h5py.Dataset)}
            if not datasets:
                return None
            def sort_key(n):
                nl = n.lower()
                if 'kernel' in nl or '_w' in nl: return 0
                if 'bias'   in nl or '_b' in nl: return 1
                return 2
            return [datasets[k] for k in sorted(datasets.keys(), key=sort_key)]

        ckpt_names = list(wg.keys())
        used = set()

        for layer in model.layers:
            nw = layer.get_weights()
            if not nw:
                continue
            # Try exact name match first
            ow = get_weights(layer.name)
            src = layer.name if ow else None
            # Fall back to any unused layer whose weights match by shape
            if ow is None:
                lt = layer.name.rstrip('_0123456789')
                for cn in ckpt_names:
                    if cn in used or lt not in cn:
                        continue
                    cw = get_weights(cn)
                    if cw and len(cw) == len(nw) and all(
                            o.shape == n.shape for o, n in zip(cw, nw)):
                        ow = cw; src = cn
                        break
            if ow is not None and len(ow) == len(nw) and all(
                    o.shape == n.shape for o, n in zip(ow, nw)):
                layer.set_weights(ow)
                if src:
                    used.add(src)

    graph = tf.get_default_graph()
    print(f"  [Model] Weights loaded from: {os.path.basename(path)}")
    return model, graph


def greedy_action(model, graph, img, speed_kmh):
    """Single forward pass - pure exploitation, no randomness."""
    img_in   = img.astype(np.float32).reshape(1, IM_HEIGHT, IM_WIDTH, 1) / 255.0
    speed_in = np.array([[min(speed_kmh / 120.0, 1.0)]], dtype=np.float32)
    with graph.as_default():
        q = model.predict([img_in, speed_in])[0]
    return int(np.argmax(q))


def launch_carla():
    print("[CARLA] Launching simulator...")
    proc = subprocess.Popen(
        [CARLA_EXE_PATH] + CARLA_LAUNCH_FLAGS,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[CARLA] PID {proc.pid} - waiting {CARLA_STARTUP_WAIT}s...")
    time.sleep(CARLA_STARTUP_WAIT)
    return proc


def kill_carla(proc):
    pid = getattr(proc, 'pid', None)
    if pid:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            try: proc.terminate()
            except Exception: pass
    gc.collect()
    time.sleep(20)     # wait for OS to fully reclaim CARLA's memory


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
    raise RuntimeError("[CARLA] Could not connect after all retries.")


# Test environment

class TestEnv:
    def __init__(self):
        self.client            = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world             = self.client.load_world('Town02')  # match training map
        self.world.set_weather(carla.WeatherParameters.ClearNoon)  # max visibility
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3           = self.blueprint_library.filter("model3")[0]
        self._lock             = threading.Lock()
        self._spec_lock        = threading.Lock()
        self.front_camera      = None
        self.spectator_frame   = None   # third-person colour frame for preview
        self.actor_list        = []
        self.sensor_list       = []

    def reset(self):
        self.collision_hist    = []
        self.lane_invasion_hist = []
        self.actor_list        = []
        self.sensor_list       = []

        spawn_points = self.world.get_map().get_spawn_points()

        if USE_FIXED_SPAWN:
            sp = spawn_points[FIXED_SPAWN_INDEX]
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, sp)
            except RuntimeError:
                sp_shifted = carla.Transform(
                    carla.Location(x=sp.location.x, y=sp.location.y,
                                   z=sp.location.z + 0.5),
                    sp.rotation
                )
                self.vehicle = self.world.spawn_actor(self.model_3, sp_shifted)
            if self.vehicle is None:
                raise RuntimeError(f"Could not spawn at fixed point {FIXED_SPAWN_INDEX}.")
        else:
            random.shuffle(spawn_points)
            self.vehicle = None
            for sp in spawn_points:
                try:
                    self.vehicle = self.world.spawn_actor(self.model_3, sp)
                    break
                except RuntimeError as e:
                    if "collision" in str(e).lower(): continue
                    raise
            if self.vehicle is None:
                raise RuntimeError("All spawn points occupied.")
        self.actor_list.append(self.vehicle)

        sensor_spawn = carla.Transform(carla.Location(x=2.5, z=0.7))

        rgb_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute("image_size_x", str(IM_WIDTH))
        rgb_bp.set_attribute("image_size_y", str(IM_HEIGHT))
        rgb_bp.set_attribute("fov", "110")
        self.sensor = self.world.spawn_actor(rgb_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor_list.append(self.sensor)
        self.sensor.listen(lambda d: self._process_img(d))

        col_bp = self.blueprint_library.find("sensor.other.collision")
        col_s  = self.world.spawn_actor(col_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(col_s); self.sensor_list.append(col_s)
        col_s.listen(lambda e: self.collision_hist.append(e))

        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        lane_s  = self.world.spawn_actor(lane_bp, sensor_spawn, attach_to=self.vehicle)
        self.actor_list.append(lane_s); self.sensor_list.append(lane_s)
        lane_s.listen(lambda e: self.lane_invasion_hist.append(e))

        # Third-person spectator camera - behind and above the vehicle.
        # Captures full colour BGR for the preview window only; not used for
        # inference. x=-7 = 7m behind, z=4 = 4m above, pitch=-15 = angled down
        spec_bp = self.blueprint_library.find('sensor.camera.rgb')
        spec_bp.set_attribute("image_size_x", "640")
        spec_bp.set_attribute("image_size_y", "480")
        spec_bp.set_attribute("fov", "90")
        spec_transform = carla.Transform(
            carla.Location(x=-7.0, z=4.0),
            carla.Rotation(pitch=-15.0)
        )
        self.spec_sensor = self.world.spawn_actor(
            spec_bp, spec_transform, attach_to=self.vehicle)
        self.actor_list.append(self.spec_sensor)
        self.sensor_list.append(self.spec_sensor)
        self.spec_sensor.listen(lambda d: self._process_spectator(d))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(1)

        with self._lock: self.front_camera = None
        while True:
            with self._lock:
                if self.front_camera is not None: break
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        with self._lock:
            return self.front_camera.copy()

    def _process_img(self, image):
        i    = np.array(image.raw_data)
        i2   = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        gray = np.mean(i2[:, :, :3], axis=2, keepdims=True).astype(np.uint8)
        with self._lock:
            self.front_camera = gray

    def _process_spectator(self, image):
        """Convert CARLA BGRA frame to BGR for cv2 display."""
        i   = np.array(image.raw_data)
        i2  = i.reshape((480, 640, 4))
        bgr = i2[:, :, :3]          # drop alpha channel
        with self._spec_lock:
            self.spectator_frame = bgr.copy()

    def step(self, action):
        self.vehicle.apply_control(carla.VehicleControl(**ACTION_MAP[action]))
        v   = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        collided     = len(self.collision_hist) > 0
        lane_invaded = len(self.lane_invasion_hist) > 0
        self.lane_invasion_hist.clear()
        elapsed = time.time() - self.episode_start
        done    = collided or elapsed >= EPISODE_SECONDS
        with self._lock:
            img = self.front_camera.copy()
        # Display third-person view on main thread
        if SHOW_PREVIEW:
            with self._spec_lock:
                spec = self.spectator_frame
            if spec is not None:
                display = spec.copy()
                label   = f"{ACTION_NAMES.get(action, str(action))}  {kmh}km/h"
                cv2.putText(display, label, (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Agent View", display)
                cv2.waitKey(1)
        return img, kmh, collided, lane_invaded, done

    def cleanup(self):
        for s in self.sensor_list:
            try: s.stop()
            except Exception: pass
        for a in self.actor_list:
            try: a.destroy()
            except Exception: pass


# Single-model evaluation

def evaluate_model(model, graph, env, label):
    """
    Run NUM_EPISODES episodes and return an aggregate results dict
    Prints a per-episode line for transparency
    """
    ep_results = []

    for ep in range(1, NUM_EPISODES + 1):
        try:
            img = env.reset()
        except Exception as e:
            print(f"    Ep {ep:>2}: reset failed - {e}")
            continue

        v   = env.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        step = 0; total_kmh = 0; lane_steps = 0; collided = False

        while True:
            action    = greedy_action(model, graph, img, kmh)
            img, kmh, collided, lane_invaded, done = env.step(action)
            step      += 1
            total_kmh += kmh
            if lane_invaded: lane_steps += 1
            if done: break

        duration = time.time() - env.episode_start
        avg_kmh  = total_kmh / max(step, 1)
        lane_pct = 100 * lane_steps / max(step, 1)
        outcome  = "COLLISION" if collided else "survived "

        print(f"    ep {ep:>2}  steps={step:>5}  {duration:>5.1f}s  "
              f"{avg_kmh:>5.1f}km/h  lane={lane_pct:>5.1f}%  {outcome}")

        ep_results.append({
            "collided":   collided,
            "steps":      step,
            "duration":   duration,
            "avg_kmh":    avg_kmh,
            "lane_pct":   lane_pct,
        })

        env.cleanup()
        time.sleep(0.3)

    if not ep_results:
        return None

    n = len(ep_results)
    return {
        "label":          label,
        "episodes":       n,
        "collision_rate": sum(r["collided"]  for r in ep_results) / n,
        "survival_rate":  sum(not r["collided"] for r in ep_results) / n,
        "avg_steps":      sum(r["steps"]     for r in ep_results) / n,
        "avg_duration":   sum(r["duration"]  for r in ep_results) / n,
        "avg_kmh":        sum(r["avg_kmh"]   for r in ep_results) / n,
        "avg_lane_pct":   sum(r["lane_pct"]  for r in ep_results) / n,
    }


# Scoring

def composite_score(r):
    """
    Single number to rank models. Higher = better
    Weights:
      - Survival rate (most important):  40%
      - Avg steps (longevity):           30%
      - Lane discipline (inverse pct):   20%
      - Avg speed in target range:       10%
    All components normalised to [0, 1] within the tested set before weighting
    """
    # Raw components (pre-normalisation)
    return {
        "survival":   r["survival_rate"],
        "steps_norm": r["avg_steps"] / 1500.0,          # 1500 = ~30s full episode
        "lane_norm":  1.0 - r["avg_lane_pct"] / 100.0,  # lower invasion = better
        "speed_norm": min(r["avg_kmh"], 60.0) / 60.0,   # reward speed up to 60 km/h
    }


def save_results(ranked, score_fn, skipped, timestamp):
    """
    Save results to two files in the current directory:
      - test_results_<timestamp>.txt  - human-readable summary (mirrors terminal output)
      - test_results_<timestamp>.csv  - machine-readable table for further analysis
    """
    txt_path = f"test_results_{timestamp}.txt"
    csv_path = f"test_results_{timestamp}.csv"

    # Text summary
    lines = []
    lines.append(f"EVALUATION RESULTS - {NUM_EPISODES} episodes × {EPISODE_SECONDS}s each")
    lines.append(f"Run timestamp : {timestamp}")
    lines.append("")
    lines.append(f"{'Rank':<5} {'Model':<16} {'Surv%':>6} {'CollRate':>9} "
                 f"{'AvgSteps':>9} {'AvgKmh':>7} {'Lane%':>7} {'Score':>7}  Note")
    lines.append("-" * 90)

    for rank, r in enumerate(ranked, 1):
        sc   = score_fn(r)
        note = r.get("note", "")
        lines.append(
            f"{rank:<5} {r['label']:<16} {r['survival_rate']*100:>5.1f}% "
            f"{r['collision_rate']:>9.2f} {r['avg_steps']:>9.0f} "
            f"{r['avg_kmh']:>7.1f} {r['avg_lane_pct']:>6.1f}% {sc:>7.3f}  {note}"
        )

    lines.append("-" * 90)
    lines.append(f"Scoring weights: survival 40% | steps 30% | lane 20% | speed 10%")
    lines.append("")

    best = ranked[0]
    lines.append(f"RECOMMENDED MODEL : {best['label']}")
    lines.append(f"  Survival rate   : {best['survival_rate']*100:.1f}%")
    lines.append(f"  Collision rate  : {best['collision_rate']:.2f}")
    lines.append(f"  Avg steps       : {best['avg_steps']:.0f}")
    lines.append(f"  Avg speed       : {best['avg_kmh']:.1f} km/h")
    lines.append(f"  Lane invasion   : {best['avg_lane_pct']:.1f}% of steps")
    if best.get("note"):
        lines.append(f"  Training note   : {best['note']}")

    if skipped:
        lines.append("")
        lines.append(f"Skipped: {', '.join(skipped)}")

    with open(txt_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Results saved to: {txt_path}")

    # CSV
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "label", "survival_rate", "collision_rate",
            "avg_steps", "avg_duration_s", "avg_kmh", "avg_lane_pct",
            "composite_score", "note"
        ])
        for rank, r in enumerate(ranked, 1):
            writer.writerow([
                rank, r["label"],
                f"{r['survival_rate']:.4f}", f"{r['collision_rate']:.4f}",
                f"{r['avg_steps']:.1f}", f"{r['avg_duration']:.1f}",
                f"{r['avg_kmh']:.2f}", f"{r['avg_lane_pct']:.2f}",
                f"{score_fn(r):.4f}", r.get("note", "")
            ])
    print(f"  Results saved to: {csv_path}")


def rank_results(all_results):
    """Return results sorted by weighted composite score, best first"""
    def score(r):
        c = composite_score(r)
        return (0.40 * c["survival"] +
                0.30 * c["steps_norm"] +
                0.20 * c["lane_norm"] +
                0.10 * c["speed_norm"])
    return sorted(all_results, key=score, reverse=True), score


# Main

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create the preview window once so it persists across all models
    if SHOW_PREVIEW:
        cv2.namedWindow("Agent View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Agent View", 480, 360)

    all_results = []
    skipped     = []

    total_models = len(MODELS_TO_TEST)

    for model_idx, cfg in enumerate(MODELS_TO_TEST, 1):
        label = cfg["label"]
        path  = cfg["path"]
        note  = cfg.get("note", "")

        print(f"\n{'='*65}")
        print(f"  Model {model_idx}/{total_models}: {label}")
        if note: print(f"  Note: {note}")
        print(f"  Path: {path}")
        print(f"{'='*65}")

        if not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}")
            skipped.append(label)
            continue

        # Load model
        try:
            model, graph = load_checkpoint(path)
        except Exception as e:
            print(f"  [SKIP] Could not load model: {e}")
            skipped.append(label)
            continue

        # Launch CARLA for this model
        carla_proc = launch_carla()
        wait_for_carla()
        env = TestEnv()

        try:
            result = evaluate_model(model, graph, env, label)
            if result:
                result["note"] = note
                all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] Evaluation failed: {e}")
            skipped.append(label)

        # Tear down CARLA between models to prevent memory accumulation
        env.cleanup()
        kill_carla(carla_proc)

        # Release model from memory before loading the next one
        del model
        gc.collect()
        time.sleep(5)

    # Results table
    if not all_results:
        print("\n[Done] No results collected.")
        sys.exit(0)

    ranked, score_fn = rank_results(all_results)

    print(f"\n{'='*75}")
    print(f"  EVALUATION RESULTS - {NUM_EPISODES} episodes × {EPISODE_SECONDS}s each")
    print(f"{'='*75}")
    print(f"{'Rank':<5} {'Model':<15} {'Surv%':>6} {'CollRate':>9} "
          f"{'AvgSteps':>9} {'AvgKmh':>7} {'Lane%':>7} {'Score':>7}")
    print(f"{'-'*75}")

    for rank, r in enumerate(ranked, 1):
        sc    = score_fn(r)
        surv  = r["survival_rate"] * 100
        coll  = r["collision_rate"]
        steps = r["avg_steps"]
        kmh   = r["avg_kmh"]
        lane  = r["avg_lane_pct"]
        print(f"{rank:<5} {r['label']:<15} {surv:>5.1f}% {coll:>9.2f} "
              f"{steps:>9.0f} {kmh:>7.1f} {lane:>6.1f}% {sc:>7.3f}")

    print(f"{'-'*75}")
    print(f"\n  Scoring weights: survival 40% | steps 30% | lane 20% | speed 10%")

    best = ranked[0]
    print(f"\n  RECOMMENDED MODEL: {best['label']}")
    print(f"  Survival: {best['survival_rate']*100:.1f}%  |  "
          f"Avg steps: {best['avg_steps']:.0f}  |  "
          f"Lane invasion: {best['avg_lane_pct']:.1f}%  |  "
          f"Avg speed: {best['avg_kmh']:.1f} km/h")
    if best.get("note"):
        print(f"  Note: {best['note']}")

    if skipped:
        print(f"\n  Skipped (file not found or load error): {', '.join(skipped)}")

    print(f"\n{'='*75}\n")

    # Save results to disk
    import time as _time
    timestamp = _time.strftime("%Y%m%d_%H%M%S")
    save_results(ranked, score_fn, skipped, timestamp)

    cv2.destroyAllWindows()
    print("[Done] Batch evaluation complete.")