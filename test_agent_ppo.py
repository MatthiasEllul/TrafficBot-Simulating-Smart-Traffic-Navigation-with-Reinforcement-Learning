"""
test_agent_ppo.py - Batch evaluation of PPO model checkpoints

Loads Stable Baselines 3 PPO zip checkpoints and evaluates each one across
NUM_EPISODES episodes of pure exploitation (deterministic=True), then prints
a ranked comparison table and saves results to .txt and .csv.

Usage:
    py -3.7 test_agent_ppo.py

Notes:
    - Paths should NOT include the .zip extension (SB3 adds it automatically)
    - The environment, reward function, and map must match training exactly
    - Set USE_FIXED_SPAWN and FIXED_SPAWN_INDEX to match training settings
"""

import os
import sys
import glob
import random
import time
import threading
import subprocess
import math
import gc
import csv
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import gym
from gym import spaces
from stable_baselines3 import PPO

# Models to evaluate
# Paths WITHOUT .zip extension. SB3 adds it automatically.
MODELS_TO_TEST = [
    {
        "label": "May-7 850",
        "path":  r"models\best_PPO_CNN__474.40max_237.01avg_4.32min__ep850__1778158539",
        "note":  "transition point - agent shifts from off-road to forward driving",
    },
    {
        "label": "May-7 1250",
        "path":  r"models\best_PPO_CNN__829.28max_329.34avg_-165.96min__ep1250__1778124734",
        "note":  "best from previous run",
    },
    {
        "label": "May-10 1000",
        "path":  r"models\checkpoints\checkpoint_PPO_CNN_ep1000",
        "note":  "stable high-reward window",
    },
    {
        "label": "May-11 5800",
        "path":  r"models\best_PPO_CNN__456.38max_274.68avg_-136.00min__ep5800__1778505171",
        "note":  "PPO run",
    },
    {
        "label": "May-11 7600",
        "path":  r"models\best_PPO_CNN__685.57max_336.34avg_-70.00min__ep7600__1778540840",
        "note":  "PPO run",
    },
]

# Evaluation settings
NUM_EPISODES    = 20
EPISODE_SECONDS = 60
SHOW_PREVIEW    = True

USE_FIXED_SPAWN   = True    # must match training
FIXED_SPAWN_INDEX = 4       # must match training

# CARLA / environment settings (must match training)
CARLA_EXE_PATH = r"C:\Users\matth\Downloads\CARLA_0.9.8\WindowsNoEditor\CarlaUE4.exe"
CARLA_LAUNCH_FLAGS = [
    "-quality-level=Low",
    "-RenderOffScreen",
    "-benchmark",
    "-fps=20",
]
CARLA_STARTUP_WAIT = 65
CARLA_MAP          = 'Town02'

IM_WIDTH  = 160
IM_HEIGHT = 120

LANE_GRACE_SECONDS    = 5.0
OFFROAD_TERMINATE_DIST = 3.5

ACTIONS = [
    {"throttle": 1.0, "steer": -1.0, "brake": 0.0},
    {"throttle": 1.0, "steer": -0.5, "brake": 0.0},
    {"throttle": 1.0, "steer":  0.0, "brake": 0.0},
    {"throttle": 1.0, "steer":  0.5, "brake": 0.0},
    {"throttle": 1.0, "steer":  1.0, "brake": 0.0},
    {"throttle": 0.5, "steer":  0.0, "brake": 0.0},
    {"throttle": 0.0, "steer":  0.0, "brake": 1.0},
]
ACTION_NAMES = {
    0: "hard left", 1: "gentle left", 2: "straight",
    3: "gentle right", 4: "hard right", 5: "slow", 6: "brake",
}

BLUEPRINT_BLACKLIST = {
    'vehicle.bh.crossbike', 'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets', 'vehicle.harley-davidson.low_rider',
    'vehicle.kawasaki.ninja', 'vehicle.yamaha.yzf', 'vehicle.vespa.zx125',
}


# CARLA helpers

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
    time.sleep(20)


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


# Test environment

class PPOTestEnv:
    """
    Minimal CARLA environment for PPO testing.
    Mirrors the CarlaEnvPPO from training but does NOT inherit gym.Env -
    we drive it manually step by step rather than through SB3's VecEnv wrapper
    """

    def __init__(self):
        self.client            = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world             = self.client.load_world(CARLA_MAP)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.carla_map         = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3           = self.blueprint_library.filter("model3")[0]
        self._lock             = threading.Lock()
        self._spec_lock        = threading.Lock()
        self.front_camera      = None
        self.spectator_frame   = None
        self.actor_list        = []
        self.sensor_list       = []
        self.vehicle           = None

    def reset(self):
        for s in self.sensor_list:
            try: s.stop()
            except Exception: pass
        for a in self.actor_list:
            try: a.destroy()
            except Exception: pass
        self.actor_list        = []
        self.sensor_list       = []
        self.collision_hist    = []
        self.lane_inv_hist     = []
        self.total_lane_inv    = 0

        spawn_points = self.world.get_map().get_spawn_points()
        if USE_FIXED_SPAWN:
            sp = spawn_points[FIXED_SPAWN_INDEX]
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, sp)
            except RuntimeError:
                sp2 = carla.Transform(
                    carla.Location(x=sp.location.x, y=sp.location.y,
                                   z=sp.location.z + 0.5), sp.rotation)
                self.vehicle = self.world.spawn_actor(self.model_3, sp2)
        else:
            random.shuffle(spawn_points)
            self.vehicle = None
            for sp in spawn_points:
                try:
                    self.vehicle = self.world.spawn_actor(self.model_3, sp); break
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

        # Third-person spectator camera for preview
        spec_bp = self.blueprint_library.find('sensor.camera.rgb')
        spec_bp.set_attribute("image_size_x", "640")
        spec_bp.set_attribute("image_size_y", "480")
        spec_bp.set_attribute("fov", "90")
        spec_t = carla.Transform(carla.Location(x=-7.0, z=4.0),
                                  carla.Rotation(pitch=-15.0))
        spec = self.world.spawn_actor(spec_bp, spec_t, attach_to=self.vehicle)
        self.actor_list.append(spec); self.sensor_list.append(spec)
        spec.listen(lambda d: self._process_spec(d))

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

        loc = self.vehicle.get_location()
        wp  = self.carla_map.get_waypoint(loc, project_to_road=True,
                                           lane_type=carla.LaneType.Driving)
        nxt = wp.next(3.0)
        self.next_wp = nxt[0] if nxt else wp

        return self._get_obs(0)

    def _process_img(self, image):
        i    = np.array(image.raw_data)
        i2   = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        gray = np.mean(i2[:, :, :3], axis=2, keepdims=True).astype(np.uint8)
        with self._lock:
            self.front_camera = gray

    def _process_spec(self, image):
        i   = np.array(image.raw_data)
        i2  = i.reshape((480, 640, 4))
        with self._spec_lock:
            self.spectator_frame = i2[:, :, :3].copy()

    def _get_obs(self, kmh):
        with self._lock:
            img = self.front_camera.copy() if self.front_camera is not None \
                  else np.zeros((IM_HEIGHT, IM_WIDTH, 1), dtype=np.uint8)
        img_f    = img[:, :, 0].astype(np.float32) / 255.0
        speed_f  = float(min(kmh / 120.0, 1.0))
        speed_ch = np.full((IM_HEIGHT, IM_WIDTH), speed_f, dtype=np.float32)
        return np.stack([img_f, speed_ch], axis=0)   # (2, H, W)

    def step(self, action):
        self.vehicle.apply_control(carla.VehicleControl(**ACTIONS[action]))
        v   = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        collided     = len(self.collision_hist) > 0
        lane_invaded = len(self.lane_inv_hist)  > 0
        self.lane_inv_hist.clear()

        elapsed = time.time() - self.episode_start
        done    = collided or elapsed >= EPISODE_SECONDS

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

        return self._get_obs(kmh), kmh, collided, lane_invaded, done

    def cleanup(self):
        for s in self.sensor_list:
            try: s.stop()
            except Exception: pass
        for a in self.actor_list:
            try: a.destroy()
            except Exception: pass


# Evaluation

def evaluate_model(model, env, label):
    results = []

    for ep in range(1, NUM_EPISODES + 1):
        try:
            obs = env.reset()
        except Exception as e:
            print(f"    ep {ep:>2}: reset failed - {e}")
            continue

        step       = 0
        total_kmh  = 0
        lane_steps = 0
        collided   = False

        while True:
            # PPO deterministic inference - no exploration
            action, _ = model.predict(obs, deterministic=True)
            obs, kmh, collided, lane_invaded, done = env.step(int(action))

            step      += 1
            total_kmh += kmh
            if lane_invaded:
                lane_steps += 1

            if done:
                break

        duration = time.time() - env.episode_start
        avg_kmh  = total_kmh / max(step, 1)
        lane_pct = 100 * lane_steps / max(step, 1)
        outcome  = "COLLISION" if collided else "survived "

        print(f"    ep {ep:>2}  steps={step:>5}  {duration:>5.1f}s  "
              f"{avg_kmh:>5.1f}km/h  lane={lane_pct:>5.1f}%  {outcome}")

        results.append({
            "collided":  collided,
            "steps":     step,
            "duration":  duration,
            "avg_kmh":   avg_kmh,
            "lane_pct":  lane_pct,
        })

        env.cleanup()
        time.sleep(0.3)

    if not results:
        return None

    n = len(results)
    return {
        "label":          label,
        "episodes":       n,
        "collision_rate": sum(r["collided"]  for r in results) / n,
        "survival_rate":  sum(not r["collided"] for r in results) / n,
        "avg_steps":      sum(r["steps"]     for r in results) / n,
        "avg_duration":   sum(r["duration"]  for r in results) / n,
        "avg_kmh":        sum(r["avg_kmh"]   for r in results) / n,
        "avg_lane_pct":   sum(r["lane_pct"]  for r in results) / n,
    }


def composite_score(r):
    return (0.40 * r["survival_rate"] +
            0.30 * min(r["avg_steps"] / 3000.0, 1.0) +
            0.20 * (1.0 - r["avg_lane_pct"] / 100.0) +
            0.10 * min(r["avg_kmh"] / 60.0, 1.0))


def save_results(ranked, skipped, timestamp):
    txt = f"test_results_ppo_{timestamp}.txt"
    csv_path = f"test_results_ppo_{timestamp}.csv"

    lines = [
        f"PPO EVALUATION RESULTS - {NUM_EPISODES} episodes × {EPISODE_SECONDS}s",
        f"Timestamp: {timestamp}", "",
        f"{'Rank':<5} {'Model':<20} {'Surv%':>6} {'CollRate':>9} "
        f"{'AvgSteps':>9} {'AvgKmh':>7} {'Lane%':>7} {'Score':>7}  Note",
        "-" * 95,
    ]
    for rank, r in enumerate(ranked, 1):
        sc = composite_score(r)
        lines.append(
            f"{rank:<5} {r['label']:<20} {r['survival_rate']*100:>5.1f}% "
            f"{r['collision_rate']:>9.2f} {r['avg_steps']:>9.0f} "
            f"{r['avg_kmh']:>7.1f} {r['avg_lane_pct']:>6.1f}% {sc:>7.3f}  "
            f"{r.get('note', '')}"
        )
    lines += ["-" * 95, ""]
    best = ranked[0]
    lines += [
        f"RECOMMENDED: {best['label']}",
        f"  Survival: {best['survival_rate']*100:.1f}%  "
        f"Steps: {best['avg_steps']:.0f}  "
        f"Speed: {best['avg_kmh']:.1f}km/h  "
        f"Lane: {best['avg_lane_pct']:.1f}%",
    ]
    if skipped:
        lines += ["", f"Skipped: {', '.join(skipped)}"]

    with open(txt, 'w') as f:
        f.write("\n".join(lines) + "\n")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["rank","label","survival_rate","collision_rate",
                    "avg_steps","avg_duration_s","avg_kmh","avg_lane_pct",
                    "composite_score","note"])
        for rank, r in enumerate(ranked, 1):
            w.writerow([rank, r["label"],
                        f"{r['survival_rate']:.4f}", f"{r['collision_rate']:.4f}",
                        f"{r['avg_steps']:.1f}", f"{r['avg_duration']:.1f}",
                        f"{r['avg_kmh']:.2f}", f"{r['avg_lane_pct']:.2f}",
                        f"{composite_score(r):.4f}", r.get("note","")])
    print(f"  Saved: {txt}")
    print(f"  Saved: {csv_path}")


# Main

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    if SHOW_PREVIEW:
        cv2.namedWindow("Agent View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Agent View", 640, 480)

    all_results = []
    skipped     = []
    total       = len(MODELS_TO_TEST)

    for idx, cfg in enumerate(MODELS_TO_TEST, 1):
        label = cfg["label"]
        path  = cfg["path"]
        note  = cfg.get("note", "")

        print(f"\n{'='*65}")
        print(f"  Model {idx}/{total}: {label}")
        if note: print(f"  Note: {note}")
        print(f"  Path: {path}")
        print(f"{'='*65}")

        # Check file exists (SB3 saves as .zip)
        if not os.path.exists(path + ".zip") and not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}.zip")
            skipped.append(label)
            continue

        # Load PPO model - custom policy kwargs must match training architecture
        try:
            from train_agent_ppo import DualInputCNNExtractor
            model = PPO.load(path, device="cpu")
            print(f"  [Model] Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"  [SKIP] Could not load: {e}")
            skipped.append(label)
            continue

        # Launch CARLA for this model
        carla_proc = launch_carla()
        wait_for_carla()
        env = PPOTestEnv()

        try:
            result = evaluate_model(model, env, label)
            if result:
                result["note"] = note
                all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {e}")
            skipped.append(label)

        env.cleanup()
        kill_carla(carla_proc)
        del model
        gc.collect()
        time.sleep(5)

    if not all_results:
        print("\n[Done] No results collected.")
        cv2.destroyAllWindows()
        sys.exit(0)

    ranked = sorted(all_results, key=composite_score, reverse=True)

    print(f"\n{'='*75}")
    print(f"  PPO EVALUATION RESULTS - {NUM_EPISODES} episodes × {EPISODE_SECONDS}s")
    print(f"{'='*75}")
    print(f"{'Rank':<5} {'Model':<20} {'Surv%':>6} {'CollRate':>9} "
          f"{'AvgSteps':>9} {'AvgKmh':>7} {'Lane%':>7} {'Score':>7}")
    print(f"{'-'*75}")
    for rank, r in enumerate(ranked, 1):
        sc = composite_score(r)
        print(f"{rank:<5} {r['label']:<20} {r['survival_rate']*100:>5.1f}% "
              f"{r['collision_rate']:>9.2f} {r['avg_steps']:>9.0f} "
              f"{r['avg_kmh']:>7.1f} {r['avg_lane_pct']:>6.1f}% {sc:>7.3f}")
    print(f"{'-'*75}")

    best = ranked[0]
    print(f"\n  RECOMMENDED: {best['label']}")
    print(f"  Survival {best['survival_rate']*100:.1f}%  "
          f"Steps {best['avg_steps']:.0f}  "
          f"Speed {best['avg_kmh']:.1f}km/h  "
          f"Lane {best['avg_lane_pct']:.1f}%")
    if skipped:
        print(f"\n  Skipped: {', '.join(skipped)}")
    print(f"\n{'='*75}\n")

    import time as _t
    save_results(ranked, skipped, _t.strftime("%Y%m%d_%H%M%S"))

    cv2.destroyAllWindows()
    print("[Done] PPO evaluation complete.")
