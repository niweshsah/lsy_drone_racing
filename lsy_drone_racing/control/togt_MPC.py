import json  # noqa: D100
import os
import sys
from datetime import datetime
from pprint import pprint
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from lsy_drone_racing.control.common_functions.yaml_import import load_yaml
    from lsy_drone_racing.control.GeometryEngines.geometryEngine2 import GeometryEngine
    from lsy_drone_racing.control.model_dynamics.mpc1 import SpatialMPC, get_drone_params
    
    print("✅ All modules imported successfully!")
    print(f"GeometryEngine location: {GeometryEngine.__module__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nCurrent Python Path:")
    pprint(sys.path)
from scipy.spatial.transform import Rotation as R

# Simulation Environment Imports
try:
    from drone_models.core import load_params
    from drone_models.utils.rotation import ang_vel2rpy_rates

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.utils.utils import draw_line
except ImportError:
    print("Warning: Simulation specific modules not found. Using mocks/defaults.")

    def load_params(*args) -> None:
        return None  # noqa: D103

    def ang_vel2rpy_rates(q, w):
        return np.zeros(3)

    def draw_line(*args, **kwargs):
        pass

    class Controller:
        pass  # noqa: D101


# Use non-interactive backend
matplotlib.use("Agg")

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

# CONSTANTS = {
#     "v_max_ref": 1.5,           # m/s
#     "corner_acc": 1.95,         # m/s^2
#     "mpc_horizon": 250,          # Steps
#     "max_lateral_width": 0.35,  # m (Corridor width)
#     "safety_radius": 0.15,      # m (Obstacle radius + Drone radius buffer)
#     "tf_horizon": 1.0           # s
# }

CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")

# ==============================================================================
# 2. DYNAMICS & MODEL DEFINITION
# ==============================================================================

# ==============================================================================
# 5. CONTROLLER CLASS
# ==============================================================================


class SpatialMPCController(Controller):
    def __init__(self, obs: Dict, info: Dict, config: Dict, env=None):
        self.params = get_drone_params()
        self.v_target = CONSTANTS["v_max_ref"]
        self.env = env
        self.OBS_RADIUS = CONSTANTS["safety_radius"]
        self.W1_MAX = CONSTANTS["max_lateral_width"]
        self.W2_MAX = CONSTANTS["max_lateral_width"]

        # --- LOAD OBSTACLES ---
        raw_obstacles = config.get("env", {}).get("track", {}).get("obstacles", [])
        if not raw_obstacles and "obstacles" in info:
            raw_obstacles = info["obstacles"]

        self.obstacles_pos = []
        for o in raw_obstacles:
            if isinstance(o, dict) and "pos" in o:
                self.obstacles_pos.append(np.array(o["pos"]))
            elif isinstance(o, (list, np.ndarray)):
                self.obstacles_pos.append(np.array(o))
            elif isinstance(o, dict):
                self.obstacles_pos.append(np.array(list(o.values())))

        print(f"\n[INIT] Loaded {len(self.obstacles_pos)} obstacles.")

        gates_list = config.get("env", {}).get("track", {}).get("gates", [])
        if not gates_list and "gates" in info:
            gates_list = info["gates"]
        gates_pos = [g["pos"] for g in gates_list]
        gates_normals = self._get_gate_normals(obs["gates_quat"])
        gates_y, gates_z = self._get_gate_yz(obs["gates_quat"])
        
        starting_pos = obs["pos"]
        print(f"Starting Position for Geometry Engine: {starting_pos}")

        # --- INITIALIZE GEOMETRY WITH OFFLINE CORRIDOR GENERATION ---
        # self.geo = GeometryEngine(
        #     gates_pos, gates_normals, obs["pos"], self.obstacles_pos, self.OBS_RADIUS
        # )
        
        # start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=np.float64)
        # print("observations pos:", obs)
        
        # geom = GeometryEngine(g_pos, g_norm, g_y, g_z, obs_pos, s_pos, gate_dims=(0.3, 0.3))

        
        self.geo = GeometryEngine(
            gates_pos, gates_normals, gates_y= gates_y, gates_z= gates_z, obstacles_pos= self.obstacles_pos, start_pos= starting_pos, gate_dims=(0.2, 0.2)
        )

        self.N_horizon = CONSTANTS["mpc_horizon"]
        self.mpc = SpatialMPC(self.params, N=self.N_horizon, Tf=CONSTANTS["tf_horizon"])

        self.prev_s = 0.0
        self.episode_start_time = datetime.now()
        self.step_count = 0
        self.debug = True
        self.control_log = {
            k: []
            for k in [
                "timestamps",
                "phi_c",
                "theta_c",
                "psi_c",
                "thrust_c",
                "solver_status",
                "s",
                "w1",
                "w2",
                "ds",
            ]
        }

        subsample = 5
        self.global_viz_center = self.geo.pt_frame["pos"][::subsample]
        self.global_viz_left = self.global_viz_center + (
            self.W1_MAX * self.geo.pt_frame["n1"][::subsample]
        )
        self.global_viz_right = self.global_viz_center - (
            self.W1_MAX * self.geo.pt_frame["n1"][::subsample]
        )

        self.reset_mpc_solver()

    def _draw_static_corridor(self):
        """Draws the full pre-computed corridor boundaries in the simulation."""
        if self.env is None:
            return

        # Extract pre-computed data
        step = 10
        positions = self.geo.pt_frame["pos"][::step]
        n1_vecs = self.geo.pt_frame["n1"][::step]

        # Get bounds for these specific indices
        full_indices = np.arange(0, len(self.geo.pt_frame["s"]), step)

        lb_w1 = self.geo.corridor_map["lb_w1"][full_indices]
        ub_w1 = self.geo.corridor_map["ub_w1"][full_indices]

        # Calculate Boundary Points
        left_bound_pts = positions + (n1_vecs * ub_w1[:, np.newaxis])
        right_bound_pts = positions + (n1_vecs * lb_w1[:, np.newaxis])

        # Draw Lines
        try:
            draw_line(self.env, points=left_bound_pts, rgba=np.array([1.0, 0.65, 0.0, 0.5]))
            draw_line(self.env, points=right_bound_pts, rgba=np.array([1.0, 0.65, 0.0, 0.5]))
        except Exception as e:
            print(f"Visualization Error: {e}")

    def _draw_debug_vectors(self):
        """Draws the collision rays (r_vec) stored in the geometry engine.
        These are the Cyan lines showing which obstacle is hitting the path.
        """
        if self.env is None or not self.geo.debug_vectors:
            return

        try:
            # draw_line expects points in sequence.
            # We create a list [start1, end1, start2, end2, ...]
            # Note: draw_line implementation varies. If it connects all points, this might look zig-zaggy.
            # Assuming draw_line takes segment pairs or we call it per segment.
            # For efficiency in some envs, we'll try to batch if possible, or loop.

            # If draw_line connects sequence, we use the nan trick or just loop.
            # Given we likely can't use NaN in this specific draw_line wrapper without knowing internals,
            # we will assume we can pass a list of pairs if modified, OR we loop.

            # Safe approach: Plot points. If draw_line connects them, it might be messy.
            # Ideally, draw_line handles `mode="lines"` (segments).
            # If not, we iterate.

            # Since we want to be safe and `draw_line` usually connects points sequentially:
            # We will grab just the last few vectors to avoid clutter/lag, or just loop.

            for start, end in self.geo.debug_vectors:
                draw_line(
                    self.env, points=np.array([start, end]), rgba=np.array([0.0, 1.0, 1.0, 0.8])
                )  # Cyan

        except Exception:
            pass

    def _draw_global_track(self):
        if self.env is None:
            return
        try:
            draw_line(self.env, points=self.global_viz_center, rgba=np.array([0.0, 1.0, 0.0, 0.5]))
        except Exception:
            pass

    def _get_gate_normals(self, gates_quaternions):
        rotations = R.from_quat(gates_quaternions)
        return rotations.as_matrix()[:, :, 0]
    
    def _get_gate_yz(self, gates_quaternions):
        rotations = R.from_quat(gates_quaternions)
        return rotations.as_matrix()[:, :, 1], rotations.as_matrix()[:, :, 2]

    def reset_mpc_solver(self):
        nx = 12
        hover_T = self.params["mass"] * self.params["g"]
        for k in range(self.N_horizon + 1):
            x_guess = np.zeros(nx)
            vel_k = self.v_target * (k / self.N_horizon)
            x_guess[3] = vel_k
            x_guess[0] = vel_k * k * (self.mpc.Tf / self.N_horizon) * 0.5
            self.mpc.solver.set(k, "x", x_guess)
            if k < self.N_horizon:
                self.mpc.solver.set(k, "u", np.array([0, 0, 0, hover_T]))
        self.prev_s = 0.0

    def compute_control(self, obs: Dict, info: Optional[Dict] = None) -> np.ndarray:
        self._draw_global_track()
        self._draw_static_corridor()
        # self._draw_debug_vectors() # New Debug Call

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])

        hover_T = self.params["mass"] * -self.params["g"]

        x_spatial = self._cartesian_to_spatial(obs["pos"], obs["vel"], obs["rpy"], obs["drpy"])
        self.mpc.solver.set(0, "lbx", x_spatial)
        self.mpc.solver.set(0, "ubx", x_spatial)

        curr_s = x_spatial[0]
        curr_ds = x_spatial[3]
        dt = self.mpc.Tf / self.mpc.N

        running_s_ref = curr_s
        max_lat_acc = CONSTANTS["corner_acc"]
        epsilon = 0.01
        vis_dynamic_left, vis_dynamic_right = [], []

        for k in range(self.mpc.N):
            s_pred = curr_s + k * max(curr_ds, 1.0) * dt

            # Lookup Bounds
            lb_w1, ub_w1 = self.geo.get_static_bounds(s_pred)
            lb_w2, ub_w2 = -self.W2_MAX, self.W2_MAX

            f = self.geo.get_frame(s_pred)
            vis_dynamic_left.append(f["pos"] + ub_w1 * f["n1"])
            vis_dynamic_right.append(f["pos"] + lb_w1 * f["n1"])

            if k > 0:
                lbx = np.array([lb_w1, lb_w2, -0.5, -0.5, -0.5])
                ubx = np.array([ub_w1, ub_w2, 0.5, 0.5, 0.5])
                self.mpc.solver.set(k, "lbx", lbx)
                self.mpc.solver.set(k, "ubx", ubx)

            k_mag = np.sqrt(f["k1"] ** 2 + f["k2"] ** 2)
            v_corner = np.sqrt(max_lat_acc / (k_mag + epsilon))
            v_ref_k = min(v_corner, self.v_target)
            running_s_ref += v_ref_k * dt

            p_k = np.concatenate(
                [f["t"], f["n1"], f["n2"], [f["k1"]], [f["k2"]], [f["dk1"]], [f["dk2"]]]
            )
            self.mpc.solver.set(k, "p", p_k)
            y_ref = np.zeros(16)
            y_ref[0] = running_s_ref
            y_ref[3] = v_ref_k
            y_ref[15] = hover_T
            self.mpc.solver.set(k, "yref", y_ref)

        s_end = running_s_ref + v_ref_k * dt
        f_end = self.geo.get_frame(s_end)
        p_end = np.concatenate(
            [
                f_end["t"],
                f_end["n1"],
                f_end["n2"],
                [f_end["k1"]],
                [f_end["k2"]],
                [f_end["dk1"]],
                [f_end["dk2"]],
            ]
        )
        self.mpc.solver.set(self.mpc.N, "p", p_end)

        lb_w1_e, ub_w1_e = self.geo.get_static_bounds(s_end)
        lbx_e = np.array([lb_w1_e, -self.W2_MAX, -0.5, -0.5, -0.5])
        ubx_e = np.array([ub_w1_e, self.W2_MAX, 0.5, 0.5, 0.5])
        self.mpc.solver.set(self.mpc.N, "lbx", lbx_e)
        self.mpc.solver.set(self.mpc.N, "ubx", ubx_e)

        yref_e = np.zeros(12)
        yref_e[0] = s_end
        yref_e[3] = v_ref_k
        yref_e[11] = hover_T
        self.mpc.solver.set(self.mpc.N, "yref", yref_e)

        status = self.mpc.solver.solve()

        if status != 0:
            u_opt = np.array([0.0, 0.0, 0.0, hover_T])
        else:
            u_opt = self.mpc.solver.get(0, "u")

        if self.env is not None and self.debug:
            try:
                mpc_points = []
                for k in range(self.mpc.N + 1):
                    x_k = self.mpc.solver.get(k, "x")
                    mpc_points.append(self._spatial_to_cartesian(x_k[0], x_k[1], x_k[2]))
                # draw_line(self.env, points=np.array(mpc_points), rgba=np.array([0.0, 0.0, 1.0, 0.8]))
                # draw_line(self.env, points=np.array(vis_dynamic_left), rgba=np.array([1.0, 0.5, 0.0, 0.9]))
                # draw_line(self.env, points=np.array(vis_dynamic_right), rgba=np.array([1.0, 0.5, 0.0, 0.9]))
            except Exception:
                pass

        self._log_control_step(x_spatial, u_opt, status)
        return np.array([u_opt[0], u_opt[1], u_opt[2], u_opt[3]])

    def _spatial_to_cartesian(self, s, w1, w2):
        f = self.geo.get_frame(s)
        return f["pos"] + w1 * f["n1"] + w2 * f["n2"]

    def _cartesian_to_spatial(self, pos, vel, rpy, drpy):
        s = self.geo.get_closest_s(pos, s_guess=self.prev_s)
        self.prev_s = s
        f = self.geo.get_frame(s)
        r_vec = pos - f["pos"]
        w1 = np.dot(r_vec, f["n1"])
        w2 = np.dot(r_vec, f["n2"])
        h = max(1 - f["k1"] * w1 - f["k2"] * w2, 0.01)
        ds = np.dot(vel, f["t"]) / h
        dw1 = np.dot(vel, f["n1"])
        dw2 = np.dot(vel, f["n2"])
        return np.array(
            [s, w1, w2, ds, dw1, dw2, rpy[0], rpy[1], rpy[2], drpy[0], drpy[1], drpy[2]]
        )

    def reset(self):
        self.prev_s = 0.0
        self.reset_mpc_solver()

    def episode_reset(self):
        self.reset()

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        return False

    def episode_callback(self):
        if len(self.control_log["timestamps"]) > 0:
            self.plot_all_diagnostics()
        return

    def _log_control_step(self, x_spatial, u_opt, status):
        self.step_count += 1
        elapsed = (datetime.now() - self.episode_start_time).total_seconds()
        self.control_log["timestamps"].append(elapsed)
        self.control_log["phi_c"].append(float(u_opt[0]))
        self.control_log["theta_c"].append(float(u_opt[1]))
        self.control_log["psi_c"].append(float(u_opt[2]))
        self.control_log["thrust_c"].append(float(u_opt[3]))
        self.control_log["solver_status"].append(int(status))
        self.control_log["s"].append(float(x_spatial[0]))
        self.control_log["w1"].append(float(x_spatial[1]))
        self.control_log["w2"].append(float(x_spatial[2]))
        self.control_log["ds"].append(float(x_spatial[3]))

    def save_control_log(self, filepath=None):
        if filepath is None:
            filepath = f"control_log_{self.episode_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, "w") as f:
            json.dump(self.control_log, f, indent=2)
        return filepath

    def plot_control_values(self, figsize=(16, 10), save_path=None):
        if len(self.control_log["timestamps"]) == 0:
            return
        t = np.array(self.control_log["timestamps"])
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle("MPC Control Values", fontsize=16)
        axes[0, 0].plot(t, self.control_log["phi_c"], "b")
        axes[0, 0].set_ylabel("Roll")
        axes[0, 1].plot(t, self.control_log["theta_c"], "g")
        axes[0, 1].set_ylabel("Pitch")
        axes[1, 0].plot(t, self.control_log["thrust_c"], "r")
        axes[1, 0].set_ylabel("Thrust")
        axes[1, 1].plot(t, self.control_log["psi_c"], "m")
        axes[1, 1].set_ylabel("Yaw")
        axes[2, 0].plot(t, self.control_log["s"], "c")
        axes[2, 0].set_ylabel("s")
        axes[2, 1].plot(t, self.control_log["w1"], "orange", label="w1")
        axes[2, 1].plot(t, self.control_log["w2"], "purple", label="w2")
        axes[2, 1].axhline(y=self.W1_MAX, c="r", ls="--")
        axes[2, 1].axhline(y=-self.W1_MAX, c="r", ls="--")
        axes[2, 1].legend()
        plt.tight_layout()
        if save_path is None:
            save_path = f"control_plot_{self.episode_start_time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path)
        plt.close()

    def plot_solver_status(self, save_path=None):
        if len(self.control_log["timestamps"]) == 0:
            return
        t = np.array(self.control_log["timestamps"])
        status = np.array(self.control_log["solver_status"])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(t, status, c=["g" if s == 0 else "r" for s in status])
        ax.set_title("Solver Status")
        if save_path is None:
            save_path = "solver_status.png"
        plt.savefig(save_path)
        plt.close()

    def plot_all_diagnostics(self, save_dir=None):
        if save_dir is None:
            save_dir = (
                f"mpc_debug/mpc_diagnostics_{self.episode_start_time.strftime('%Y%m%d_%H%M%S')}"
            )
        os.makedirs(save_dir, exist_ok=True)
        self.save_control_log(os.path.join(save_dir, "control_log.json"))
        self.plot_control_values(save_path=os.path.join(save_dir, "control_values.png"))
        self.plot_solver_status(save_path=os.path.join(save_dir, "solver_status.png"))
        return save_dir
