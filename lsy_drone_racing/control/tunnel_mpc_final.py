"""
    Near Time-Optimal Model Predictive Controller using Spatial Reformulation.

    This controller implements the three-stage trajectory optimization framework proposed by 
    Chan et al. (2025) for high-speed drone navigation in constrained urban environments. 
    Instead of standard Cartesian tracking, this approach reformulates the drone's nonlinear 
    dynamics into a spatial coordinate system, enabling efficient "flight corridor" constraints 
    and real-time obstacle avoidance.

    The control strategy operates in three specific stages:

    1.  Spatial Reformulation (Parallel Transport Frame):
        The drone's state is decoupled into longitudinal progress along a reference path 
        and transverse error coordinates. Unlike the traditional 
        Frenet-Serret frame, this implementation uses the Parallel Transport (Bishop) frame to 
        prevent singularities when the path curvature vanishes or changes abruptly in 3D space
        
    2.  Dynamic Flight Corridor Generation:
        Obstacles are not modeled as complex 3D non-convex shapes in the solver. Instead, they 
        are projected onto the transverse plane of the spatial frame.
        The controller determines the "dominant side" of each obstacle (e.g., whether it restricts 
        the left or right side of the path) and dynamically updates the upper and lower bounds of the transverse states. This creates 
        a safe "tube" or corridor along the path.
        
    3.  Time-Optimal NMPC:
        The problem is solved as a Nonlinear Model Predictive Control (NMPC) problem. By using 
        spatial progress as the independent variable (explicitly or implicitly via reference tracking), 
        the controller maximizes progress along the track. The solver uses a Multiple 
        Shooting method with a Real-Time Iteration (RTI) scheme to achieve high-frequency updates 
        (~20ms) suitable for online replanning.

    Attributes:
        geo (GeometryEngine): Handles the computation of the Parallel Transport frame, 
            curvature, and coordinate transformations.
        mpc (SpatialMPC): The ACADOS/CasADi wrapper solving the NLP form of the OCP.
        obstacles_pos (List[np.ndarray]): Locations of static obstacles used to compute 
            the dominant side and clip the flight corridor bounds.

    References:
        Chan, Y.Y., et al. "Near time-optimal trajectory optimisation for drones in last-mile 
        delivery using spatial reformulation approach." Transportation Research Part C 171 (2025).
        
        
    Author: Niwesh Sah
    Email: sahniwesh@gmail.com
    """

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# Use non-interactive backend for plotting to avoid GUI blocks during simulation
matplotlib.use("Agg")

# --- Import Management ---
try:
    # Simulation Environment Imports
    from drone_models.core import load_params
    from drone_models.utils.rotation import ang_vel2rpy_rates

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.control.GeometryEngines.geometryEngine import GeometryEngine
    from lsy_drone_racing.control.model_dynamics.Spatial_NMPC import SpatialMPC, get_drone_params
    from lsy_drone_racing.control.utils.yaml_import import load_yaml
    from lsy_drone_racing.utils.utils import draw_line

    print(" All modules imported successfully!")

except ImportError as e:
    print(f" Import Warning: {e}")
    print("Running in limited mode. Some simulation-specific features may use mocks.")

    # Mocks for standalone testing or linting environments
    def load_params(*args: Any) -> None: return None
    def ang_vel2rpy_rates(q: Any, w: Any) -> np.ndarray: return np.zeros(3)
    def draw_line(*args: Any, **kwargs: Any) -> None: pass
    
    # Mock base class if not found
    if "Controller" not in locals():
        class Controller:
            pass
            
    # Mock load_yaml if not found, to prevent crash on CONSTANTS
    if "load_yaml" not in locals():
        def load_yaml(path: str) -> Dict: return {}

# Load constants globally to avoid reloading every step
CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")


class SpatialMPCController(Controller):
    """
    Path-following MPC controller operating in spatial coordinates.

    Attributes:
        params (Dict): Drone physical parameters (mass, inertia, etc.).
        env: The simulation environment interface.
        geo (GeometryEngine): Handles path generation and coordinate transforms.
        mpc (SpatialMPC): The interface to the ACADOS/CasADi solver.
    """

    def __init__(self, obs: Dict[str, Any], info: Dict[str, Any], config: Dict[str, Any], env: Optional[Any] = None):
        """
        Initialize the Spatial MPC Controller.

        Args:
            obs: Initial observation dictionary from the environment.
            info: Initial info dictionary containing track data (gates, obstacles).
            config: Configuration dictionary for the environment.
            env: The gym-like environment instance (used for visualization).
        """
        self.env = env
        self.params = get_drone_params()
        
        # --- Control Hyperparameters ---
        self.v_target = CONSTANTS.get("v_max_ref", 20.0)
        self.safety_radius = CONSTANTS.get("safety_radius", 0.3)
        self.max_width_w1 = CONSTANTS.get("max_lateral_width", 2.0)
        self.max_width_w2 = CONSTANTS.get("max_lateral_width", 2.0)
        
        # --- MPC Solver Setup ---
        self.N_horizon = CONSTANTS.get("mpc_horizon", 20)
        self.Tf_horizon = CONSTANTS.get("tf_horizon", 1.0)
        
        # --- Initialization Helpers ---
        self._init_track_and_geometry(obs, info, config)
        self._init_mpc_solver()
        self._init_logging()

        # Visualization helpers
        subsample = 5
        self.global_viz_center = self.geo.pt_frame["pos"][::subsample]
        
        print("[SpatialMPC] Initialization Complete.")

    def _init_track_and_geometry(self, obs: Dict, info: Dict, config: Dict) -> None:
        """Parses configuration to initialize obstacles and the GeometryEngine."""
        
        # 1. Parse Obstacles
        raw_obstacles = config.get("env", {}).get("track", {}).get("obstacles", [])
        if not raw_obstacles and "obstacles" in info:
            raw_obstacles = info["obstacles"]

        self.obstacles_pos: List[np.ndarray] = []
        for o in raw_obstacles:
            if isinstance(o, dict) and "pos" in o:
                self.obstacles_pos.append(np.array(o["pos"]))
            elif isinstance(o, (list, np.ndarray)):
                self.obstacles_pos.append(np.array(o))
            elif isinstance(o, dict):
                self.obstacles_pos.append(np.array(list(o.values())))
        
        print(f"[INIT] Loaded {len(self.obstacles_pos)} obstacles.")

        # 2. Parse Gates
        gates_list = config.get("env", {}).get("track", {}).get("gates", [])
        if not gates_list and "gates" in info:
            gates_list = info["gates"]
            
        gates_pos = [np.array(g["pos"]) for g in gates_list]
        gates_normals = self._get_gate_normals(obs["gates_quat"])
        gates_y, gates_z = self._get_gate_yz(obs["gates_quat"])

        # 3. Initialize Geometry Engine
        starting_pos = obs["pos"]
        print(f"[INIT] Geometry Engine Start Pos: {starting_pos}")

        self.geo = GeometryEngine(
            gates_pos=gates_pos,
            gates_normal=gates_normals,
            gates_y=gates_y,
            gates_z=gates_z,
            obstacles_pos=self.obstacles_pos,
        )

    def _init_mpc_solver(self) -> None:
        """Instantiates and warm-starts the MPC solver."""
        self.mpc = SpatialMPC(self.params, N=self.N_horizon, Tf=self.Tf_horizon)
        self.prev_s = 0.0
        self.reset_mpc_solver()

    def _init_logging(self) -> None:
        """Initializes data structures for telemetry and debugging."""
        self.episode_start_time = datetime.now()
        self.step_count = 0
        self.debug_mode = True
        self.control_log = {
            k: [] for k in [
                "timestamps", "phi_c", "theta_c", "psi_c", "thrust_c",
                "solver_status", "s", "w1", "w2", "ds"
            ]
        }

    # --- Geometry Helpers ---

    def _get_gate_normals(self, gates_quaternions: np.ndarray) -> np.ndarray:
        """Extracts normal vectors (x-axis) from gate quaternions."""
        rotations = R.from_quat(gates_quaternions)
        return rotations.as_matrix()[:, :, 0]

    def _get_gate_yz(self, gates_quaternions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts Y and Z axes from gate quaternions."""
        rotations = R.from_quat(gates_quaternions)
        matrices = rotations.as_matrix()
        return matrices[:, :, 1], matrices[:, :, 2]

    # --- Core Control Logic ---

    def reset_mpc_solver(self) -> None:
        """
        Resets the MPC solver state.
        
        Warm-starts the solver with a naive guess: constant velocity `v_target`
        along the center of the track (s-direction).
        """
        nx = 12  # State dimension
        hover_thrust = self.params["mass"] * self.params["g"]

        for k in range(self.N_horizon + 1):
            x_guess = np.zeros(nx)
            
            # Linear ramp from 0 to v_target for velocity guess
            vel_k = self.v_target * (k / self.N_horizon)
            x_guess[3] = vel_k  # ds
            # Approximate arc length progression
            x_guess[0] = vel_k * k * (self.mpc.Tf / self.N_horizon) * 0.5

            self.mpc.solver.set(k, "x", x_guess)
            
            if k < self.N_horizon:
                self.mpc.solver.set(k, "u", np.array([0, 0, 0, hover_thrust]))

        self.prev_s = 0.0

    def compute_control(self, obs: Dict[str, Any], info: Optional[Dict] = None) -> np.ndarray:
        """
        Main control loop method.

        1. Updates visualizations.
        2. Converts Cartesian observation to Spatial coordinates (s, w1, w2).
        3. Updates MPC constraints (corridor bounds) based on current path prediction.
        4. Solves the OCP (Optimal Control Problem).
        5. Logs data and returns control inputs (Roll, Pitch, Yaw_rate, Thrust).
        """
        # Visualization
        self._draw_global_track()
        self._draw_static_corridor()

        # Process State
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])

        # Convert to Spatial State [s, w1, w2, ds, dw1, dw2, euler, euler_rates]
        x_spatial = self._cartesian_to_spatial(
            obs["pos"], obs["vel"], obs["rpy"], obs["drpy"]
        )
        
        # Set Initial State Constraint
        self.mpc.solver.set(0, "lbx", x_spatial)
        self.mpc.solver.set(0, "ubx", x_spatial)

        # Prepare Horizon
        curr_s = x_spatial[0]
        curr_ds = x_spatial[3]
        dt = self.mpc.Tf / self.mpc.N
        
        hover_thrust = self.params["mass"] * -self.params["g"]
        max_lat_acc = CONSTANTS.get("corner_acc", 5.0)
        epsilon = 0.01

        running_s_ref = curr_s

        # Iterate over prediction horizon to set constraints and reference parameters
        for k in range(self.mpc.N):
            # Predict s position for this step (simple integration)
            s_pred = curr_s + k * max(curr_ds, 1.0) * dt

            # 1. Get Corridor Bounds at this predicted s
            lb_w1, ub_w1 = self.geo.get_static_bounds(s_pred)
            # Default height bounds (can be dynamic if needed)
            lb_w2, ub_w2 = -self.max_width_w2, self.max_width_w2

            # 2. Update Constraints (k > 0 only, k=0 is fixed to current state)
            if k > 0:
                # State bounds: [w1, w2, euler_angles...]
                # Relaxing orientation bounds to +/- 0.5 rad (~28 deg)
                lbx = np.array([lb_w1, lb_w2, -0.5, -0.5, -0.5])
                ubx = np.array([ub_w1, ub_w2,  0.5,  0.5,  0.5])
                self.mpc.solver.set(k, "lbx", lbx)
                self.mpc.solver.set(k, "ubx", ubx)

            # 3. Update Reference (Parameters & Setpoints)
            frenet_frame = self.geo.get_frame(s_pred)
            
            # Curvature-based velocity profiling
            k_mag = np.sqrt(frenet_frame["k1"] ** 2 + frenet_frame["k2"] ** 2)
            v_corner = np.sqrt(max_lat_acc / (k_mag + epsilon))
            v_ref_k = min(v_corner, self.v_target)
            
            running_s_ref += v_ref_k * dt

            # Parameter vector 'p': [t_vec, n1_vec, n2_vec, k1, k2, dk1, dk2]
            p_k = np.concatenate([
                frenet_frame["t"], frenet_frame["n1"], frenet_frame["n2"],
                [frenet_frame["k1"]], [frenet_frame["k2"]], 
                [frenet_frame["dk1"]], [frenet_frame["dk2"]]
            ])
            self.mpc.solver.set(k, "p", p_k)

            # Reference vector 'yref': [s, w1, w2, ds, dw1, dw2, r, p, y, dr, dp, dy, u1, u2, u3, u4]
            # We target center of path (w1=0, w2=0) and calculated velocity
            y_ref = np.zeros(16)
            y_ref[0] = running_s_ref
            y_ref[3] = v_ref_k
            y_ref[15] = hover_thrust  # Feedforward thrust
            self.mpc.solver.set(k, "yref", y_ref)

        # --- Terminal Node Setup (k = N) ---
        s_end = running_s_ref + v_ref_k * dt
        f_end = self.geo.get_frame(s_end)
        p_end = np.concatenate([
            f_end["t"], f_end["n1"], f_end["n2"],
            [f_end["k1"]], [f_end["k2"]], 
            [f_end["dk1"]], [f_end["dk2"]]
        ])
        self.mpc.solver.set(self.mpc.N, "p", p_end)

        lb_w1_e, ub_w1_e = self.geo.get_static_bounds(s_end)
        lbx_e = np.array([lb_w1_e, -self.max_width_w2, -0.5, -0.5, -0.5])
        ubx_e = np.array([ub_w1_e,  self.max_width_w2,  0.5,  0.5,  0.5])
        self.mpc.solver.set(self.mpc.N, "lbx", lbx_e)
        self.mpc.solver.set(self.mpc.N, "ubx", ubx_e)

        yref_e = np.zeros(12) # Terminal state only, no controls
        yref_e[0] = s_end
        yref_e[3] = v_ref_k
        yref_e[11] = hover_thrust # Use as proxy for vertical velocity ref if needed, or 0
        self.mpc.solver.set(self.mpc.N, "yref", yref_e)

        # --- Solve ---
        status = self.mpc.solver.solve()

        if status != 0:
            print(f"[MPC] Solver failed with status {status}. Executing hover failsafe.")
            u_opt = np.array([0.0, 0.0, 0.0, hover_thrust])
        else:
            u_opt = self.mpc.solver.get(0, "u")

        # Logging
        self._log_control_step(x_spatial, u_opt, status)
        
        # Return control action: [roll, pitch, yaw_rate, thrust]
        return np.array([u_opt[0], u_opt[1], u_opt[2], u_opt[3]])

    def _compute_corridor_bounds(
        self, s_pred: float, frame_pos: np.ndarray, frame_t: np.ndarray, 
        frame_n1: np.ndarray, frame_n2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes dynamic [lb, ub] for w1 and w2 at a specific path location s.
        Projects obstacles onto the transverse plane to restrict the flight corridor.

        Args:
            s_pred: The predicted arc length.
            frame_pos: Position of the path at s_pred.
            frame_t: Tangent vector at s_pred.
            frame_n1: Normal vector (w1 direction) at s_pred.
            frame_n2: Binormal vector (w2 direction) at s_pred.

        Returns:
            Tuple containing lower_bounds (np.array([lb_w1, lb_w2])) and 
            upper_bounds (np.array([ub_w1, ub_w2])).
        """
        # 1. Initialize with full corridor width
        lb_w1, ub_w1 = -self.max_width_w1, self.max_width_w1
        lb_w2, ub_w2 = -self.max_width_w2, self.max_width_w2

        # Sensitivity: How far along s (longitudinal) do we care about an obstacle?
        longitudinal_threshold = 0.3
        
        # Combine obstacles and gates for collision checking if geometry requires it
        # (Assuming gates might have posts to avoid)
        all_hazards = self.obstacles_pos  # Add self.gates_pos if gates are solid constraints

        for hazard_pos in all_hazards:
            # Vector from Path Center -> Hazard
            r_vec = hazard_pos - frame_pos

            # Project onto Tangent (s-direction)
            s_dist = np.dot(r_vec, frame_t)

            # Only constrain if the obstacle is "at this slice" of s
            if abs(s_dist) < longitudinal_threshold:
                # Project onto Transverse Plane (n1, n2)
                w1_obs = np.dot(r_vec, frame_n1)
                w2_obs = np.dot(r_vec, frame_n2)

                # Check if obstacle is actually inside our max corridor
                if (lb_w1 < w1_obs < ub_w1) and (lb_w2 < w2_obs < ub_w2):
                    
                    # --- Dominant Side Logic ---
                    # If hazard is to the LEFT (w1 > 0), pass RIGHT (reduce Upper Bound).
                    if w1_obs > 0:
                        dist_to_surface = w1_obs - self.safety_radius
                        ub_w1 = min(ub_w1, dist_to_surface)
                    
                    # If hazard is to the RIGHT (w1 <= 0), pass LEFT (increase Lower Bound).
                    else:
                        dist_to_surface = w1_obs + self.safety_radius
                        lb_w1 = max(lb_w1, dist_to_surface)

        # Safety Check: Ensure bounds didn't cross (lb < ub).
        # If they cross, the corridor is blocked. We force a tiny gap to prevent solver NaNs.
        if lb_w1 >= ub_w1:
            mid = (lb_w1 + ub_w1) / 2
            lb_w1 = mid - 0.05
            ub_w1 = mid + 0.05

        return np.array([lb_w1, lb_w2]), np.array([ub_w1, ub_w2])

    def _cartesian_to_spatial(self, pos, vel, rpy, drpy) -> np.ndarray:
        """Converts Cartesian state to Spatial (Frenet) state."""
        # Find closest s on the path
        s = self.geo.get_closest_s(pos, s_guess=self.prev_s)
        self.prev_s = s
        
        # Get Frame at s
        f = self.geo.get_frame(s)
        
        # Position Errors (w1, w2)
        r_vec = pos - f["pos"]
        w1 = np.dot(r_vec, f["n1"])
        w2 = np.dot(r_vec, f["n2"])
        
        # Velocity Projection
        # h is the scale factor for arc length in curvilinear coords
        h = max(1 - f["k1"] * w1 - f["k2"] * w2, 0.01)
        
        ds = np.dot(vel, f["t"]) / h
        dw1 = np.dot(vel, f["n1"])
        dw2 = np.dot(vel, f["n2"])
        
        return np.array([
            s, w1, w2, ds, dw1, dw2, 
            rpy[0], rpy[1], rpy[2], 
            drpy[0], drpy[1], drpy[2]
        ])

    def _spatial_to_cartesian(self, s: float, w1: float, w2: float) -> np.ndarray:
        """Helper to convert spatial coords back to world frame (for debug viz)."""
        f = self.geo.get_frame(s)
        return f["pos"] + w1 * f["n1"] + w2 * f["n2"]

    # --- Reset & Callbacks ---

    def reset(self) -> None:
        """Reset internal controller state."""
        self.prev_s = 0.0
        self.reset_mpc_solver()

    def episode_reset(self) -> None:
        """Called by the environment when an episode restarts."""
        self.reset()

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        """Standard Gym step callback."""
        return False

    def episode_callback(self) -> None:
        """Called at the end of an episode to save logs."""
        if len(self.control_log["timestamps"]) > 0:
            print("[SpatialMPC] Saving diagnostics...")
            self.plot_all_diagnostics()

    # --- Visualization & Debugging ---

    def _draw_static_corridor(self) -> None:
        """Draws the static flight corridor boundaries in the simulation."""
        if self.env is None:
            return

        try:
            step = 10
            positions = self.geo.pt_frame["pos"][::step]
            n1_vecs = self.geo.pt_frame["n1"][::step]

            full_indices = np.arange(0, len(self.geo.pt_frame["s"]), step)
            lb_w1 = self.geo.corridor_map["lb_w1"][full_indices]
            ub_w1 = self.geo.corridor_map["ub_w1"][full_indices]

            left_bound_pts = positions + (n1_vecs * ub_w1[:, np.newaxis])
            right_bound_pts = positions + (n1_vecs * lb_w1[:, np.newaxis])

            # Orange lines for boundaries
            color = np.array([1.0, 0.65, 0.0, 0.5])
            draw_line(self.env, points=left_bound_pts, rgba=color)
            draw_line(self.env, points=right_bound_pts, rgba=color)
        except Exception:
            # Suppress visualization errors to prevent crashing the simulation
            pass

    def _draw_global_track(self) -> None:
        """Draws the reference path center line."""
        if self.env is None:
            return
        try:
            draw_line(self.env, points=self.global_viz_center, rgba=np.array([0.0, 1.0, 0.0, 0.5]))
        except Exception:
            pass

    def _log_control_step(self, x_spatial: np.ndarray, u_opt: np.ndarray, status: int) -> None:
        """Appends current step data to the log buffers."""
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

    def plot_all_diagnostics(self, save_dir: Optional[str] = None) -> str:
        """Generates and saves plots for control inputs and solver health."""
        if save_dir is None:
            timestamp = self.episode_start_time.strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join("mpc_debug", f"mpc_diagnostics_{timestamp}")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Save JSON Log
        log_path = os.path.join(save_dir, "control_log.json")
        with open(log_path, "w") as f:
            json.dump(self.control_log, f, indent=2)

        # Plot Controls
        self._plot_control_values(save_path=os.path.join(save_dir, "control_values.png"))
        
        return save_dir

    def _plot_control_values(self, figsize=(16, 10), save_path: str = "controls.png") -> None:
        """Internal helper to plot control values using Matplotlib."""
        if not self.control_log["timestamps"]:
            return
            
        t = np.array(self.control_log["timestamps"])
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle("MPC Control & State Log", fontsize=16)

        # Plot Data
        axes[0, 0].plot(t, self.control_log["phi_c"], "b")
        axes[0, 0].set_ylabel("Roll Command")
        
        axes[0, 1].plot(t, self.control_log["theta_c"], "g")
        axes[0, 1].set_ylabel("Pitch Command")
        
        axes[1, 0].plot(t, self.control_log["thrust_c"], "r")
        axes[1, 0].set_ylabel("Thrust Command")
        
        axes[1, 1].plot(t, self.control_log["psi_c"], "m")
        axes[1, 1].set_ylabel("Yaw Rate Command")
        
        axes[2, 0].plot(t, self.control_log["s"], "c")
        axes[2, 0].set_ylabel("Arc Length (s)")
        
        axes[2, 1].plot(t, self.control_log["w1"], "orange", label="Lateral Error (w1)")
        axes[2, 1].plot(t, self.control_log["w2"], "purple", label="Vertical Error (w2)")
        axes[2, 1].axhline(y=self.max_width_w1, c="r", ls="--", alpha=0.5)
        axes[2, 1].axhline(y=-self.max_width_w1, c="r", ls="--", alpha=0.5)
        axes[2, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()