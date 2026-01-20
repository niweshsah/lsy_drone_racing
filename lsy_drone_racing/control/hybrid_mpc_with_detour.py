"""
Hybrid Drone Racing Controller: Dynamic Spline Planning with NMPC Tracking.

This module implements a two-layer control architecture for autonomous drone racing:

1. High-Level Trajectory Planner:
   - Dynamically generates a smooth 3D trajectory using Cubic Splines.
   - Computes gate-aligned approach waypoints and handles sharp turns via detour logic.
   - Implements a bisector-based obstacle avoidance algorithm to modify path geometry 
     in real-time based on environmental feedback.

2. Low-Level NMPC Tracker:
   - Utilizes the Acados framework to solve a Nonlinear Model Predictive Control 
     problem over a sliding time horizon.
   - Synchronizes spline sampling with the MPC shooting nodes to minimize tracking 
     error across position, velocity, and orientation.
   - Respects physical drone constraints (thrust limits, tilt angles) while 
     maintaining high-frequency control throughput.

The architecture ensures reactive replanning when obstacles are detected or gates 
are cleared, providing a robust balance between aggressive racing and safety.

Author: Niwesh Sah
Email: sahniwesh@gmail.com
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import scipy.linalg

# Acados Imports
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# Domain Specific Imports
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PlannerConfig:
    """Configuration parameters for the Trajectory Planner."""
    flight_duration: float = 9.0
    replan_radius: float = 1.0
    obstacle_clearance: float = 0.3
    gate_approach_dist: float = 0.5
    gate_approach_points: int = 5
    detour_angle_threshold: float = 120.0
    detour_radius: float = 0.65


@dataclass
class MpcConfig:
    """Configuration parameters for the NMPC Controller."""
    T_horizon: float = 2.0  # Time horizon in seconds
    N_horizon: int = 50     # Number of shooting nodes
    
    # Weights
    Q_pos: float = 100.0
    Q_pos_z: float = 400.0
    Q_angle: float = 1.0
    Q_vel: float = 10.0
    Q_rate: float = 5.0
    R_angle: float = 1.0
    R_thrust: float = 50.0

    # Constraints
    max_angle: float = 0.5  # rad
    max_thrust_factor: float = 4.0

    # Solver
    qp_solver: str = "FULL_CONDENSING_HPIPM"
    hessian_approx: str = "GAUSS_NEWTON"
    nlp_tol: float = 1e-6
    max_qp_iter: int = 20
    max_nlp_iter: int = 50
    generated_code_dir: str = "c_generated_code"


# ==============================================================================
# TRAJECTORY PLANNER
# ==============================================================================

class TrajectoryPlanner:
    """Handles high-level path planning, gate sequencing, and obstacle avoidance.
    
    This class generates a smooth CubicSpline trajectory that navigates through
    gates while avoiding static obstacles.
    """

    def __init__(self, config: PlannerConfig, control_freq: float):
        self.cfg = config
        self.control_freq = control_freq
        
        # State
        self.spline: Optional[CubicSpline] = None
        self._last_gate_flags: Optional[NDArray[np.bool_]] = None
        self._last_obs_flags: Optional[NDArray[np.bool_]] = None

    def plan(self, obs: dict, start_pos: NDArray[np.float64]) -> None:
        """Generates or regenerates the trajectory spline based on current observations."""
        gate_pos = obs["gates_pos"]
        obstacle_pos = obs["obstacles_pos"]
        
        # Extract gate geometry
        gate_normals, gate_y, gate_z = self._extract_gate_frames(obs["gates_quat"])

        # 1. Generate base waypoints through gates
        path_points = self._generate_gate_approach(start_pos, gate_pos, gate_normals)

        # 2. Add detours for sharp turns
        path_points = self._add_detours(path_points, gate_pos, gate_normals, gate_y, gate_z)

        # 3. Handle Obstacles (modifies timing and points)
        time_knots, path_points = self._avoid_obstacles(path_points, obstacle_pos)

        # 4. Generate Spline
        self.spline = self._compute_spline(self.cfg.flight_duration, path_points, time_knots)


    def get_reference(self, time: float | NDArray[np.float64], nu: int = 0) -> NDArray[np.float64]:
        """Samples the spline at specific time(s).
        
        Args:
            time: Time or array of times to sample.
            nu: Derivative order (0=pos, 1=vel, 2=acc).
        """
        if self.spline is None:
            raise RuntimeError("Trajectory has not been planned yet.")
        
        # Pass 'nu' directly to the underlying CubicSpline
        return self.spline(time, nu=nu)

    def check_replan_trigger(self, obs: dict) -> bool:
        """Determines if the environment state has changed enough to warrant replanning."""
        # 1. Check State Transitions (Gate/Obstacle hits)
        curr_gate_flags = np.array(obs["gates_visited"], dtype=bool)
        curr_obs_flags = np.array(obs["obstacles_visited"], dtype=bool)

        if self._last_gate_flags is None:
            self._last_gate_flags = curr_gate_flags
            self._last_obs_flags = curr_obs_flags
            return False

        gate_change = np.any((~self._last_gate_flags) & curr_gate_flags)
        obs_change = np.any((~self._last_obs_flags) & curr_obs_flags)

        self._last_gate_flags = curr_gate_flags
        self._last_obs_flags = curr_obs_flags

        # 2. Check Proximity (Replanning Window Integrity)
        drone_pos = obs["pos"]
        gate_dists = np.linalg.norm(obs["gates_pos"] - drone_pos, axis=1)
        obs_dists = np.linalg.norm(obs["obstacles_pos"][:, :2] - drone_pos[:2], axis=1)

        proximity_alert = np.any(gate_dists < self.cfg.replan_radius) or \
                          np.any(obs_dists < self.cfg.replan_radius)

        return gate_change or obs_change or proximity_alert

    # --- Private Geometric Helpers ---

    def _extract_gate_frames(self, quats: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        rot_matrices = R.from_quat(quats).as_matrix()
        return rot_matrices[:, :, 0], rot_matrices[:, :, 1], rot_matrices[:, :, 2]

    def _generate_gate_approach(self, start: NDArray, gates: NDArray, normals: NDArray) -> NDArray:
        """Generates linear approach points aligned with gate normals."""
        offsets = np.linspace(-self.cfg.gate_approach_dist, self.cfg.gate_approach_dist, self.cfg.gate_approach_points)
        # Broadcasting to create waypoints: Gate + Offset * Normal
        waypoints = gates[:, None, :] + offsets[None, :, None] * normals[:, None, :]
        return np.vstack([start, waypoints.reshape(-1, 3)])

    def _add_detours(self, points: NDArray, g_pos: NDArray, g_norm: NDArray, 
                     g_y: NDArray, g_z: NDArray) -> NDArray:
        """Injects intermediate waypoints if the turn angle into a gate is too sharp."""
        pts_list = list(points)
        inserts = 0
        num_pts = self.cfg.gate_approach_points
        n_gates = g_pos.shape[0]

        for i in range(n_gates - 1):
            # Calculate indices based on fixed approach points per gate
            idx_prev_gate_end = 1 + (i + 1) * num_pts - 1 + inserts
            idx_next_gate_start = 1 + (i + 1) * num_pts + inserts

            p1, p2 = pts_list[idx_prev_gate_end], pts_list[idx_next_gate_start]
            vec = p2 - p1
            norm_vec = np.linalg.norm(vec)

            if norm_vec < 1e-6:
                continue

            # Check angle
            cos_a = np.dot(vec, g_norm[i]) / norm_vec
            angle_deg = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

            if angle_deg > self.cfg.detour_angle_threshold:
                # Calculate detour direction
                v_proj = vec - np.dot(vec, g_norm[i]) * g_norm[i]
                detour_dir = self._get_detour_direction(v_proj, g_y[i], g_z[i])
                
                detour_pt = g_pos[i] + self.cfg.detour_radius * detour_dir
                pts_list.insert(idx_prev_gate_end + 1, detour_pt)
                inserts += 1

        return np.array(pts_list)

    def _get_detour_direction(self, v_proj: NDArray, y_axis: NDArray, z_axis: NDArray) -> NDArray:
        norm = np.linalg.norm(v_proj)
        if norm < 1e-6: 
            return y_axis

        angle = np.degrees(np.arctan2(np.dot(v_proj, z_axis), np.dot(v_proj, y_axis)))
        
        if -90 <= angle < 45:
            return y_axis     # Right
        if 45 <= angle < 135:
            return z_axis     # Top
        return -y_axis                          # Left

    def _avoid_obstacles(self, path_points: NDArray, obstacles: NDArray) -> Tuple[NDArray, NDArray]:
        """Modifies the path to skirt around obstacles using a bisector method."""
        # Create temporary spline to sample dense points
        temp_spline = self._compute_spline(self.cfg.flight_duration, path_points)
        n_samples = int(self.control_freq * self.cfg.flight_duration)
        times = np.linspace(0, self.cfg.flight_duration, n_samples)
        points = temp_spline(times)

        for obs_center in obstacles:
            times, points = self._process_single_obstacle(obs_center, points, times)
            
        return times, points

    def _process_single_obstacle(self, obs_center: NDArray, points: NDArray, times: NDArray) -> Tuple[NDArray, NDArray]:
        """Filters points through a single obstacle constraint."""
        safe_times, safe_points = [], []
        inside_obstacle = False
        entry_idx = -1
        obs_xy = obs_center[:2]
        
        for i, pt in enumerate(points):
            dist = np.linalg.norm(obs_xy - pt[:2])

            if dist < self.cfg.obstacle_clearance:
                if not inside_obstacle:
                    inside_obstacle = True
                    entry_idx = i
            elif inside_obstacle:
                # Just exited
                inside_obstacle = False
                entry_pt, exit_pt = points[entry_idx], points[i]
                
                # Calculate detour point (bisector)
                vec_entry = entry_pt[:2] - obs_xy
                vec_exit = exit_pt[:2] - obs_xy
                avoid_vec = vec_entry + vec_exit
                
                norm = np.linalg.norm(avoid_vec)
                avoid_vec = (avoid_vec / norm) if norm > 0 else np.array([1.0, 0.0])
                
                new_xy = obs_xy + avoid_vec * self.cfg.obstacle_clearance
                new_z = (entry_pt[2] + exit_pt[2]) / 2.0
                
                safe_points.append(np.array([*new_xy, new_z]))
                safe_times.append((times[entry_idx] + times[i]) / 2.0)
            else:
                safe_times.append(times[i])
                safe_points.append(pt)

        return np.array(safe_times), np.array(safe_points)

    def _compute_spline(self, duration: float, points: NDArray, knots: Optional[NDArray] = None) -> CubicSpline:
        if knots is None:
            # Heuristic: Distribute knots based on segment length
            dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cum_dist = np.concatenate([[0], np.cumsum(dists)])
            knots = (cum_dist / cum_dist[-1]) * duration
        return CubicSpline(knots, points)


# ==============================================================================
# NMPC TRACKER (ACADOS WRAPPER)
# ==============================================================================

class NMPCTracker:
    """Wrapper for Acados NMPC solver generation and execution."""

    def __init__(self, mpc_config: MpcConfig, drone_params: dict):
        self.cfg = mpc_config
        self.params = drone_params
        
        # Dimensions
        self.model = self._create_model()
        self.nx = self.model.x.rows()
        self.nu = self.model.u.rows()
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        # Setup Solver
        self.solver, self.ocp = self._create_solver()
        
    def solve(self, x0: NDArray, references: NDArray) -> NDArray:
        """Solves the OCP for the current state and reference horizon."""
        # Set initial condition
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Set references over horizon (0 to N-1)
        for k in range(self.cfg.N_horizon):
            self.solver.set(k, "yref", references[k])
        
        # Set terminal reference (N)
        # FIX: Slice the reference to only include state variables (:self.nx)
        # Acados expects size 12 here, not 16.
        terminal_ref = references[self.cfg.N_horizon, :self.nx]
        self.solver.set(self.cfg.N_horizon, "y_ref", terminal_ref)

        # Solve
        status = self.solver.solve()
        if status != 0:
            pass 

        return self.solver.get(0, "u")

    def reset(self) -> None:
        self.solver.reset()

    # --- Acados Setup Helpers ---

    def _create_model(self) -> AcadosModel:
        X_dot, X, U, _ = symbolic_dynamics_euler(
            mass=self.params["mass"],
            gravity_vec=self.params["gravity_vec"],
            J=self.params["J"],
            J_inv=self.params["J_inv"],
            acc_coef=self.params["acc_coef"],
            cmd_f_coef=self.params["cmd_f_coef"],
            rpy_coef=self.params["rpy_coef"],
            rpy_rates_coef=self.params["rpy_rates_coef"],
            cmd_rpy_coef=self.params["cmd_rpy_coef"],
        )
        model = AcadosModel()
        model.name = "mpc_spline_tracker"
        model.f_expl_expr = X_dot
        model.x = X
        model.u = U
        return model

    def _create_solver(self) -> Tuple[AcadosOcpSolver, AcadosOcp]:
        ocp = AcadosOcp()
        ocp.model = self.model
        
        # Horizon
        ocp.solver_options.N_horizon = self.cfg.N_horizon
        ocp.solver_options.tf = self.cfg.T_horizon

        # Costs (Linear Least Squares)
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        
        # Weight Matrices
        Q = np.diag([
            self.cfg.Q_pos, self.cfg.Q_pos, self.cfg.Q_pos_z,   # Pos
            self.cfg.Q_angle, self.cfg.Q_angle, self.cfg.Q_angle, # RPY
            self.cfg.Q_vel, self.cfg.Q_vel, self.cfg.Q_vel,     # Vel
            self.cfg.Q_rate, self.cfg.Q_rate, self.cfg.Q_rate     # Rates
        ])
        R_mat = np.diag([
            self.cfg.R_angle, self.cfg.R_angle, self.cfg.R_angle, # Cmd RPY
            self.cfg.R_thrust                                     # Cmd Thrust
        ])
        
        ocp.cost.W = scipy.linalg.block_diag(Q, R_mat)
        ocp.cost.W_e = Q

        # Mappings (Vx * x + Vu * u = y)
        ocp.cost.Vx = np.zeros((self.ny, self.nx))
        ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        ocp.cost.Vu = np.zeros((self.ny, self.nu))
        ocp.cost.Vu[self.nx:, :] = np.eye(self.nu)
        ocp.cost.Vx_e = np.eye(self.nx)

        # Initial references (placeholders)
        ocp.cost.yref = np.zeros((self.ny,))
        ocp.cost.yref_e = np.zeros((self.ny_e,))

        # Constraints
        ocp.constraints.idxbx = np.array([3, 4, 5]) # RPY indices
        ocp.constraints.lbx = np.array([-self.cfg.max_angle] * 3)
        ocp.constraints.ubx = np.array([self.cfg.max_angle] * 3)

        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, self.params["thrust_min"] * 4])
        ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, self.params["thrust_max"] * 4])
        ocp.constraints.x0 = np.zeros(self.nx)

        # Solver Options
        ocp.solver_options.qp_solver = self.cfg.qp_solver
        ocp.solver_options.hessian_approx = self.cfg.hessian_approx
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.tol = self.cfg.nlp_tol
        ocp.solver_options.qp_solver_cond_N = self.cfg.N_horizon
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = self.cfg.max_qp_iter
        ocp.solver_options.nlp_solver_max_iter = self.cfg.max_nlp_iter

        # Helper to avoid recompilation path issues
        json_file = os.path.join(self.cfg.generated_code_dir, f"{self.model.name}.json")
        
        solver = AcadosOcpSolver(
            ocp, json_file=json_file, verbose=False, build=True, generate=True
        )
        return solver, ocp


# ==============================================================================
# MAIN CONTROLLER
# ==============================================================================

class MPCSplineController(Controller):
    """Main controller class interfacing with the Gym Environment.
    
    Orchestrates the High-Level Trajectory Planner and the Low-Level NMPC Tracker.
    """

    def __init__(self, initial_obs: dict, info: dict, sim_config: dict, env=None):
        super().__init__(initial_obs, info, sim_config)
        self.env = env
        
        # 1. Setup Configurations
        self.dt = 1.0 / sim_config.env.freq
        self.drone_params = load_params("so_rpy", sim_config.sim.drone_model)
        
        self.planner_cfg = PlannerConfig(flight_duration=10.0) # Adjust as needed
        self.mpc_cfg = MpcConfig(
            T_horizon=2.0,
            N_horizon=40 # Increase resolution
        )
        self.mpc_cfg.T_horizon = self.mpc_cfg.N_horizon * self.dt # Sync horizon with dt

        # 2. Initialize Modules
        self.planner = TrajectoryPlanner(self.planner_cfg, sim_config.env.freq)
        self.tracker = NMPCTracker(self.mpc_cfg, self.drone_params)

        # 3. Initial Plan
        self.planner.plan(initial_obs, initial_obs["pos"])

        # 4. Runtime State
        self.current_step = 0
        self.is_finished = False

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray[np.float64]:
        """Main control loop."""
        current_time = min(self.current_step * self.dt, self.planner_cfg.flight_duration)
        
        # Check termination condition
        if current_time >= self.planner_cfg.flight_duration:
            self.is_finished = True

        # --- 1. Replan if environment changes ---
        if self.planner.check_replan_trigger(obs):
            self.planner.plan(obs, obs["pos"])

        # --- 2. Prepare State for MPC ---
        # Convert quat to Euler for this specific MPC formulation
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        
        # State vector: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        # --- 3. Generate Reference Trajectory (Horizon) ---
        horizon_times = np.linspace(
            current_time, 
            current_time + self.mpc_cfg.T_horizon, 
            self.mpc_cfg.N_horizon + 1
        )
        # Clip to ensure we don't query spline past flight duration
        horizon_times = np.clip(horizon_times, 0, self.planner_cfg.flight_duration)

        ref_pos = self.planner.get_reference(horizon_times)       # (N+1, 3)
        ref_vel = self.planner.get_reference(horizon_times, nu=1) # (N+1, 3)
        
        # Construct full reference matrix [N+1, ny]
        # Although Nth element only needs ny_e, we build uniform array for easy slicing
        references = np.zeros((self.mpc_cfg.N_horizon + 1, self.tracker.ny))
        
        references[:, 0:3] = ref_pos
        # Orientation ref (0,0,0) - could be improved to align with velocity
        references[:, 3:6] = 0.0 
        references[:, 6:9] = ref_vel
        # Rate ref (0,0,0)
        references[:, 9:12] = 0.0
        # Input ref (Hover thrust)
        hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        references[:, 15] = hover_thrust

        # --- 4. Solve MPC ---
        u_opt = self.tracker.solve(x0, references)

        # --- 5. Visual Debugging ---
        self._visualize_trajectory()

        return u_opt

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self.current_step += 1
        return self.is_finished

    def episode_callback(self):
        self.current_step = 0
        self.tracker.reset()
        self.is_finished = False

    def _visualize_trajectory(self):
        """Draws the planned spline in the environment if supported."""
        if hasattr(self, "env") and self.env is not None and self.planner.spline:
            try:
                # Sample entire spline for visualization
                ts = np.linspace(0, self.planner_cfg.flight_duration, 100)
                pts = self.planner.get_reference(ts)
                draw_line(self.env, pts, rgba=np.array([1.0, 1.0, 1.0, 0.2]))
            except (AttributeError, TypeError):
                pass