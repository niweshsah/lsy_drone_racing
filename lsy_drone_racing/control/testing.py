"""
Merged Controller: MPC implementation following a Dynamic Spline Trajectory.
Combines Level 3 replanning/obstacle avoidance with Acados NMPC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import scipy.linalg
import casadi as cs
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

# Acados Imports
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# Drone Environment Imports
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from lsy_drone_racing.control import Controller
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# ACADOS MODEL & SOLVER GENERATION
# ==============================================================================

def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model with obstacle constraints."""
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    model = AcadosModel()
    model.name = "mpc_spline_tracker"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    # --- ADDED: Obstacle Avoidance Constraints ---
    # Define parameters: [obs_x, obs_y, min_safe_dist_squared]
    # We use a single 'closest obstacle' for the horizon to keep it fast
    p = cs.SX.sym('p', 3) 
    model.p = p

    # Constraint Expression: distance_sq >= min_safe_dist_squared
    # h(x) = (x - obs_x)^2 + (y - obs_y)^2 - min_safe_dist_squared >= 0
    # We only care about XY distance (Cylindrical obstacle)
    dist_sq = (X[0] - p[0])**2 + (X[1] - p[1])**2
    model.con_h_expr = dist_sq - p[2]

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    # Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    # Cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    Q = np.diag([
        50.0, 50.0, 400.0,  # Position (x, y, z)
        1.0, 1.0, 1.0,      # Orientation (roll, pitch, yaw)
        10.0, 10.0, 10.0,   # Velocity (vx, vy, vz)
        5.0, 5.0, 5.0       # Angular Rates (p, q, r)
    ])
    
    # Input weights
    R_mat = np.diag([
        1.0, 1.0, 1.0,  # Cmd Orientation
        50.0            # Cmd Thrust
    ])

    ocp.cost.W = scipy.linalg.block_diag(Q, R_mat)
    ocp.cost.W_e = Q

    # Output selection matrices
    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    # Initial references (will be updated online)
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # --- Constraints ---
    # 1. Box Constraints (State/Input)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5]) # Max roll/pitch/yaw
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = np.zeros((nx))

    # 2. Nonlinear Constraints (Obstacle Avoidance)
    # We constrained con_h_expr >= 0
    # So lower bound is 0, upper bound is infinity (large number)
    ocp.constraints.lh = np.array([0.0])
    ocp.constraints.uh = np.array([1e9])
    
    # Slack variables for soft constraints (prevents solver crash if too close)
    # Penalize violations of the obstacle constraint
    ocp.constraints.lsh = np.zeros(1)
    ocp.constraints.ush = np.zeros(1)
    ocp.constraints.idxsh = np.array([0])
    
    # Weights for slack variables (High weight = behave like hard constraint)
    ocp.cost.Zl = np.array([1000.0])
    ocp.cost.Zu = np.array([1000.0])
    ocp.cost.zl = np.array([1000.0])
    ocp.cost.zu = np.array([1000.0])

    # Initial parameter values (far away obstacle)
    ocp.parameter_values = np.array([100.0, 100.0, 0.1])

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tf = Tf

    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mpc_spline_tracker.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return solver, ocp


# ==============================================================================
# MERGED CONTROLLER CLASS
# ==============================================================================

class MPCSplineController(Controller):
    """
    Controller that combines High-Level Spline (GATES ONLY)
    with Low-Level MPC tracking (OBSTACLE DEFLECTION).
    """
    
    # Planner Constants
    FLIGHT_DURATION = 25.0
    REPLAN_RADIUS = 0.5
    
    # MPC Constants
    MPC_HORIZON_STEPS = 25  # N
    OBSTACLE_RADIUS_REAL = 0.3 # Approximate radius of obstacle
    DRONE_RADIUS = 0.15
    SAFETY_MARGIN = 0.2

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], info: dict, sim_config: dict):
        super().__init__(initial_obs, info, sim_config)

        # ---------------------------------------------------------
        # 1. Initialize Planner State (From MyController)
        # ---------------------------------------------------------
        self.__initialize_planner_state(initial_obs, sim_config)
        self.__plan_initial_trajectory(initial_obs)

        # ---------------------------------------------------------
        # 2. Initialize MPC Solver
        # ---------------------------------------------------------
        self._dt = 1 / sim_config.env.freq
        self._T_HORIZON = self.MPC_HORIZON_STEPS * self._dt

        self.drone_params = load_params("so_rpy", sim_config.sim.drone_model)
        
        # Create Solver
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self.MPC_HORIZON_STEPS, self.drone_params
        )

        # Cache dimensions
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        
        self._finished = False

    # --------------------------------------------------------------------------
    # PLANNER LOGIC (Pure Gates)
    # --------------------------------------------------------------------------

    def __initialize_planner_state(self, initial_obs, sim_config):
        self.__current_step = 0
        self.__control_freq = sim_config.env.freq
        
        # Replan flags
        self.__last_gate_flags = None
        self.__last_obstacle_flags = None

        # Environment Geometry
        self.__gate_positions = initial_obs["gates_pos"]
        self.__obstacle_positions = initial_obs["obstacles_pos"]
        self.__start_position = initial_obs["pos"]
        self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
            initial_obs["gates_quat"]
        )
        self.__trajectory_spline = None

    def __plan_initial_trajectory(self, initial_obs):
        """Generates trajectory STRICTLY through gates."""
        # 1. Generate waypoints (Center + Entry/Exit alignment points)
        path_points = self.__generate_gate_approach_points(
            self.__start_position,
            self.__gate_positions,
            self.__gate_normals
        )
        
        # REMOVED: __add_detour_logic
        # REMOVED: __insert_obstacle_avoidance_points
        
        # 2. Generate Spline
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points
        )

    def __extract_gate_frames(self, gates_quaternions):
        rotations = R.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        normals = rotation_matrices[:, :, 0]
        y_axes = rotation_matrices[:, :, 1]
        z_axes = rotation_matrices[:, :, 2]
        return normals, y_axes, z_axes

    def __generate_gate_approach_points(self, initial_pos, gate_pos, gate_norm, approach_dist=0.5, num_pts=5):
        offsets = np.linspace(-approach_dist, approach_dist, num_pts)
        gate_pos_exp = gate_pos[:, np.newaxis, :]
        gate_norm_exp = gate_norm[:, np.newaxis, :]
        offsets_exp = offsets[np.newaxis, :, np.newaxis]
        waypoints_matrix = gate_pos_exp + offsets_exp * gate_norm_exp
        flat_waypoints = waypoints_matrix.reshape(-1, 3)
        return np.vstack([initial_pos, flat_waypoints])

    def __compute_trajectory_spline(self, total_time, path_points, custom_time_knots=None):
        if custom_time_knots is not None:
            return CubicSpline(custom_time_knots, path_points)
        
        path_segments = np.diff(path_points, axis=0)
        segment_distances = np.linalg.norm(path_segments, axis=1)
        cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)])
        time_knots = cumulative_distance / cumulative_distance[-1] * total_time
        return CubicSpline(time_knots, path_points)

    def __check_for_env_update(self, current_obs) -> bool:
        """Trigger replan if gates visited. Obstacle proximity is now handled by MPC, not Replanning."""
        # 1. State Transitions
        if self.__last_gate_flags is None:
            self.__last_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
            # We don't track obstacles for replanning anymore, as spline ignores them
            return False

        current_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
        gate_newly_hit = np.any((~self.__last_gate_flags) & current_gate_flags)
        self.__last_gate_flags = current_gate_flags
        
        # 2. Proximity (RWI)
        # Only check Gates for replanning logic (if gates move/update)
        drone_pos = current_obs["pos"]
        gate_dists = np.linalg.norm(current_obs["gates_pos"] - drone_pos, axis=1)
        gate_alert = np.any(gate_dists < self.REPLAN_RADIUS)
        
        return gate_newly_hit or gate_alert

    def __regenerate_flight_plan(self, current_obs, elapsed_time):
        """Replans trajectory (gates only) based on new observations."""
        self.__gate_positions = current_obs["gates_pos"]
        self.__gate_normals, _, _ = self.__extract_gate_frames(
            current_obs["gates_quat"]
        )

        path_points = self.__generate_gate_approach_points(
            self.__start_position, self.__gate_positions, self.__gate_normals
        )
        # Direct spline generation without detour/avoidance
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points
        )

    # ==========================================================================
    # CORE CONTROL FUNCTION
    # ==========================================================================

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """
        1. Checks for replanning (Gates).
        2. Identifies Closest Obstacle.
        3. Updates MPC constraints (Deflection).
        4. Solves.
        """
        current_time = min(self.__current_step * self._dt, self.FLIGHT_DURATION)
        
        if self.__current_step >= int(self.FLIGHT_DURATION * self.__control_freq):
            self._finished = True

        # --- 1. Replanning Logic (Spline Update) ---
        if self.__check_for_env_update(obs):
            self.__regenerate_flight_plan(obs, current_time)

        # --- 2. Obstacle Detection for MPC ---
        drone_pos_2d = obs["pos"][:2]
        obstacle_positions = obs["obstacles_pos"][:, :2] # 2D positions
        
        # Calculate distances to all obstacles
        dists = np.linalg.norm(obstacle_positions - drone_pos_2d, axis=1)
        closest_idx = np.argmin(dists)
        closest_dist = dists[closest_idx]
        
        closest_obs_pos = obstacle_positions[closest_idx]
        
        # Calculate required safety distance squared
        # (r_obs + r_drone + margin)^2
        safe_dist_total = self.OBSTACLE_RADIUS_REAL + self.DRONE_RADIUS + self.SAFETY_MARGIN
        min_safe_dist_sq = safe_dist_total**2
        
        # Create parameter vector: [obs_x, obs_y, min_safe_dist_sq]
        # If obstacle is very far (> 3m), we can relax constraint effectively by moving it far away
        # But keeping it active is safer.
        p_val = np.array([closest_obs_pos[0], closest_obs_pos[1], min_safe_dist_sq])

        # --- 3. Prepare MPC State (x0) ---
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # --- 4. Generate Horizon References & Update Parameters ---
        horizon_times = np.linspace(current_time, current_time + self._T_HORIZON, self.MPC_HORIZON_STEPS)
        horizon_times = np.clip(horizon_times, 0, self.FLIGHT_DURATION)

        ref_pos_horizon = self.__trajectory_spline(horizon_times)       # (N, 3)
        ref_vel_horizon = self.__trajectory_spline(horizon_times, nu=1) # (N, 3)
        
        for k in range(self.MPC_HORIZON_STEPS):
            yref = np.zeros(self._ny)
            yref[0:3] = ref_pos_horizon[k]
            yref[6:9] = ref_vel_horizon[k]
            yref[15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

            self._acados_ocp_solver.set(k, "yref", yref)
            
            # UPDATE PARAMETER (Obstacle Constraint)
            # We assume the obstacle is static or we use the current closest one for the whole horizon
            self._acados_ocp_solver.set(k, "p", p_val)

        # Terminal
        t_final = min(current_time + self._T_HORIZON, self.FLIGHT_DURATION)
        ref_pos_final = self.__trajectory_spline(t_final)
        ref_vel_final = self.__trajectory_spline(t_final, nu=1)

        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = ref_pos_final
        yref_e[6:9] = ref_vel_final
        self._acados_ocp_solver.set(self.MPC_HORIZON_STEPS, "y_ref", yref_e)
        
        # Need to set parameter for terminal stage too if using cost/constraints there depending on formulation
        # Acados parameters are usually required for all stages defined in model
        # Our model defines 'p', so we must set it even if not strictly used in a terminal cost
        # (Though constraints are usually distinct for terminal)
        # Note: We didn't add the constraint to the terminal stage explicitly in create_ocp, 
        # but parameters might still be expected by the C-function interfaces.
        # self._acados_ocp_solver.set(self.MPC_HORIZON_STEPS, "p", p_val) 
        # (Safest to skip unless we explicitly added constraints to terminal)

        # --- 5. Solve ---
        status = self._acados_ocp_solver.solve()
        
        # Get control input
        u0 = self._acados_ocp_solver.get(0, "u")
        
        # Visualization (Optional)
        try:
            draw_line(
                self.env,
                self.__trajectory_spline(self.__trajectory_spline.x),
                rgba=np.array([1.0, 1.0, 1.0, 0.2]),
            )
        except (AttributeError, TypeError):
            pass

        return u0

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self.__current_step += 1
        return self._finished

    def episode_callback(self):
        self.__current_step = 0
        self._acados_ocp_solver.reset()