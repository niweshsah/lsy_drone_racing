"""
Merged Controller: MPC implementation following a Dynamic Spline Trajectory.
Combines Level 3 replanning/obstacle avoidance with Acados NMPC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import scipy.linalg
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
# ACADOS MODEL & SOLVER GENERATION (From your MPC Code)
# ==============================================================================


# X-dot is the dynamics function f(X, U)
# X is the state vector
# U is the control input vector
def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
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
    Q = np.diag(
        [
            50.0,
            50.0,
            400.0,  # Position (x, y, z)
            1.0,
            1.0,
            1.0,  # Orientation (roll, pitch, yaw)
            10.0,
            10.0,
            10.0,  # Velocity (vx, vy, vz)
            5.0,
            5.0,
            5.0,  # Angular Rates (p, q, r)
        ]
    )

    # Input weights
    R_mat = np.diag(
        [
            1.0,
            1.0,
            1.0,  # Cmd Orientation
            50.0,  # Cmd Thrust
        ]
    )

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

    # Constraints
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])  # Max roll/pitch/yaw
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = np.zeros((nx))

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
    Controller that combines High-Level Spline Replanning (Obstacles/Gates)
    with Low-Level MPC tracking.
    """

    # Planner Constants
    FLIGHT_DURATION = 25.0
    REPLAN_RADIUS = 0.5
    OBSTACLE_CLEARANCE = 0.2

    # MPC Constants
    MPC_HORIZON_STEPS = 25  # N

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], info: dict, sim_config: dict):
        super().__init__(initial_obs, info, sim_config)

        # ---------------------------------------------------------
        # 1. Initialize Planner State (From MyController)
        # ---------------------------------------------------------
        self.__initialize_planner_state(initial_obs, sim_config)
        self.__plan_initial_trajectory(initial_obs)

        # ---------------------------------------------------------
        # 2. Initialize MPC Solver (From AttitudeMPC)
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
    # PLANNER LOGIC (Private methods from MyController)
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
        # 1. Gates
        path_points = self.__generate_gate_approach_points(
            self.__start_position, self.__gate_positions, self.__gate_normals
        )
        # 2. Detours
        path_points = self.__add_detour_logic(
            path_points,
            self.__gate_positions,
            self.__gate_normals,
            self.__gate_y_axes,
            self.__gate_z_axes,
        )
        # 3. Obstacles
        time_knots, path_points = self.__insert_obstacle_avoidance_points(
            path_points, self.__obstacle_positions, self.OBSTACLE_CLEARANCE
        )
        # 4. Spline
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points, custom_time_knots=time_knots
        )

    def __extract_gate_frames(self, gates_quaternions):
        rotations = R.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        normals = rotation_matrices[:, :, 0]
        y_axes = rotation_matrices[:, :, 1]
        z_axes = rotation_matrices[:, :, 2]
        return normals, y_axes, z_axes

    def __generate_gate_approach_points(
        self, initial_pos, gate_pos, gate_norm, approach_dist=0.5, num_pts=5
    ):
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

    def __process_single_obstacle(self, obs_center, sampled_points, sampled_times, clearance):
        collision_free_times = []
        collision_free_points = []
        is_inside = False
        entry_idx = None
        obs_xy = obs_center[:2]

        for i, point in enumerate(sampled_points):
            dist_xy = np.linalg.norm(obs_xy - point[:2])

            if dist_xy < clearance:
                if not is_inside:
                    is_inside = True
                    entry_idx = i
            elif is_inside:
                # Exiting zone
                is_inside = False
                exit_idx = i

                entry_pt = sampled_points[entry_idx]
                exit_pt = sampled_points[exit_idx]

                # Bisector avoidance
                entry_vec = entry_pt[:2] - obs_xy
                exit_vec = exit_pt[:2] - obs_xy
                avoid_vec = entry_vec + exit_vec
                norm_v = np.linalg.norm(avoid_vec)
                if norm_v > 0:
                    avoid_vec /= norm_v

                new_pos_xy = obs_xy + avoid_vec * clearance
                new_pos_z = (entry_pt[2] + exit_pt[2]) / 2
                new_wp = np.concatenate([new_pos_xy, [new_pos_z]])

                avg_time = (sampled_times[entry_idx] + sampled_times[exit_idx]) / 2
                collision_free_times.append(avg_time)
                collision_free_points.append(new_wp)
            else:
                collision_free_times.append(sampled_times[i])
                collision_free_points.append(point)

        return np.array(collision_free_times), np.array(collision_free_points)

    def __insert_obstacle_avoidance_points(self, path_points, obstacle_centers, clearance):
        temp_spline = self.__compute_trajectory_spline(self.FLIGHT_DURATION, path_points)
        num_samples = int(self.__control_freq * self.FLIGHT_DURATION)
        sampled_times = np.linspace(0, self.FLIGHT_DURATION, num_samples)
        sampled_points = temp_spline(sampled_times)

        for obs_center in obstacle_centers:
            sampled_times, sampled_points = self.__process_single_obstacle(
                obs_center, sampled_points, sampled_times, clearance
            )
        return sampled_times, sampled_points

    def __check_for_env_update(self, current_obs) -> bool:
        """Trigger replan if gates/obstacles visited OR proximity danger."""
        # 1. State Transitions
        if self.__last_gate_flags is None:
            self.__last_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
            self.__last_obstacle_flags = np.array(current_obs["obstacles_visited"], dtype=bool)
            return False

        current_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
        current_obstacle_flags = np.array(current_obs["obstacles_visited"], dtype=bool)

        gate_newly_hit = np.any((~self.__last_gate_flags) & current_gate_flags)
        obstacle_newly_hit = np.any((~self.__last_obstacle_flags) & current_obstacle_flags)

        self.__last_gate_flags = current_gate_flags
        self.__last_obstacle_flags = current_obstacle_flags

        # 2. Proximity (RWI)
        drone_pos = current_obs["pos"]

        # Gates (3D)
        gate_dists = np.linalg.norm(current_obs["gates_pos"] - drone_pos, axis=1)
        gate_alert = np.any(gate_dists < self.REPLAN_RADIUS)

        # Obstacles (2D)
        obs_dists = np.linalg.norm(current_obs["obstacles_pos"][:, :2] - drone_pos[:2], axis=1)
        obs_alert = np.any(obs_dists < self.REPLAN_RADIUS)

        return gate_newly_hit or obstacle_newly_hit or gate_alert or obs_alert

    def __regenerate_flight_plan(self, current_obs, elapsed_time):
        """Replans trajectory based on new observations."""
        self.__gate_positions = current_obs["gates_pos"]
        self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
            current_obs["gates_quat"]
        )

        path_points = self.__generate_gate_approach_points(
            self.__start_position, self.__gate_positions, self.__gate_normals
        )
        path_points = self.__add_detour_logic(
            path_points,
            self.__gate_positions,
            self.__gate_normals,
            self.__gate_y_axes,
            self.__gate_z_axes,
        )
        time_knots, path_points = self.__insert_obstacle_avoidance_points(
            path_points, current_obs["obstacles_pos"], self.OBSTACLE_CLEARANCE
        )
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points, custom_time_knots=time_knots
        )

    def __determine_detour_direction(self, v_proj, v_proj_norm, y_axis, z_axis):
        if v_proj_norm < 1e-6:
            return y_axis, "right", 0.0

        v_proj_y = np.dot(v_proj, y_axis)
        v_proj_z = np.dot(v_proj, z_axis)
        angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi

        if -90 <= angle_deg < 45:
            return y_axis, "right", angle_deg
        elif 45 <= angle_deg < 135:
            return z_axis, "top", angle_deg
        else:
            return -y_axis, "left", angle_deg

    def __add_detour_logic(
        self, path_points, g_pos, g_norm, g_y, g_z, num_pts=5, angle_deg=120.0, rad=0.65
    ):
        num_gates = g_pos.shape[0]
        pts_list = list(path_points)
        inserts = 0

        for i in range(num_gates - 1):
            last_idx = 1 + (i + 1) * num_pts - 1 + inserts
            first_idx_next = 1 + (i + 1) * num_pts + inserts

            p1 = pts_list[last_idx]
            p2 = pts_list[first_idx_next]
            vec = p2 - p1
            norm = np.linalg.norm(vec)

            if norm < 1e-6:
                continue

            cos_a = np.dot(vec, g_norm[i]) / norm
            if np.arccos(np.clip(cos_a, -1, 1)) * 180 / np.pi > angle_deg:
                v_proj = vec - np.dot(vec, g_norm[i]) * g_norm[i]
                detour_vec, _, _ = self.__determine_detour_direction(
                    v_proj, np.linalg.norm(v_proj), g_y[i], g_z[i]
                )

                detour_pt = g_pos[i] + rad * detour_vec
                pts_list.insert(last_idx + 1, detour_pt)
                inserts += 1

        return np.array(pts_list)

    # ==========================================================================
    # CORE CONTROL FUNCTION (Merges Planner + MPC)
    # ==========================================================================

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """
        1. Checks for replanning triggers.
        2. Samples the spline for the MPC horizon.
        3. Solves the OCP.
        """
        current_time = min(self.__current_step * self._dt, self.FLIGHT_DURATION)

        if self.__current_step >= int(self.FLIGHT_DURATION * self.__control_freq):
            self._finished = True

        # --- 1. Replanning Logic ---
        if self.__check_for_env_update(obs):
            # This updates self.__trajectory_spline
            self.__regenerate_flight_plan(obs, current_time)

        # --- 2. Prepare MPC State (x0) ---
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        # Constrain initial state for solver
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # --- 3. Generate Horizon References from Spline ---
        # Generate time steps for the prediction horizon: [t, t+dt, t+2dt, ..., t+N*dt]
        horizon_times = np.linspace(
            current_time, current_time + self._T_HORIZON, self.MPC_HORIZON_STEPS
        )

        # Clamp times to flight duration (spline is only defined up to FLIGHT_DURATION)
        horizon_times = np.clip(horizon_times, 0, self.FLIGHT_DURATION)

        # Sample Position and Velocity from the Spline
        ref_pos_horizon = self.__trajectory_spline(horizon_times)  # (N, 3)
        ref_vel_horizon = self.__trajectory_spline(horizon_times, nu=1)  # (N, 3)

        # Set References in Solver
        for k in range(self.MPC_HORIZON_STEPS):
            yref = np.zeros(self._ny)

            # Position Reference
            yref[0:3] = ref_pos_horizon[k]

            # Orientation Reference (Roll, Pitch, Yaw)
            # Roll/Pitch = 0. Yaw = 0 (Can be improved to face velocity vector)
            yref[3:6] = np.zeros(3)

            # Velocity Reference
            yref[6:9] = ref_vel_horizon[k]

            # Angular Rate Reference (0)
            yref[9:12] = np.zeros(3)

            # Input Reference (Hover Thrust)
            yref[15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

            self._acados_ocp_solver.set(k, "yref", yref)

        # --- 4. Final Step Reference (Terminal Cost) ---
        # Sample terminal point
        t_final = min(current_time + self._T_HORIZON, self.FLIGHT_DURATION)
        ref_pos_final = self.__trajectory_spline(t_final)
        ref_vel_final = self.__trajectory_spline(t_final, nu=1)

        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = ref_pos_final
        yref_e[6:9] = ref_vel_final
        self._acados_ocp_solver.set(self.MPC_HORIZON_STEPS, "y_ref", yref_e)

        # --- 5. Solve ---
        status = self._acados_ocp_solver.solve()

        # Check status if needed (0 = success)
        # if status != 0: print(f"Acados returned status {status}")

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
