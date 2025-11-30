"""
Merged Controller: MPC implementation with Spatial "Tunnel" Constraints.
Single Class Implementation.
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
# HELPER FUNCTIONS (Model Generation)
# ==============================================================================

def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates the acados model with Spatial Tunnel Constraints."""
    
    # X-dot is the dynamics function f(X, U)
    # X is the state vector
    # U is the control input vector
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
    model.name = "mpc_spatial_tunnel"
    model.f_expl_expr = X_dot # Explicit dynamics
    model.f_impl_expr = None # No implicit dynamics
    model.x = X
    model.u = U

    # --- SPATIAL FRAME PARAMETERS ---
    # ref_pos (3), normal_vec (3), binormal_vec (3)
    p = cs.MX.sym('p', 9) 
    model.p = p
    
    ref_pos = p[0:3]
    vec_n   = p[3:6] # Transverse 1
    vec_b   = p[6:9] # Transverse 2

    # Current Drone Position
    pos_drone = X[0:3]
    
    # Calculate Deviation from Path Center
    err_vec = pos_drone - ref_pos
    
    # Project error onto transverse vectors to get w1, w2
    w1 = cs.mtimes(err_vec.T, vec_n)
    w2 = cs.mtimes(err_vec.T, vec_b)

    # Constraint Expression: [w1, w2]
    model.con_h_expr = cs.vertcat(w1, w2)
    
    # IMPORTANT: Define for initial stage as well to ensure dimensions match at k=0
    model.con_h_expr_0 = model.con_h_expr

    return model


def create_ocp_solver(Tf: float, N: int, parameters: dict) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates the OCP solver with Tunnel Constraints."""
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # EXPICITLY SET DIMENSIONS
    # This prevents the 'dimension is 0' error
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.nh = 2   # Number of nonlinear path constraints (w1, w2)
    ocp.dims.nsh = 2  # Number of slack variables for h

    ocp.solver_options.N_horizon = N

    # Cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Tracking weights
    Q = np.diag([
        20.0, 20.0, 20.0,   # Pos
        5.0, 5.0, 5.0,      # RPY
        10.0, 10.0, 10.0,   # Vel
        1.0, 1.0, 1.0       # Rates
    ])
    R_mat = np.diag([5.0, 5.0, 5.0, 1.0])

    ocp.cost.W = scipy.linalg.block_diag(Q, R_mat)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((ny, nx)); ocp.cost.Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu)); ocp.cost.Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx_e = np.zeros((ny_e, nx)); ocp.cost.Vx_e[0:nx, 0:nx] = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # Constraints
    ocp.constraints.lbx = np.array([-1.0, -1.0, -1.0]) 
    ocp.constraints.ubx = np.array([1.0, 1.0, 1.0])
    ocp.constraints.idxbx = np.array([3, 4, 5]) # RPY limits

    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((nx))

    # TUNNEL CONSTRAINTS (h = [w1, w2])
    # Must explicitly initialize lh/uh so generator knows they exist
    ocp.constraints.lh = np.array([-5.0, -5.0]) 
    ocp.constraints.uh = np.array([5.0, 5.0])
    
    # Slack variables (Soft constraints)
    ocp.constraints.lsh = np.zeros(2)
    ocp.constraints.ush = np.zeros(2)
    ocp.constraints.idxsh = np.array([0, 1])
    
    # Slack weights
    ocp.cost.zl = 1e4 * np.ones((2,))
    ocp.cost.zu = 1e4 * np.ones((2,))
    ocp.cost.Zl = 1e4 * np.ones((2,))
    ocp.cost.Zu = 1e4 * np.ones((2,))

    ocp.parameter_values = np.zeros(9)

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = Tf

    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mpc_spatial_tunnel.json",
        verbose=False,
        build=True,
        generate=True,
    )
    return solver, ocp


# ==============================================================================
# MAIN CONTROLLER CLASS
# ==============================================================================

class MPCSplineController(Controller):
    """Unified Controller.
    Handles Spline Generation, Parallel Transport Frame calculation, 
    Tunnel Bounds (Gates/Obstacles), and MPC Solving.
    """
    
    # Constants
    FLIGHT_DURATION = 25.0
    MPC_HORIZON_STEPS = 20
    TRACK_WIDTH = 3.0
    GATE_WIDTH = 0.6
    
    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], info: dict, sim_config: dict):
        super().__init__(initial_obs, info, sim_config)

        self.__current_step = 0
        self.__control_freq = sim_config.env.freq
        self._dt = 1 / self.__control_freq
        self._T_HORIZON = self.MPC_HORIZON_STEPS * self._dt

        # 1. Initialize Model & Solver
        self.drone_params = load_params("so_rpy", sim_config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self.MPC_HORIZON_STEPS, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._ny = self._nx + self._ocp.model.u.rows()
        
        # 2. Initialize Planner and Frames
        self.__init_planner(initial_obs)
        
        self._finished = False

    def __init_planner(self, obs):
        """Generates Spline, pre-computes Parallel Transport Frame."""
        self.gates_pos = obs["gates_pos"]
        self.obs_pos = obs["obstacles_pos"]
        
        # --- A. Generate Spline Through Gates ---
        waypoints = [obs["pos"]]
        quats = obs["gates_quat"]
        gate_rot = R.from_quat(quats).as_matrix()
        
        for i, g_pos in enumerate(self.gates_pos):
            # Use gate normal to align entry/exit
            normal = gate_rot[i, :, 0] 
            waypoints.append(g_pos - normal * 0.5)
            waypoints.append(g_pos)
            waypoints.append(g_pos + normal * 0.5)
            
        waypoints = np.array(waypoints)
        
        # Fit Cubic Spline
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        t_knots = cum_dist / cum_dist[-1] * self.FLIGHT_DURATION
        
        self.spline = CubicSpline(t_knots, waypoints)
        
        # --- B. Pre-compute Parallel Transport Frame ---
        self.s_grid = np.linspace(0, self.FLIGHT_DURATION, 1000)
        self.pos_grid = self.spline(self.s_grid)
        self.vel_grid = self.spline(self.s_grid, 1)
        
        # Tangents
        tangents = self.vel_grid / np.linalg.norm(self.vel_grid, axis=1)[:, None]
        
        # Initialize Normals
        self.normals = np.zeros_like(tangents)
        self.binormals = np.zeros_like(tangents)
        
        # Initial Frame
        t0 = tangents[0]
        guide = np.array([0, 0, 1]) if abs(t0[2]) < 0.9 else np.array([0, 1, 0])
        n0 = np.cross(t0, guide)
        n0 /= np.linalg.norm(n0)
        
        self.normals[0] = n0
        self.binormals[0] = np.cross(t0, n0)
        
        # Propagate Frame (minimize twist)
        for i in range(1, len(self.s_grid)):
            t_prev = tangents[i-1]
            t_curr = tangents[i]
            n_prev = self.normals[i-1]
            
            # Rotation from t_prev to t_curr
            v = np.cross(t_prev, t_curr)
            c = np.dot(t_prev, t_curr)
            if c > -0.99: 
                s_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R_trans = np.eye(3) + s_skew + s_skew @ s_skew * (1 / (1 + c))
                n_curr = R_trans @ n_prev
            else:
                n_curr = n_prev
                
            self.normals[i] = n_curr
            self.binormals[i] = np.cross(t_curr, n_curr)

    def _get_spline_frame(self, t):
        """Lookup pre-computed frame at time t."""
        idx = np.searchsorted(self.s_grid, t)
        idx = np.clip(idx, 0, len(self.s_grid)-1)
        return self.pos_grid[idx], self.normals[idx], self.binormals[idx]

    def _get_tunnel_bounds(self, t, pos_t, n_t, b_t):
        """
        Calculates corridor bounds [w1_min, w1_max, w2_min, w2_max] 
        based on Gate Funnels and Obstacle Denting.
        """
        # 1. Base Tunnel
        bounds = np.array([-self.TRACK_WIDTH, self.TRACK_WIDTH, -self.TRACK_WIDTH, self.TRACK_WIDTH])
        
        # 2. Gate Funnels (Pinching)
        for g_pos in self.gates_pos:
            dist = np.linalg.norm(pos_t - g_pos)
            if dist < 2.0:
                ratio = dist / 2.0
                current_w = self.GATE_WIDTH + (self.TRACK_WIDTH - self.GATE_WIDTH) * ratio
                
                # Symmetrical constraint for gates
                bounds[0] = max(bounds[0], -current_w)
                bounds[1] = min(bounds[1],  current_w)
                bounds[2] = max(bounds[2], -current_w)
                bounds[3] = min(bounds[3],  current_w)

        # 3. Obstacle Avoidance (Denting)
        obs_radius = 0.4
        margin = 0.3
        
        for obs_p in self.obs_pos:
            dist_vec = obs_p - pos_t
            dist = np.linalg.norm(dist_vec)
            
            if dist < 4.0:
                # Project into frame
                w1_obs = np.dot(dist_vec, n_t)
                w2_obs = np.dot(dist_vec, b_t)
                
                # Check Dominant Side (Horizontal vs Vertical)
                if abs(w1_obs) > abs(w2_obs):
                    # Horizontal Obstacle
                    if w1_obs > 0: # Obstacle is Right
                        new_max = w1_obs - obs_radius - margin
                        bounds[1] = min(bounds[1], new_max)
                    else: # Obstacle is Left
                        new_min = w1_obs + obs_radius + margin
                        bounds[0] = max(bounds[0], new_min)
                
        return bounds

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        current_time = min(self.__current_step * self._dt, self.FLIGHT_DURATION)
        
        # 1. Prepare State
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # 2. Update Constraints over Horizon
        horizon_times = np.linspace(current_time, current_time + self._T_HORIZON, self.MPC_HORIZON_STEPS)
        
        for k in range(self.MPC_HORIZON_STEPS):
            t_k = min(horizon_times[k], self.FLIGHT_DURATION)
            
            # Get Reference & Frame
            ref_pos, vec_n, vec_b = self._get_spline_frame(t_k)
            ref_vel = self.spline(t_k, 1)
            
            # Set Cost Reference
            yref = np.zeros(self._ny)
            yref[0:3] = ref_pos
            yref[6:9] = ref_vel
            yref[15] = 9.81 * self.drone_params["mass"]
            self._acados_ocp_solver.set(k, "yref", yref)
            
            # Set Frame Parameters [ref_pos, n, b]
            p_val = np.concatenate([ref_pos, vec_n, vec_b])
            self._acados_ocp_solver.set(k, "p", p_val)
            
            # Set Tunnel Bounds
            bounds = self._get_tunnel_bounds(t_k, ref_pos, vec_n, vec_b)
            # lh = [w1_min, w2_min], uh = [w1_max, w2_max]
            lh_k = np.array([bounds[0], bounds[2]])
            uh_k = np.array([bounds[1], bounds[3]])
            
            self._acados_ocp_solver.constraints_set(k, "lh", lh_k)
            self._acados_ocp_solver.constraints_set(k, "uh", uh_k)

        # Terminal Set (Safely set terminal references and parameters)
        t_end = min(current_time + self._T_HORIZON, self.FLIGHT_DURATION)
        ref_pos_e, vec_n_e, vec_b_e = self._get_spline_frame(t_end)
        yref_e = np.zeros(self._nx)
        yref_e[0:3] = ref_pos_e
        self._acados_ocp_solver.set(self.MPC_HORIZON_STEPS, "yref", yref_e)
        
        # Important: Set Parameters for terminal stage too!
        p_val_e = np.concatenate([ref_pos_e, vec_n_e, vec_b_e])
        self._acados_ocp_solver.set(self.MPC_HORIZON_STEPS, "p", p_val_e)
        
        # 3. Solve
        status = self._acados_ocp_solver.solve()
        if status != 0:
            print(f"Acados status: {status}")
            
        u0 = self._acados_ocp_solver.get(0, "u")
        
        # Debug Vis
        try:
            draw_line(self.env, self.spline(self.spline.x), rgba=np.array([0,1,0,0.5]))
        except: pass

        return u0

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self.__current_step += 1
        return self.__current_step >= int(self.FLIGHT_DURATION * self.__control_freq)

    def episode_callback(self):
        self.__current_step = 0
        self._acados_ocp_solver.reset()