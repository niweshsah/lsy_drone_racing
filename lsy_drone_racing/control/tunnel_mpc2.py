import json
import os
import shutil
from datetime import datetime
from multiprocessing import Array

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller

# Use non-interactive backend for saving plots
matplotlib.use('Agg')

# ==============================================================================
# 1. PARAMETERS & DYNAMICS
# ==============================================================================

def get_drone_params():  # noqa: ANN201
    """Defines physical parameters and System-ID coefficients."""
    # Attempt to load from file, otherwise use defaults
    params = load_params("so_rpy", "cf21B_500")

    if params is not None:
        # Ensure consistency in return type if loaded from file
        return params
    else:
        return {
            "mass": 0.04338,
            "gravity_vec": np.array([0.0, 0.0, -9.81]),
            "g": 9.81, # Added for convenience
            "J": np.diag([25e-6, 28e-6, 49e-6]),
            "J_inv": np.linalg.inv(np.diag([25e-6, 28e-6, 49e-6])),
            "thrust_min": 0.0,
            "thrust_max": 0.8, # Normalized or Newtons depending on cmd_f_coef
            # System ID Coefficients (Linear Response Model)
            "acc_coef": 0.0, # Bias term for thrust curve
            "cmd_f_coef": 0.96836458, # Slope for thrust curve
            "rpy_coef": [-188.9910, -188.9910, -138.3109],       # Stiffness
            "rpy_rates_coef": [-12.7803, -12.7803, -16.8485],    # Damping
            "cmd_rpy_coef": [138.0834, 138.0834, 198.5161]       # Input Gain
        }

def symbolic_dynamics_spatial(
    mass: float,
    gravity_vec: np.ndarray,
    J: np.ndarray,
    J_inv: np.ndarray,
    acc_coef: float,
    cmd_f_coef: float,
    rpy_coef: list,
    rpy_rates_coef: list,
    cmd_rpy_coef: list,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    
    # --- 1. State Vector X (12 States) ---
    # [cite: 302] States: s, w1, w2, ds, dw1, dw2, phi, theta, psi, dphi, dtheta, dpsi
    s, w1, w2 = ca.SX.sym('s'), ca.SX.sym('w1'), ca.SX.sym('w2')
    ds, dw1, dw2 = ca.SX.sym('ds'), ca.SX.sym('dw1'), ca.SX.sym('dw2')
    
    phi, theta, psi = ca.SX.sym('phi'), ca.SX.sym('theta'), ca.SX.sym('psi')
    dphi, dtheta, dpsi = ca.SX.sym('dphi'), ca.SX.sym('dtheta'), ca.SX.sym('dpsi')

    rpy = ca.vertcat(phi, theta, psi)
    drpy = ca.vertcat(dphi, dtheta, dpsi)

    X = ca.vertcat(s, w1, w2, ds, dw1, dw2, rpy, drpy)

    # --- 2. Control Input U (4 Inputs) ---
    phi_c, theta_c, psi_c, T_c = ca.SX.sym('phi_c'), ca.SX.sym('theta_c'), ca.SX.sym('psi_c'), ca.SX.sym('T_c')
    cmd_rpy = ca.vertcat(phi_c, theta_c, psi_c)
    U = ca.vertcat(cmd_rpy, T_c)

    # --- 3. Parameters P (13 Elements) ---
    # [cite: 299] Dependencies on t, n1, n2, k1, k2 and their derivatives
    t_vec = ca.SX.sym('t_vec', 3)
    n1_vec = ca.SX.sym('n1_vec', 3)
    n2_vec = ca.SX.sym('n2_vec', 3)
    k1, k2 = ca.SX.sym('k1'), ca.SX.sym('k2') 
    dk1, dk2 = ca.SX.sym('dk1'), ca.SX.sym('dk2') # Spatial derivatives of curvature
    
    # ORDER MATTERS: This must match the order in the Controller loop exactly
    P = ca.vertcat(t_vec, n1_vec, n2_vec, k1, k2, dk1, dk2)

    # --- 4. Physics Engine ---

    # A. Rotational Dynamics (Fitted Linear Model)
    # ddrpy = Stiffness * angle + Damping * rate + Gain * command
    c_rpy = ca.DM(rpy_coef)
    c_drpy = ca.DM(rpy_rates_coef)
    c_cmd = ca.DM(cmd_rpy_coef)

    ddrpy = c_rpy * rpy + c_drpy * drpy + c_cmd * cmd_rpy 

    # B. Translational Acceleration (Inertial Frame)
    thrust_mag = acc_coef + cmd_f_coef * T_c
    F_body = ca.vertcat(0, 0, thrust_mag)
    
    # Rotation Matrix (Body -> Inertial)
    cx, cy, cz = ca.cos(phi), ca.cos(theta), ca.cos(psi)
    sx, sy, sz = ca.sin(phi), ca.sin(theta), ca.sin(psi)
    
    R_IB = ca.vertcat(
        ca.horzcat(cy*cz,              cz*sx*sy - cx*sz,   sx*sz + cx*cz*sy),
        ca.horzcat(cy*sz,              cx*cz + sx*sy*sz,   cx*sy*sz - cz*sx),
        ca.horzcat(-sy,                cy*sx,              cx*cy)
    )

    # Global Acceleration [cite: 166]
    g_vec_sym = ca.DM(gravity_vec)
    acc_world = g_vec_sym + (R_IB @ F_body) / mass

    # C. Spatial Dynamics Reconstruction 
    # h is the scaling factor for path curvature
    h = 1 - k1*w1 - k2*w2
    
    # h_dot requires dk1/dk2 (Chain rule: d/dt = d/ds * ds/dt)
    h_dot = -(k1*dw1 + k2*dw2 + (dk1*w1 + dk2*w2)*ds) 

    coriolis = (
        (ds * h_dot) * t_vec +
        (ds**2 * h * k1) * n1_vec +
        (ds**2 * h * k2) * n2_vec -
        (ds * dw1 * k1) * t_vec -
        (ds * dw2 * k2) * t_vec
    )

    # Project World Acceleration onto Path Frame
    proj_t = ca.dot(t_vec, acc_world - coriolis) 
    dds = proj_t / h 
    ddw1 = ca.dot(n1_vec, acc_world - coriolis)
    ddw2 = ca.dot(n2_vec, acc_world - coriolis)

    # --- 5. Final Time Derivative ---
    X_Dot = ca.vertcat(
        ds,     # s_dot
        dw1,    # w1_dot
        dw2,    # w2_dot
        dds,    # s_ddot
        ddw1,   # w1_ddot
        ddw2,   # w2_ddot
        drpy,   # rpy_dot
        ddrpy   # rpy_ddot
    )

    return X_Dot, X, U, P

def export_model(params: dict) -> AcadosModel:
    X_dot, X, U, P = symbolic_dynamics_spatial(
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        J=params["J"],
        J_inv=params.get("J_inv"),
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
    )

    model = AcadosModel()
    model.name = "spatial_mpc_drone"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    model.p = P

    return model

# ==============================================================================
# 2. GEOMETRY ENGINE
# ==============================================================================

class GeometryEngine:
    def __init__(self, gates_pos, start_pos):
        self.gates_pos = np.asarray(gates_pos)
        self.start_pos = np.asarray(start_pos)

        # 1. Waypoints
        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        
        # 2. Tangents (Finite difference)
        tangents = []
        for i in range(len(self.waypoints)):
            if i < len(self.waypoints) - 1:
                t = self.waypoints[i+1] - self.waypoints[i]
            else:
                t = self.waypoints[i] - self.waypoints[i-1]
            norm = np.linalg.norm(t)
            tangents.append(t / norm if norm > 1e-6 else np.array([1,0,0]))
        self.tangents = np.array(tangents)

        # 3. Spline Construction
        # Calculate approximate arc length 's' for knots
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        
        # Cubic Hermite Spline for position
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)
        
        # 4. Generate Pre-computed Frame Data
        self.pt_frame = self._generate_parallel_transport_frame()

    def _generate_parallel_transport_frame(self, num_points=3000):
        """Generates the Parallel Transport frame and pre-calculates curvature derivatives."""
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]
        
        # Initialize dictionary to hold arrays
        frames = {
            "s": s_eval, 
            "pos": [], "t": [], "n1": [], "n2": [], 
            "k1": [], "k2": [], 
            "dk1": [], "dk2": []
        }

        # Initial Frame Setup [cite: 310-313]
        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)
        g_vec = np.array([0, 0, -1]) 
        
        # Handle case where start is vertical
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            # Project gravity to find normal
            n2_0 = g_vec - np.dot(g_vec, t0) * t0
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        # Pass 1: Integrate Frame and Calculate Curvature (k1, k2)
        k1_list = []
        k2_list = []

        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            
            # Curvature vector (k * n) is the second derivative of position wrt s
            k_vec = self.spline(s, 2)
            
            # Project curvature onto normals [cite: 211-218]
            k1 = np.dot(k_vec, curr_n1) # Sign convention depends on definition, usually k = dT/ds
            k2 = np.dot(k_vec, curr_n2)

            # Store
            frames["pos"].append(pos); frames["t"].append(curr_t)
            frames["n1"].append(curr_n1); frames["n2"].append(curr_n2)
            k1_list.append(k1); k2_list.append(k2)
            
            # Propagate Frame using Parallel Transport [cite: 213]
            # dT/ds = k1*n1 + k2*n2
            # dN1/ds = -k1*T
            # dN2/ds = -k2*T
            
            if i < len(s_eval) - 1:
                # Next tangent
                next_t = self.spline(s_eval[i+1], 1)
                next_t /= np.linalg.norm(next_t)
                
                # Approximate integration for normals (small angle approximation)
                # This keeps n1 perpendicular to t without inducing twist
                axis = np.cross(curr_t, next_t)
                angle = np.arccos(np.clip(np.dot(curr_t, next_t), -1.0, 1.0))
                
                if np.linalg.norm(axis) > 1e-6:
                    axis /= np.linalg.norm(axis)
                    r_vec = R.from_rotvec(axis * angle)
                    next_n1 = r_vec.apply(curr_n1)
                    next_n2 = r_vec.apply(curr_n2)
                else:
                    next_n1 = curr_n1
                    next_n2 = curr_n2
                
                curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        frames["k1"] = np.array(k1_list)
        frames["k2"] = np.array(k2_list)

        # Pass 2: Calculate Derivatives of Curvature (dk1, dk2)
        # We need these for the term h_dot in the dynamics 
        frames["dk1"] = np.gradient(frames["k1"], ds)
        frames["dk2"] = np.gradient(frames["k2"], ds)

        for k in frames: 
            if isinstance(frames[k], list):
                frames[k] = np.array(frames[k])
                
        return frames

    def get_frame(self, s_query):
        """Looks up pre-calculated frame data for a given s."""
        idx = np.searchsorted(self.pt_frame['s'], s_query) - 1
        idx = np.clip(idx, 0, len(self.pt_frame['s'])-1)
        return {k: self.pt_frame[k][idx] for k in self.pt_frame if k != 's'}

    def get_closest_s(self, pos_query, s_guess=0.0, window=5.0):
        """Finds s* [cite: 239] locally around s_guess."""
        mask = (self.pt_frame['s'] >= s_guess - 1.0) & (self.pt_frame['s'] <= s_guess + window)
        
        if not np.any(mask):
            candidates_pos = self.pt_frame['pos']
            candidates_s = self.pt_frame['s']
        else:
            candidates_pos = self.pt_frame['pos'][mask]
            candidates_s = self.pt_frame['s'][mask]

        dists = np.linalg.norm(candidates_pos - pos_query, axis=1)
        idx_min = np.argmin(dists)
        return candidates_s[idx_min]

# ==============================================================================
# 3. ACADOS SOLVER SETUP
# ==============================================================================

class SpatialMPC:
    def __init__(self, params, N=20, Tf=1.0):
        self.N = N
        self.Tf = Tf
        self.params = params
        
        # Clean compile directory
        if os.path.exists('c_generated_code'): 
            try: shutil.rmtree('c_generated_code')
            except: pass
        
        self.solver = self._build_solver()

    def _build_solver(self):
        model = export_model(self.params)
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf

        # --- DIMENSIONS ---
        nx = 12 
        nu = 4
        ny = nx + nu
        ny_e = nx

        # --- COST CONFIGURATION ---
        # Weights: s, w1, w2, ds, dw1, dw2, phi, th, psi, dphi, dth, dpsi
        # High penalty on w1/w2 (corridor). 
        # Low penalty on s (we drive s via reference).
        q_diag = np.array([
            0.0, 20.0, 20.0,    # Pos
            10.0, 5.0,  5.0,    # Vel
            1.0,  1.0,  1.0,    # Att
            0.1,  0.1,  0.1     # Rate
        ])
        
        r_diag = np.array([5.0, 5.0, 5.0, 0.1]) # Input weights

        ocp.cost.W = scipy.linalg.block_diag(np.diag(q_diag), np.diag(r_diag))
        ocp.cost.W_e = np.diag(q_diag)
        
        ocp.cost.Vx = np.zeros((ny, nx)); ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu)); ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # --- CONSTRAINTS ---
        # Hard Inputs
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-0.8, -0.8, -0.8, self.params['thrust_min']])
        ocp.constraints.ubu = np.array([+0.8, +0.8, +0.8, self.params['thrust_max']])
        
        # Soft State Bounds (Corridor) [cite: 305]
        # w1, w2 indices in x are 1 and 2
        ocp.constraints.idxbx = np.array([1, 2])
        ocp.constraints.lbx = np.array([-0.5, -0.5]) 
        ocp.constraints.ubx = np.array([+0.5, +0.5])
        
        # Slack Config
        ns = 2
        ocp.constraints.idxsbx = np.array([0, 1]) # Slack on 0th and 1st element of idxbx
        
        BIG_COST = 1000.0
        ocp.cost.zl = BIG_COST * np.ones(ns)
        ocp.cost.zu = BIG_COST * np.ones(ns)
        ocp.cost.Zl = BIG_COST * np.ones(ns)
        ocp.cost.Zu = BIG_COST * np.ones(ns)

        ocp.constraints.x0 = np.zeros(nx)

        # --- OPTIONS ---
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp.solver_options.qp_solver_tol_stat = 1e-4

        # --- PARAMETERS ---
        # Initialize P (size 13)
        # [t(3), n1(3), n2(3), k1, k2, dk1, dk2]
        p0 = np.concatenate([
            [1,0,0], [0,1,0], [0,0,1], 
            [0,0], [0,0]
        ])
        ocp.parameter_values = p0

        return AcadosOcpSolver(ocp, json_file='acados_spatial.json')

# ==============================================================================
# 4. CONTROLLER CLASS
# ==============================================================================

class SpatialMPCController(Controller):
    def __init__(self, obs: dict, info: dict, config: dict):
        # 1. Setup
        self.params = get_drone_params()
        self.v_target = 4.0 # Target speed (aggressive)
        
        # 2. Geometry
        gates_list = config.get("env", {}).get("track", {}).get("gates", [])
        if not gates_list and "gates" in info:
            gates_list = info["gates"]
            
        gates_pos = [g["pos"] for g in gates_list]
        start_pos = obs["pos"]
        
        self.geo = GeometryEngine(gates_pos, start_pos)
        
        # 3. Solver
        self.N_horizon = 20
        self.mpc = SpatialMPC(self.params, N=self.N_horizon, Tf=1.0)
        
        # 4. State
        self.prev_s = 0.0
        self.episode_start_time = datetime.now()
        self.step_count = 0
        self.control_log = {k: [] for k in ['timestamps', 'phi_c', 'theta_c', 'psi_c', 'thrust_c', 'solver_status', 's', 'w1', 'w2', 'ds']}
        self.debug = True

        self.reset_mpc_solver()

    def reset_mpc_solver(self):
        """Warm starts the solver with a forward guess."""
        nx = 12
        hover_T = self.params['mass'] * self.params['g']
        
        for k in range(self.N_horizon + 1):
            x_guess = np.zeros(nx)
            # Guess forward progress
            x_guess[0] = self.v_target * k * (self.mpc.Tf / self.N_horizon)
            x_guess[3] = self.v_target # Target velocity
            
            self.mpc.solver.set(k, "x", x_guess)
            if k < self.N_horizon:
                self.mpc.solver.set(k, "u", np.array([0, 0, 0, hover_T]))
        
        self.prev_s = 0.0

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        
        # Derived states
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        
        hover_T = self.params['mass'] * self.params['g']
        
        # 1. State Feedback (World -> Spatial) [cite: 239-245]
        x_spatial = self._cartesian_to_spatial(obs["pos"], obs["vel"], obs["rpy"], obs["drpy"])
        
        # Update current state constraint (x0)
        self.mpc.solver.set(0, "lbx", x_spatial)
        self.mpc.solver.set(0, "ubx", x_spatial)
        
        # 2. Horizon Updates
        curr_s = x_spatial[0]
        curr_ds = x_spatial[3]
        dt = self.mpc.Tf / self.mpc.N
        
        # "Carrot" approach: Set target velocity high [cite: 586]
        target_vel = self.v_target 

        for k in range(self.mpc.N):
            # A. Predict s for parameter lookup
            s_pred = curr_s + k * max(curr_ds, 1.0) * dt # Use max to prevent stagnation lookups
            
            # B. Set Parameters P [cite: 213]
            f = self.geo.get_frame(s_pred)
            # P = [t(3), n1(3), n2(3), k1, k2, dk1, dk2]
            p_k = np.concatenate([
                f['t'], f['n1'], f['n2'], 
                [f['k1']], [f['k2']], 
                [f['dk1']], [f['dk2']]
            ])
            self.mpc.solver.set(k, "p", p_k)
            
            # C. Set Reference yref [cite: 558]
            # Drive s forward aggressively
            s_ref = curr_s + (k + 1) * target_vel * dt
            
            y_ref = np.zeros(16)
            y_ref[0] = s_ref        # Target progress
            y_ref[3] = target_vel   # Target speed
            y_ref[15] = hover_T     # Feedforward thrust
            
            self.mpc.solver.set(k, "yref", y_ref)

        # 3. Terminal Node Update
        s_end = curr_s + self.mpc.N * target_vel * dt
        f_end = self.geo.get_frame(s_end)
        p_end = np.concatenate([
             f_end['t'], f_end['n1'], f_end['n2'], 
             [f_end['k1']], [f_end['k2']], 
             [f_end['dk1']], [f_end['dk2']]
        ])
        self.mpc.solver.set(self.mpc.N, "p", p_end)
        
        yref_e = np.zeros(12)
        yref_e[0] = s_end
        yref_e[3] = target_vel
        self.mpc.solver.set(self.mpc.N, "yref", yref_e)

        # 4. Solve
        status = self.mpc.solver.solve()
        
        if status != 0:
            print(f"MPC Warning: Solver status {status}")
            # Fallback to hover if failed
            u_opt = np.array([0.0, 0.0, 0.0, hover_T])
        else:
            u_opt = self.mpc.solver.get(0, "u")
            
        # Log
        self._log_control_step(x_spatial, u_opt, status)

        # Return [roll, pitch, yaw, thrust]
        return np.array([u_opt[0], u_opt[1], u_opt[2], u_opt[3]])

    def _cartesian_to_spatial(self, pos, vel, rpy, drpy):
        """Projects global state onto the path frame [cite: 282-284]."""
        # 1. Find s
        s = self.geo.get_closest_s(pos, s_guess=self.prev_s)
        self.prev_s = s 
        
        # 2. Get Frame
        f = self.geo.get_frame(s)
        
        # 3. Calculate Errors (w1, w2)
        r_vec = pos - f['pos']
        w1 = np.dot(r_vec, f['n1'])
        w2 = np.dot(r_vec, f['n2'])
        
        # 4. Scaling Factor h
        h = 1 - f['k1'] * w1 - f['k2'] * w2
        # Safety clamp to avoid division by zero in tight loops
        h = max(h, 0.1) 
        
        # 5. Calculate Spatial Velocities
        ds = np.dot(vel, f['t']) / h
        dw1 = np.dot(vel, f['n1'])
        dw2 = np.dot(vel, f['n2'])
        
        return np.array([s, w1, w2, ds, dw1, dw2, rpy[0], rpy[1], rpy[2], drpy[0], drpy[1], drpy[2]])

    def _log_control_step(self, x_spatial, u_opt, status):
        # Basic logging implementation
        self.step_count += 1
        elapsed = (datetime.now() - self.episode_start_time).total_seconds()
        
        self.control_log['timestamps'].append(elapsed)
        self.control_log['s'].append(x_spatial[0])
        self.control_log['w1'].append(x_spatial[1])
        self.control_log['thrust_c'].append(u_opt[3])
        self.control_log['solver_status'].append(status)

    def episode_callback(self):
        # Placeholder for plotting logic
        pass

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return False