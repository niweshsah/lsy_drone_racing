import json
import os
import shutil
from datetime import datetime
from multiprocessing import Array
from typing import List

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
from lsy_drone_racing.utils.utils import draw_line

# Use non-interactive backend for saving plots
matplotlib.use('Agg')

# Parameters:
v_max_ref = 1.5  # m/s
corner_acc = 1.95
mpc_horizons_global = 50
max_lateral_width = 0.3
safety_radius = 0.1

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
            "thrust_max": 0.2, # Normalized or Newtons depending on cmd_f_coef
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

    # Global Acceleration
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
    def __init__(
        self, 
        gates_pos: List[List[float]], 
        gates_normals: List[List[float]], 
        start_pos: List[float]
    ):
        """
        Initializes the geometry engine and generates the safe flight path.
        """
        # --- 1. Configuration Constants ---
        self.DETOUR_ANGLE_THRESHOLD = 60.0  # Degrees. If turn > this, add detour.
        self.DETOUR_RADIUS = 0.3            # Meters. How far to swing out for detours.
        self.TANGENT_SCALE_FACTOR = 1     # Controls how "aggressive" the curves are.

        # --- 2. Data Ingestion ---
        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)

        # --- 3. Pipeline Execution ---
        
        # A. Initialize Waypoints (Start + Gates)
        # We track 'types': 0=Start, 1=Gate, 2=Detour
        self.waypoints, self.wp_types, self.wp_normals = self._initialize_waypoints()

        # B. Insert Detour Points for Sharp Turns
        self.waypoints, self.wp_types, self.wp_normals = self._add_detour_logic(
            self.waypoints, self.wp_types, self.wp_normals
        )

        # C. Compute Tangents (Strict constraints for Gates, Smooth for others)
        self.tangents = self._compute_hermite_tangents()

        # D. Generate Cubic Hermite Spline
        # Parameterize by cumulative Euclidean distance (Arc Length approximation)
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        
        # P(s) -> Returns (x,y,z)
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # E. Generate Parallel Transport Frame (for visualization/physics)
        self.pt_frame = self._generate_parallel_transport_frame(num_points=1000)

    def _initialize_waypoints(self):
        """Creates the initial ordered list of waypoints starting with Start Pos."""
        wps = [self.start_pos]
        types = [0]             # 0 = Start
        normals = [np.zeros(3)] # Start has no forced orientation
        
        for i in range(len(self.gates_pos)):
            wps.append(self.gates_pos[i])
            types.append(1)     # 1 = Gate
            normals.append(self.gate_normals[i])
            
        return np.array(wps), np.array(types), np.array(normals)

    def _add_detour_logic(self, wps, types, normals):
        """
        Analyzes consecutive waypoints. If the angle required to hit the next point
        is too sharp relative to the current gate's normal, inserts a detour point.
        """
        new_wps = [wps[0]]
        new_types = [types[0]]
        new_normals = [normals[0]]

        for i in range(len(wps) - 1):
            curr_p = wps[i]
            next_p = wps[i+1]
            curr_type = types[i]
            
            # Only apply detour logic if we are LEAVING a GATE (Type 1)
            if curr_type == 1: 
                # Identify which gate index this corresponds to in original arrays
                # (Since wps includes start, gate index is i-1)
                gate_idx = i - 1 
                gate_norm = self.gate_normals[gate_idx]
                
                vec_to_next = next_p - curr_p
                dist = np.linalg.norm(vec_to_next)
                
                if dist > 1e-6:
                    vec_to_next /= dist
                    
                    # Dot product: 1.0 (Straight), 0.0 (90 deg), -1.0 (180 deg)
                    alignment = np.dot(gate_norm, vec_to_next)
                    # Convert to angle
                    angle_deg = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))

                    if angle_deg > self.DETOUR_ANGLE_THRESHOLD:
                        # --- Create Detour ---
                        # Project vector onto the Gate's Plane (Y-Z plane) to find "sideways" direction
                        # Formula: v_proj = v - (v . n) * n
                        proj = vec_to_next - (np.dot(vec_to_next, gate_norm) * gate_norm)
                        
                        if np.linalg.norm(proj) < 1e-3:
                            # Perfectly backwards (180 deg). Default to "Up" relative to gate
                            # (Assuming we have gate Z vector, but here simple approximation)
                            detour_dir = np.array([0, 0, 1])
                        else:
                            detour_dir = proj / np.linalg.norm(proj)
                        
                        # Place detour point:
                        # 1. Start at gate center
                        # 2. Move 'out' by radius
                        # 3. Move 'forward' slightly along normal so we don't clip the frame
                        detour_pos = curr_p + (detour_dir * self.DETOUR_RADIUS) + (gate_norm * 1.5)
                        
                        new_wps.append(detour_pos)
                        new_types.append(2) # 2 = Detour
                        new_normals.append(np.zeros(3)) # No forced normal

            # Always add the target point
            new_wps.append(next_p)
            new_types.append(types[i+1])
            new_normals.append(normals[i+1])

        return np.array(new_wps), np.array(new_types), np.array(new_normals)

    def _compute_hermite_tangents(self):
        """
        Calculates the tangent (velocity) vectors for the Cubic Hermite Spline.
        - Gates: Tangent MUST be aligned with Gate Normal.
        - Detours/Start: Tangent is heuristic (Catmull-Rom / Finite Difference).
        """
        num_pts = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)

        for i in range(num_pts):
            # 1. Determine Scale (Speed) based on segment lengths
            dist_prev = np.linalg.norm(self.waypoints[i] - self.waypoints[i-1]) if i > 0 else 0
            dist_next = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i]) if i < num_pts - 1 else 0
            
            # Use minimum neighbor distance to prevent loops/overshoot
            base_scale = min(dist_prev if dist_prev > 0 else dist_next, 
                             dist_next if dist_next > 0 else dist_prev)
            
            scale = base_scale * self.TANGENT_SCALE_FACTOR

            if self.wp_types[i] == 1: 
                # --- GATE: Strict Alignment ---
                normal = self.wp_normals[i].copy()
                
                # Auto-Flip: If the natural path flow opposes the normal, flip the normal
                if i > 0 and i < num_pts - 1:
                    flow_vec = self.waypoints[i+1] - self.waypoints[i-1]
                    if np.dot(normal, flow_vec) < 0:
                        normal = -normal
                
                tangents[i] = normal * scale

            else:
                # --- START / DETOUR: Smooth Curve ---
                if i == 0:
                    t = self.waypoints[i+1] - self.waypoints[i]
                elif i == num_pts - 1:
                    t = self.waypoints[i] - self.waypoints[i-1]
                else:
                    t = self.waypoints[i+1] - self.waypoints[i-1]
                
                if np.linalg.norm(t) > 1e-6:
                    t = t / np.linalg.norm(t)
                
                tangents[i] = t * scale

        return tangents
        
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

        # [cite_start]Initial Frame Setup [cite: 310-313]
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
            
            # Project curvature onto normals
            k1 = np.dot(k_vec, curr_n1) 
            k2 = np.dot(k_vec, curr_n2)

            # Store
            frames["pos"].append(pos); frames["t"].append(curr_t)
            frames["n1"].append(curr_n1); frames["n2"].append(curr_n2)
            k1_list.append(k1); k2_list.append(k2)
            
            # [cite_start]Propagate Frame using Parallel Transport [cite: 213]
            if i < len(s_eval) - 1:
                # Next tangent
                next_t = self.spline(s_eval[i+1], 1)
                next_t /= np.linalg.norm(next_t)
                
                # Approximate integration for normals (small angle approximation)
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
    def __init__(self, params, N=50, Tf=1.0):
        self.N = N
        self.Tf = Tf
        params['g'] = params['gravity_vec'][2]  # Ensure g is consistent
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
        q_diag = np.array([
            1.0, 20.0, 20.0,    # Pos (s, w1, w2)
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
        ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, self.params['thrust_min'] * 4] )
        ocp.constraints.ubu = np.array([+0.5, +0.5, +0.5, self.params['thrust_max'] * 4])
        
        # Soft State Bounds (Corridor) - Path Stage
        # w1, w2 indices in x are 1 and 2
        ocp.constraints.idxbx = np.array([1, 2 , 6 , 7 , 8])  # w1, w2, phi, theta, psi
        ocp.constraints.lbx = np.array([-0.4, -0.4 , -0.5, -0.5, -0.5]) 
        ocp.constraints.ubx = np.array([+0.4, +0.4, +0.5, +0.5, +0.5])
        
        # --- FIXED: Terminal Constraints ---
        # Ensure terminal node N also respects the flight corridor
        ocp.constraints.idxbx_e = np.array([1, 2 , 6 , 7 , 8])
        ocp.constraints.lbx_e = np.array([-0.4, -0.4 , -0.5, -0.5, -0.5]) 
        ocp.constraints.ubx_e = np.array([+0.4, +0.4, +0.5, +0.5, +0.5])

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
    def __init__(self, obs: dict, info: dict, config: dict, env=None):
        # 1. Setup
        self.params = get_drone_params()
        self.v_target = v_max_ref # Target speed
        self.env = env
        
        # --- FIXED: Use Upper Case for Constants to match usage ---
        self.OBS_RADIUS = safety_radius
        self.W1_MAX = max_lateral_width
        self.W2_MAX = max_lateral_width
        
        # --- FIXED: Robust Obstacle Parsing ---
        # Attempt to load obstacles from config, fallback to info
        raw_obstacles = config.get("env", {}).get("track", {}).get("obstacles", [])
        if not raw_obstacles and "obstacles" in info:
            raw_obstacles = info["obstacles"]
            
        # Parse into list of numpy arrays
        self.obstacles_pos = []
        for o in raw_obstacles:
            if isinstance(o, dict) and "pos" in o:
                self.obstacles_pos.append(np.array(o["pos"]))
            elif isinstance(o, (list, np.ndarray)):
                self.obstacles_pos.append(np.array(o))
            elif isinstance(o, dict): 
                # Fallback if 'pos' key isn't present but dict itself is the pos (unlikely but safe)
                self.obstacles_pos.append(np.array(list(o.values())))

        # 2. Geometry
        gates_list = config.get("env", {}).get("track", {}).get("gates", [])
        if not gates_list and "gates" in info:
            gates_list = info["gates"]
            
        gates_pos = [g["pos"] for g in gates_list]
        start_pos = obs["pos"]
        
        gates_quaternions = obs['gates_quat']
        gates_normals = self._get_gate_normals(gates_quaternions)
        
        self.geo = GeometryEngine(gates_pos, gates_normals , start_pos)
        
        # 3. Solver
        self.N_horizon = mpc_horizons_global
        self.mpc = SpatialMPC(self.params, N=self.N_horizon, Tf=1.0)
        
        # 4. State
        self.prev_s = 0.0
        self.episode_start_time = datetime.now()
        self.step_count = 0
        self.control_log = {k: [] for k in ['timestamps', 'phi_c', 'theta_c', 'psi_c', 'thrust_c', 'solver_status', 's', 'w1', 'w2', 'ds']}
        self.debug = True

        self.reset_mpc_solver()
    
    def _compute_corridor_bounds(self, s_pred, frame_pos, frame_t, frame_n1, frame_n2):
        """Computes dynamic [lb, ub] for w1 and w2 at a specific path location s.
        Projects 'thin rod' obstacles onto the transverse plane.
        """
        # 1. Initialize with full corridor width
        lb_w1, ub_w1 = -self.W1_MAX, self.W1_MAX
        lb_w2, ub_w2 = -self.W2_MAX, self.W2_MAX
        
        # Sensitivity: How far along s (longitudinal) do we care about an obstacle?
        longitudinal_threshold = 0.5 

        # --- FIXED: Loop over self.obstacles_pos instead of undefined self.obstacles ---
        for obs_pos in self.obstacles_pos:
            # Vector from Path Center -> Obstacle
            r_vec = obs_pos - frame_pos
            
            # Project onto Tangent (s-direction)
            s_dist = np.dot(r_vec, frame_t)
            
            if abs(s_dist) < longitudinal_threshold:
                # Project onto Transverse Plane (n1, n2)
                w1_obs = np.dot(r_vec, frame_n1)
                w2_obs = np.dot(r_vec, frame_n2)
                
                # --- Dominant Side Logic ---
                # Check if obstacle is actually inside our max corridor
                if (lb_w1 < w1_obs < ub_w1) and (lb_w2 < w2_obs < ub_w2):
                    
                    # DECISION: Pass Left or Pass Right?
                    if w1_obs > 0:
                        # Obstacle on Left -> Pass Right (Trim Upper Bound)
                        dist_to_surface = w1_obs - self.OBS_RADIUS
                        ub_w1 = min(ub_w1, dist_to_surface)
                    else:
                        # Obstacle on Right -> Pass Left (Trim Lower Bound)
                        dist_to_surface = w1_obs + self.OBS_RADIUS
                        lb_w1 = max(lb_w1, dist_to_surface)

        # --- FIXED: Gap Closing Logic ---
        # Prevent infeasibility if bounds cross
        if lb_w1 >= ub_w1:
            mid = (lb_w1 + ub_w1) / 2
            lb_w1 = mid - 0.05
            ub_w1 = mid + 0.05
            if self.debug and self.step_count % 50 == 0:
                print(f"Warning: Corridor gap closed at s={s_pred:.2f}. Forcing narrow passage.")

        return np.array([lb_w1, lb_w2]), np.array([ub_w1, ub_w2])
        
    def _get_gate_normals(self, gates_quaternions : np.ndarray) -> np.ndarray:
        rotations = R.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        gates_normals = rotation_matrices[:, :, 0]  # X-axis (normal)
        return gates_normals

    def reset_mpc_solver(self):
        """Warm starts the solver with a forward guess."""
        nx = 12
        hover_T = self.params['mass'] * self.params['g']
        
        for k in range(self.N_horizon + 1):
            x_guess = np.zeros(nx)
            # Linear ramp from 0 to v_target
            vel_k = self.v_target * (k / self.N_horizon) 
            x_guess[3] = vel_k 
            x_guess[0] = vel_k * k * (self.mpc.Tf / self.N_horizon) * 0.5 
            
            self.mpc.solver.set(k, "x", x_guess)
            if k < self.N_horizon:
                self.mpc.solver.set(k, "u", np.array([0, 0, 0, hover_T]))
        
        self.prev_s = 0.0

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        
        # Derived states
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        
        ANGLE_LB = np.array([-0.5, -0.5, -0.5])
        ANGLE_UB = np.array([0.5, 0.5, 0.5])
        
        hover_T = self.params['mass'] * -self.params['g']
        
        # 1. State Feedback (World -> Spatial)
        x_spatial = self._cartesian_to_spatial(obs["pos"], obs["vel"], obs["rpy"], obs["drpy"])
        
        # Update current state constraint (x0)
        self.mpc.solver.set(0, "lbx", x_spatial)
        self.mpc.solver.set(0, "ubx", x_spatial)
        
        # 2. Horizon Updates
        curr_s = x_spatial[0]
        curr_ds = x_spatial[3]
        dt = self.mpc.Tf / self.mpc.N
        
        # "Carrot" approach: Set target velocity high
        target_vel = self.v_target 
        
        # if self.env is not None and self.step_count % 5 == 0:
        #     try:
        #         path_points = self.geo.pt_frame['pos'][::5]
        #         draw_line(self.env, points=path_points, rgba=np.array([0.0, 1.0, 0.0, 0.5]))
        #     except Exception:
        #         pass
            
        if self.env is not None and self.step_count % 5 == 0:
            try:
                # 1. Draw Center Line (Green)
                path_points = self.geo.pt_frame['pos'][::5]
                draw_line(self.env, points=path_points, rgba=np.array([0.0, 1.0, 0.0, 0.5]))

                # 2. Draw Left & Right Bounds (Red)
                # We need the normal vector n1 at each point to offset the position
                # Offset = Position + (Width * n1)
                
                # Slicing [::5] to match the path_points downsampling
                positions = self.geo.pt_frame['pos'][::5]
                normals_n1 = self.geo.pt_frame['n1'][::5]
                
                
                
                # Calculate Left and Right Boundary Points
                # Note: w1 is along n1. 
                # Left Bound  = pos + (W1_MAX * n1)
                # Right Bound = pos + (-W1_MAX * n1)
                left_bound_points = positions + (self.W1_MAX * normals_n1)
                right_bound_points = positions - (self.W1_MAX * normals_n1)
                
                left_curr_bound_points = positions + (-self.W1_MAX * normals_n1)

                # Draw them
                draw_line(self.env, points=left_bound_points, rgba=np.array([1.0, 0.0, 0.0, 0.3])) # Red, semi-transparent
                draw_line(self.env, points=right_bound_points, rgba=np.array([1.0, 0.0, 0.0, 0.3])) 

            except Exception as e:
                pass

        # Initialize a running reference s starting from current position
        running_s_ref = curr_s
        max_lat_acc = corner_acc 
        epsilon = 0.01 

        # --- LOOP 0 to N-1 ---
        for k in range(self.mpc.N):
            # A. Predict s for parameter lookup
            s_pred = curr_s + k * max(curr_ds, 1.0) * dt 
            
            # B. Get Frame & Curvature
            f = self.geo.get_frame(s_pred)
            
            # C. Dynamic Corridor Bounds
            w_lb, w_ub = self._compute_corridor_bounds(
                s_pred, f['pos'], f['t'], f['n1'], f['n2']
            )
            vis_curr_l_points = []
            vis_curr_r_points = []
            vis_curr_l_points.append(f['pos'] + w_ub[0] * f['n1'])
            vis_curr_r_points.append(f['pos'] + w_lb[0] * f['n1'])
            
            # Update Constraints
            if k > 0:
                lbx_k = np.concatenate([w_lb, ANGLE_LB])
                # print("this is lbx_k:", lbx_k)
                ubx_k = np.concatenate([w_ub, ANGLE_UB])
                self.mpc.solver.set(k, "lbx", lbx_k)
                self.mpc.solver.set(k, "ubx", ubx_k)
            
            # D. Speed Logic
            k_mag = np.sqrt(f['k1']**2 + f['k2']**2)
            v_corner = np.sqrt(max_lat_acc / (k_mag + epsilon))
            v_ref_k = min(v_corner, target_vel)
        
            # E. Integrate s_ref
            running_s_ref += v_ref_k * dt

            # F. Set Parameters P
            p_k = np.concatenate([
                f['t'], f['n1'], f['n2'], 
                [f['k1']], [f['k2']], 
                [f['dk1']], [f['dk2']]
            ])
            self.mpc.solver.set(k, "p", p_k)
            
            # G. Set Reference yref
            y_ref = np.zeros(16)
            y_ref[0] = running_s_ref
            y_ref[3] = v_ref_k
            y_ref[15] = hover_T
            self.mpc.solver.set(k, "yref", y_ref)

        # --- FIXED: Terminal Node N Update ---
        # 1. Integrate one last step for terminal s
        s_end = running_s_ref + v_ref_k * dt 
        
        # 2. Get Frame for N
        f_end = self.geo.get_frame(s_end)
        p_end = np.concatenate([
             f_end['t'], f_end['n1'], f_end['n2'], 
             [f_end['k1']], [f_end['k2']], 
             [f_end['dk1']], [f_end['dk2']]
        ])
        self.mpc.solver.set(self.mpc.N, "p", p_end)
        
        # 3. Compute Terminal Corridor Bounds (Safety check at end of horizon)
        w_lb_e, w_ub_e = self._compute_corridor_bounds(
             s_end, f_end['pos'], f_end['t'], f_end['n1'], f_end['n2']
        )
        lbx_e = np.concatenate([w_lb_e, ANGLE_LB])
        ubx_e = np.concatenate([w_ub_e, ANGLE_UB])
        self.mpc.solver.set(self.mpc.N, "lbx", lbx_e)
        self.mpc.solver.set(self.mpc.N, "ubx", ubx_e)
        
        # 4. Set Terminal Reference
        yref_e = np.zeros(12)
        yref_e[0] = s_end
        yref_e[3] = v_ref_k 
        yref_e[11] = hover_T
        self.mpc.solver.set(self.mpc.N, "yref", yref_e)

        if self.debug and self.step_count % 50 == 0:
            print(f"Terminal Step, s_end: {s_end:.2f}, k_mag: {k_mag:.3f}, v_ref: {v_ref_k:.3f}")

        # 4. Solve
        status = self.mpc.solver.solve()
        
        # Visualization (Blue Line)
        if self.env is not None:
            try:
                mpc_points_cartesian = []
                for k in range(self.mpc.N + 1):
                    x_k = self.mpc.solver.get(k, "x")
                    pos_k = self._spatial_to_cartesian(x_k[0], x_k[1], x_k[2])
                    mpc_points_cartesian.append(pos_k)
                
                draw_line(self.env, points=np.array(mpc_points_cartesian), rgba=np.array([0.0, 0.0, 1.0, 0.8]))
                draw_line(self.env, points=np.array(vis_curr_l_points), rgba=np.array([0.0, 1.0, 0.0, 0.8]))
                draw_line(self.env, points=np.array(vis_curr_r_points), rgba=np.array([0.0, 1.0, 0.0, 0.8]))
            except Exception:
                pass
        
        if status != 0:
            print(f"MPC Warning: Solver status {status}")
            u_opt = np.array([0.0, 0.0, 0.0, hover_T])
        else:
            u_opt = self.mpc.solver.get(0, "u")
            
        self._log_control_step(x_spatial, u_opt, status)
        return np.array([u_opt[0], u_opt[1], u_opt[2], u_opt[3]])
    
    def _spatial_to_cartesian(self, s, w1, w2):
        """Reconstructs global position from spatial coordinates."""
        f = self.geo.get_frame(s)
        pos_world = f['pos'] + w1 * f['n1'] + w2 * f['n2']
        return pos_world

    def _cartesian_to_spatial(self, pos, vel, rpy, drpy):
        """Projects global state onto the path frame."""
        s = self.geo.get_closest_s(pos, s_guess=self.prev_s)
        self.prev_s = s 
        f = self.geo.get_frame(s)
        
        r_vec = pos - f['pos']
        w1 = np.dot(r_vec, f['n1'])
        w2 = np.dot(r_vec, f['n2'])
        
        h = 1 - f['k1'] * w1 - f['k2'] * w2
        h = max(h, 0.01) 
        
        ds = np.dot(vel, f['t']) / h
        dw1 = np.dot(vel, f['n1'])
        dw2 = np.dot(vel, f['n2'])
        
        return np.array([s, w1, w2, ds, dw1, dw2, rpy[0], rpy[1], rpy[2], drpy[0], drpy[1], drpy[2]])

    def reset(self):
        self.prev_s = 0.0
        self.reset_mpc_solver()

    def episode_reset(self):
        self.reset()

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return False

    def episode_callback(self):
        if len(self.control_log['timestamps']) > 0:
            self.plot_all_diagnostics()
        return

    def _log_control_step(self, x_spatial: np.ndarray, u_opt: np.ndarray, solver_status: int):
        self.step_count += 1
        elapsed_time = (datetime.now() - self.episode_start_time).total_seconds()
        
        self.control_log['timestamps'].append(elapsed_time)
        self.control_log['phi_c'].append(float(u_opt[0]))
        self.control_log['theta_c'].append(float(u_opt[1]))
        self.control_log['psi_c'].append(float(u_opt[2]))
        self.control_log['thrust_c'].append(float(u_opt[3]))
        self.control_log['solver_status'].append(int(solver_status))
        self.control_log['s'].append(float(x_spatial[0]))
        self.control_log['w1'].append(float(x_spatial[1]))
        self.control_log['w2'].append(float(x_spatial[2]))
        self.control_log['ds'].append(float(x_spatial[3]))
        
        if self.debug and self.step_count % 10 == 0:
            print(f"[Step {self.step_count}] t={elapsed_time:.3f}s | "
                  f"s={x_spatial[0]:.2f} w1={x_spatial[1]:+.4f} w2={x_spatial[2]:+.4f} | "
                  f"Status={solver_status}")

    def save_control_log(self, filepath: str = None):
        if filepath is None:
            timestamp = self.episode_start_time.strftime("%Y%m%d_%H%M%S")
            filepath = f"control_log_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.control_log, f, indent=2)
        return filepath

    def plot_control_values(self, figsize=(16, 10), save_path: str = None):
        if len(self.control_log['timestamps']) == 0:
            return
        
        t = np.array(self.control_log['timestamps'])
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('MPC Control Values & State Feedback', fontsize=16, fontweight='bold')
        
        ax = axes[0, 0]
        ax.plot(t, self.control_log['phi_c'], 'b-', label='φ_c')
        ax.set_ylabel('Roll (rad)')
        ax.legend()
        
        ax = axes[0, 1]
        ax.plot(t, self.control_log['theta_c'], 'g-', label='θ_c')
        ax.set_ylabel('Pitch (rad)')
        ax.legend()
        
        ax = axes[1, 0]
        ax.plot(t, self.control_log['thrust_c'], 'r-', label='Thrust')
        ax.set_ylabel('Thrust (N)')
        ax.legend()
        
        ax = axes[1, 1]
        ax.plot(t, self.control_log['psi_c'], 'm-', label='ψ_c')
        ax.set_ylabel('Yaw (rad)')
        ax.legend()
        
        ax = axes[2, 0]
        ax.plot(t, self.control_log['s'], 'c-', label='s')
        ax.set_ylabel('Progress (m)')
        ax.legend()
        
        ax = axes[2, 1]
        ax.plot(t, self.control_log['w1'], 'orange', label='w1')
        ax.plot(t, self.control_log['w2'], 'purple', label='w2')
        ax.axhline(y=0.5, color='r', linestyle='--')
        ax.axhline(y=-0.5, color='r', linestyle='--')
        ax.set_ylabel('Deviation (m)')
        ax.legend()
        
        plt.tight_layout()
        if save_path is None:
            timestamp = self.episode_start_time.strftime("%Y%m%d_%H%M%S")
            save_path = f"control_plot_{timestamp}.png"
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_solver_status(self, save_path: str = None):
        if len(self.control_log['timestamps']) == 0:
            return
        t = np.array(self.control_log['timestamps'])
        status = np.array(self.control_log['solver_status'])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(t, status, c=['g' if s==0 else 'r' for s in status])
        ax.set_title('Solver Status (0=Success)')
        if save_path is None:
            save_path = f"solver_status.png"
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    def plot_all_diagnostics(self, save_dir: str = None):
        if save_dir is None:
            timestamp = self.episode_start_time.strftime("%Y%m%d_%H%M%S")
            save_dir = f"mpc_debug/mpc_diagnostics_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        self.save_control_log(os.path.join(save_dir, "control_log.json"))
        self.plot_control_values(save_path=os.path.join(save_dir, "control_values.png"))
        self.plot_solver_status(save_path=os.path.join(save_dir, "solver_status.png"))
        return save_dir