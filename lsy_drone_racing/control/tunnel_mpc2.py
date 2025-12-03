"""Spatial MPC Controller with Quaternion Support."""

from __future__ import annotations
import os
import shutil
import uuid
import numpy as np
import casadi as ca
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from lsy_drone_racing.control import Controller

# Constraint: SCIPY_ARRAY_API must be set before importing scipy logic
os.environ["SCIPY_ARRAY_API"] = "1"

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ==============================================================================
# 1. HELPER CLASSES (Geometry & Modeling)
# ==============================================================================

class GeometryEngine:
    """Handles Path Parameterization and Parallel Transport Frames."""
    def __init__(self, gates_pos, gates_rpy, start_pos):
        self.gates_pos = np.asarray(gates_pos)
        self.gates_rpy = np.asarray(gates_rpy)
        self.start_pos = np.asarray(start_pos)

        # Safety check for empty arrays
        if len(self.gates_pos) == 0:
            print("[WARN] GeometryEngine received 0 gates. Using dummy path.")
            self.gates_pos = np.array([[1.0, 0.0, 1.0]])
            self.gates_rpy = np.array([[0.0, 0.0, 0.0]])
        
        # Ensure 2D array for rotation conversion
        if self.gates_rpy.ndim == 1:
            self.gates_rpy = self.gates_rpy[np.newaxis, :]
            
        # 1. Calculate Gate Orientations
        rot = R.from_euler("xyz", self.gates_rpy)
        self.Rm = rot.as_matrix()
        self.gate_normals = self.Rm[:, :, 0] # Assume X-axis is the pass-through direction

        # 2. Waypoints & Tangents
        # Assumption: Start tangent points toward first gate
        start_tan = self.gates_pos[0] - self.start_pos
        norm_tan = np.linalg.norm(start_tan)
        start_tan = start_tan / norm_tan if norm_tan > 1e-6 else np.array([1.0, 0.0, 0.0])

        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        self.tangents = np.vstack((start_tan, self.gate_normals))

        # 3. Parameterize Path
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        
        if self.total_length < 0.1: self.total_length = 1.0

        # 4. Spline
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)
        
        # 5. Precompute Dense Frame for lookups
        self.pt_frame = self._generate_parallel_transport_frame()

    def _generate_parallel_transport_frame(self, num_points=2000):
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]

        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []}

        # Initial Frame
        t0 = self.spline(0, 1); t0 /= np.linalg.norm(t0)
        g_vec = np.array([0, 0, -1])
        
        # Twist-free initialization against gravity
        n2_0 = g_vec - np.dot(g_vec, t0) * t0
        if np.linalg.norm(n2_0) < 1e-3: n2_0 = np.cross(t0, np.array([1, 0, 0]))
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            # Curvature vector
            k_vec = self.spline(s, 2)
            k1 = -np.dot(k_vec, curr_n1)
            k2 = -np.dot(k_vec, curr_n2)

            # Parallel Transport Propagation
            next_n1 = curr_n1 + (k1 * curr_t) * ds
            next_n2 = curr_n2 + (k2 * curr_t) * ds

            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i+1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

            # Re-orthogonalize (Gram-Schmidt)
            next_n1 = next_n1 - np.dot(next_n1, next_t) * next_t
            next_n1 /= np.linalg.norm(next_n1)
            next_n2 = np.cross(next_t, next_n1)

            frames["pos"].append(pos); frames["t"].append(curr_t)
            frames["n1"].append(curr_n1); frames["n2"].append(curr_n2)
            frames["k1"].append(k1); frames["k2"].append(k2)
            
            curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        for k in frames: frames[k] = np.array(frames[k])
        return frames

    def get_frame_at_s(self, s_query):
        idx = np.searchsorted(self.pt_frame['s'], s_query) - 1
        idx = np.clip(idx, 0, len(self.pt_frame['s'])-1)
        return {
            'k1': self.pt_frame['k1'][idx], 'k2': self.pt_frame['k2'][idx],
            't': self.pt_frame['t'][idx], 'n1': self.pt_frame['n1'][idx],
            'n2': self.pt_frame['n2'][idx], 'pos': self.pt_frame['pos'][idx]
        }
    
    def find_closest_s(self, pos_inertial):
        dists = np.linalg.norm(self.pt_frame['pos'] - pos_inertial, axis=1)
        idx_min = np.argmin(dists)
        return self.pt_frame['s'][idx_min]


def export_spatial_drone_model(drone_params):
    """Generates the CASADI/Acados model for the drone in spatial coordinates."""
    model = AcadosModel()
    model.name = 'spatial_drone'
    
    mass = drone_params["mass"]
    g = 9.81
    J = drone_params["J"]
    Ixx, Iyy, Izz = J[0,0], J[1,1], J[2,2]
    L = drone_params["arm_length"]
    Cm = drone_params["cm"]

    # --- Symbolics ---
    s = ca.SX.sym('s'); w1 = ca.SX.sym('w1'); w2 = ca.SX.sym('w2')
    ds = ca.SX.sym('ds'); dw1 = ca.SX.sym('dw1'); dw2 = ca.SX.sym('dw2')
    phi = ca.SX.sym('phi'); theta = ca.SX.sym('theta'); psi = ca.SX.sym('psi')
    p = ca.SX.sym('p'); q = ca.SX.sym('q'); r = ca.SX.sym('r')
    
    x = ca.vertcat(s, w1, w2, ds, dw1, dw2, phi, theta, psi, p, q, r)
    f = ca.SX.sym('f', 4)
    u = f
    
    k1 = ca.SX.sym('k1'); k2 = ca.SX.sym('k2')
    t_vec = ca.SX.sym('t', 3); n1_vec = ca.SX.sym('n1', 3); n2_vec = ca.SX.sym('n2', 3)
    param = ca.vertcat(k1, k2, t_vec, n1_vec, n2_vec)

    # --- Dynamics ---
    R_z = ca.vertcat(ca.horzcat(ca.cos(psi), -ca.sin(psi), 0), ca.horzcat(ca.sin(psi), ca.cos(psi), 0), ca.horzcat(0, 0, 1))
    R_y = ca.vertcat(ca.horzcat(ca.cos(theta), 0, ca.sin(theta)), ca.horzcat(0, 1, 0), ca.horzcat(-ca.sin(theta), 0, ca.cos(theta)))
    R_x = ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, ca.cos(phi), -ca.sin(phi)), ca.horzcat(0, ca.sin(phi), ca.cos(phi)))
    R_mat = ca.mtimes(R_z, ca.mtimes(R_y, R_x))

    F_total = ca.sum1(f)
    tau_x = (L/1.4142)*(f[0]+f[3]-f[1]-f[2])
    tau_y = (L/1.4142)*(f[0]+f[1]-f[2]-f[3])
    tau_z = Cm*(f[0]-f[1]+f[2]-f[3])

    acc_inertial = ca.vertcat(0,0,g) - (1/mass) * ca.mtimes(R_mat, ca.vertcat(0,0,F_total))
    
    h = 1 - k1*w1 - k2*w2
    acc_t = ca.dot(acc_inertial, t_vec)
    dds = acc_t / h
    ddw1 = ca.dot(acc_inertial, n1_vec) - k1 * ds**2
    ddw2 = ca.dot(acc_inertial, n2_vec) - k2 * ds**2

    dp = (tau_x - (Izz-Iyy)*q*r)/Ixx
    dq = (tau_y - (Ixx-Izz)*p*r)/Iyy
    dr = (tau_z - (Iyy-Ixx)*p*q)/Izz
    
    dphi = p + q*ca.sin(phi)*ca.tan(theta) + r*ca.cos(phi)*ca.tan(theta)
    dtheta = q*ca.cos(phi) - r*ca.sin(phi)
    dpsi = q*ca.sin(phi)/ca.cos(theta) + r*ca.cos(phi)/ca.cos(theta)

    model.f_expl_expr = ca.vertcat(ds*h, dw1, dw2, dds, ddw1, ddw2, dphi, dtheta, dpsi, dp, dq, dr)
    model.f_impl_expr = x - x
    model.x = x; model.u = u; model.p = param
    return model


def generate_flight_corridor(geo_engine, s_horizon, obstacles, default_r=0.4):
    """Dynamically slices the flight tunnel based on obstacle locations."""
    N = len(s_horizon)
    lb_w1 = np.full(N, -default_r); ub_w1 = np.full(N, default_r)
    lb_w2 = np.full(N, -default_r); ub_w2 = np.full(N, default_r)
    
    if not obstacles: return lb_w1, ub_w1, lb_w2, ub_w2

    for obs in obstacles:
        obs_pos = np.array(obs['pos'])
        obs_r = obs['radius'] + 0.1 # Safety buffer
        
        for k, s_val in enumerate(s_horizon):
            s_clamped = np.clip(s_val, 0, geo_engine.total_length - 1e-3)
            frame = geo_engine.get_frame_at_s(s_clamped)
            
            r_vec = obs_pos - frame['pos']
            w1_obs = np.dot(r_vec, frame['n1'])
            dist_long = np.dot(r_vec, frame['t'])
            
            if abs(dist_long) < obs_r:
                slice_r = np.sqrt(max(0, obs_r**2 - dist_long**2))
                obs_inner = w1_obs - slice_r
                obs_outer = w1_obs + slice_r
                
                if w1_obs > 0: # Obstacle is to the Left
                    new_ub = max(lb_w1[k] + 0.1, obs_inner)
                    if new_ub < ub_w1[k]: ub_w1[k] = new_ub
                else: # Obstacle is to the Right
                    new_lb = min(ub_w1[k] - 0.1, obs_outer)
                    if new_lb > lb_w1[k]: lb_w1[k] = new_lb
                    
    return lb_w1, ub_w1, lb_w2, ub_w2


# ==============================================================================
# 2. CONTROLLER BASE CLASS
# ==============================================================================


# ==============================================================================
# 3. SPATIAL MPC CHILD IMPLEMENTATION
# ==============================================================================

class SpatialMPCController(Controller):
    """
    Controller implementing Near-Time-Optimal Spatial MPC.
    Robustly handles Quaternion inputs from observation dictionary.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        
        # 1. Physics Parameters
        self.drone_params = {
            "mass": 0.04338, 
            "gravity_vec": np.array([0.0, 0.0, -9.81]),
            "J": np.diag([25e-6, 28e-6, 49e-6]),
            "arm_length": 0.035355, 
            "cm": 0.00593893393599368, 
            "thrust_max_per_motor": 0.2
        }
        
        # 2. Geometry Setup - Prioritize data from 'obs' if available
        # This fixes the "ValueError: Expected angles" error by using the raw Quats from obs
        
        gates_pos = None
        gates_rpy = None
        start_pos = obs.get('pos', np.array([0,0,0]))

        # Attempt to get Gates from OBS (Most reliable source based on your log)
        if 'gates_pos' in obs and 'gates_quat' in obs:
            gates_pos = obs['gates_pos']
            gates_quat = obs['gates_quat']
            # Convert Quat (x,y,z,w) -> Euler (x,y,z)
            gates_rpy = R.from_quat(gates_quat).as_euler('xyz')
            print(f"[SpatialMPC] Loaded {len(gates_pos)} gates directly from Observation.")

        # Fallback to Config if Obs failed
        if gates_pos is None:
            track_config = None
            if 'gates' in config: track_config = config
            elif 'env' in config and 'track' in config['env']: track_config = config['env']['track']
            
            if track_config and 'gates' in track_config:
                gates = track_config['gates']
                gates_pos = np.array([g['pos'] for g in gates])
                gates_rpy = np.array([g.get('rpy', [0,0,0]) for g in gates])
        
        # Final safety net
        if gates_pos is None or len(gates_pos) == 0:
            print("[WARN] No gates found. Using dummy gate.")
            gates_pos = np.array([[1.0, 0.0, 1.0]])
            gates_rpy = np.array([[0.0, 0.0, 0.0]])

        self.geo = GeometryEngine(gates_pos, gates_rpy, start_pos)
        
        # 3. Load Obstacles (From Obs or Config)
        self.obstacles = []
        if 'obstacles_pos' in obs:
            # Obs usually only gives pos, assume radius
            for pos in obs['obstacles_pos']:
                self.obstacles.append({'pos': pos, 'radius': 0.5})
            print(f"[SpatialMPC] Loaded {len(self.obstacles)} obstacles from Observation.")
        elif 'obstacles' in config:
            for obs_c in config['obstacles']:
                self.obstacles.append({'pos': obs_c['pos'], 'radius': obs_c.get('radius', 0.5)})

        # 4. Initialize MPC Solver
        self.N = 20
        self.Tf = 1.0
        self.v_target = 3.5
        self.tunnel_r = 0.4
        
        self.solver = self._init_acados_solver()
        self.last_u = np.full(4, (self.drone_params['mass']*9.81)/4.0)

    def _init_acados_solver(self):
        """Initializes Acados OCP solver with unique directory."""
        code_dir = f'c_generated_code_{uuid.uuid4().hex[:8]}'
        if os.path.exists(code_dir): shutil.rmtree(code_dir)
            
        model = export_spatial_drone_model(self.drone_params)
        
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        
        ocp.parameter_values = np.zeros(11)
        
        # Cost
        Q_diag = np.array([0, 50, 50, 10, 5, 5, 10, 10, 1, 0.1, 0.1, 0.1])
        R_diag = np.array([0.1, 0.1, 0.1, 0.1])
        
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = np.diag(np.concatenate([Q_diag, R_diag]))
        ocp.cost.W_e = np.diag(Q_diag) * 10.0 
        
        ocp.cost.Vx = np.zeros((16, 12)); ocp.cost.Vx[:12, :] = np.eye(12)
        ocp.cost.Vu = np.zeros((16, 4)); ocp.cost.Vu[12:, :] = np.eye(4)
        ocp.cost.Vx_e = np.eye(12)
        ocp.cost.yref = np.zeros(16); ocp.cost.yref_e = np.zeros(12)

        # Constraints
        max_thrust = self.drone_params['thrust_max_per_motor'] * 4.0
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.zeros(4)
        ocp.constraints.ubu = np.full(4, max_thrust/4.0)
        
        ocp.constraints.idxbx = np.array([1, 2, 6, 7]) # w1, w2, phi, theta
        ocp.constraints.lbx = np.array([-self.tunnel_r, -self.tunnel_r, -1.0, -1.0])
        ocp.constraints.ubx = np.array([ self.tunnel_r,  self.tunnel_r,  1.0,  1.0])

        ocp.solver_options.tf = self.Tf
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.code_export_directory = code_dir
        
        return AcadosOcpSolver(ocp, json_file=f'{code_dir}/acados_ocp.json')

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """
        Input: obs with 'pos', 'quat', 'vel', 'ang_vel'
        Output: [Thrust, Roll, Pitch, Yaw]
        """
        
        # 1. Parse Observation
        pos = obs.get('pos')
        vel = obs.get('vel')
        ang_vel = obs.get('ang_vel', np.zeros(3))
        
        # Handle Attitude: Obs gives Quaternion, MPC needs Euler (Roll, Pitch, Yaw)
        if 'quat' in obs:
            # Normalize quaternion just in case
            q = obs['quat'] / np.linalg.norm(obs['quat'])
            rpy = R.from_quat(q).as_euler('xyz')
        elif 'rpy' in obs:
            rpy = obs['rpy']
        else:
            rpy = np.zeros(3)

        # 2. Map Inertial -> Spatial State
        s_est = self.geo.find_closest_s(pos)
        frame = self.geo.get_frame_at_s(s_est)
        t, n1, n2 = frame['t'], frame['n1'], frame['n2']
        k1, k2 = frame['k1'], frame['k2']
        
        r_vec = pos - frame['pos']
        w1_val = np.dot(r_vec, n1)
        w2_val = np.dot(r_vec, n2)
        
        h = 1 - k1*w1_val - k2*w2_val
        if h < 0.1: h = 0.1 
        
        v_t = np.dot(vel, t)
        v_n1 = np.dot(vel, n1)
        v_n2 = np.dot(vel, n2)
        
        ds_val = v_t / h
        dw1_val = v_n1
        dw2_val = v_n2
        
        # State: [s, w1, w2, ds, dw1, dw2, phi, theta, psi, p, q, r]
        x0 = np.concatenate([
            [s_est, w1_val, w2_val, ds_val, dw1_val, dw2_val],
            rpy,
            ang_vel
        ])

        # 3. Predict Horizon & Set Constraints
        v_pred = max(ds_val, 1.0)
        s_horizon = s_est + np.arange(self.N) * v_pred * (self.Tf/self.N)
        
        lb_w1, ub_w1, lb_w2, ub_w2 = generate_flight_corridor(
            self.geo, s_horizon, self.obstacles, default_r=self.tunnel_r
        )
        
        # 4. Configure Solver
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        
        hover_thrust = (self.drone_params['mass'] * 9.81) / 4.0
        
        for k in range(self.N):
            # Corridor Constraints
            lbx_k = np.array([lb_w1[k], lb_w2[k], -1.0, -1.0])
            ubx_k = np.array([ub_w1[k], ub_w2[k],  1.0,  1.0])
            self.solver.set(k, "lbx", lbx_k)
            self.solver.set(k, "ubx", ubx_k)
            
            # Parameters
            s_k = s_horizon[k]
            f_k = self.geo.get_frame_at_s(np.clip(s_k, 0, self.geo.total_length-0.1))
            p_val = np.concatenate([[f_k['k1'], f_k['k2']], f_k['t'], f_k['n1'], f_k['n2']])
            self.solver.set(k, "p", p_val)
            
            # Ref
            yref = np.zeros(16)
            yref[3] = self.v_target 
            yref[12:] = hover_thrust
            self.solver.set(k, "yref", yref)
            
        self.solver.set(self.N, "yref", np.zeros(12))
        f_N = self.geo.get_frame_at_s(np.clip(s_horizon[-1], 0, self.geo.total_length-0.1))
        p_val_N = np.concatenate([[f_N['k1'], f_N['k2']], f_N['t'], f_N['n1'], f_N['n2']])
        self.solver.set(self.N, "p", p_val_N)

        # 5. Solve
        status = self.solver.solve()
        
        if status != 0:
            total_thrust = self.drone_params['mass'] * 9.81
            return np.array([total_thrust, 0.0, 0.0, 0.0])

        # 6. Extract Command
        u_opt = self.solver.get(0, "u")
        total_thrust_newtons = np.sum(u_opt)
        
        # Use next state prediction as Attitude Setpoint
        x_next = self.solver.get(1, "x")
        roll_cmd = x_next[6]
        pitch_cmd = x_next[7]
        yaw_cmd = x_next[8]
        
        return np.array([total_thrust_newtons, roll_cmd, pitch_cmd, yaw_cmd])

    def episode_reset(self):
        self.solver.reset()