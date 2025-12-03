import numpy as np
import scipy.linalg
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import os
import shutil
from lsy_drone_racing.control import Controller

# Import the Base Class (assuming it's in a file named base_controller.py or similar context)
# from base_controller import Controller 
# Since I am writing the implementation here, I will treat the class provided in the prompt as the parent.

# ==============================================================================
# 1. HELPER CLASSES (Geometry & MPC)
# ==============================================================================

def get_drone_params():
    return {
        "mass": 0.04338, 
        "g": 9.81,
        "thrust_max": 0.60,
        "thrust_min": 0.05,
        "tau_att": 0.1,
    }

class GeometryEngine:
    def __init__(self, gates_pos, start_pos):
        self.gates_pos = np.asarray(gates_pos)
        self.start_pos = np.asarray(start_pos)

        # 1. Waypoints
        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        
        # 2. Tangents
        tangents = []
        for i in range(len(self.waypoints)):
            if i < len(self.waypoints) - 1:
                t = self.waypoints[i+1] - self.waypoints[i]
            else:
                t = self.waypoints[i] - self.waypoints[i-1]
            norm = np.linalg.norm(t)
            tangents.append(t / norm if norm > 1e-6 else np.array([1,0,0]))
        self.tangents = np.array(tangents)

        # 3. Spline
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)
        
        # 4. Parallel Transport Frame
        self.pt_frame = self._generate_parallel_transport_frame()

    def _generate_parallel_transport_frame(self, num_points=3000):
        # Increased density for better lookup accuracy
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]
        
        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []}

        t0 = self.spline(0, 1); t0 /= np.linalg.norm(t0)
        g_vec = np.array([0, 0, -1]) 
        
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = g_vec - np.dot(g_vec, t0) * t0
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            k_vec = self.spline(s, 2)
            
            k1 = -np.dot(k_vec, curr_n1)
            k2 = -np.dot(k_vec, curr_n2)

            next_n1 = curr_n1 + (k1 * curr_t) * ds
            next_n2 = curr_n2 + (k2 * curr_t) * ds

            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i+1], 1); next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t
            
            next_n1 -= np.dot(next_n1, next_t) * next_t
            next_n1 /= np.linalg.norm(next_n1)
            next_n2 = np.cross(next_t, next_n1)

            frames["pos"].append(pos); frames["t"].append(curr_t)
            frames["n1"].append(curr_n1); frames["n2"].append(curr_n2)
            frames["k1"].append(k1); frames["k2"].append(k2)
            
            curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        for k in frames: frames[k] = np.array(frames[k])
        return frames

    def get_frame(self, s_query):
        idx = np.searchsorted(self.pt_frame['s'], s_query) - 1
        idx = np.clip(idx, 0, len(self.pt_frame['s'])-1)
        return {k: self.pt_frame[k][idx] for k in self.pt_frame if k != 's'}

    def get_closest_s(self, pos_query, s_guess=0.0, window=5.0):
        """Finds the s coordinate closest to the drone position."""
        # Restrict search to a window around s_guess to save time and prevent jumping
        mask = (self.pt_frame['s'] >= s_guess - 1.0) & (self.pt_frame['s'] <= s_guess + window)
        
        # If mask is empty (end of track), look at the end
        if not np.any(mask):
            candidates_pos = self.pt_frame['pos'][-100:]
            candidates_s = self.pt_frame['s'][-100:]
        else:
            candidates_pos = self.pt_frame['pos'][mask]
            candidates_s = self.pt_frame['s'][mask]

        dists = np.linalg.norm(candidates_pos - pos_query, axis=1)
        idx_min = np.argmin(dists)
        return candidates_s[idx_min]

def export_model(params):
    model = AcadosModel()
    model.name = 'spatial_drone'
    m, g, tau = params['mass'], params['g'], params['tau_att']

    # States
    s, w1, w2 = ca.SX.sym('s'), ca.SX.sym('w1'), ca.SX.sym('w2')
    ds, dw1, dw2 = ca.SX.sym('ds'), ca.SX.sym('dw1'), ca.SX.sym('dw2')
    phi, th, psi = ca.SX.sym('phi'), ca.SX.sym('th'), ca.SX.sym('psi')
    x = ca.vertcat(s, w1, w2, ds, dw1, dw2, phi, th, psi)

    # Controls
    phi_c, th_c, psi_c, T_c = ca.SX.sym('phi_c'), ca.SX.sym('th_c'), ca.SX.sym('psi_c'), ca.SX.sym('T_c')
    u = ca.vertcat(phi_c, th_c, psi_c, T_c)

    # Params
    k1, k2 = ca.SX.sym('k1'), ca.SX.sym('k2')
    t_vec = ca.SX.sym('t', 3)
    n1_vec = ca.SX.sym('n1', 3)
    n2_vec = ca.SX.sym('n2', 3)
    p = ca.vertcat(k1, k2, t_vec, n1_vec, n2_vec)

    # Dynamics
    c_p, s_p = ca.cos(phi), ca.sin(phi)
    c_t, s_t = ca.cos(th), ca.sin(th)
    c_ps, s_ps = ca.cos(psi), ca.sin(psi)
    R_b = ca.vertcat(
        ca.horzcat(c_t*c_ps, s_p*s_t*c_ps - c_p*s_ps, c_p*s_t*c_ps + s_p*s_ps),
        ca.horzcat(c_t*s_ps, s_p*s_t*s_ps + c_p*c_ps, c_p*s_t*s_ps - s_p*c_ps),
        ca.horzcat(-s_t,     s_p*c_t,                 c_p*c_t)
    )
    acc_inertial = (1/m) * ca.mtimes(R_b, ca.vertcat(0,0,T_c)) - ca.vertcat(0,0,g)
    
    at = ca.dot(acc_inertial, t_vec)
    an1 = ca.dot(acc_inertial, n1_vec)
    an2 = ca.dot(acc_inertial, n2_vec)

    h = 1 - k1*w1 - k2*w2
    # Approx dh_dt for stability
    dh_dt = -(k1*dw1 + k2*dw2) 

    dds = (at - ds * dh_dt) / h
    ddw1 = an1 - k1 * ds**2 * h
    ddw2 = an2 - k2 * ds**2 * h

    d_phi = (phi_c - phi) / tau
    d_th = (th_c - th) / tau
    d_psi = (psi_c - psi) / tau

    model.f_expl_expr = ca.vertcat(ds, dw1, dw2, dds, ddw1, ddw2, d_phi, d_th, d_psi)
    model.x = x
    model.u = u
    model.p = p
    return model

class SpatialMPC:
    def __init__(self, params, N=20, Tf=1.0):
        self.N = N
        self.Tf = Tf
        self.params = params
        
        # Ensure clean state
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

        nx, nu = 9, 4

        # --- COST CONFIGURATION ---
        # State Weights (Q): Low penalty on s, High on deviations
        # Index:      s    w1    w2    ds    dw1   dw2   att...
        q_diag = [0.0, 20.0, 20.0, 10.0, 5.0,  5.0,  1.0, 1.0, 1.0]
        r_diag = [5.0, 5.0, 5.0, 0.1] # Inputs

        ocp.cost.W = scipy.linalg.block_diag(np.diag(q_diag), np.diag(r_diag))
        ocp.cost.W_e = np.diag(q_diag)
        
        ocp.cost.Vx = np.zeros((nx+nu, nx)); ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((nx+nu, nu)); ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(nx+nu)
        ocp.cost.yref_e = np.zeros(nx)

        # --- CONSTRAINTS ---
        # 1. Inputs (Hard Constraints - Physical limits)
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-0.8, -0.8, -0.8, self.params['thrust_min']])
        ocp.constraints.ubu = np.array([+0.8, +0.8, +0.8, self.params['thrust_max']])
        
        # 2. State/Corridor (SOFT Constraints - The Fix)
        # We define bounds, but allow violation for a cost
        ocp.constraints.idxbx = np.array([1, 2]) # w1, w2
        ocp.constraints.lbx = np.array([-0.5, -0.5]) 
        ocp.constraints.ubx = np.array([+0.5, +0.5])
        
        # SLACK CONFIGURATION
        # We add slack variables (Z) to the bounds on x (bx)
        # J_slack = Zl * slack_lower + Zu * slack_upper + ... (quadratic terms)
        ns = 2 # Number of soft constraints (w1, w2)
        ocp.constraints.idxsbx = np.array([0, 1]) # Slack indices corresponding to idxbx
        
        # High cost for violating the tunnel (1000.0), but FINITE.
        # Linear cost (zl/zu) + Quadratic cost (Zl/Zu)
        BIG_COST = 1000.0
        ocp.cost.zl = BIG_COST * np.ones(ns)
        ocp.cost.zu = BIG_COST * np.ones(ns)
        ocp.cost.Zl = BIG_COST * np.ones(ns)
        ocp.cost.Zu = BIG_COST * np.ones(ns)

        ocp.constraints.x0 = np.zeros(nx)

        # --- SOLVER OPTIONS ---
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        # Increase tolerance slightly to prevent numerical noise crashes
        ocp.solver_options.qp_solver_tol_stat = 1e-4

        # Init Params
        ocp.parameter_values = np.array([0,0, 1,0,0, 0,1,0, 0,0,1])

        return AcadosOcpSolver(ocp, json_file='acados_spatial.json')

# ==============================================================================
# 2. CONTROLLER IMPLEMENTATION
# ==============================================================================

class SpatialMPCController(Controller):
    """
    Spatial MPC implementation inheriting from the base Controller.
    Uses Acados to solve a path-following OCP in Frenet-Serret coordinates.
    """

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Initialization of the controller."""
        
        # 1. Physics & Configuration
        self.params = get_drone_params()
        self.v_target = 6.0  # m/s target speed
        
        # 2. Geometry Setup
        # Assuming config structure based on provided context.
        # If gates are not in config, we might need to look in info.
        gates_list = config.get("env", {}).get("track", {}).get("gates", [])
        if not gates_list and "gates" in info:
            gates_list = info["gates"] # Fallback
            
        gates_pos = [g["pos"] for g in gates_list]
        start_pos = obs["pos"] # Use current pos as start if not in config
        
        self.geo = GeometryEngine(gates_pos, start_pos)
        
        # 3. MPC Setup
        self.N_horizon = 20
        self.mpc = SpatialMPC(self.params, N=self.N_horizon, Tf=1.0)
        
        # 4. Internal State
        self.prev_s = 0.0
        self.last_u = np.array([0.0, 0.0, 0.0, self.params['mass'] * self.params['g']])
        
        # Initial Warm Start
        self.reset_mpc_solver()

    def reset_mpc_solver(self):
        """Resets the MPC solver guess to a forward constant velocity."""
        nx = 9
        hover_T = self.params['mass'] * self.params['g']
        
        for k in range(self.N_horizon + 1):
            x_guess = np.zeros(nx)
            # Assuming starting at s=0
            x_guess[0] = self.v_target * k * (1.0/self.N_horizon)
            x_guess[3] = self.v_target
            
            self.mpc.solver.set(k, "x", x_guess)
            if k < self.N_horizon:
                self.mpc.solver.set(k, "u", np.array([0, 0, 0, hover_T]))
        
        self.prev_s = 0.0

    def compute_control(
        self, obs: dict[str, np.ndarray], info: dict | None = None
    ) -> np.ndarray:
        """Compute the next control action."""
        
        # 1. State Extraction & Conversion
        pos = obs["pos"]
        vel = obs["vel"]
        quat = obs["quat"] # Assuming [x,y,z,w] or [w,x,y,z]
        rot = obs["rpy"] if "rpy" in obs else R.from_quat(quat).as_euler('zyx')[::-1]

        
        # Note: Model expects [phi, theta, psi] (Roll, Pitch, Yaw)
        
        # Map Cartesian -> Spatial
        x_spatial = self._cartesian_to_spatial(pos, vel, rot)
        
        # 2. Update Constraints (Current State)
        self.mpc.solver.set(0, "lbx", x_spatial)
        self.mpc.solver.set(0, "ubx", x_spatial)
        
        # 3. Update Reference & Parameters along Horizon
        s_val = x_spatial[0]
        ds_val = np.clip(x_spatial[3], 1.0, 8.0)
        dt = self.mpc.Tf / self.mpc.N
        hover_T = self.params['mass'] * self.params['g']

        for k in range(self.mpc.N):
            s_pred = min(s_val + ds_val * k * dt, self.geo.total_length)
            f = self.geo.get_frame(s_pred)
            
            # Curvature Clamping for numerical stability
            k1_safe = np.clip(f['k1'], -1.5, 1.5)
            k2_safe = np.clip(f['k2'], -1.5, 1.5)
            
            p = np.concatenate([[k1_safe, k2_safe], f['t'], f['n1'], f['n2']])
            self.mpc.solver.set(k, "p", p)
            
            # Reference: [s, w1, w2, ds, dw1, dw2, phi, th, psi, u1, u2, u3, u4]
            # We only track velocity (idx 3) and hover thrust (idx 12)
            yref = np.zeros(13)
            yref[3] = self.v_target
            yref[12] = hover_T
            self.mpc.solver.set(k, "yref", yref)

        # Terminal Reference
        s_end = min(s_val + ds_val * self.mpc.N * dt, self.geo.total_length)
        f = self.geo.get_frame(s_end)
        p_end = np.concatenate([[f['k1'], f['k2']], f['t'], f['n1'], f['n2']])
        self.mpc.solver.set(self.mpc.N, "p", p_end)
        
        yref_e = np.zeros(9)
        yref_e[3] = self.v_target
        self.mpc.solver.set(self.mpc.N, "yref", yref_e)

        # 4. Solve
        status = self.mpc.solver.solve()
        
        if status != 0:
            # Fallback: simple hover or maintain last valid input
            # print(f"MPC Failed: {status}")
            u_opt = np.array([0, 0, 0, hover_T])
        else:
            u_opt = self.mpc.solver.get(0, "u")
            self.last_u = u_opt

        # 5. Output Formatting
        # MPC outputs: [phi_c, th_c, psi_c, Thrust_c]
        # Return: [roll, pitch, yaw, thrust]
        # Ensure Thrust is scalar
        return np.array([u_opt[0], u_opt[1], u_opt[2], u_opt[3]])

    def _cartesian_to_spatial(self, pos, vel, rpy):
        """Projects global state onto the path frame."""
        # 1. Find s
        s = self.geo.get_closest_s(pos, s_guess=self.prev_s)
        self.prev_s = s # Update internal tracking
        
        # 2. Get Frame
        f = self.geo.get_frame(s)
        
        # 3. Calculate Errors (w1, w2)
        r_vec = pos - f['pos']
        w1 = np.dot(r_vec, f['n1'])
        w2 = np.dot(r_vec, f['n2'])
        
        # 4. Calculate Spatial Velocities (Approximate)
        # In reality, need to account for curvature h factor, 
        # but for state feedback this projection is usually sufficient.
        ds = np.dot(vel, f['t'])
        dw1 = np.dot(vel, f['n1'])
        dw2 = np.dot(vel, f['n2'])
        
        # 5. Attitude
        phi, th, psi = rpy[0], rpy[1], rpy[2]
        
        return np.array([s, w1, w2, ds, dw1, dw2, phi, th, psi])

    def reset(self):
        """Reset internal variables."""
        self.prev_s = 0.0
        self.reset_mpc_solver()

    def episode_reset(self):
        """Reset the controller's internal state and models."""
        self.reset()