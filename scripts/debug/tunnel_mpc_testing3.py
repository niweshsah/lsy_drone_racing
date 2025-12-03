import os
os.environ["SCIPY_ARRAY_API"] = "1"

import shutil
import time
import scipy.linalg
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import toml
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

# 
# The spatial reformulation relies on a moving frame (t, n1, n2) along the path. 
# s is progress, w1 is lateral error, w2 is vertical error.

# --- MOCK PARAMETERS (Keep as provided) ---
def load_params(model_type, config_node):
    # Defaulting to 'cf21B_500' logic
    return {
        "mass": 0.04338,
        "gravity_vec": np.array([0.0, 0.0, -9.81]),
        "J": np.diag([25e-6, 28e-6, 49e-6]),
        "L": 0.035355,
        "thrust2torque": 0.00593893393599368,
        "thrust_max": 0.1625,
        "thrust_min": 0.03,
    }

# ==============================================================================
# 1. GEOMETRY ENGINE (Unchanged but validated)
# ==============================================================================
class GeometryEngine:
    def __init__(self, gates_pos, gates_rpy, start_pos):
        self.gates_pos = np.asarray(gates_pos)
        self.gates_rpy = np.asarray(gates_rpy)
        self.start_pos = np.asarray(start_pos)

        rot = R.from_euler("xyz", self.gates_rpy)
        self.Rm = rot.as_matrix()
        self.gate_normals = self.Rm[:, :, 0]
        
        start_tan = self.gates_pos[0] - self.start_pos
        norm_tan = np.linalg.norm(start_tan)
        if norm_tan > 1e-6: start_tan = start_tan / norm_tan
        else: start_tan = np.array([1.0, 0.0, 0.0])

        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        self.tangents = np.vstack((start_tan, self.gate_normals))

        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]

        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)
        self.pt_frame = self._generate_parallel_transport_frame(num_points=2000)

    def _generate_parallel_transport_frame(self, num_points=2000):
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]
        
        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []}

        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)
        # Assuming World Z is UP for the track, Gravity Vector points DOWN
        g_vec = np.array([0, 0, -1]) 
        
        cross_chk = np.cross(t0, g_vec)
        if np.linalg.norm(cross_chk) < 1e-3:
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
                next_t = self.spline(s_eval[i+1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

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
        return {k: self.pt_frame[k][idx] for k in ['k1','k2','t','n1','n2','pos']}

# ==============================================================================
# 2. ACADOS MODEL (Fixing Coordinate System)
# ==============================================================================
def export_spatial_drone_model(drone_params: dict):
    model = AcadosModel()
    model.name = 'spatial_drone'

    mass = drone_params["mass"]
    # Gravity is 9.81 magnitude
    g = 9.81 
    
    tau_phi = 0.1; tau_theta = 0.1; tau_psi = 0.1

    # States
    s, w1, w2 = ca.SX.sym('s'), ca.SX.sym('w1'), ca.SX.sym('w2')
    ds, dw1, dw2 = ca.SX.sym('ds'), ca.SX.sym('dw1'), ca.SX.sym('dw2')
    phi, th, psi = ca.SX.sym('phi'), ca.SX.sym('th'), ca.SX.sym('psi')
    x = ca.vertcat(s, w1, w2, ds, dw1, dw2, phi, th, psi)

    # Inputs
    phi_cmd, th_cmd, psi_cmd, T_cmd = ca.SX.sym('phi_cmd'), ca.SX.sym('th_cmd'), ca.SX.sym('psi_cmd'), ca.SX.sym('T_cmd')
    u = ca.vertcat(phi_cmd, th_cmd, psi_cmd, T_cmd)

    # Parameters
    k1, k2 = ca.SX.sym('k1'), ca.SX.sym('k2')
    tx, ty, tz = ca.SX.sym('tx'), ca.SX.sym('ty'), ca.SX.sym('tz')
    n1x, n1y, n1z = ca.SX.sym('n1x'), ca.SX.sym('n1y'), ca.SX.sym('n1z')
    n2x, n2y, n2z = ca.SX.sym('n2x'), ca.SX.sym('n2y'), ca.SX.sym('n2z')
    p = ca.vertcat(k1, k2, tx, ty, tz, n1x, n1y, n1z, n2x, n2y, n2z)

    # Dynamics (Z-Up Assumption for Track compatibility)
    # R_body (ZYX)
    c_phi, s_phi = ca.cos(phi), ca.sin(phi)
    c_th, s_th = ca.cos(th), ca.sin(th)
    c_psi, s_psi = ca.cos(psi), ca.sin(psi)

    R_body = ca.vertcat(
        ca.horzcat(c_th*c_psi, s_phi*s_th*c_psi - c_phi*s_psi, c_phi*s_th*c_psi + s_phi*s_psi),
        ca.horzcat(c_th*s_psi, s_phi*s_th*s_psi + c_phi*c_psi, c_phi*s_th*s_psi - s_phi*c_psi),
        ca.horzcat(-s_th,      s_phi*c_th,                     c_phi*c_th)
    )

    # ACCELERATION LOGIC:
    # If Track Z is UP (Heights 0->2m), Gravity points DOWN (-Z).
    # Thrust points along Body Z.
    F_vec = ca.vertcat(0, 0, T_cmd) 
    
    # acc = (R * F)/m - [0,0,g]
    acc_inertial = (1/mass) * ca.mtimes(R_body, F_vec) - ca.vertcat(0, 0, g)

    # [cite_start]Spatial Projections [cite: 23]
    t_vec = ca.vertcat(tx, ty, tz)
    n1_vec = ca.vertcat(n1x, n1y, n1z)
    n2_vec = ca.vertcat(n2x, n2y, n2z)
    
    acc_t  = ca.dot(acc_inertial, t_vec)
    acc_n1 = ca.dot(acc_inertial, n1_vec)
    acc_n2 = ca.dot(acc_inertial, n2_vec)

    h = 1 - k1*w1 - k2*w2
    dh_dt = -(k1 * dw1 + k2 * dw2)

    # Reformulated Accelerations
    dds = (acc_t - ds * dh_dt) / h
    ddw1 = acc_n1 - k1 * (ds**2) * h
    ddw2 = acc_n2 - k2 * (ds**2) * h
    
    # Inner loop
    d_phi = (phi_cmd - phi) / tau_phi
    d_th  = (th_cmd - th)  / tau_theta
    d_psi = (psi_cmd - psi) / tau_psi

    f_expl = ca.vertcat(ds, dw1, dw2, dds, ddw1, ddw2, d_phi, d_th, d_psi)
    
    model.f_impl_expr = x - x 
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = p
    
    return model

# ==============================================================================
# 3. SPATIAL MPC (With Warm Start)
# ==============================================================================
class SpatialMPC:
    def __init__(self, N=20, Tf=1.0, max_thrust=None, min_thrust=None, tunnel_radius=0.5, safety_margin=0.05, v_target=5.0):
        self.N = N
        self.Tf = Tf
        self.v_target = v_target
        self.drone_params = load_params("so_rpy", "cf21B_500") # simplified loading
        self.tunnel_r = tunnel_radius
        
        # Cleanup
        if os.path.exists('c_generated_code'): 
            try: shutil.rmtree('c_generated_code')
            except: pass
        
        self.max_thrust = self.drone_params["thrust_max"] * 4.0 if max_thrust is None else max_thrust
        self.min_thrust = self.drone_params["thrust_min"] * 4.0 if min_thrust is None else min_thrust
        
        self.solver = self._create_solver()
        self._update_thrust_bounds()

    def _create_solver(self):
        model = export_spatial_drone_model(self.drone_params)
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf

        nx = model.x.size()[0]
        nu = model.u.size()[0]

        # Cost
        Q = np.diag([10.0, 20.0, 20.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        R = np.diag([10.0, 10.0, 10.0, 0.1])
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        ocp.cost.Vx = np.zeros((nx + nu, nx)); ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((nx + nu, nu)); ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(nx + nu)
        ocp.cost.yref_e = np.zeros(nx)

        # Constraints
        w_lim = self.tunnel_r
        angle_lim = 0.6
        ocp.constraints.idxbx = np.array([1, 2])
        ocp.constraints.lbx = np.array([-w_lim, -w_lim])
        ocp.constraints.ubx = np.array([+w_lim, +w_lim])
        
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-angle_lim, -angle_lim, -angle_lim, self.min_thrust])
        ocp.constraints.ubu = np.array([+angle_lim, +angle_lim, +angle_lim, self.max_thrust])
        ocp.constraints.x0 = np.zeros(nx)

        # Solver Options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.qp_solver_iter_max = 100 # Increased for robustness
        
        ocp.parameter_values = np.array([0,0, 1,0,0, 0,1,0, 0,0,1]) # Default params

        return AcadosOcpSolver(ocp, json_file=f'acados_ocp_{self.N}.json')

    def _update_thrust_bounds(self):
        for i in range(self.N):
            self.solver.set(i, "ubu", np.array([0.6, 0.6, 0.6, self.max_thrust]))

    def warm_start_solver(self, x0, v_guess):
        """CRITICAL: Initialize solver with feasible guess to avoid Status 3"""
        nx = 9
        nu = 4
        hover_thrust = self.drone_params['mass'] * 9.81
        dt = self.Tf / self.N
        
        # Initial guess: Flying forward at v_guess with hover thrust
        u_guess = np.array([0, 0, 0, hover_thrust])
        
        for k in range(self.N + 1):
            x_guess = np.zeros(nx)
            x_guess[0] = x0[0] + v_guess * k * dt # s
            x_guess[3] = v_guess                  # ds
            
            self.solver.set(k, "x", x_guess)
            if k < self.N:
                self.solver.set(k, "u", u_guess)

    def solve(self, x0, geometry_engine):
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        
        s_curr = x0[0]
        v_est = max(x0[3], 0.1)
        dt_step = self.Tf / self.N
        
        nx = 9; nu = 4
        hover_thrust = self.drone_params['mass'] * 9.81
        
        for k in range(self.N):
            s_pred = s_curr + v_est * k * dt_step
            # Clamp s_pred to track length
            s_pred = min(s_pred, geometry_engine.total_length)
            
            frame = geometry_engine.get_frame_at_s(s_pred)
            p_val = np.concatenate([[frame['k1'], frame['k2']], frame['t'], frame['n1'], frame['n2']])
            self.solver.set(k, "p", p_val)
            
            yref = np.zeros(nx + nu)
            yref[3] = self.v_target      # Target ds
            yref[nx+3] = hover_thrust    # Target Thrust
            self.solver.set(k, "yref", yref)
            
        # Terminal
        s_end = min(s_curr + v_est * self.N * dt_step, geometry_engine.total_length)
        f_e = geometry_engine.get_frame_at_s(s_end)
        p_e = np.concatenate([[f_e['k1'], f_e['k2']], f_e['t'], f_e['n1'], f_e['n2']])
        self.solver.set(self.N, "p", p_e)
        
        yref_e = np.zeros(nx)
        yref_e[3] = self.v_target
        self.solver.set(self.N, "yref", yref_e)

        status = self.solver.solve()
        return status, self.solver.get(0, "u")

# ==============================================================================
# 4. RUNNER
# ==============================================================================
if __name__ == "__main__":
    # Mock Track (Straight then up)
    gates_pos = [[5,0,2], [10,2,2], [15,-2,2]]
    gates_rpy = [[0,0,0], [0,0,0], [0,0,0]]
    start_pos = [0,0,0]

    geo = GeometryEngine(gates_pos, gates_rpy, start_pos)
    mpc = SpatialMPC(N=20, Tf=1.0, v_target=4.0)

    # Simulation Init
    x_curr = np.zeros(9)
    x_curr[0] = 0.0 # s
    x_curr[3] = 1.0 # Initial ds (must be non-zero for spatial stability)

    # --- FIX: CALL WARM START BEFORE LOOP ---
    print("[INFO] Warm starting solver...")
    mpc.warm_start_solver(x_curr, v_guess=1.0)

    history_x = []
    
    print(f"{'Step':<5} | {'s':<6} | {'ds':<6} | {'Status'}")
    for i in range(100):
        status, u_opt = mpc.solve(x_curr, geo)
        
        if status != 0:
            print(f"Solver Failed at step {i} with status {status}")
            break
            
        # Integration (Simple Euler for viz)
        x_next = mpc.solver.get(1, "x") 
        x_curr = x_next
        history_x.append(x_curr)
        
        if i % 10 == 0:
            print(f"{i:<5} | {x_curr[0]:<6.2f} | {x_curr[3]:<6.2f} | {status}")
            
        if x_curr[0] > geo.total_length:
            print("Finished Track!")
            break

    # Viz
    hist = np.array(history_x)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(hist[:,0], hist[:,3])
    plt.title("Velocity Profile")
    plt.xlabel("s"); plt.ylabel("ds")
    
    plt.subplot(1,2,2)
    plt.plot(hist[:,0], hist[:,1], label="w1")
    plt.plot(hist[:,0], hist[:,2], label="w2")
    plt.title("Path Deviation")
    plt.legend()
    plt.show()