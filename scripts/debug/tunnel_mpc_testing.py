import os
# Constraint from drone_models: SCIPY_ARRAY_API must be set before importing scipy
os.environ["SCIPY_ARRAY_API"] = "1"

import shutil
import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3 # Added for 3D visualization
import toml
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# --- USER LIBRARY IMPORTS ---
# We wrap these in try-except blocks to allow the script to be tested 
# standalone if the 'drone_models' package path isn't set in the shell.
try:
    from drone_models.core import load_params
except ImportError:
    print("[WARN] Could not import 'drone_models'. Using mock parameter loader.")
    def load_params(model_type, config_node):
        """
        Mock parameter loader using the physical parameters provided by the user.
        Defaulting to 'cf21B_500' if specific config not requested.
        """
        # Dictionary of available models based on user input
        models = {
            "cf21B_500": {
                "mass": 0.04338,
                "gravity_vec": np.array([0.0, 0.0, -9.81]),
                "J": np.diag([25e-6, 28e-6, 49e-6]),
                "L": 0.035355, # 'L' in config
                "thrust2torque": 0.00593893393599368, # 'thrust2torque'
                "thrust_max_per_motor": 0.2,
            },
            "cf2x_L250": {
                "mass": 0.033,
                "gravity_vec": np.array([0.0, 0.0, -9.81]),
                "J": np.diag([16.8e-6, 16.8e-6, 29.8e-6]),
                "L": 0.03253,
                # Calculated cm ~ rpm2torque / rpm2thrust (7.94e-12 / 3.16e-10)
                "thrust2torque": 0.0251, 
                "thrust_max_per_motor": 0.1125,
            },
            "cf2x_P250": {
                "mass": 0.03454,
                "gravity_vec": np.array([0.0, 0.0, -9.81]),
                "J": np.diag([14e-6, 14e-6, 21.7e-6]),
                "L": 0.03253,
                # Calculated cm ~ 7.94e-12 / 5.79e-10
                "thrust2torque": 0.0137,
                "thrust_max_per_motor": 0.1125,
            }
        }
        
        # Default to cf21B_500 if unknown or testing
        selected_model = models["cf21B_500"]
        return selected_model

# ==============================================================================
# 1. GEOMETRY ENGINE (Parallel Transport Frame)
# ==============================================================================
class GeometryEngine:
    def __init__(self, gates_pos, gates_rpy, start_pos):
        self.gates_pos = np.asarray(gates_pos)
        self.gates_rpy = np.asarray(gates_rpy)
        self.start_pos = np.asarray(start_pos)

        # 1. Calculate Gate Orientations
        # Convert roll-pitch-yaw to rotation matrices
        rot = R.from_euler("xyz", self.gates_rpy)
        self.Rm = rot.as_matrix()
        self.gate_normals = self.Rm[:, :, 0] # X-axis is forward through gate

        # 2. Prepare Waypoints & Tangents
        # We assume the start tangent points towards the first gate
        # We calculate the vector from start_pos to the first gate to establish the initial direction.
        start_tan = self.gates_pos[0] - self.start_pos
        norm_tan = np.linalg.norm(start_tan)
        print(f"Start tangent vector (unnormalized): {start_tan}, norm: {norm_tan}")
        if norm_tan > 1e-6:
            start_tan = start_tan / norm_tan
        else:
            start_tan = np.array([1.0, 0.0, 0.0])

        # We stack the start and gate positions to create a list of "knots" (control points).
        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        
        # Tangents at each waypoint
        self.tangents = np.vstack((start_tan, self.gate_normals))

        # 3. Parameterize Path (Knots)
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]

        # 4. Create Spline
        # Generates a C2-continuous curve. Crucial: We provide tangents so the drone flies straight through gates.
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # 5. Compute Dense Frame for Lookups
        self.pt_frame = self._generate_parallel_transport_frame(num_points=2000)

    def _generate_parallel_transport_frame(self, num_points=2000):
        
        # Evaluate parameter values along the path
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]

        frames = {
            "s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []
        }

        # Initial Frame Setup
        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)
        g_vec = np.array([0, 0, -1]) # World Down
        
        # Gram-Schmidt for initial n2 (projection of gravity)
        cross_chk = np.cross(t0, g_vec)
        
        
        if np.linalg.norm(cross_chk) < 1e-3:
            # Singularity: Path is vertical. Arbitrary n2.
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = g_vec - np.dot(g_vec, t0) * t0 # We take component orthogonal to t0
        
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0) # This gives us another orthogonal vector

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        # Propagate frame
        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            
            # 2nd derivative (curvature vector direction)
            k_vec = self.spline(s, 2) 
            
            # Project curvature onto normal plane
            k1 = -np.dot(k_vec, curr_n1)
            k2 = -np.dot(k_vec, curr_n2)

            # This is the main part of the Parallel Transport algorithm mentioned in the paper
            # Parallel Transport Propagation (Rotation minimizing)
            next_n1 = curr_n1 + (k1 * curr_t) * ds
            next_n2 = curr_n2 + (k2 * curr_t) * ds

            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i+1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

            # Re-orthogonalize
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
        # Find index via binary search
        idx = np.searchsorted(self.pt_frame['s'], s_query) - 1
        idx = np.clip(idx, 0, len(self.pt_frame['s'])-1)
        return {
            'k1': self.pt_frame['k1'][idx], 
            'k2': self.pt_frame['k2'][idx],
            't': self.pt_frame['t'][idx], 
            'n1': self.pt_frame['n1'][idx],
            'n2': self.pt_frame['n2'][idx], 
            'pos': self.pt_frame['pos'][idx]
        }

# ==============================================================================
# 2. ACADOS MODEL DEFINITION (Using Loaded Parameters)
# ==============================================================================
def export_spatial_drone_model(drone_params: dict):
    model = AcadosModel()
    model.name = 'spatial_drone'

    # ---------------------------------------------------------
    # 1. Extract Parameters Safely
    # ---------------------------------------------------------
    mass = drone_params["mass"]
    
    # Gravity: Handle vector or scalar
    g_raw = drone_params.get("gravity_vec", 9.81)
    print(f"[INFO] Using gravity vector = {g_raw}")
    if hasattr(g_raw, "__len__"):
        g = abs(g_raw[2]) # Assume NED or ENU, g is magnitude approx
    else:
        g = float(g_raw)

    # Inertia: Handle 3x3 Matrix or list
    J_val = drone_params["J"]
    
    if hasattr(J_val, "shape") and J_val.shape == (3,3):
        Ixx, Iyy, Izz = J_val[0,0], J_val[1,1], J_val[2,2]
    else:
        # Fallback if it's a list/array
        Ixx, Iyy, Izz = J_val[0], J_val[1], J_val[2]

    # Geometry: Handle missing keys with fallbacks
    # 'L' in user config is arm_length
    arm_length = drone_params.get("L", 0.25) # L is the arm length
    print(f"[INFO] Using arm_length = {arm_length} m")
    
    # 'thrust2torque' in user config is 'cm'
    
    Cm = drone_params.get("thrust2torque", 0.01)
    print(f"[INFO] Using thrust2torque (Cm) = {Cm} Nm/N")

    # ---------------------------------------------------------
    # 2. Define CasADi Symbolics
    # ---------------------------------------------------------
    # Spatial States
    s = ca.SX.sym('s') # Progress along the center-line (meters
    w1 = ca.SX.sym('w1'); w2 = ca.SX.sym('w2') # Lateral & Vertical deviations (meters)
    ds = ca.SX.sym('ds') # Progress rate along center-line (m/s)
    
    dw1 = ca.SX.sym('dw1'); dw2 = ca.SX.sym('dw2') # Lateral & Vertical deviation rates (m/s)
    
    # Attitude States (Euler ZYX)
    phi = ca.SX.sym('phi'); theta = ca.SX.sym('theta'); psi = ca.SX.sym('psi')
    
    # Body Rates
    # rotation rates in body frame
    p = ca.SX.sym('p'); q = ca.SX.sym('q'); r = ca.SX.sym('r')
    
    # Full State Vector -> These describe the drone's state in spatial frame
    x = ca.vertcat(s, w1, w2, ds, dw1, dw2, phi, theta, psi, p, q, r)


    # Controls (Motor Thrusts)
    f1 = ca.SX.sym('f1'); f2 = ca.SX.sym('f2'); f3 = ca.SX.sym('f3'); f4 = ca.SX.sym('f4')
    
    # full Control Vector
    # These are 
    u = ca.vertcat(f1, f2, f3, f4)

    # Online Parameters (Path Data)
    k1 = ca.SX.sym('k1'); k2 = ca.SX.sym('k2')
    tx = ca.SX.sym('tx'); ty = ca.SX.sym('ty'); tz = ca.SX.sym('tz')
    n1x = ca.SX.sym('n1x'); n1y = ca.SX.sym('n1y'); n1z = ca.SX.sym('n1z')
    n2x = ca.SX.sym('n2x'); n2y = ca.SX.sym('n2y'); n2z = ca.SX.sym('n2z')
    param = ca.vertcat(k1, k2, tx, ty, tz, n1x, n1y, n1z, n2x, n2y, n2z)

    # ---------------------------------------------------------
    # 3. Dynamics (Physics + Reformulation)
    # ---------------------------------------------------------
    
    # Rotation Matrix (Body -> Inertial, ZYX convention)
    R_z = ca.vertcat(ca.horzcat(ca.cos(psi), -ca.sin(psi), 0), 
                     ca.horzcat(ca.sin(psi), ca.cos(psi), 0), 
                     ca.horzcat(0, 0, 1))
    R_y = ca.vertcat(ca.horzcat(ca.cos(theta), 0, ca.sin(theta)), 
                     ca.horzcat(0, 1, 0), 
                     ca.horzcat(-ca.sin(theta), 0, ca.cos(theta)))
    R_x = ca.vertcat(ca.horzcat(1, 0, 0), 
                     ca.horzcat(0, ca.cos(phi), -ca.sin(phi)), 
                     ca.horzcat(0, ca.sin(phi), ca.cos(phi)))
    R_mat = ca.mtimes(R_z, ca.mtimes(R_y, R_x))

    # Forces & Torques (Quad X Configuration)
    F_total = f1 + f2 + f3 + f4
    tau_x = (arm_length / 1.4142) * (f1 + f4 - f2 - f3)
    tau_y = (arm_length / 1.4142) * (f1 + f2 - f3 - f4)
    tau_z = Cm * (f1 - f2 + f3 - f4)

    # Inertial Linear Acceleration
    acc_inertial = ca.vertcat(0, 0, g) - (1/mass) * ca.mtimes(R_mat, ca.vertcat(0, 0, F_total))
    
    # Projections onto Spatial Frame
    t_vec = ca.vertcat(tx, ty, tz)
    n1_vec = ca.vertcat(n1x, n1y, n1z)
    n2_vec = ca.vertcat(n2x, n2y, n2z)
    
    acc_t = ca.dot(acc_inertial, t_vec)
    acc_n1 = ca.dot(acc_inertial, n1_vec)
    acc_n2 = ca.dot(acc_inertial, n2_vec)

    # Reformulation Terms
    # h is the scale factor relating path arc-length to state progress
    h = 1 - k1*w1 - k2*w2
    
    # Second Derivatives (State Dynamics)
    dds = acc_t / h 
    ddw1 = acc_n1 - k1 * ds**2
    ddw2 = acc_n2 - k2 * ds**2

    # Angular Dynamics (Euler Rates from Body Rates)
    d_phi = p + q*ca.sin(phi)*ca.tan(theta) + r*ca.cos(phi)*ca.tan(theta)
    d_theta = q*ca.cos(phi) - r*ca.sin(phi)
    d_psi = q*ca.sin(phi)/ca.cos(theta) + r*ca.cos(phi)/ca.cos(theta)
    
    # Angular Acceleration (Euler's Equations)
    dp = (tau_x - (Izz - Iyy)*q*r) / Ixx
    dq = (tau_y - (Ixx - Izz)*p*r) / Iyy
    dr = (tau_z - (Iyy - Ixx)*p*q) / Izz

    # Complete Explicit Dynamics
    f_expl = ca.vertcat(ds * h, dw1, dw2, dds, ddw1, ddw2, d_phi, d_theta, d_psi, dp, dq, dr)
    
    model.f_impl_expr = x - x 
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = param
    
    return model

# ==============================================================================
# 3. SPATIAL MPC SOLVER CLASS
# ==============================================================================
class SpatialMPC:  # noqa: D101
    def __init__(self, N=20, Tf=1.0, max_thrust=None, tunnel_radius=0.4, safety_margin=0.05, v_target=5.0):  # noqa: ANN001, D107
        self.N = N
        self.Tf = Tf
        self.tunnel_r = tunnel_radius
        self.safety_margin = safety_margin
        self.v_target = v_target 
        
        self.safe_tunnel_r = max(0.01, self.tunnel_r - self.safety_margin)
        
        self.code_export_dir = 'c_generated_code'
        if os.path.exists(self.code_export_dir): 
            try:
                shutil.rmtree(self.code_export_dir)
            except OSError:
                pass 
                
        self.solver = self._create_solver()
        
        # If max_thrust was not passed explicitly, assume limit from the loaded drone model
        if max_thrust is None:
            # Approx: 4 motors * max_per_motor
            if "thrust_max_per_motor" in self.drone_params:
                self.max_thrust = self.drone_params["thrust_max_per_motor"] * 4.0
            else:
                self.max_thrust = 1.0 # Fallback for nanodrone
        else:
            self.max_thrust = max_thrust

        # Re-update the solver bounds now that we have the specific max_thrust
        self._update_thrust_bounds()

    def _create_solver(self):  # noqa: ANN202
        # 1. Parameter Loading Logic
        if 'config' in globals():
            print("[INFO] Loading parameters from global config.")
            self.drone_params = load_params("so_rpy", globals()['config'].sim.drone_model)
        else:
            print("[INFO] Config not found, using default drone parameters (cf21B_500).")
            self.drone_params = load_params("so_rpy", "cf21B_500")

        # 2. Export Model
        model = export_spatial_drone_model(self.drone_params)
        
        # 3. Define OCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        
        ocp.parameter_values = np.zeros(11)
        
        # 4. Cost Function (Linear Least Squares)
        Q_diag = np.array([0.0, 50.0, 50.0, 10.0, 5.0, 5.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
        R_diag = np.array([1.0, 1.0, 1.0, 1.0]) * 0.1
        
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        ocp.cost.W = np.diag(np.concatenate([Q_diag, R_diag]))
        ocp.cost.W_e = np.diag(Q_diag) * 10.0 
        
        ocp.cost.Vx = np.zeros((16, 12))
        ocp.cost.Vx[:12, :] = np.eye(12)
        ocp.cost.Vu = np.zeros((16, 4))
        ocp.cost.Vu[12:, :] = np.eye(4)
        ocp.cost.Vx_e = np.eye(12)
        
        ocp.cost.yref = np.zeros(16)
        ocp.cost.yref_e = np.zeros(12)

        # 5. Constraints
        ocp.constraints.x0 = np.zeros(12)
        ocp.constraints.lbu = np.zeros(4)
        ocp.constraints.ubu = np.full(4, 20.0) 
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        
        ocp.constraints.lbx = np.array([-self.safe_tunnel_r, -self.safe_tunnel_r, -0.6, -0.6])
        ocp.constraints.ubx = np.array([ self.safe_tunnel_r,  self.safe_tunnel_r,  0.6,  0.6])
        ocp.constraints.idxbx = np.array([1, 2, 6, 7]) 

        # 6. Solver Options
        ocp.solver_options.tf = self.Tf
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.levenberg_marquardt = 1e-3
        
        ocp.code_export_directory = self.code_export_dir
        return AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    def _update_thrust_bounds(self):
        for i in range(self.N):
            self.solver.set(i, "ubu", np.full(4, self.max_thrust/4.0)) 

    def solve(self, x0, geometry_engine):
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        
        s_curr = x0[0]
        v_est = max(x0[3], 1.0)
        dt_step = self.Tf / self.N
        
        for k in range(self.N):
            s_pred = s_curr + v_est * k * dt_step
            frame = geometry_engine.get_frame_at_s(s_pred)
            p_val = np.concatenate([
                [frame['k1'], frame['k2']], 
                frame['t'], frame['n1'], frame['n2']
            ])
            self.solver.set(k, "p", p_val)
            
            # Target hover thrust = mg/4 roughly (for cost regularization)
            hover_thrust = (self.drone_params['mass'] * 9.81) / 4.0
            yref = np.zeros(16)
            yref[3] = self.v_target
            yref[12:] = hover_thrust
            self.solver.set(k, "yref", yref)

        s_end = s_curr + v_est * self.N * dt_step
        frame_e = geometry_engine.get_frame_at_s(s_end)
        p_val_e = np.concatenate([
            [frame_e['k1'], frame_e['k2']], 
            frame_e['t'], frame_e['n1'], frame_e['n2']
        ])
        self.solver.set(self.N, "p", p_val_e)
        
        yref_e = np.zeros(12)
        yref_e[3] = self.v_target
        self.solver.set(self.N, "yref", yref_e)

        status = self.solver.solve()
        u_opt = self.solver.get(0, "u")
        return status, u_opt

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
def load_track_from_toml(filepath):
    # Dummy loader if file doesn't exist
    if not os.path.exists(filepath):
        print(f"[WARN] {filepath} not found. Using default track.")
        return [[5,0,2], [10,5,2], [5,10,2], [0,5,2]], [[0,0,0]]*4, [0,0,2]
        
    with open(filepath, "r") as f:
        data = toml.load(f)
        
    gates_raw = data["env"]["track"]["gates"]
    gates_pos = np.array([g["pos"] for g in gates_raw], dtype=float)
    gates_rpy = np.array([g.get("rpy", [0, 0, 0]) for g in gates_raw], dtype=float)
    start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=float)
    return gates_pos, gates_rpy, start_pos

if __name__ == "__main__":
    # Load Track
    toml_path = "config/level2_noObstacle.toml"
    gates_pos, gates_rpy, start_pos = load_track_from_toml(toml_path)

    # Initialize Geometry
    geo = GeometryEngine(gates_pos, gates_rpy, start_pos)
    
    # Initialize MPC
    mpc = SpatialMPC(N=20, Tf=1.0, tunnel_radius=0.5, safety_margin=0.05, v_target=4.0)

    # Simulation Setup
    x_curr = np.zeros(12)
    x_curr[0] = 0.0 # Start at s=0
    x_curr[3] = 1.0 # Initial velocity guess
    
    history_x = []
    history_u = []
    solve_times = []
    
    # Gate Timing Logic
    gate_s_locs = geo.s_knots[1:] 
    gate_passed = [False] * len(gate_s_locs)
    gate_metrics = [] 

    print("Starting Simulation...")
    print(f"Drone Mass: {mpc.drone_params['mass']} kg")
    print(f"Track Length: {geo.total_length:.2f}m")
    print(f"Number of Gates: {len(gate_s_locs)}")
    print(f"Target Velocity: {mpc.v_target} m/s")
    print(f"Safety Margin: {mpc.safety_margin} m")
    print(f"Max Thrust per Motor: {mpc.max_thrust/4.0:.2f} N")
    print(f"Arm Length: {mpc.drone_params.get('L')} m")
    print(f"Thrust to Torque (Cm): {mpc.drone_params.get('thrust2torque')} Nm/N")
    
    
    print("-" * 75)
    print(f"{'Step':<5} | {'s [m]':<8} | {'v [m/s]':<8} | {'w1':<6} | {'w2':<6} | {'Status'}")
    
    sim_steps = 300
    dt_sim = 0.05 
    start_time_total = time.time()
    failed_steps = 0

    for i in range(sim_steps):
        t_start = time.time()
        status, u_opt = mpc.solve(x_curr, geo)
        solve_times.append((time.time() - t_start) * 1000)
        
        if status != 0:
            failed_steps += 1
            if failed_steps > 5:
                print("Too many consecutive failures. Aborting.")
                break
            # Fallback hover for sim if solver fails
            u_opt = np.full(4, (mpc.drone_params['mass']*9.81)/4.0)
        else:
            failed_steps = 0

        # Gate Check
        curr_s = x_curr[0]
        for g_idx, g_s in enumerate(gate_s_locs):
            if not gate_passed[g_idx] and curr_s >= g_s:
                gate_passed[g_idx] = True
                gate_metrics.append((i * dt_sim, x_curr[3]))

        # Simple Integration
        if status == 0:
            x_next = mpc.solver.get(1, "x")
            x_curr = x_next
        else:
            x_curr[0] += x_curr[3] * dt_sim 
        
        history_x.append(x_curr)
        history_u.append(u_opt)
        
        if i % 10 == 0:
            print(f"{i:<5} | {x_curr[0]:<8.2f} | {x_curr[3]:<8.2f} | {x_curr[1]:<6.2f} | {x_curr[2]:<6.2f} | {status}")
            
        if x_curr[0] >= geo.total_length:
            print("-" * 75)
            print(f"FINISHED! Track Length {geo.total_length:.2f}m completed in {i} steps.")
            break

    total_time = time.time() - start_time_total
    sim_flight_time = i * dt_sim

    # --- METRICS CALCULATIONS ---
    hist_x = np.array(history_x)
    hist_u = np.array(history_u)
    
    # 1. Trajectory Reconstruction
    traj_3d = []
    for x in hist_x:
        frame = geo.get_frame_at_s(x[0])
        p_drone = frame['pos'] + frame['n1']*x[1] + frame['n2']*x[2]
        traj_3d.append(p_drone)
    traj_3d = np.array(traj_3d)
    
    # 2. Path Efficiency
    if len(traj_3d) > 1:
        flown_dist = np.sum(np.linalg.norm(np.diff(traj_3d, axis=0), axis=1))
    else:
        flown_dist = 0.0
    
    # 3. Error Metrics
    w1_rmse = np.sqrt(np.mean(hist_x[:, 1]**2))
    w2_rmse = np.sqrt(np.mean(hist_x[:, 2]**2))
    max_dev = np.max(np.abs(hist_x[:, 1:3]))
    
    # 4. Control Metrics
    if len(hist_u) > 1:
        u_diff = np.diff(hist_u, axis=0)
        smoothness = np.mean(np.sum(np.abs(u_diff), axis=1))
    else:
        smoothness = 0.0
    avg_thrust = np.mean(hist_u)

    print("\n" + "="*40)
    print("      DETAILED PERFORMANCE REPORT      ")
    print("="*40)
    print(f"Simulated Flight Time: {sim_flight_time:.2f} s")
    print(f"Total Compute Time:    {total_time:.4f} s")
    print(f"Path Efficiency:       {flown_dist:.2f}m / {geo.total_length:.2f}m")
    print("-" * 40)
    print(f"Avg Speed:             {np.mean(hist_x[:, 3]):.2f} m/s")
    print(f"Max Speed:             {np.max(hist_x[:, 3]):.2f} m/s")
    print("-" * 40)
    print(f"Lateral RMSE (w1):     {w1_rmse:.4f} m")
    print(f"Vertical RMSE (w2):    {w2_rmse:.4f} m")
    print(f"Max Deviation:         {max_dev:.4f} m")
    print("-" * 40)
    print(f"Avg Thrust per Motor:  {avg_thrust:.2f} N")
    print(f"Control Smoothness:    {smoothness:.2f}")
    print(f"Avg Solve Time:        {np.mean(solve_times):.2f} ms")
    print("-" * 40)
    
    if gate_metrics:
        print("GATE SPLITS:")
        print(f"{'Gate':<6} | {'Time [s]':<10} | {'Vel [m/s]':<10}")
        for k, (t, v) in enumerate(gate_metrics):
            print(f"G{k:<5} | {t:<10.2f} | {v:<10.2f}")
    print("="*40)

    # Plots
    fig = plt.figure(figsize=(15, 10))
    
    # Tunnel Deviation
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(hist_x[:,0], hist_x[:,1], label="w1 (Lat)")
    ax1.plot(hist_x[:,0], hist_x[:,2], label="w2 (Vert)")
    ax1.axhline(mpc.tunnel_r, color='r', linestyle='--', label="Physical Wall")
    ax1.axhline(-mpc.tunnel_r, color='r', linestyle='--')
    ax1.set_title(f"Tunnel Errors"); ax1.legend(); ax1.grid(True)

    # Speed Profile
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(hist_x[:,0], hist_x[:,3], 'g-')
    ax2.set_title("Speed Profile"); ax2.grid(True)

    # 3D Trajectory
    ax3 = fig.add_subplot(2, 2, (3, 4), projection='3d')
    s_dense = np.linspace(0, geo.total_length, 200)
    c_3d = np.array([geo.spline(s) for s in s_dense])
    ax3.plot(c_3d[:,0], c_3d[:,1], c_3d[:,2], 'k--', alpha=0.5, label="Center")
    ax3.plot(traj_3d[:,0], traj_3d[:,1], traj_3d[:,2], 'b-', linewidth=2, label="Flown")
    
    # Plot Gates
    for i, g in enumerate(geo.gates_pos): 
        ax3.text(g[0], g[1], g[2], f"G{i}", color='r')
    
    ax3.set_title("3D Trajectory"); ax3.legend()
    plt.tight_layout()
    plt.show()