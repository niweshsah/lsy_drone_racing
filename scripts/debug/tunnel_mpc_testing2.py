# import os
# import shutil
# import time
# import numpy as np
# import casadi as ca
# import toml
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# from scipy.interpolate import CubicHermiteSpline
# from scipy.spatial.transform import Rotation as R
# from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# # IMPORTANT: Set this environment variable for Acados/CasADi compatibility
# os.environ["SCIPY_ARRAY_API"] = "1"

# # ==============================================================================
# # 0. CONFIGURATION & PARAMETER LOADING
# # ==============================================================================

# def load_drone_params(model_name="cf21B_500"):
#     """
#     Returns a dictionary of physical parameters based on the selected model.
#     """
#     print(f"\n[INIT] Loading drone parameters for model: '{model_name}'...")
    
#     models = {
#         "cf2x_L250": {
#             "mass": 0.033, "gravity_vec": [0.0, 0.0, -9.81],
#             "thrust_max_total": 0.1125 * 4, "tau_att": 0.1, "L": 0.03253,
#             "J": np.diag([16.8e-6, 16.8e-6, 29.8e-6])
#         },
#         "cf21B_500": {
#             "mass": 0.04338, "gravity_vec": [0.0, 0.0, -9.81],
#             "thrust_max_total": 0.2 * 4, # 0.8 N
#             "tau_att": 0.1, "L": 0.035355,
#             "J": np.diag([25e-6, 28e-6, 49e-6])
#         }
#     }

#     if model_name not in models:
#         print(f"[WARN] Model '{model_name}' not found. Defaulting to cf21B_500.")
#         return models["cf21B_500"]
    
#     p = models[model_name]
#     # Calculate Max Theoretical Acceleration (used for reference generation)
#     p["max_accel"] = p["thrust_max_total"] / p["mass"]
#     print(f"[INIT] Params: Mass={p['mass']}kg, MaxThrust={p['thrust_max_total']}N, MaxAccel={p['max_accel']:.2f}m/s2")
#     return p

# # ==============================================================================
# # 1. GEOMETRY ENGINE (Spatial Reformulation Logic)
# # ==============================================================================

# class GeometryEngine:
#     def __init__(self, gates_pos, gates_rpy, start_pos):
#         print("\n[GEOMETRY] Initializing Geometry Engine...")
#         self.gates_pos = np.asarray(gates_pos)
#         self.gates_rpy = np.asarray(gates_rpy)
#         self.start_pos = np.asarray(start_pos)

#         # 1. Gate Normals
#         rot = R.from_euler("xyz", self.gates_rpy)
#         self.Rm = rot.as_matrix()
#         self.gate_normals = self.Rm[:, :, 0] # X-axis is forward

#         # 2. Tangents & Waypoints
#         start_tan = self.gates_pos[0] - self.start_pos
#         norm_tan = np.linalg.norm(start_tan)
#         if norm_tan > 1e-6: start_tan /= norm_tan
#         else: start_tan = np.array([1.0, 0.0, 0.0])

#         self.waypoints = np.vstack((self.start_pos, self.gates_pos))
#         self.tangents = np.vstack((start_tan, self.gate_normals))

#         # 3. Spline Parameterization
#         dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
#         self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
#         self.total_length = self.s_knots[-1]
#         print(f"[GEOMETRY] Spline generated. Length: {self.total_length:.4f} m")
        
#         self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)
        
#         # 4. Frame Generation (Parallel Transport)
#         self.pt_frame = self._generate_parallel_transport_frame(num_points=2000)

#     def _generate_parallel_transport_frame(self, num_points):
#         s_eval = np.linspace(0, self.total_length, num_points)
#         ds = s_eval[1] - s_eval[0]
#         frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []}

#         # Initial Frame
#         t0 = self.spline(0, 1); t0 /= np.linalg.norm(t0)
#         g_vec = np.array([0, 0, -1])
#         n2_0 = g_vec - np.dot(g_vec, t0) * t0 
#         if np.linalg.norm(n2_0) < 1e-3: n2_0 = np.cross(t0, np.array([1, 0, 0]))
#         n2_0 /= np.linalg.norm(n2_0)
#         n1_0 = np.cross(n2_0, t0)

#         curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

#         for i, s in enumerate(s_eval):
#             pos = self.spline(s)
#             k_vec = self.spline(s, 2) # Curvature vector
#             k1 = -np.dot(k_vec, curr_n1)
#             k2 = -np.dot(k_vec, curr_n2)

#             next_n1 = curr_n1 + (k1 * curr_t) * ds
#             next_n2 = curr_n2 + (k2 * curr_t) * ds

#             if i < len(s_eval) - 1:
#                 next_t = self.spline(s_eval[i+1], 1)
#                 next_t /= np.linalg.norm(next_t)
#             else:
#                 next_t = curr_t

#             next_n1 = next_n1 - np.dot(next_n1, next_t) * next_t
#             next_n1 /= np.linalg.norm(next_n1)
#             next_n2 = np.cross(next_t, next_n1)

#             frames["pos"].append(pos); frames["t"].append(curr_t)
#             frames["n1"].append(curr_n1); frames["n2"].append(curr_n2)
#             frames["k1"].append(k1); frames["k2"].append(k2)
#             curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

#         for k in frames: frames[k] = np.array(frames[k])
#         return frames

#     def get_frame_at_s(self, s_query):
#         idx = np.searchsorted(self.pt_frame['s'], s_query) - 1
#         idx = np.clip(idx, 0, len(self.pt_frame['s'])-1)
#         return {
#             'k1': self.pt_frame['k1'][idx], 'k2': self.pt_frame['k2'][idx],
#             't': self.pt_frame['t'][idx], 
#             'n1': self.pt_frame['n1'][idx], 'n2': self.pt_frame['n2'][idx], 
#             'pos': self.pt_frame['pos'][idx]
#         }

# # ==============================================================================
# # 2. ACADOS MODEL: Spatial Dynamics
# # ==============================================================================

# def export_spatial_drone_model(drone_params: dict):
#     model = AcadosModel()
#     model.name = 'spatial_drone_attitude'

#     # Parameters
#     mass = drone_params["mass"]
#     tau = drone_params.get("tau_att", 0.1)
#     g_raw = drone_params.get("gravity_vec", [0,0,-9.81])
#     g = abs(g_raw[2]) if hasattr(g_raw, "__len__") else float(g_raw)

#     # States [9]: s, w1, w2, ds, dw1, dw2, phi, theta, psi
#     s, w1, w2 = ca.SX.sym('s'), ca.SX.sym('w1'), ca.SX.sym('w2')
#     ds, dw1, dw2 = ca.SX.sym('ds'), ca.SX.sym('dw1'), ca.SX.sym('dw2')
#     phi, theta, psi = ca.SX.sym('phi'), ca.SX.sym('theta'), ca.SX.sym('psi')
#     x = ca.vertcat(s, w1, w2, ds, dw1, dw2, phi, theta, psi)

#     # Controls [4]: Thrust, Roll_cmd, Pitch_cmd, Yaw_cmd
#     u_thrust = ca.SX.sym('u_thrust')
#     u_roll, u_pitch, u_yaw = ca.SX.sym('u_roll'), ca.SX.sym('u_pitch'), ca.SX.sym('u_yaw')
#     u = ca.vertcat(u_thrust, u_roll, u_pitch, u_yaw)

#     # Online Parameters [11]
#     k1, k2 = ca.SX.sym('k1'), ca.SX.sym('k2')
#     tx, ty, tz = ca.SX.sym('tx'), ca.SX.sym('ty'), ca.SX.sym('tz')
#     n1x, n1y, n1z = ca.SX.sym('n1x'), ca.SX.sym('n1y'), ca.SX.sym('n1z')
#     n2x, n2y, n2z = ca.SX.sym('n2x'), ca.SX.sym('n2y'), ca.SX.sym('n2z')
#     param = ca.vertcat(k1, k2, tx, ty, tz, n1x, n1y, n1z, n2x, n2y, n2z)

#     # Dynamics
#     # Rotation Matrix (ZYX)
#     R_z = ca.vertcat(ca.horzcat(ca.cos(psi), -ca.sin(psi), 0), 
#                      ca.horzcat(ca.sin(psi), ca.cos(psi), 0), ca.horzcat(0, 0, 1))
#     R_y = ca.vertcat(ca.horzcat(ca.cos(theta), 0, ca.sin(theta)), 
#                      ca.horzcat(0, 1, 0), ca.horzcat(-ca.sin(theta), 0, ca.cos(theta)))
#     R_x = ca.vertcat(ca.horzcat(1, 0, 0), 
#                      ca.horzcat(0, ca.cos(phi), -ca.sin(phi)), ca.horzcat(0, ca.sin(phi), ca.cos(phi)))
#     R_mat = ca.mtimes(R_z, ca.mtimes(R_y, R_x))

#     # Accelerations
#     acc_inertial = ca.vertcat(0, 0, g) - (1/mass) * ca.mtimes(R_mat, ca.vertcat(0, 0, u_thrust))
    
#     t_vec = ca.vertcat(tx, ty, tz)
#     n1_vec = ca.vertcat(n1x, n1y, n1z)
#     n2_vec = ca.vertcat(n2x, n2y, n2z)
    
#     acc_t = ca.dot(acc_inertial, t_vec)
#     acc_n1 = ca.dot(acc_inertial, n1_vec)
#     acc_n2 = ca.dot(acc_inertial, n2_vec)

#     # Spatial Reformulation
#     h = 1 - k1*w1 - k2*w2
#     dds = acc_t / h 
#     ddw1 = acc_n1 - k1 * ds**2
#     ddw2 = acc_n2 - k2 * ds**2

#     # Attitude Dynamics (First Order Lag)
#     d_phi = (u_roll - phi) / tau
#     d_theta = (u_pitch - theta) / tau
#     d_psi = (u_yaw - psi) / tau

#     f_expl = ca.vertcat(ds * h, dw1, dw2, dds, ddw1, ddw2, d_phi, d_theta, d_psi)
    
#     model.f_impl_expr = x - x 
#     model.f_expl_expr = f_expl
#     model.x = x; model.u = u; model.p = param
#     return model

# # ==============================================================================
# # 3. SOLVER: TIME-OPTIMAL MPC
# # ==============================================================================

# class SpatialMPC:
#     def __init__(self, N=20, Tf=1.0, tunnel_radius=0.4, v_max_ref=15.0, model_name="cf21B_500"):
#         print(f"\n[MPC] Initializing Time-Optimal MPC (Model: {model_name})")
#         self.N = N; self.Tf = Tf
#         self.tunnel_r = tunnel_radius
#         self.v_max_ref = v_max_ref # Used for reference generation
        
#         # Robustness: Slightly shrink constraint to allow slack to work
#         self.safety_margin = 0.05
#         self.safe_tunnel_r = max(0.01, self.tunnel_r - self.safety_margin)
        
#         self.code_export_dir = 'c_generated_code'
#         if os.path.exists(self.code_export_dir): 
#             try: shutil.rmtree(self.code_export_dir)
#             except OSError: pass 
                
#         self.drone_params = load_drone_params(model_name)
#         self.solver = self._create_solver()
#         print("[MPC] Initialization complete.")

#     def _create_solver(self):
#         model = export_spatial_drone_model(self.drone_params)
#         ocp = AcadosOcp()
#         ocp.model = model
#         ocp.dims.N = self.N
#         ocp.parameter_values = np.zeros(11)

#         # --- Cost Function (UPDATED FOR TIME OPTIMALITY) ---
#         # State Q: [s, w1, w2, ds, dw1, dw2, phi, theta, psi]
        
#         # Q[0] (s): HIGH penalty -> Pulls drone forward to match "infeasible" reference
#         # Q[1,2] (w1, w2): LOW penalty -> Allow drone to use full track width (racing line)
#         Q_diag = np.array([
#             20.0,  # s (Progress) - Key for time optimality
#             0.5,   # w1 (Lateral) - Low, rely on constraints
#             0.5,   # w2 (Vertical) - Low, rely on constraints
#             2.0,   # ds (Speed)
#             1.0, 1.0, # Transverse velocity
#             2.0, 2.0, # Roll/Pitch
#             50.0   # Yaw (Keep aligned with path)
#         ])
        
#         # Control R: [Thrust, Roll, Pitch, Yaw]
#         # Low penalties on control to allow aggressive maneuvering
#         R_diag = np.array([0.1, 1.0, 1.0, 5.0])
        
#         ocp.cost.cost_type = 'LINEAR_LS'
#         ocp.cost.cost_type_e = 'LINEAR_LS'
#         ocp.cost.W = np.diag(np.concatenate([Q_diag, R_diag]))
#         ocp.cost.W_e = np.diag(Q_diag) * 10.0 # Terminal cost
        
#         nx = 9; nu = 4
#         ocp.cost.Vx = np.zeros((nx+nu, nx)); ocp.cost.Vx[:nx, :] = np.eye(nx)
#         ocp.cost.Vu = np.zeros((nx+nu, nu)); ocp.cost.Vu[nx:, :] = np.eye(nu)
#         ocp.cost.Vx_e = np.eye(nx)
#         ocp.cost.yref = np.zeros(nx+nu); ocp.cost.yref_e = np.zeros(nx)

#         # --- Constraints ---
#         ocp.constraints.x0 = np.zeros(nx)

#         max_T = self.drone_params["thrust_max_total"]
#         max_angle = 0.78 # 45 degrees
        
#         # Input Bounds
#         ocp.constraints.lbu = np.array([0.0, -max_angle, -max_angle, -1.5])
#         ocp.constraints.ubu = np.array([max_T, max_angle, max_angle, 1.5])
#         ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        
#         # Tunnel Bounds (Soft Constraints)
#         ocp.constraints.lbx = np.array([-self.safe_tunnel_r, -self.safe_tunnel_r])
#         ocp.constraints.ubx = np.array([ self.safe_tunnel_r,  self.safe_tunnel_r])
#         ocp.constraints.idxbx = np.array([1, 2]) # w1, w2
        
#         # Slacks for robustness
#         ns = 2
#         ocp.constraints.idxsbx = np.array([0, 1]) 
#         ocp.cost.zl = 1000.0 * np.ones((ns,))
#         ocp.cost.zu = 1000.0 * np.ones((ns,))
#         ocp.cost.Zl = 1000.0 * np.ones((ns,))
#         ocp.cost.Zu = 1000.0 * np.ones((ns,))

#         # --- Solver Options ---
#         ocp.solver_options.tf = self.Tf
#         ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
#         ocp.solver_options.nlp_solver_type = 'SQP_RTI'
#         ocp.solver_options.integrator_type = 'ERK'
#         ocp.solver_options.qp_solver_iter_max = 50
#         ocp.solver_options.levenberg_marquardt = 1e-2 
#         ocp.solver_options.print_level = 0
#         ocp.code_export_directory = self.code_export_dir
        
#         return AcadosOcpSolver(ocp, json_file='acados_ocp.json')

#     def solve(self, x0, geometry_engine):
#         # Set initial state
#         self.solver.set(0, "lbx", x0); self.solver.set(0, "ubx", x0)
        
#         s_curr = x0[0]
#         # Current speed guess (prevent division by zero or negative)
#         v_est = max(x0[3], 1.0) 
#         dt = self.Tf / self.N
#         hover_T = self.drone_params['mass'] * 9.81

#         # --- REFERENCE GENERATION LOOP ---
#         for k in range(self.N):
#             # 1. Update Parameters (Path curvature/frame) based on current estimate
#             s_pred = s_curr + v_est * k * dt
#             f = geometry_engine.get_frame_at_s(s_pred)
#             p_val = np.concatenate([[f['k1'], f['k2']], f['t'], f['n1'], f['n2']])
#             self.solver.set(k, "p", p_val)
            
#             # 2. Time-Optimal Reference (Paper Section 2.5)
#             # We set s_ref to be ahead of the drone at max velocity.
#             # This creates a constant "pull" along the track.
#             s_target = s_curr + (k + 1) * dt * self.v_max_ref
            
#             yref = np.zeros(13)
#             yref[0] = s_target      # Target position (Advanced)
#             yref[3] = self.v_max_ref # Target max velocity
#             yref[9] = hover_T       # Feedforward thrust
#             self.solver.set(k, "yref", yref)

#         # Terminal Node
#         s_end = s_curr + v_est * self.N * dt
#         f_e = geometry_engine.get_frame_at_s(s_end)
#         p_val_e = np.concatenate([[f_e['k1'], f_e['k2']], f_e['t'], f_e['n1'], f_e['n2']])
#         self.solver.set(self.N, "p", p_val_e)
        
#         yref_e = np.zeros(9)
#         yref_e[0] = s_curr + (self.N + 1) * dt * self.v_max_ref
#         yref_e[3] = self.v_max_ref
#         self.solver.set(self.N, "yref", yref_e)

#         # Solve
#         status = self.solver.solve()
#         return status, self.solver.get(0, "u")

# # ==============================================================================
# # 4. MAIN SIMULATION
# # ==============================================================================

# def dummy_track_loader():
#     """Generates a simple 3D track if TOML file is missing"""
#     gates = [
#         [0, 0, 2], [5, 0, 2], [10, 2, 3], [15, 5, 4], 
#         [10, 10, 5], [5, 12, 4], [0, 10, 2], [-5, 5, 2]
#     ]
#     gates = np.array(gates, dtype=float)
#     rpy = np.zeros((len(gates), 3))
#     start = np.array([-2, 0, 2], dtype=float)
#     return gates, rpy, start

# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print(" TIME-OPTIMAL SPATIAL MPC SIMULATION ")
#     print("="*60)
    
#     # 1. Load Track
#     toml_path = "config/level2_noObstacle.toml"
#     if os.path.exists(toml_path):
#         try:
#             with open(toml_path, "r") as f:
#                 data = toml.load(f)
#             gates_raw = data["env"]["track"]["gates"]
#             gates_pos = np.array([g["pos"] for g in gates_raw], dtype=float)
#             gates_rpy = np.array([g.get("rpy", [0, 0, 0]) for g in gates_raw], dtype=float)
#             start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=float)
#         except Exception as e:
#             print(f"[WARN] Failed to load TOML: {e}. Using dummy track.")
#             gates_pos, gates_rpy, start_pos = dummy_track_loader()
#     else:
#         print("[INFO] No TOML file found. Using dummy track.")
#         gates_pos, gates_rpy, start_pos = dummy_track_loader()

#     # 2. Init Components
#     geo = GeometryEngine(gates_pos, gates_rpy, start_pos)
    
#     # Target reference velocity (set high for time optimality)
#     # The drone will try to reach this but be limited by dynamics constraints
#     mpc = SpatialMPC(
#         N=20, Tf=1.0, 
#         tunnel_radius=0.5, 
#         v_max_ref=20.0, # High 'virtual' speed
#         model_name="cf21B_500"
#     )

#     # 3. Setup Simulation
#     # State: [s, w1, w2, ds, dw1, dw2, phi, theta, psi]
#     x_curr = np.zeros(9)
#     x_curr[3] = 0.5 # Initial small velocity
    
#     history_x = []; history_u = []; solve_times = []
    
#     # Run loop
#     sim_steps = 400
#     dt_sim = 0.05 # 20Hz simulation
    
#     print("\nStarting Sim...")
#     for i in range(sim_steps):
#         t0 = time.time()
        
#         status, u_opt = mpc.solve(x_curr, geo)
        
#         t_solve = (time.time() - t0) * 1000
#         solve_times.append(t_solve)
        
#         if status != 0:
#             print(f"Step {i}: Solver Error {status}")
#             u_opt = np.zeros(4) # Failsafe
#             u_opt[0] = mpc.drone_params['mass'] * 9.81
            
#         # Get next state from Acados integrator (simulate forward)
#         x_curr = mpc.solver.get(1, "x")
        
#         history_x.append(x_curr)
#         history_u.append(u_opt)
        
#         if x_curr[0] > geo.total_length:
#             print(f"Goal Reached at Step {i}!")
#             break

#     # 4. Results & Detailed Report
#     hist_x = np.array(history_x)
#     hist_u = np.array(history_u)
    
#     # Reconstruct 3D path for plotting and metrics
#     traj_3d = []
#     for x in hist_x:
#         f = geo.get_frame_at_s(x[0])
#         pos = f['pos'] + f['n1']*x[1] + f['n2']*x[2]
#         traj_3d.append(pos)
#     traj_3d = np.array(traj_3d)

#     # --- Metrics Calculation ---
#     total_time = len(hist_x) * dt_sim
#     avg_speed = np.mean(hist_x[:, 3])
#     max_speed = np.max(hist_x[:, 3])
    
#     # Deviations
#     w1_error = hist_x[:, 1]
#     w2_error = hist_x[:, 2]
#     rmse_w1 = np.sqrt(np.mean(w1_error**2))
#     rmse_w2 = np.sqrt(np.mean(w2_error**2))
#     max_w1 = np.max(np.abs(w1_error))
#     max_w2 = np.max(np.abs(w2_error))
    
#     # Distance
#     segment_dists = np.linalg.norm(np.diff(traj_3d, axis=0), axis=1)
#     dist_flown = np.sum(segment_dists)
#     dist_ref = geo.total_length

#     # Attitude & Control
#     # Yaw error (State index 8) relative to path frame
#     max_yaw_error = np.degrees(np.max(np.abs(hist_x[:, 8])))
#     avg_thrust = np.mean(hist_u[:, 0])
#     max_roll_pitch = np.degrees(np.max(np.abs(hist_x[:, 6:8]))) # Indices 6, 7 are phi, theta

#     print("\n" + "="*60)
#     print("              DETAILED PERFORMANCE REPORT              ")
#     print("="*60)
#     print(f"TRACK METRICS:")
#     print(f"  Reference Length:   {dist_ref:.2f} m")
#     print(f"  Distance Flown:     {dist_flown:.2f} m (Efficiency: {dist_flown/dist_ref:.2f})")
#     print(f"  Total Flight Time:  {total_time:.2f} s")
#     print(f"  Average Speed:      {avg_speed:.2f} m/s (Max: {max_speed:.2f} m/s)")
    
#     print(f"\nPATH ACCURACY (Spatial Deviations):")
#     print(f"  Lateral (w1) RMSE:  {rmse_w1:.4f} m  (Max: {max_w1:.4f} m)")
#     print(f"  Vertical (w2) RMSE: {rmse_w2:.4f} m  (Max: {max_w2:.4f} m)")
    
#     print(f"\nATTITUDE & CONTROL:")
#     print(f"  Yaw Tracking Max Error: {max_yaw_error:.2f} degrees")
#     print(f"  Average Thrust:         {avg_thrust:.2f} N")
#     print(f"  Max Roll/Pitch Used:    {max_roll_pitch:.2f} degrees")
    
#     print(f"\nCOMPUTATION:")
#     print(f"  Total Solver Time:  {sum(solve_times)/1000:.4f} s")
#     print(f"  Average Step Time:  {np.mean(solve_times):.2f} ms")
#     print("="*60)
    
#     # Visualization
#     fig = plt.figure(figsize=(12, 10))
    
#     # 2D Overhead
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax1.plot(traj_3d[:,0], traj_3d[:,1], 'b-', label="Trajectory")
#     ax1.plot(gates_pos[:,0], gates_pos[:,1], 'rx', label="Gates")
#     ax1.set_title("Top View (XY)")
#     ax1.legend()
#     ax1.axis('equal')
#     ax1.grid(True)
    
#     # Tunnel Deviation
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax2.plot(hist_x[:,0], hist_x[:,1], label="w1 (Lat)")
#     ax2.plot(hist_x[:,0], hist_x[:,2], label="w2 (Vert)")
#     ax2.axhline(mpc.tunnel_r, color='r', linestyle='--', alpha=0.5)
#     ax2.axhline(-mpc.tunnel_r, color='r', linestyle='--', alpha=0.5)
#     ax2.set_title("Tunnel Errors (w1/w2)")
#     ax2.set_xlabel("Progress s (m)")
#     ax2.legend()
#     ax2.grid(True)
    
#     # Speed Profile
#     ax3 = fig.add_subplot(2, 2, 3)
#     ax3.plot(hist_x[:,0], hist_x[:,3], 'g-', linewidth=2)
#     ax3.set_title("Longitudinal Velocity (ds)")
#     ax3.set_xlabel("Progress s (m)")
#     ax3.set_ylabel("Speed (m/s)")
#     ax3.grid(True)
    
#     # 3D View
#     ax4 = fig.add_subplot(2, 2, 4, projection='3d')
#     ax4.plot(traj_3d[:,0], traj_3d[:,1], traj_3d[:,2], 'b-', linewidth=2)
#     ax4.scatter(gates_pos[:,0], gates_pos[:,1], gates_pos[:,2], c='r', marker='x')
#     for i, g in enumerate(geo.gates_pos):
#         ax4.text(g[0], g[1], g[2], f"G{i}", color='k', fontsize=8)
#     ax4.set_title("3D Trajectory")
    
#     plt.tight_layout()
#     plt.show()