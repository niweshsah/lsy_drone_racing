import os
import shutil
import time
import numpy as np
import scipy.linalg
import toml
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# --- CONFIGURATION ---
os.environ["SCIPY_ARRAY_API"] = "1"

# ==============================================================================
# 1. PARAMETERS & PHYSICS
# ==============================================================================
def get_drone_params():
    """Returns physical parameters for the drone (based on Paper/Iris+ specs)."""
    return {
        "mass": 0.04338,  # kg (Crazyflie scale for testing, scale up for Iris+)
        "g": 9.81,        # m/s^2
        "thrust_max": 0.60, # N (Total thrust, approx 4x weight for racing)
        "thrust_min": 0.05,
        "tau_att": 0.1,   # Time constant for attitude tracking
    }

# ==============================================================================
# 2. GEOMETRY ENGINE (Spatial Frame Generation)
# ==============================================================================
class GeometryEngine:
    def __init__(self, gates_pos, start_pos):
        self.gates_pos = np.asarray(gates_pos)
        self.start_pos = np.asarray(start_pos)

        # 1. Waypoints: Start -> Gates
        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        
        # 2. Tangents: Heuristic (Point to next gate)
        tangents = []
        for i in range(len(self.waypoints)):
            if i < len(self.waypoints) - 1:
                t = self.waypoints[i+1] - self.waypoints[i]
            else:
                t = self.waypoints[i] - self.waypoints[i-1]
            tangents.append(t / np.linalg.norm(t))
        self.tangents = np.array(tangents)

        # 3. Spline Parameterization
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]

        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)
        
        # 4. Generate Dense Lookup Table (Parallel Transport)
        self.pt_frame = self._generate_parallel_transport_frame()

    def _generate_parallel_transport_frame(self, num_points=2000):
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]
        
        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []}

        # Initial Frame (Gram-Schmidt with Gravity)
        t0 = self.spline(0, 1); t0 /= np.linalg.norm(t0)
        g_vec = np.array([0, 0, -1]) # World Down
        
        # Check singularity (vertical start)
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = g_vec - np.dot(g_vec, t0) * t0
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        for i, s in enumerate(s_eval):
            # Calculate Curvature Vector k
            pos = self.spline(s)
            k_vec = self.spline(s, 2) # Second derivative
            
            # Project k onto normal plane
            k1 = -np.dot(k_vec, curr_n1)
            k2 = -np.dot(k_vec, curr_n2)

            # Propagate Frame (Parallel Transport)
            next_n1 = curr_n1 + (k1 * curr_t) * ds
            next_n2 = curr_n2 + (k2 * curr_t) * ds

            # Re-orthogonalize
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i+1], 1); next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t
            
            next_n1 -= np.dot(next_n1, next_t) * next_t
            next_n1 /= np.linalg.norm(next_n1)
            next_n2 = np.cross(next_t, next_n1)

            # Store
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

# ==============================================================================
# 3. ACADOS MODEL (Spatial Reformulation)
# ==============================================================================
def export_model(params):
    model = AcadosModel()
    model.name = 'spatial_drone'

    # Constants
    m = params['mass']
    g = params['g']
    tau = params['tau_att']

    # --- States (9) ---
    s, w1, w2 = ca.SX.sym('s'), ca.SX.sym('w1'), ca.SX.sym('w2')
    ds, dw1, dw2 = ca.SX.sym('ds'), ca.SX.sym('dw1'), ca.SX.sym('dw2')
    phi, th, psi = ca.SX.sym('phi'), ca.SX.sym('th'), ca.SX.sym('psi')
    
    # 9 state variables
    # s is the path progress
    # w1, w2 are lateral/vertical deviations in the path frame
    # ds, dw1, dw2 are their derivatives
    # phi, th, psi are the drone attitudes (roll, pitch, yaw)
    x = ca.vertcat(s, w1, w2, ds, dw1, dw2, phi, th, psi)

    # --- Controls (4) ---
    phi_c, th_c, psi_c, T_c = ca.SX.sym('phi_c'), ca.SX.sym('th_c'), ca.SX.sym('psi_c'), ca.SX.sym('T_c')
    # phi_c, th_c, psi_c are commanded attitudes
    # T_c is the collective thrust
    u = ca.vertcat(phi_c, th_c, psi_c, T_c)

    # --- Parameters (11) ---
    k1, k2 = ca.SX.sym('k1'), ca.SX.sym('k2')
    # Frame vectors (t, n1, n2) passed as params to project Gravity/Thrust correctly
    t_vec = ca.SX.sym('t', 3)
    n1_vec = ca.SX.sym('n1', 3)
    n2_vec = ca.SX.sym('n2', 3)
    p = ca.vertcat(k1, k2, t_vec, n1_vec, n2_vec)

    # --- Dynamics ---
    # 1. Inertial Acceleration
    # Rotation Body->Inertial (ZYX)
    c_p, s_p = ca.cos(phi), ca.sin(phi)
    c_t, s_t = ca.cos(th), ca.sin(th)
    c_ps, s_ps = ca.cos(psi), ca.sin(psi)

    # This is to get R_b (Body to Inertial)
    R_b = ca.vertcat(
        ca.horzcat(c_t*c_ps, s_p*s_t*c_ps - c_p*s_ps, c_p*s_t*c_ps + s_p*s_ps),
        ca.horzcat(c_t*s_ps, s_p*s_t*s_ps + c_p*c_ps, c_p*s_t*s_ps - s_p*c_ps),
        ca.horzcat(-s_t,     s_p*c_t,                 c_p*c_t)
    )

    # Force: Thrust (Body Z) + Gravity (World -Z)
    acc_inertial = (1/m) * ca.mtimes(R_b, ca.vertcat(0,0,T_c)) - ca.vertcat(0,0,g)

    # 2. Project to Path Frame
    at = ca.dot(acc_inertial, t_vec)
    an1 = ca.dot(acc_inertial, n1_vec)
    an2 = ca.dot(acc_inertial, n2_vec)

    # 3. Spatial Reformulation Equations
    h = 1 - k1*w1 - k2*w2
    dh_dt = -(k1*dw1 + k2*dw2) # assuming k_dot ~ 0 locally for stability

    dds = (at - ds * dh_dt) / h
    ddw1 = an1 - k1 * ds**2 * h
    ddw2 = an2 - k2 * ds**2 * h

    # 4. Attitude Lag
    d_phi = (phi_c - phi) / tau
    d_th = (th_c - th) / tau
    d_psi = (psi_c - psi) / tau

    model.f_expl_expr = ca.vertcat(ds, dw1, dw2, dds, ddw1, ddw2, d_phi, d_th, d_psi)
    model.x = x
    model.u = u
    model.p = p
    
    return model

# ==============================================================================
# 4. ROBUST SPATIAL MPC (With Soft Constraints)
# ==============================================================================
class SpatialMPC:
    def __init__(self, N=20, Tf=1.0, v_target=5.0):
        self.N = N
        self.Tf = Tf
        self.v_target = v_target
        self.params = get_drone_params()
        
        # Cleanup
        if os.path.exists('c_generated_code'): 
            try: shutil.rmtree('c_generated_code')
            except: pass
        if os.path.exists('acados_spatial.json'):
            os.remove('acados_spatial.json')
        
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

    def warm_start(self, x0):
        nx = 9
        hover_T = self.params['mass'] * self.params['g']
        v_guess = max(x0[3], 2.0)
        
        for k in range(self.N + 1):
            x_guess = np.zeros(nx)
            x_guess[0] = x0[0] + v_guess * k * (self.Tf/self.N) 
            x_guess[3] = v_guess 
            
            self.solver.set(k, "x", x_guess)
            if k < self.N:
                self.solver.set(k, "u", np.array([0, 0, 0, hover_T]))

    def solve(self, x_curr, geo):
        self.solver.set(0, "lbx", x_curr)
        self.solver.set(0, "ubx", x_curr)

        s_val = x_curr[0]
        # Lookahead speed: don't look too far if we are stuck
        ds_val = np.clip(x_curr[3], 0.5, 8.0)
        dt = self.Tf / self.N
        hover_T = self.params['mass'] * self.params['g']

        for k in range(self.N):
            s_pred = min(s_val + ds_val * k * dt, geo.total_length)
            f = geo.get_frame(s_pred)
            
            # --- CURVATURE CLAMPING (Safety Feature) ---
            # If curvature > 1.8, h = 1 - k*w can hit 0. We clamp k to +/- 1.5
            k1_safe = np.clip(f['k1'], -1.5, 1.5)
            k2_safe = np.clip(f['k2'], -1.5, 1.5)
            
            p = np.concatenate([[k1_safe, k2_safe], f['t'], f['n1'], f['n2']])
            self.solver.set(k, "p", p)
            
            yref = np.zeros(13)
            yref[3] = self.v_target
            yref[12] = hover_T 
            self.solver.set(k, "yref", yref)

        # Terminal
        s_end = min(s_val + ds_val * self.N * dt, geo.total_length)
        f = geo.get_frame(s_end)
        k1_safe = np.clip(f['k1'], -1.5, 1.5)
        k2_safe = np.clip(f['k2'], -1.5, 1.5)
        p = np.concatenate([[k1_safe, k2_safe], f['t'], f['n1'], f['n2']])
        
        self.solver.set(self.N, "p", p)
        yref_e = np.zeros(9); yref_e[3] = self.v_target
        self.solver.set(self.N, "yref", yref_e)

        status = self.solver.solve()
        u_opt = self.solver.get(0, "u")
        return status, u_opt


# ==============================================================================
# 5. MAIN EXECUTION & ENHANCED REPORTING
# ==============================================================================
def draw_3d_gate(ax, pos, tangent, size=1.0):
    """Helper to draw a square gate in 3D oriented along the path tangent."""
    # 1. Create an orthogonal basis (t, n1, n2)
    t = tangent / np.linalg.norm(tangent)
    # Arbitrary vector to cross with
    tmp = np.array([0, 0, 1]) if np.abs(t[2]) < 0.9 else np.array([1, 0, 0])
    n1 = np.cross(t, tmp)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(t, n1)
    
    # 2. Define corners relative to center
    w = size / 2.0
    # Square corners in the local n1-n2 plane
    c1 = pos + w*n1 + w*n2
    c2 = pos - w*n1 + w*n2
    c3 = pos - w*n1 - w*n2
    c4 = pos + w*n1 - w*n2
    
    # 3. Plot
    corners = np.array([c1, c2, c3, c4, c1]) # Loop back to close
    ax.plot(corners[:,0], corners[:,1], corners[:,2], color='crimson', linewidth=3, alpha=0.8)
    ax.text(pos[0], pos[1], pos[2]+0.6, "GATE", color='red', fontsize=8, ha='center')

if __name__ == "__main__":
    # --- 1. Load or Generate Track ---
    # (Same logic as before)
    if os.path.exists("config/level2_noObstacle.toml"):
        print("[INFO] Loading Level 2 Track from TOML...")
        data = toml.load("config/level2_noObstacle.toml")
        gates_raw = data["env"]["track"]["gates"]
        gates_pos = np.array([g["pos"] for g in gates_raw])
        start_pos = np.array(data["env"]["track"]["drones"][0]["pos"])
    else:
        print("[WARN] TOML not found. Generating COMPLEX HELIX TRACK.")
        theta = np.linspace(0, 4*np.pi, 8)
        gates_pos = np.column_stack((2*np.cos(theta), 2*np.sin(theta), 1 + 0.5*theta))
        start_pos = np.array([2.0, -1.0, 1.0])

    # --- 2. Initialize ---
    geo = GeometryEngine(gates_pos, start_pos)
    mpc = SpatialMPC(N=20, Tf=1.0, v_target=6.0) # N=20 is usually sufficient for NMPC

    x = np.zeros(9)
    x[0] = 0.0
    x[3] = 1.0 

    print("[INFO] Warm Starting...")
    mpc.warm_start(x)

    # --- 3. Sim Loop ---
    history_x = []
    history_u = []
    solve_times = []
    dt_sim = 0.02
    
    print(f"{'Step':<5} | {'s [m]':<7} | {'v [m/s]':<7} | {'Lat Err':<7} | {'Thrust':<6} | {'Status'}")
    
    # Run loop
    for i in range(300):
        t0 = time.time()
        status, u = mpc.solve(x, geo)
        dt_solve = (time.time() - t0) * 1000 # ms
        solve_times.append(dt_solve)

        if status != 0:
            print(f"Solver Fail: {status}")
            u = np.array([0,0,0, mpc.params['mass']*9.81])
        
        # Integration
        dx = mpc.solver.get(1, "x") - x 
        x = x + dx
        
        history_x.append(x)
        history_u.append(u)

        if i % 10 == 0:
            print(f"{i:<5} | {x[0]:<7.2f} | {x[3]:<7.2f} | {x[1]:<7.2f} | {u[3]:<6.2f} | {status}")
        
        if x[0] >= geo.total_length:
            print(f"[INFO] Track Finished at Step {i}.")
            break

    # --- 4. ADVANCED ANALYTICS ---
    H_x = np.array(history_x)
    H_u = np.array(history_u)
    
    # State Extraction
    s_hist = H_x[:, 0]
    w_error = np.linalg.norm(H_x[:, 1:3], axis=1)
    vel_hist = H_x[:, 3]
    att_hist = H_x[:, 6:9] # phi, th, psi
    
    # Input Extraction
    att_cmd_hist = H_u[:, 0:3]
    thrust_hist = H_u[:, 3]

    # Metrics
    rmse = np.sqrt(np.mean(w_error**2))
    max_dev = np.max(w_error)
    avg_v = np.mean(vel_hist)
    
    # Saturation Checks
    u_min = np.array([-0.8, -0.8, -0.8, mpc.params['thrust_min']])
    u_max = np.array([+0.8, +0.8, +0.8, mpc.params['thrust_max']])
    # Check what % of time inputs were within 1% of limits
    sat_count = np.sum(np.isclose(H_u, u_min, atol=0.01) | np.isclose(H_u, u_max, atol=0.01))
    sat_pct = (sat_count / (H_u.size)) * 100

    print("\n" + "="*60)
    print("      SPATIAL MPC: ENGINEERING REPORT      ")
    print("="*60)
    print(f"TRACK STATS:")
    print(f"  Total Length:      {geo.total_length:.2f} m")
    print(f"  Completion Time:   {len(H_x)*dt_sim:.2f} s")
    print(f"  Avg Velocity:      {avg_v:.2f} m/s")
    print(f"  Max Velocity:      {np.max(vel_hist):.2f} m/s")
    print("-" * 60)
    print(f"CONTROLLER ACCURACY:")
    print(f"  RMSE (Path Error): {rmse:.4f} m")
    print(f"  Max Deviation:     {max_dev:.4f} m  [Limit: 0.50m]")
    print(f"  End Progress:      {s_hist[-1]:.2f} / {geo.total_length:.2f} m")
    print("-" * 60)
    print(f"ACTUATOR HEALTH:")
    print(f"  Input Saturation:  {sat_pct:.1f}% (Time spent at limits)")
    print(f"  Max Thrust Used:   {np.max(thrust_hist):.2f} N")
    print(f"  Avg Solve Time:    {np.mean(solve_times):.2f} ms")
    print("="*60)

    # --- 5. DETAILED VISUALIZATION ---
    # --- 5. DETAILED VISUALIZATION ---
    # FIGURE 1: 3D Visualization of Reference vs. Actual
    fig1 = plt.figure(figsize=(12, 10))
    ax3d = fig1.add_subplot(111, projection='3d')

    # --- 1. Plot Reference Path (Ideal Spline) ---
    # We sample the spline densely to get a smooth curve
    s_dense = np.linspace(0, geo.total_length, 500)
    p_ref = np.array([geo.spline(s) for s in s_dense])
    
    ax3d.plot(p_ref[:,0], p_ref[:,1], p_ref[:,2], 
              color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
              label="Reference Path (Ideal)")

    # --- 2. Plot Followed MPC Path (Actual Trajectory) ---
    # Reconstruct Cartesian coordinates from Spatial States (s, w1, w2)
    p_flown = []
    for val in H_x:
        f = geo.get_frame(val[0]) # Get frame at current progress 's'
        # Position = Centerline + n1*w1 + n2*w2
        pos_xyz = f['pos'] + f['n1']*val[1] + f['n2']*val[2]
        p_flown.append(pos_xyz)
    p_flown = np.array(p_flown)

    # Plot the line for continuity
    ax3d.plot(p_flown[:,0], p_flown[:,1], p_flown[:,2], 
              color='royalblue', linewidth=2, alpha=0.9, label="Followed MPC Path")

    # Add Scatter points colored by Velocity to show speed profile
    sc = ax3d.scatter(p_flown[:,0], p_flown[:,1], p_flown[:,2], 
                      c=vel_hist, cmap='plasma', s=20, alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax3d, pad=0.1, shrink=0.5)
    cbar.set_label("Drone Velocity (m/s)")

    # --- 3. Mark Starting Position ---
    start_pt = p_flown[0]
    ax3d.scatter(start_pt[0], start_pt[1], start_pt[2], 
                 color='lime', marker='*', s=400, edgecolors='black', 
                 label='Start Position', zorder=100)
    ax3d.text(start_pt[0], start_pt[1], start_pt[2] + 0.5, 
              "  START", color='green', fontweight='bold', fontsize=12)

    # --- 4. Draw Gates for Context ---
    # for i, g_pos in enumerate(geo.gates_pos):
    #     # Heuristic: use tangent at gate index for orientation
    #     t_gate = geo.tangents[i] if i > 0 else geo.tangents[0]
    #     draw_3d_gate(ax3d, g_pos, t_gate, size=1.2)

    # Final Plot Settings
    ax3d.set_title(f"Trajectory Analysis: Reference vs. Actual\nRMSE: {rmse:.3f}m | Max Dev: {max_dev:.3f}m")
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.legend(loc='upper right')
    
    # Set equal aspect ratio for realistic geometric view
    try:
        ax3d.set_box_aspect((np.ptp(p_flown[:,0]), np.ptp(p_flown[:,1]), np.ptp(p_flown[:,2])))
    except:
        pass # Fallback if numpy version is old

    plt.show()