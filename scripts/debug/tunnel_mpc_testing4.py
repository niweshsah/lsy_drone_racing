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
    x = ca.vertcat(s, w1, w2, ds, dw1, dw2, phi, th, psi)

    # --- Controls (4) ---
    phi_c, th_c, psi_c, T_c = ca.SX.sym('phi_c'), ca.SX.sym('th_c'), ca.SX.sym('psi_c'), ca.SX.sym('T_c')
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
# 4. MPC SOLVER WRAPPER
# ==============================================================================
class SpatialMPC:
    def __init__(self, N=20, Tf=1.0, v_target=5.0):
        self.N = N
        self.Tf = Tf
        self.v_target = v_target
        self.params = get_drone_params()
        
        # Clean previous builds to ensure recompilation with new weights
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

        # --- COST FUNCTION FIX ---
        nx, nu = 9, 4
        
        # State Weights (Q Matrix)
        # Index:      s    w1    w2    ds    dw1   dw2   phi   th    psi
        # FIX: Set s weight to 0.0 so it doesn't pull drone back to start
        q_diag = [0.0, 50.0, 50.0, 20.0, 1.0,  1.0,  5.0,  5.0,  5.0]
        
        # Input Weights (R Matrix)
        # Index:      phi   th    psi   Thrust
        r_diag = [10.0, 10.0, 10.0, 0.1]

        Q = np.diag(q_diag)
        R = np.diag(r_diag)

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        
        # Set Up Reference Structure
        ocp.cost.Vx = np.zeros((nx+nu, nx)); ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((nx+nu, nu)); ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        
        # Initialize yref (Will be updated in loop, but defaults matter)
        ocp.cost.yref = np.zeros(nx+nu)
        ocp.cost.yref_e = np.zeros(nx)

        # --- CONSTRAINTS ---
        # Inputs
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-0.6, -0.6, -0.6, self.params['thrust_min']])
        ocp.constraints.ubu = np.array([+0.6, +0.6, +0.6, self.params['thrust_max']])
        
        # Corridor (w1, w2)
        ocp.constraints.idxbx = np.array([1, 2])
        ocp.constraints.lbx = np.array([-0.5, -0.5]) 
        ocp.constraints.ubx = np.array([+0.5, +0.5])

        ocp.constraints.x0 = np.zeros(nx)

        # --- SOLVER OPTIONS ---
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        
        # Initialize Params
        ocp.parameter_values = np.array([0,0, 1,0,0, 0,1,0, 0,0,1])

        return AcadosOcpSolver(ocp, json_file='acados_spatial.json')

    def warm_start(self, x0):
        """Initializes solver with a forward-moving guess."""
        nx, nu = 9, 4
        hover_T = self.params['mass'] * self.params['g']
        
        # If x0 has 0 velocity, force a guess to help the optimizer start
        v_guess = max(x0[3], 1.0) 
        
        for k in range(self.N + 1):
            x_guess = np.zeros(nx)
            x_guess[0] = x0[0] + v_guess * k * (self.Tf/self.N) # Forward s
            x_guess[3] = v_guess # Forward ds
            
            self.solver.set(k, "x", x_guess)
            if k < self.N:
                self.solver.set(k, "u", np.array([0, 0, 0, hover_T]))

    def solve(self, x_curr, geo):
        # 1. Update Initial Condition
        self.solver.set(0, "lbx", x_curr)
        self.solver.set(0, "ubx", x_curr)

        # 2. Update Path Params & References
        s_val = x_curr[0]
        ds_val = max(x_curr[3], 0.5) # Lookahead speed
        dt = self.Tf / self.N
        hover_T = self.params['mass'] * self.params['g']

        for k in range(self.N):
            s_pred = min(s_val + ds_val * k * dt, geo.total_length)
            f = geo.get_frame(s_pred)
            
            # Update Params (Curvature)
            p = np.concatenate([[f['k1'], f['k2']], f['t'], f['n1'], f['n2']])
            self.solver.set(k, "p", p)
            
            # Update Reference
            yref = np.zeros(13) # 9 states + 4 inputs
            yref[3] = self.v_target # Target Velocity (ds)
            yref[12] = hover_T      # Target Thrust
            self.solver.set(k, "yref", yref)

        # Terminal Node
        s_end = min(s_val + ds_val * self.N * dt, geo.total_length)
        f = geo.get_frame(s_end)
        p = np.concatenate([[f['k1'], f['k2']], f['t'], f['n1'], f['n2']])
        self.solver.set(self.N, "p", p)
        
        yref_e = np.zeros(9)
        yref_e[3] = self.v_target
        self.solver.set(self.N, "yref", yref_e)

        # 3. Solve
        status = self.solver.solve()
        u_opt = self.solver.get(0, "u")
        return status, u_opt
# ==============================================================================
# 5. MAIN EXECUTION & REPORTING
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Load or Generate Track ---
    if os.path.exists("config/level2_noObstacle.toml"):
        print("[INFO] Loading Level 2 Track from TOML...")
        data = toml.load("config/level2_noObstacle.toml")
        gates_raw = data["env"]["track"]["gates"]
        gates_pos = np.array([g["pos"] for g in gates_raw])
        start_pos = np.array(data["env"]["track"]["drones"][0]["pos"])
    else:
        print("[WARN] TOML not found. Generating COMPLEX HELIX TRACK.")
        # Generate a rising helix (Radius 2m, climbing 1m per turn)
        theta = np.linspace(0, 4*np.pi, 8)
        gates_pos = np.column_stack((2*np.cos(theta), 2*np.sin(theta), 1 + 0.5*theta))
        start_pos = np.array([2.0, -1.0, 1.0])

    # --- 2. Initialize ---
    geo = GeometryEngine(gates_pos, start_pos)
    mpc = SpatialMPC(N=25, Tf=1.0, v_target=6.0) # Faster target

    # Initial State [s, w1, w2, ds, dw1, dw2, phi, th, psi]
    x = np.zeros(9)
    x[0] = 0.0
    x[3] = 1.0 # Initial push

    print("[INFO] Warm Starting...")
    mpc.warm_start(x)

    # --- 3. Sim Loop ---
    history_x, history_u, solve_times = [], [], []
    dt_sim = 0.02
    
    print(f"{'Step':<5} | {'s [m]':<7} | {'v [m/s]':<7} | {'w1':<6} | {'Status'}")
    
    for i in range(250):
        t0 = time.time()
        status, u = mpc.solve(x, geo)
        dt_solve = time.time() - t0
        solve_times.append(dt_solve * 1000)

        if status != 0:
            print(f"Solver Fail: {status}")
            # Simple fallback integration to prevent crash logging
            u = np.array([0,0,0, mpc.params['mass']*9.81])
        
        # Mock Integration (Euler)
        dx = mpc.solver.get(1, "x") - x # Use MPC prediction as update
        x = x + dx # Ideally use real dynamics, but this validates the solver output
        
        history_x.append(x)
        history_u.append(u)

        if i % 10 == 0:
            print(f"{i:<5} | {x[0]:<7.2f} | {x[3]:<7.2f} | {x[1]:<6.2f} | {status}")
        
        if x[0] >= geo.total_length:
            print("[INFO] Track Finished.")
            break

    # --- 4. DETAILED PERFORMANCE REVIEW ---
    H_x = np.array(history_x)
    H_u = np.array(history_u)
    
    # Calculate Deviation Metrics
    w_error = np.linalg.norm(H_x[:, 1:3], axis=1) # Norm of [w1, w2]
    rmse = np.sqrt(np.mean(w_error**2))
    max_err = np.max(w_error)
    
    # Smoothness (Jerk proxy: diff of u)
    u_jerk = np.mean(np.abs(np.diff(H_u, axis=0)))

    print("\n" + "="*50)
    print("      SPATIAL MPC PERFORMANCE REVIEW      ")
    print("="*50)
    print(f"Total Distance:      {geo.total_length:.2f} m")
    print(f"Completion Time:     {len(H_x)*dt_sim:.2f} s")
    print(f"Avg Velocity:        {np.mean(H_x[:,3]):.2f} m/s")
    print("-" * 50)
    print(f"Tracking Accuracy (RMSE): {rmse:.4f} m")
    print(f"Max Corridor Deviation:   {max_err:.4f} m (Limit: 0.50m)")
    print("-" * 50)
    print(f"Avg Solve Time:      {np.mean(solve_times):.2f} ms")
    print(f"Max Solve Time:      {np.max(solve_times):.2f} ms")
    print(f"Control Smoothness:  {u_jerk:.4f}")
    print("="*50)

    # --- 5. Visualization ---
    fig = plt.figure(figsize=(12, 5))
    
    # 3D Trajectory
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # Plot Centerline
    s_dense = np.linspace(0, geo.total_length, 200)
    p_center = np.array([geo.spline(s) for s in s_dense])
    ax.plot(p_center[:,0], p_center[:,1], p_center[:,2], 'k--', alpha=0.3, label="Ref Path")
    
    # Reconstruct Flown Path
    p_flown = []
    for val in H_x:
        f = geo.get_frame(val[0])
        p_flown.append(f['pos'] + f['n1']*val[1] + f['n2']*val[2])
    p_flown = np.array(p_flown)
    ax.plot(p_flown[:,0], p_flown[:,1], p_flown[:,2], 'b-', linewidth=2, label="MPC Path")
    
    # Draw Gates
    ax.scatter(geo.gates_pos[:,0], geo.gates_pos[:,1], geo.gates_pos[:,2], c='r', marker='s', s=50, label="Gates")
    ax.set_title("3D Spatial Trajectory")
    ax.legend()

    # Errors
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(H_x[:,0], H_x[:,1], label="Lat Error (w1)")
    ax2.plot(H_x[:,0], H_x[:,2], label="Vert Error (w2)")
    ax2.axhline(0.5, c='r', ls='--'); ax2.axhline(-0.5, c='r', ls='--')
    ax2.set_xlabel("Progress (s)"); ax2.set_ylabel("Deviation (m)")
    ax2.set_title("Corridor Compliance")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()