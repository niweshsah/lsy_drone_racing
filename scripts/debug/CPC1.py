import numpy as np
import casadi as ca
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Any, Tuple

class TimeOptimalPlanner:
    def __init__(
        self,
        start_pos: List[float],
        gates_pos: List[List[float]],
        # 'Race Quad' Config [Table I]
        mass: float = 0.8,         # kg
        T_max: float = 8.0,        # N (per rotor)
        T_min: float = 0.0,        # N (per rotor)
        arm_length: float = 0.15,  # m
        drag_coeff: float = 0.4,   # 1/s (Linear drag)
        gate_radius: float = 0.3   # m
    ):
        self.start_pos = np.array(start_pos)
        self.gates_pos = np.array(gates_pos)
        self.waypoints = self.gates_pos
        
        # Physics
        self.m = mass
        self.g = np.array([0, 0, -9.81])
        self.J = np.diag([0.01, 0.01, 0.017])
        self.l = arm_length
        self.c_tau = 0.013
        self.D_drag = np.diag([drag_coeff]*3) # [cite: 235]
        
        # Limits
        self.T_min = T_min
        self.T_max = T_max
        # Collective thrust limit approx for point mass
        self.Thrust_total_max = 4 * T_max 
        
        # Grid
        self.N_per_gate = 40
        self.N = self.N_per_gate * len(self.waypoints)
        self.d_tol = gate_radius

    def solve(self):
        print(f"--- STAGE 1: Solving Convex Point-Mass Model (N={self.N}) ---")
        #  "reducing the non-linear quadrotor model into a linear point-mass model"
        sol_pm = self._solve_point_mass()
        
        if not sol_pm['success']:
            print("Stage 1 Failed. Cannot proceed.")
            return sol_pm

        print(f"Stage 1 Success! Approx Time: {sol_pm['T']:.4f}s")
        print(f"--- STAGE 2: Solving Full Quadrotor Model with Warm Start ---")
        # [cite: 423] "allows us to find a solution from which the original problem... can be initialized"
        return self._solve_full_quadrotor(warm_start=sol_pm)

    def _solve_point_mass(self):
        """
        Stage 1: Convex optimization using simple a = F/m dynamics.
        Resolves the path (p, v) and total time (T) robustly.
        """
        opti = ca.Opti()
        
        # Variables
        T = opti.variable()
        opti.subject_to(T >= 0.1)
        
        # Guess T based on straight line distance / avg speed (10m/s)
        dist = np.sum(np.linalg.norm(np.diff(np.vstack([self.start_pos, self.gates_pos]), axis=0), axis=1))
        opti.set_initial(T, dist / 10.0)
        
        dt = T / self.N
        
        # States: p (position), v (velocity)
        p = opti.variable(3, self.N + 1)
        v = opti.variable(3, self.N + 1)
        # Input: a (acceleration vector) - represents Thrust direction
        a = opti.variable(3, self.N)
        
        # Progress Vars
        M = len(self.waypoints)
        Lam = opti.variable(M, self.N + 1)
        Mu = opti.variable(M, self.N)
        Nu = opti.variable(M, self.N)

        # Objective
        opti.minimize(T)
        
        # Initial Conditions
        opti.subject_to(p[:,0] == self.start_pos)
        opti.subject_to(v[:,0] == [0,0,0])
        opti.subject_to(Lam[:,0] == 1.0)
        
        # Dynamics (Euler integration for Point Mass is sufficient for init)
        for k in range(self.N):
            # p_next = p + v*dt + 0.5*a*dt^2
            opti.subject_to(p[:,k+1] == p[:,k] + v[:,k]*dt + 0.5*a[:,k]*dt**2)
            # v_next = v + a*dt
            opti.subject_to(v[:,k+1] == v[:,k] + a[:,k]*dt)
            
            # Acceleration Limit (Approximate Thrust Limit) [cite: 423]
            # ||a - g|| <= T_max_total / m
            # We add gravity because 'a' here is the net acceleration, but thrust fights gravity
            thrust_acc = a[:,k] - self.g
            opti.subject_to(ca.sumsqr(thrust_acc) <= (self.Thrust_total_max / self.m)**2)
            
            # Progress Logic (Same as full model) [cite: 123, 161]
            opti.subject_to(Lam[:,k+1] == Lam[:,k] - Mu[:,k])
            opti.subject_to(Mu[:,k] >= 0)
            
            for j in range(M):
                dist_sq = ca.sumsqr(p[:,k] - self.waypoints[j])
                opti.subject_to(opti.bounded(0, Nu[j,k], self.d_tol**2))
                # Relaxed CPC
                opti.subject_to(opti.bounded(-1e-3, Mu[j,k]*(dist_sq - Nu[j,k]), 1e-3))

        # Global constraints
        opti.subject_to(Lam[:,-1] == 0.0)
        for k in range(self.N+1):
            for j in range(M-1):
                opti.subject_to(Lam[j,k] <= Lam[j+1,k])

        # Init Guess (Linear)
        p_lin = np.linspace(self.start_pos, self.gates_pos[-1], self.N+1).T
        opti.set_initial(p, p_lin)
        opti.set_initial(Lam, np.linspace(1, 0, self.N+1))
        
        # Solver
        opti.solver('ipopt', {'expand':True, 'ipopt.print_level':3}, {'max_iter':2000})
        
        try:
            sol = opti.solve()
            # Calculate thrust vector for quaternion init
            acc_vals = sol.value(a)
            thrust_vecs = acc_vals - self.g.reshape(3,1) # The direction the drone must point
            return {
                "success": True,
                "T": sol.value(T),
                "p": sol.value(p),
                "v": sol.value(v),
                "thrust_vec": thrust_vecs,
                "Lam": sol.value(Lam),
                "Mu": sol.value(Mu),
                "Nu": sol.value(Nu)
            }
        except Exception as e:
            print(f"Point mass failed: {e}")
            return {"success": False}

    def _solve_full_quadrotor(self, warm_start):
        """
        Stage 2: Full Non-Convex Dynamics initialized with Stage 1 results.
        """
        opti = ca.Opti()
        
        # --- Variables ---
        T = opti.variable()
        opti.subject_to(T >= 0.1)
        dt = T / self.N
        
        X = opti.variable(13, self.N + 1) # p, q, v, w
        p, q, v, w = X[0:3,:], X[3:7,:], X[7:10,:], X[10:13,:]
        U = opti.variable(4, self.N) # Single rotor thrusts
        
        M = len(self.waypoints)
        Lam = opti.variable(M, self.N + 1)
        Mu = opti.variable(M, self.N)
        Nu = opti.variable(M, self.N)
        
        # --- Warm Start Initialization [cite: 424] ---
        opti.set_initial(T, warm_start['T'])
        opti.set_initial(p, warm_start['p'])
        opti.set_initial(v, warm_start['v'])
        opti.set_initial(Lam, warm_start['Lam'])
        opti.set_initial(Mu, warm_start['Mu'])
        opti.set_initial(Nu, warm_start['Nu'])
        
        # Initialize Quaternions from Point-Mass Thrust Vectors
        # We need to rotate the body Z-axis (0,0,1) to align with thrust_vec
        q_init = np.zeros((4, self.N + 1))
        thrust_vecs = warm_start['thrust_vec']
        
        for k in range(self.N):
            # Desired z-axis direction
            z_des = thrust_vecs[:,k]
            norm = np.linalg.norm(z_des)
            if norm > 1e-6:
                z_des /= norm
            else:
                z_des = np.array([0,0,1])
                
            # Compute rotation from [0,0,1] to z_des
            # Simplest way: Shortest arc rotation
            z_ref = np.array([0,0,1])
            cross = np.cross(z_ref, z_des)
            dot = np.dot(z_ref, z_des)
            
            # Quaternion construction (x,y,z,w)
            q_k = np.array([cross[0], cross[1], cross[2], 1 + dot])
            q_k /= np.linalg.norm(q_k)
            # Reorder to [w, x, y, z] for code consistency
            q_init[:, k] = [q_k[3], q_k[0], q_k[1], q_k[2]]
            
        q_init[:, -1] = q_init[:, -2] # Duplicate last
        opti.set_initial(q, q_init)
        
        # Initialize Inputs (Gravity compensation distributed)
        hover = (self.m * 9.81) / 4.0
        opti.set_initial(U, np.ones((4, self.N)) * hover)

        # --- Objective ---
        opti.minimize(T) # [cite: 173]

        # --- Constraints ---
        opti.subject_to(p[:,0] == self.start_pos)
        opti.subject_to(v[:,0] == [0,0,0])
        opti.subject_to(q[:,0] == [1,0,0,0]) # Upright
        opti.subject_to(Lam[:,0] == 1.0)
        
        f_dyn = self._get_dynamics()

        for k in range(self.N):
            # RK4 Integration [cite: 83]
            k1 = f_dyn(X[:,k], U[:,k])
            k2 = f_dyn(X[:,k] + dt/2*k1, U[:,k])
            k3 = f_dyn(X[:,k] + dt/2*k2, U[:,k])
            k4 = f_dyn(X[:,k] + dt*k3, U[:,k])
            x_next = X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:,k+1] == x_next)
            
            # Unit Quaternion
            opti.subject_to(ca.sumsqr(q[:,k+1]) == 1)
            
            # Actuator Limits [cite: 198]
            opti.subject_to(opti.bounded(self.T_min, U[:,k], self.T_max))
            
            # Progress [cite: 123]
            opti.subject_to(Lam[:,k+1] == Lam[:,k] - Mu[:,k])
            opti.subject_to(Mu[:,k] >= 0)
            
            # CPC [cite: 161]
            for j in range(M):
                dist_sq = ca.sumsqr(p[:,k] - self.waypoints[j])
                opti.subject_to(opti.bounded(0, Nu[j,k], self.d_tol**2))
                # Slightly tighter tolerance than point mass
                opti.subject_to(opti.bounded(-1e-4, Mu[j,k]*(dist_sq - Nu[j,k]), 1e-4))

        opti.subject_to(Lam[:,-1] == 0.0)
        for k in range(self.N+1):
            for j in range(M-1):
                opti.subject_to(Lam[j,k] <= Lam[j+1,k])

        # --- Solver Options ---
        # Adaptive strategy is critical for Complementarity problems
        opts = {
            "ipopt.max_iter": 5000,
            "ipopt.print_level": 5,
            "ipopt.mu_strategy": "adaptive", 
            "ipopt.tol": 1e-4,
            "expand": True
        }
        opti.solver("ipopt", opts)
        
        try:
            sol = opti.solve()
            return {
                "success": True,
                "T": sol.value(T),
                "pos": sol.value(p),
                "vel": sol.value(v),
                "input": sol.value(U)
            }
        except Exception as e:
            print("Optimization Failed.")
            # Return debug values
            return {
                "success": False,
                "T": opti.debug.value(T),
                "pos": opti.debug.value(p)
            }

    def _get_dynamics(self):
        # Full Quadrotor Dynamics with Linear Drag [cite: 233]
        p = ca.SX.sym('p', 3)
        q = ca.SX.sym('q', 4) # w, x, y, z
        v = ca.SX.sym('v', 3)
        w = ca.SX.sym('w', 3)
        x = ca.vertcat(p, q, v, w)
        u = ca.SX.sym('u', 4)
        
        # Quaternion to Rotation Matrix
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        Rot = ca.vertcat(
            ca.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            ca.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            ca.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
        )
        
        # Forces
        f_thrust = ca.vertcat(0, 0, ca.sum1(u))
        
        # Torques (X-config)
        l_eff = self.l / np.sqrt(2)
        tau = ca.vertcat(
            l_eff * (u[0] + u[1] - u[2] - u[3]),
            l_eff * (-u[0] + u[1] + u[2] - u[3]),
            self.c_tau * (u[0] - u[1] + u[2] - u[3])
        )
        
        # Dynamics derivatives
        p_dot = v
        
        # q_dot = 0.5 * q * w_pure
        Omega = ca.vertcat(
            ca.horzcat(0, -w[0], -w[1], -w[2]),
            ca.horzcat(w[0], 0, w[2], -w[1]),
            ca.horzcat(w[1], -w[2], 0, w[0]),
            ca.horzcat(w[2], w[1], -w[0], 0)
        )
        q_dot = 0.5 * ca.mtimes(Omega, q)
        
        # v_dot = g + R*T/m - Drag
        v_body = ca.mtimes(Rot.T, v)
        drag_force_world = ca.mtimes(Rot, ca.mtimes(self.D_drag, v_body))
        v_dot = self.g + (1/self.m) * ca.mtimes(Rot, f_thrust) - drag_force_world
        
        # w_dot = J_inv * (tau - w x Jw)
        Jw = ca.mtimes(self.J, w)
        w_dot = ca.mtimes(np.linalg.inv(self.J), (tau - ca.cross(w, Jw)))
        
        return ca.Function('f_dyn', [x, u], [ca.vertcat(p_dot, q_dot, v_dot, w_dot)])

def plot_trajectory(res, gates):
    if res['pos'] is None: return
    path = res['pos'].T
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=path[:,0], y=path[:,1], z=path[:,2],
        mode='lines+markers', marker=dict(size=3, color='blue'),
        name=f"Trajectory (T={res['T']:.3f}s)"
    ))
    
    for i, g in enumerate(gates):
        r = 0.3
        # Simple box for gate
        x, y, z = g
        corners = np.array([
            [x, y-r, z-r], [x, y+r, z-r],
            [x, y+r, z+r], [x, y-r, z+r],
            [x, y-r, z-r]
        ])
        fig.add_trace(go.Scatter3d(
            x=corners[:,0], y=corners[:,1], z=corners[:,2],
            mode='lines', line=dict(color='red', width=5), name=f"Gate {i}"
        ))
        
    fig.update_layout(scene=dict(aspectmode='data'), title="Time-Optimal Planner Result")
    fig.show()

if __name__ == "__main__":
    # Test Setup
    start = [0, 0, 1]
    gates = [[5, 0, 2], [10, 5, 2], [5, 10, 2]]
    
    planner = TimeOptimalPlanner(start, gates)
    res = planner.solve()
    
    if res['success']:
        plot_trajectory(res, gates)