import os
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import toml
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

class GeometryEngine:
    def __init__(
        self,
        gates_pos: List[List[float]],
        gates_normals: List[List[float]],
        gates_y: List[List[float]],
        gates_z: List[List[float]],
        obstacles_pos: List[List[float]],
        start_pos: List[float],
        gate_dims: Tuple[float, float] = (1.0, 1.0),
    ):
        """Initializes the geometry engine and generates the Gate-Traversing path."""
        # --- 1. Configuration Constants ---
        self.MAX_LATERAL_WIDTH = 0.45
        self.SAFETY_RADIUS = 0.15 
        
        # Tangent Parameters
        # A factor < 1.0 tightens turns. 1.0 is standard Catmull-Rom.
        self.TANGENT_SCALE_FACTOR = 1.0 
        
        # [NEW] U-Turn Sensitivity
        # If we detect a reversal, how much do we shrink the tangent? 
        # 0.3 means the drone turns within 30% of the segment distance.
        self.U_TURN_SHRINK_FACTOR = 0.3 

        # Gate Constraints
        self.GATE_WIDTH = gate_dims[0]
        self.GATE_HEIGHT = gate_dims[1]
        self.GATE_MARGIN = 0.1 

        # --- 2. Data Ingestion ---
        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.gate_y = np.asarray(gates_y, dtype=np.float64)
        self.gate_z = np.asarray(gates_z, dtype=np.float64)
        self.obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)

        # --- 3. Pipeline Execution ---
        self.optimal_pass_points = self._optimize_gate_traversal()
        self.waypoints, self.wp_types, self.wp_normals = self._initialize_waypoints()
        self.tangents = self._compute_adaptive_tangents() # [UPDATED NAME]
        
        # Spline Generation
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # Frame and Corridor
        num_frame_points = int(self.total_length * 100)
        self.pt_frame = self._generate_parallel_transport_frame(num_points=num_frame_points)
        self.corridor_map = self._generate_static_corridor()

    def _optimize_gate_traversal(self) -> NDArray:
        """
        Solves constrained optimization to find point P_i on Gate i.
        Includes directionality check to penalize entering gates from behind.
        """
        num_gates = len(self.gates_pos)
        print(f"[Geometry] Optimizing traversal for {num_gates} gates...")
        
        initial_guess = np.zeros(num_gates * 2) 
        
        # Bounds for u and v
        safe_w = (self.GATE_WIDTH / 2.0) - self.GATE_MARGIN
        safe_h = (self.GATE_HEIGHT / 2.0) - self.GATE_MARGIN
        bounds = [(-safe_w, safe_w), (-safe_h, safe_h)] * num_gates
        
        def path_cost(uv_flat):
            total_cost = 0.0
            prev_pos = self.start_pos
            
            for i in range(num_gates):
                u = uv_flat[2*i]
                v = uv_flat[2*i + 1]
                
                curr_pos = (self.gates_pos[i] + 
                           (u * self.gate_y[i]) + 
                           (v * self.gate_z[i]))
                
                path_vec = curr_pos - prev_pos
                dist = np.linalg.norm(path_vec)
                
                # Directionality Penalty
                gate_normal = self.gate_normals[i]
                if dist > 1e-6:
                    dir_vec = path_vec / dist
                    alignment = np.dot(dir_vec, gate_normal)
                    
                    if alignment < 0.1: 
                        # Penalty for entering backwards
                        alignment_penalty = 5.0 * abs(alignment - 0.1) 
                        total_cost += alignment_penalty

                total_cost += dist
                prev_pos = curr_pos
            
            return total_cost

        result = minimize(path_cost, initial_guess, bounds=bounds, method='SLSQP', options={'ftol': 1e-4})
        
        if result.success:
            print(f"[Geometry] Optimization converged! Cost: {result.fun:.4f}")
        else:
            print("[Geometry] Optimization failed, reverting to gate centers.")
            return self.gates_pos
            
        optimized_points = []
        for i in range(num_gates):
            u = result.x[2*i]
            v = result.x[2*i + 1]
            pos = (self.gates_pos[i] + 
                   (u * self.gate_y[i]) + 
                   (v * self.gate_z[i]))
            optimized_points.append(pos)
            
        return np.array(optimized_points)

    def _initialize_waypoints(self):
        wps = [self.start_pos]
        types = [0] # 0 = Start/Normal
        normals = [np.zeros(3)]

        for i in range(len(self.optimal_pass_points)):
            wps.append(self.optimal_pass_points[i])
            types.append(1) # 1 = Gate
            normals.append(self.gate_normals[i])

        return np.array(wps), np.array(types), np.array(normals)

    def _compute_adaptive_tangents(self):
        """
        Calculates tangents with 'U-Turn Detection'.
        If the gate geometry requires a reversal, the tangent magnitude is 
        drastically reduced to tighten the turn and prevent swinging.
        """
        num_pts = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)

        for i in range(num_pts):
            # 1. Determine local segment length
            dist_prev = np.linalg.norm(self.waypoints[i] - self.waypoints[i - 1]) if i > 0 else 0
            dist_next = np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i]) if i < num_pts - 1 else 0
            
            # 2. Determine base direction and type
            if self.wp_types[i] == 1: # Gate
                # Tangent direction is FIXED by Gate Normal
                t_dir = self.wp_normals[i].copy()
                
                # 3. [NEW] Adaptive Scaling Logic
                # We check the alignment between the Gate Normal (t_dir) and the path to the NEXT waypoint.
                scale_factor = self.TANGENT_SCALE_FACTOR
                
                if i < num_pts - 1:
                    # Vector to next point
                    vec_to_next = self.waypoints[i+1] - self.waypoints[i]
                    dist_to_next = np.linalg.norm(vec_to_next)
                    
                    if dist_to_next > 1e-6:
                        dir_to_next = vec_to_next / dist_to_next
                        # Alignment: 1.0 = Straight, -1.0 = U-Turn
                        alignment = np.dot(t_dir, dir_to_next)
                        
                        # If alignment is negative (obtsuse angle), we are "overshooting".
                        # We must shrink the tangent to turn back sooner.
                        if alignment < 0:
                            # Linearly interpolate between 1.0 (at 90 deg) and U_TURN_SHRINK (at 180 deg)
                            # alignment is between 0 and -1
                            # alpha 0 -> 0, alpha -1 -> 1
                            alpha = -alignment 
                            current_shrink = (1.0 - alpha) * 1.0 + alpha * self.U_TURN_SHRINK_FACTOR
                            scale_factor *= current_shrink
                            
                # Apply scaling to the relevant distance (usually the shorter leg controls the constraint)
                # For a U-turn, we usually want to be tight relative to the gap distance.
                base_dist = min(dist_prev, dist_next) if dist_prev > 0 and dist_next > 0 else (dist_next if dist_next > 0 else dist_prev)
                
                tangents[i] = t_dir * base_dist * scale_factor

            else:
                # [Standard Catmull-Rom for non-gate points]
                if i == 0:
                    t = self.waypoints[i + 1] - self.waypoints[i]
                elif i == num_pts - 1:
                    t = self.waypoints[i] - self.waypoints[i - 1]
                else:
                    t = self.waypoints[i + 1] - self.waypoints[i - 1]
                
                if np.linalg.norm(t) > 1e-6:
                    t = t / np.linalg.norm(t)
                
                base_dist = min(dist_prev, dist_next) if dist_prev > 0 and dist_next > 0 else (dist_next if dist_next > 0 else dist_prev)
                tangents[i] = t * base_dist * self.TANGENT_SCALE_FACTOR

        return tangents

    def _generate_parallel_transport_frame(self, num_points=1000):
        # [Unchanged Logic]
        s_eval = np.linspace(0, self.total_length, num_points)
        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": []}

        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)
        
        g_vec = np.array([0, 0, -1])
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = np.cross(t0, np.cross(g_vec, t0)) 

        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

            axis = np.cross(curr_t, next_t)
            sin_phi = np.linalg.norm(axis)
            cos_phi = np.dot(curr_t, next_t)

            next_n1 = curr_n1
            next_n2 = curr_n2

            if sin_phi > 1e-6:
                axis /= sin_phi
                phi = np.arctan2(sin_phi, cos_phi)
                rot = R.from_rotvec(axis * phi)
                next_n1 = rot.apply(curr_n1)
                next_n2 = rot.apply(curr_n2)

            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            
            curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        for k in frames:
            frames[k] = np.array(frames[k])
        return frames

    def _generate_static_corridor(self) -> Dict[str, NDArray]:
        # [Unchanged Logic]
        num_pts = len(self.pt_frame["s"])
        lb_w1 = np.full(num_pts, -self.MAX_LATERAL_WIDTH)
        ub_w1 = np.full(num_pts, self.MAX_LATERAL_WIDTH)

        if len(self.obstacles_pos) == 0:
            return {"lb_w1": lb_w1, "ub_w1": ub_w1}

        for i in range(num_pts):
            frame_pos = self.pt_frame["pos"][i]
            frame_t = self.pt_frame["t"][i]
            pos_2d = np.array([frame_pos[0], frame_pos[1], 0.0])
            t_2d = np.array([frame_t[0], frame_t[1], 0.0])
            
            norm_t = np.linalg.norm(t_2d)
            if norm_t < 1e-3: continue
            t_2d /= norm_t
            n1_2d = np.array([-t_2d[1], t_2d[0], 0.0])

            for obs in self.obstacles_pos:
                obs_2d = np.array([obs[0], obs[1], 0.0])
                r_vec_2d = obs_2d - pos_2d
                d_long = np.dot(r_vec_2d, t_2d)
                if abs(d_long) > self.SAFETY_RADIUS: continue
                w1_obs = np.dot(r_vec_2d, n1_2d)
                if abs(w1_obs) > (self.MAX_LATERAL_WIDTH + self.SAFETY_RADIUS + 0.5): continue

                if w1_obs > 0:
                    safe_edge = w1_obs - self.SAFETY_RADIUS
                    if safe_edge < ub_w1[i]: ub_w1[i] = safe_edge
                else:
                    safe_edge = w1_obs + self.SAFETY_RADIUS
                    if safe_edge > lb_w1[i]: lb_w1[i] = safe_edge

        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05

        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    def plot(self):
        # [Unchanged Logic]
        print("[Geometry] Generating plot...")
        fig = go.Figure()

        path = self.pt_frame["pos"]
        fig.add_trace(go.Scatter3d(x=path[:,0], y=path[:,1], z=path[:,2], 
                                   mode="lines", line=dict(color="black", width=4), name="Optimized Path"))

        fig.add_trace(go.Scatter3d(x=self.gates_pos[:,0], y=self.gates_pos[:,1], z=self.gates_pos[:,2],
                                   mode="markers", marker=dict(size=4, color="gray", symbol="x"), name="Gate Center (Ref)"))

        fig.add_trace(go.Scatter3d(x=self.optimal_pass_points[:,0], y=self.optimal_pass_points[:,1], z=self.optimal_pass_points[:,2],
                                   mode="markers", marker=dict(size=6, color="blue"), name="Optimal Pass Point"))

        gate_w, gate_h = self.GATE_WIDTH/2, self.GATE_HEIGHT/2
        for i, pos in enumerate(self.gates_pos):
            y_v, z_v = self.gate_y[i], self.gate_z[i]
            c = [pos + y_v*gate_w + z_v*gate_h, pos - y_v*gate_w + z_v*gate_h,
                 pos - y_v*gate_w - z_v*gate_h, pos + y_v*gate_w - z_v*gate_h,
                 pos + y_v*gate_w + z_v*gate_h]
            c = np.array(c)
            fig.add_trace(go.Scatter3d(x=c[:,0], y=c[:,1], z=c[:,2], mode="lines", 
                                       line=dict(color="rgba(0,0,255,0.5)"), showlegend=False))
            # Visualize Normal
            norm_end = pos + self.gate_normals[i] * 1.0
            fig.add_trace(go.Scatter3d(x=[pos[0], norm_end[0]], y=[pos[1], norm_end[1]], z=[pos[2], norm_end[2]],
                                     mode="lines", line=dict(color="green", width=2), showlegend=False))

        step = 5
        idx = np.arange(0, len(path), step)
        p_vis = path[idx]
        n1_vis = self.pt_frame["n1"][idx]
        lb = self.corridor_map["lb_w1"][idx]
        ub = self.corridor_map["ub_w1"][idx]
        
        wall_l = p_vis + (n1_vis * ub[:, None])
        wall_r = p_vis + (n1_vis * lb[:, None])
        
        fig.add_trace(go.Scatter3d(x=wall_l[:,0], y=wall_l[:,1], z=wall_l[:,2], mode="lines", line=dict(color="orange"), name="Left Bound"))
        fig.add_trace(go.Scatter3d(x=wall_r[:,0], y=wall_r[:,1], z=wall_r[:,2], mode="lines", line=dict(color="red"), name="Right Bound"))

        fig.update_layout(scene=dict(aspectmode="data"), title="Paper-Based Gate Traversing Planner")
        fig.show()

# --- Config Loader Stub (Unchanged) ---
def load_from_toml(filepath: str):
    print(f"Loading config from: {filepath}")
    with open(filepath, "r") as f:
        data = toml.load(f)
    gates_raw = data["env"]["track"]["gates"]
    gates_pos = np.array([g["pos"] for g in gates_raw], dtype=np.float64)
    gates_rpy = np.array([g.get("rpy", [0, 0, 0]) for g in gates_raw], dtype=np.float64)
    rot = R.from_euler("xyz", gates_rpy, degrees=False)
    matrices = rot.as_matrix()
    gates_normals = matrices[:, :, 0]
    gates_y = matrices[:, :, 1]
    gates_z = matrices[:, :, 2]
    obs_raw = data["env"]["track"].get("obstacles", [])
    obstacles_pos = (
        np.array([o["pos"] for o in obs_raw], dtype=np.float64) if obs_raw else np.empty((0, 3))
    )
    start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=np.float64)
    return gates_pos, gates_normals, gates_y, gates_z, obstacles_pos, start_pos

if __name__ == "__main__":
    # Test Data: A sharp "U-Turn" scenario
    # Gate 1 and Gate 2 are almost stacked, requiring a tight reversal.
    toml_path = "config/level1.toml"
    
    s_pos = [0, 0, 1.0]
    
    # Gate 1 at X=5. Gate 2 at X=3 (Behind Gate 1). Both face +X.
    # The drone must pass G1, loop back tightly, and pass G2.
    g_pos = [[5, 0, 1.0], [3, 0, 1.0]] 
    g_norm = [[1, 0, 0], [1, 0, 0]]
    g_y = [[0, 1, 0], [0, 1, 0]]
    g_z = [[0, 0, 1], [0, 0, 1]]
    obs_pos = []

    if os.path.exists(toml_path):
        try:
            g_pos, g_norm, g_y, g_z, obs_pos, s_pos = load_from_toml(toml_path)
            print("start_pos:", s_pos)
        except Exception:
            pass

    geom = GeometryEngine(g_pos, g_norm, g_y, g_z, obs_pos, s_pos, gate_dims=(0.3, 0.3))
    geom.plot()