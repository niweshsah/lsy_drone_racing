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
        gate_dims: Tuple[float, float] = (1.0, 1.0), # (width, height) in meters
    ):
        """Initializes the geometry engine and generates the Gate-Traversing path."""
        # --- 1. Configuration Constants ---
        self.TANGENT_SCALE_FACTOR = 1.0
        self.MAX_LATERAL_WIDTH = 0.45
        self.SAFETY_RADIUS = 0.15 
        self.POLE_HEIGHT = 2.0
        
        # Paper Logic: Gate Constraints
        self.GATE_WIDTH = gate_dims[0]
        self.GATE_HEIGHT = gate_dims[1]
        self.GATE_MARGIN = 0.1 # Safety margin from frame (paper mentions 0.3m usually)

        # --- Debug Storage ---
        self.debug_vectors = [] 

        # --- 2. Data Ingestion ---
        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.gate_y = np.asarray(gates_y, dtype=np.float64)
        self.gate_z = np.asarray(gates_z, dtype=np.float64)
        self.obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)

        # --- 3. Pipeline Execution (Updated for Gate Traversing) ---

        # A. Gate Traversing Optimization (The Core Update)
        # Instead of just picking centers, we find the optimal point on the gate plane.
        self.optimal_pass_points = self._optimize_gate_traversal()
        
        # B. Initialize Waypoints with OPTIMAL points, not centers
        self.waypoints, self.wp_types, self.wp_normals = self._initialize_waypoints()

        # C. Compute Tangents
        self.tangents = self._compute_hermite_tangents()

        # D. Generate Cubic Hermite Spline
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # E. Generate Parallel Transport Frame 
        num_frame_points = int(self.total_length * 100)
        self.pt_frame = self._generate_parallel_transport_frame(num_points=num_frame_points)

        # F. Generate Static Corridor Boundaries
        self.corridor_map = self._generate_static_corridor()

    def _optimize_gate_traversal(self) -> NDArray:
        """
        Implementation of the Gate-Traversing logic.
        Solves a constrained optimization problem to find point P_i on Gate i
        that minimizes total path distance.
        
        Based on paper Section III-F: "Gate Constraints Elimination"
        We treat the pass-through point as a variable constrained by the gate geometry.
        """
        num_gates = len(self.gates_pos)
        print(f"[Geometry] Optimizing traversal for {num_gates} gates (Paper Logic)...")
        
        # Optimization variables: [u1, v1, u2, v2, ... un, vn]
        # u = horizontal offset along gate_y
        # v = vertical offset along gate_z
        initial_guess = np.zeros(num_gates * 2) 
        
        # Bounds for u and v (Gate size - margin)
        safe_w = (self.GATE_WIDTH / 2.0) - self.GATE_MARGIN
        safe_h = (self.GATE_HEIGHT / 2.0) - self.GATE_MARGIN
        bounds = [(-safe_w, safe_w), (-safe_h, safe_h)] * num_gates
        
        # Objective Function: Minimize Euclidean distance between sequential points
        def path_cost(uv_flat):
            total_dist = 0.0
            
            # Start Point
            prev_pos = self.start_pos
            
            for i in range(num_gates):
                # Reconstruct 3D position of the point on the gate
                u = uv_flat[2*i]
                v = uv_flat[2*i + 1]
                
                # Pos = Center + u*Y + v*Z
                curr_pos = (self.gates_pos[i] + 
                           (u * self.gate_y[i]) + 
                           (v * self.gate_z[i]))
                
                dist = np.linalg.norm(curr_pos - prev_pos)
                total_dist += dist
                prev_pos = curr_pos
            
            # Note: If there's a goal point, add dist to goal here.
            # Assuming last gate is the end for now, or loop back to start?
            # Let's assume open track, minimizing distance through gates.
            return total_dist

        # Run Optimization (SLSQP is good for bound-constrained problems)
        result = minimize(path_cost, initial_guess, bounds=bounds, method='SLSQP', options={'ftol': 1e-4})
        
        if result.success:
            print(f"[Geometry] Optimization converged! Cost reduction: {result.fun:.4f}")
        else:
            print("[Geometry] Optimization failed, reverting to gate centers.")
            return self.gates_pos
            
        # Reconstruct the 3D points from the optimized u, v results
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
        """Updated to use self.optimal_pass_points instead of self.gates_pos"""
        wps = [self.start_pos]
        types = [0] # 0 = Start/Normal
        normals = [np.zeros(3)]

        for i in range(len(self.optimal_pass_points)):
            wps.append(self.optimal_pass_points[i])
            types.append(1) # 1 = Gate
            normals.append(self.gate_normals[i])

        return np.array(wps), np.array(types), np.array(normals)

    def _compute_hermite_tangents(self):
        num_pts = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)

        for i in range(num_pts):
            # Calculate distance to neighbors for scaling
            dist_prev = np.linalg.norm(self.waypoints[i] - self.waypoints[i - 1]) if i > 0 else 0
            dist_next = (
                np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i]) if i < num_pts - 1 else 0
            )
            
            # Safe heuristics for tangent magnitude
            # If it's the last point, we only have dist_prev
            if i == num_pts - 1:
                base_scale = dist_prev
            elif i == 0:
                base_scale = dist_next
            else:
                base_scale = min(dist_prev, dist_next)
                
            scale = base_scale * self.TANGENT_SCALE_FACTOR

            if self.wp_types[i] == 1:
                # --- FIX START ---
                # Retrieve the geometric normal of the gate
                normal = self.wp_normals[i].copy()
                
                # Check Flow Direction: 
                # Determine the vector of the drone arriving at this gate
                if i > 0:
                    incoming_vec = self.waypoints[i] - self.waypoints[i - 1]
                    
                    # If the drone is moving AGAINST the gate normal (dot product < 0),
                    # it implies we are entering from the "front".
                    # If dot product > 0, we are entering from the "back".
                    # We want the tangent to match the flow, so:
                    
                    # If the normal points OPPOSITE to movement, flip it to match movement? 
                    # No, we want the tangent to carry the drone THROUGH.
                    
                    # 1. Project flow onto the normal to see alignment
                    alignment = np.dot(normal, incoming_vec)
                    
                    # 2. If alignment is negative, the Normal and Flow are opposed.
                    #    (e.g. Normal is [1,0,0], Flow is [-1,0,0]).
                    #    Ideally, the tangent should be [-1,0,0] to continue momentum.
                    if alignment < 0:
                        normal = -normal # Flip normal to match incoming flow
                        
                # Assign the flow-aligned normal as the tangent
                tangents[i] = normal * scale
                # --- FIX END ---
                
            else:
                # Catmull-Rom style tangent for non-gate points (Start/End points)
                if i == 0:
                    t = self.waypoints[i + 1] - self.waypoints[i]
                elif i == num_pts - 1:
                    t = self.waypoints[i] - self.waypoints[i - 1]
                else:
                    t = self.waypoints[i + 1] - self.waypoints[i - 1]
                
                if np.linalg.norm(t) > 1e-6:
                    t = t / np.linalg.norm(t)
                tangents[i] = t * scale
                
        return tangents
    
    
    def _generate_parallel_transport_frame(self, num_points=1000):
        s_eval = np.linspace(0, self.total_length, num_points)
        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": []}

        # Initialize Frame using Bishop's Frame (Parallel Transport)
        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)
        
        # Robust initial normal finding
        g_vec = np.array([0, 0, -1]) # Gravity down
        
        # If taking off vertically, use X axis as reference
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = np.cross(t0, np.cross(g_vec, t0)) # Horizontal-ish vector

        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            
            # Calculate next tangent
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

            # Rotation from curr_t to next_t
            axis = np.cross(curr_t, next_t)
            sin_phi = np.linalg.norm(axis)
            cos_phi = np.dot(curr_t, next_t)

            next_n1 = curr_n1
            next_n2 = curr_n2

            # Apply parallel transport rotation if direction changes
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
        """Calculates safe corridor bounds (w1) using 2D GROUND PROJECTION."""
        print(f"[Geometry] Generating bounds (2D Ground Projection)...")
        num_pts = len(self.pt_frame["s"])
        
        # Initialize with max width
        lb_w1 = np.full(num_pts, -self.MAX_LATERAL_WIDTH)
        ub_w1 = np.full(num_pts, self.MAX_LATERAL_WIDTH)

        if len(self.obstacles_pos) == 0:
            return {"lb_w1": lb_w1, "ub_w1": ub_w1}

        # Iterate path points
        for i in range(num_pts):
            frame_pos = self.pt_frame["pos"][i]
            frame_t = self.pt_frame["t"][i]

            # Project to 2D Ground plane (Z=0)
            pos_2d = np.array([frame_pos[0], frame_pos[1], 0.0])
            t_2d = np.array([frame_t[0], frame_t[1], 0.0])
            
            norm_t = np.linalg.norm(t_2d)
            if norm_t < 1e-3: continue # Skip vertical segments
            t_2d /= norm_t
            
            # 2D Normal vector [-y, x]
            n1_2d = np.array([-t_2d[1], t_2d[0], 0.0])

            for obs in self.obstacles_pos:
                obs_2d = np.array([obs[0], obs[1], 0.0])
                r_vec_2d = obs_2d - pos_2d

                # Longitudinal distance (Are we alongside the obstacle?)
                d_long = np.dot(r_vec_2d, t_2d)
                if abs(d_long) > self.SAFETY_RADIUS: continue

                # Lateral distance (How far left/right?)
                w1_obs = np.dot(r_vec_2d, n1_2d)

                # Filter far obstacles
                if abs(w1_obs) > (self.MAX_LATERAL_WIDTH + self.SAFETY_RADIUS + 0.5): continue

                # Constrain corridor
                if w1_obs > 0: # Obstacle on Left -> Shrink UB
                    safe_edge = w1_obs - self.SAFETY_RADIUS
                    if safe_edge < ub_w1[i]:
                        ub_w1[i] = safe_edge
                        # Debug logic could go here
                else: # Obstacle on Right -> Shrink LB
                    safe_edge = w1_obs + self.SAFETY_RADIUS
                    if safe_edge > lb_w1[i]:
                        lb_w1[i] = safe_edge

        # Cleanup collapsed corridors
        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05

        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    def plot(self):
        """Visualizes Path, Optimized Gates, and Corridor."""
        print("[Geometry] Generating plot...")
        fig = go.Figure()

        # 1. Plot Path
        path = self.pt_frame["pos"]
        fig.add_trace(go.Scatter3d(x=path[:,0], y=path[:,1], z=path[:,2], 
                                   mode="lines", line=dict(color="black", width=4), name="Optimized Path"))

        # 2. Plot Original Gate Centers (Ghost)
        fig.add_trace(go.Scatter3d(x=self.gates_pos[:,0], y=self.gates_pos[:,1], z=self.gates_pos[:,2],
                                   mode="markers", marker=dict(size=4, color="gray", symbol="x"), name="Gate Center (Ref)"))

        # 3. Plot Actual Pass-Through Points (Optimized)
        fig.add_trace(go.Scatter3d(x=self.optimal_pass_points[:,0], y=self.optimal_pass_points[:,1], z=self.optimal_pass_points[:,2],
                                   mode="markers", marker=dict(size=6, color="blue"), name="Optimal Pass Point"))

        # 4. Plot Gates Geometry
        gate_w, gate_h = self.GATE_WIDTH/2, self.GATE_HEIGHT/2
        for i, pos in enumerate(self.gates_pos):
            # Create rectangle corners in local frame
            y_v, z_v = self.gate_y[i], self.gate_z[i]
            c = [pos + y_v*gate_w + z_v*gate_h, pos - y_v*gate_w + z_v*gate_h,
                 pos - y_v*gate_w - z_v*gate_h, pos + y_v*gate_w - z_v*gate_h,
                 pos + y_v*gate_w + z_v*gate_h]
            c = np.array(c)
            fig.add_trace(go.Scatter3d(x=c[:,0], y=c[:,1], z=c[:,2], mode="lines", 
                                       line=dict(color="rgba(0,0,255,0.5)"), showlegend=False))

        # 5. Plot Corridor Bounds
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

# --- Config Loader Stub ---
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
    # Test Data: A sharp 90 degree turn
    # Gate 1: Facing Y, at (5, 5, 2)
    # Gate 2: Facing X, at (10, 0, 2)
    toml_path = "config/level1.toml"
    
    s_pos = [0, -4, 0]
    g_pos = [[0, 0, 0], [4, 4, 0]]
    g_norm = [[1, 0, 0], [0, 1, 0]]
    g_y = [[0, 1, 0], [-1, 0, 0]]
    g_z = [[0, 0, 1], [0, 0, 1]]
    # Add a dummy obstacle near the path to test boundaries
    obs_pos = [[2.0, 1.5, 0]]

    if os.path.exists(toml_path):
        try:
            g_pos, g_norm, g_y, g_z, obs_pos, s_pos = load_from_toml(toml_path)
        except Exception:
            pass


    geom = GeometryEngine(g_pos, g_norm, g_y, g_z, obs_pos, s_pos, gate_dims=(0.3, 0.3))
    geom.plot()