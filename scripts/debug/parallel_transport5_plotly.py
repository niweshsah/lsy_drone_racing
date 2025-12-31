import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import toml
import os
import sys

# Type hinting for clarity
from typing import List, Tuple, Optional, Dict
from numpy.typing import NDArray

class GeometryEngine:
    def __init__(
        self, 
        gates_pos: List[List[float]], 
        gates_normals: List[List[float]], 
        gates_y: List[List[float]], 
        gates_z: List[List[float]], 
        obstacles_pos: List[List[float]], 
        start_pos: List[float]
    ):
        """
        Initializes the geometry engine and generates the safe flight path.
        """
        # --- 1. Configuration Constants ---
        self.DETOUR_ANGLE_THRESHOLD = 60.0  # Degrees. If turn > this, add detour.
        self.DETOUR_RADIUS = 1.0            # Meters. Detour offset.
        self.TANGENT_SCALE_FACTOR = 1.0     # Controls curve aggression.
        
        # --- Corridor / Obstacle Config ---
        self.MAX_LATERAL_WIDTH = 0.45       # Meters (w1 max).
        self.SAFETY_RADIUS = 0.05           # Meters (Pole radius + Drone buffer).
        self.POLE_HEIGHT = 2.0              # Meters (For visualization).

        # --- Debug Storage ---
        self.debug_vectors = []             # Stores (start, end) tuples for visualization

        # --- 2. Data Ingestion ---
        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.gate_y = np.asarray(gates_y, dtype=np.float64)
        self.gate_z = np.asarray(gates_z, dtype=np.float64)
        self.obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)

        # --- 3. Pipeline Execution ---
        
        # A. Initialize Waypoints
        self.waypoints, self.wp_types, self.wp_normals = self._initialize_waypoints()

        # B. Insert Detour Points
        self.waypoints, self.wp_types, self.wp_normals = self._add_detour_logic(
            self.waypoints, self.wp_types, self.wp_normals
        )

        # C. Compute Tangents
        self.tangents = self._compute_hermite_tangents()

        # D. Generate Cubic Hermite Spline
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # E. Generate Parallel Transport Frame (High resolution for bounds checking)
        num_frame_points = int(self.total_length * 100) 
        self.pt_frame = self._generate_parallel_transport_frame(num_points=num_frame_points)

        # F. Generate Static Corridor Boundaries (Offline)
        self.corridor_map = self._generate_static_corridor()

    def _initialize_waypoints(self):
        wps = [self.start_pos]
        types = [0]              
        normals = [np.zeros(3)] 
        
        for i in range(len(self.gates_pos)):
            wps.append(self.gates_pos[i])
            types.append(1)      
            normals.append(self.gate_normals[i])
            
        return np.array(wps), np.array(types), np.array(normals)

    def _add_detour_logic(self, wps, types, normals):
        new_wps = [wps[0]]
        new_types = [types[0]]
        new_normals = [normals[0]]

        for i in range(len(wps) - 1):
            curr_p = wps[i]
            next_p = wps[i+1]
            curr_type = types[i]
            
            if curr_type == 1: 
                gate_idx = i - 1 
                gate_norm = self.gate_normals[gate_idx]
                vec_to_next = next_p - curr_p
                dist = np.linalg.norm(vec_to_next)
                
                if dist > 1e-6:
                    vec_to_next /= dist
                    alignment = np.dot(gate_norm, vec_to_next)
                    angle_deg = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))

                    if angle_deg > self.DETOUR_ANGLE_THRESHOLD:
                        proj = vec_to_next - (np.dot(vec_to_next, gate_norm) * gate_norm)
                        if np.linalg.norm(proj) < 1e-3:
                            detour_dir = self.gate_z[gate_idx]
                        else:
                            detour_dir = proj / np.linalg.norm(proj)
                        
                        detour_pos = curr_p + (detour_dir * self.DETOUR_RADIUS) + (gate_norm * 1.5)
                        new_wps.append(detour_pos)
                        new_types.append(2) 
                        new_normals.append(np.zeros(3)) 

            new_wps.append(next_p)
            new_types.append(types[i+1])
            new_normals.append(normals[i+1])

        return np.array(new_wps), np.array(new_types), np.array(new_normals)

    def _compute_hermite_tangents(self):
        num_pts = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)

        for i in range(num_pts):
            dist_prev = np.linalg.norm(self.waypoints[i] - self.waypoints[i-1]) if i > 0 else 0
            dist_next = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i]) if i < num_pts - 1 else 0
            base_scale = min(dist_prev if dist_prev > 0 else dist_next, 
                             dist_next if dist_next > 0 else dist_prev)
            scale = base_scale * self.TANGENT_SCALE_FACTOR

            if self.wp_types[i] == 1: 
                normal = self.wp_normals[i].copy()
                if i > 0 and i < num_pts - 1:
                    flow_vec = self.waypoints[i+1] - self.waypoints[i-1]
                    if np.dot(normal, flow_vec) < 0:
                        normal = -normal
                tangents[i] = normal * scale
            else:
                if i == 0: t = self.waypoints[i+1] - self.waypoints[i]
                elif i == num_pts - 1: t = self.waypoints[i] - self.waypoints[i-1]
                else: t = self.waypoints[i+1] - self.waypoints[i-1]
                if np.linalg.norm(t) > 1e-6: t = t / np.linalg.norm(t)
                tangents[i] = t * scale
        return tangents

    def _generate_parallel_transport_frame(self, num_points=1000):
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
                next_t = self.spline(s_eval[i+1], 1)
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

        for k in frames: frames[k] = np.array(frames[k])
        return frames

    def _generate_static_corridor(self) -> Dict[str, NDArray]:
        """
        Calculates safe corridor bounds (w1) with strict 3D bounds checking.
        """
        print(f"[Geometry] Generating bounds. Safety Radius: {self.SAFETY_RADIUS}")
        num_pts = len(self.pt_frame['s'])
        
        # Initialize with max corridor width
        lb_w1 = np.full(num_pts, -self.MAX_LATERAL_WIDTH)
        ub_w1 = np.full(num_pts, self.MAX_LATERAL_WIDTH)
        
        if len(self.obstacles_pos) == 0:
            return {"lb_w1": lb_w1, "ub_w1": ub_w1}

        # Iterate through every point on the path
        for i in range(num_pts):
            frame_pos = self.pt_frame['pos'][i]
            frame_n1 = self.pt_frame['n1'][i]  # Lateral Vector (Right/Left)
            frame_n2 = self.pt_frame['n2'][i]  # Vertical Vector (Up/Down)
            frame_t = self.pt_frame['t'][i]    # Tangent Vector (Forward)
            
            for obs_idx, obs in enumerate(self.obstacles_pos):
                r_vec = obs - frame_pos
                
                # --- 1. Longitudinal Check ---
                d_long = np.dot(r_vec, frame_t)
                if abs(d_long) > self.SAFETY_RADIUS:
                    continue

                # --- 2. Vertical Check ---
                d_vert = np.dot(r_vec, frame_n2)
                if abs(d_vert) > 1.0: 
                    continue

                # --- 3. Lateral Check ---
                w1_obs = np.dot(r_vec, frame_n1)
                
                if abs(w1_obs) > (self.MAX_LATERAL_WIDTH + self.SAFETY_RADIUS + 0.5):
                    continue

                # --- 4. Apply Constraints ---
                # Left side obstacle (limits Upper Bound)
                if w1_obs > 0:
                    safe_edge = w1_obs - self.SAFETY_RADIUS
                    if safe_edge < ub_w1[i]:
                        # --- DEBUG: STORE VECTOR ---
                        # Store tuple (start_pos, end_pos) for visualization
                        self.debug_vectors.append((frame_pos, obs))
                        
                        # Print
                        print(f"\n[DEBUG] HIT at PathIdx {i}")
                        print(f"  Drone Pos:   {frame_pos}")
                        print(f"  Obs Pos:     {obs}")
                        print(f"  Distances:   Long={d_long:.4f}, Vert={d_vert:.4f}, Lat={w1_obs:.4f}")
                        print(f"  Action:      Shrink UB {ub_w1[i]:.4f} -> {safe_edge:.4f}")
                        
                        ub_w1[i] = safe_edge
                
                # Right side obstacle (limits Lower Bound)
                else:
                    safe_edge = w1_obs + self.SAFETY_RADIUS
                    if safe_edge > lb_w1[i]:
                        # --- DEBUG: STORE VECTOR ---
                        self.debug_vectors.append((frame_pos, obs))
                        
                        print(f"\n[DEBUG] HIT at PathIdx {i}")
                        print(f"  Drone Pos:   {frame_pos}")
                        print(f"  Obs Pos:     {obs}")
                        print(f"  Distances:   Long={d_long:.4f}, Vert={d_vert:.4f}, Lat={w1_obs:.4f}")
                        print(f"  Action:      Shrink LB {lb_w1[i]:.4f} -> {safe_edge:.4f}")
                        
                        lb_w1[i] = safe_edge

        # Cleanup: Check for collapse
        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            print(f"[Geometry] WARNING: Corridor collapsed at {np.sum(collapsed)} points.")
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05
            
        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    def plot(self):
        """Visualizes Path, Gates, Obstacles, and Corridor using Plotly."""
        print("[Geometry] Generating interactive Plotly visualization...")
        fig = go.Figure()

        # --- 1. Plot Flight Path (Centerline) ---
        path = self.pt_frame["pos"]
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines',
            line=dict(color='black', width=4),
            name='Centerline'
        ))

        # --- 2. Plot Static Wall Boundaries ---
        step = 1 
        p_vis = path[::step]
        n1_vis = self.pt_frame["n1"][::step]
        indices = np.arange(0, len(path), step)
        
        lb = self.corridor_map["lb_w1"][indices]
        ub = self.corridor_map["ub_w1"][indices]
        
        # Compute Wall Coordinates
        wall_left = p_vis + (n1_vis * ub[:, np.newaxis])
        wall_right = p_vis + (n1_vis * lb[:, np.newaxis])
        
        fig.add_trace(go.Scatter3d(
            x=wall_left[:, 0], y=wall_left[:, 1], z=wall_left[:, 2],
            mode='lines',
            line=dict(color='orange', width=3),
            name='Safe Bound (Left)'
        ))

        fig.add_trace(go.Scatter3d(
            x=wall_right[:, 0], y=wall_right[:, 1], z=wall_right[:, 2],
            mode='lines',
            line=dict(color='red', width=3,),
            name='Safe Bound (Right)'
        ))
        
        rung_x, rung_y, rung_z = [], [], []
        for i in range(len(wall_left)):
            rung_x.extend([wall_left[i, 0], wall_right[i, 0], None])
            rung_y.extend([wall_left[i, 1], wall_right[i, 1], None])
            rung_z.extend([wall_left[i, 2], wall_right[i, 2], None])

        fig.add_trace(go.Scatter3d(
            x=rung_x, y=rung_y, z=rung_z,
            mode='lines',
            line=dict(color='orange', width=1),
            opacity=0.3,
            name='Corridor Width'
        ))

        # --- 3. Plot Waypoints ---
        colors = ['green' if t == 0 else 'blue' if t == 1 else 'red' for t in self.wp_types]
        fig.add_trace(go.Scatter3d(
            x=self.waypoints[:, 0], y=self.waypoints[:, 1], z=self.waypoints[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors),
            name='Waypoints'
        ))

        # --- 4. Plot Obstacles ---
        if len(self.obstacles_pos) > 0:
            u = np.linspace(0, 2 * np.pi, 25)
            z_pole = np.linspace(0, self.POLE_HEIGHT, 2)
            U, Z_pole = np.meshgrid(u, z_pole)
            
            first_obs = True
            for obs in self.obstacles_pos:
                X_pole = self.SAFETY_RADIUS * np.cos(U) + obs[0]
                Y_pole = self.SAFETY_RADIUS * np.sin(U) + obs[1]
                Z_shifted = Z_pole + (obs[2] - self.POLE_HEIGHT/2)
                
                fig.add_trace(go.Surface(
                    x=X_pole, y=Y_pole, z=Z_shifted,
                    colorscale=[[0, 'red'], [1, 'red']],
                    opacity=0.6,
                    showscale=False,
                    name='Obstacle',
                    showlegend=first_obs 
                ))
                first_obs = False

        # --- 5. Plot Debug Vectors (The r_vec lines) ---
        if len(self.debug_vectors) > 0:
            vec_x, vec_y, vec_z = [], [], []
            for start, end in self.debug_vectors:
                vec_x.extend([start[0], end[0], None])
                vec_y.extend([start[1], end[1], None])
                vec_z.extend([start[2], end[2], None])
            
            fig.add_trace(go.Scatter3d(
                x=vec_x, y=vec_y, z=vec_z,
                mode='lines',
                line=dict(color='cyan', width=2),
                name='DEBUG: Hit Vectors'
            ))

        # Start Position Marker
        fig.add_trace(go.Scatter3d(
            x=[self.start_pos[0]], y=[self.start_pos[1]], z=[self.start_pos[2]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Start Pos'
        ))

        # --- Layout Settings ---
        fig.update_layout(
            title="Interactive Flight Corridor (Plotly)",
            scene=dict(
                aspectmode='data', 
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        fig.show()

# --- Config Loader ---
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
    obstacles_pos = np.array([o["pos"] for o in obs_raw], dtype=np.float64) if obs_raw else np.empty((0, 3))
    start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=np.float64)
    return gates_pos, gates_normals, gates_y, gates_z, obstacles_pos, start_pos

if __name__ == "__main__":
    toml_path = "config/level1.toml"
    # toml_path = "config/level1_noObstacle.toml"
    
    # Fallback Test Case
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

    geom = GeometryEngine(g_pos, g_norm, g_y, g_z, obs_pos, s_pos)
    geom.plot()