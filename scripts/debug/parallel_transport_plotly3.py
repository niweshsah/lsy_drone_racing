import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import toml
import os

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
        self.DETOUR_ANGLE_THRESHOLD = 60.0  # Degrees
        self.DETOUR_RADIUS = 1.5            # Meters
        self.TANGENT_SCALE_FACTOR = 1.0     
        
        # --- Gate / Obstacle Config ---
        self.GATE_WIDTH = 1.0               # Meters (Inner width)
        self.GATE_HEIGHT = 1.0              # Meters (Inner height)
        self.POLE_HEIGHT = 2.0              # Meters (Obstacle height)
        
        self.MAX_LATERAL_WIDTH = 0.60       
        self.SAFETY_RADIUS = 0.10           # Physical Pole radius
        self.OBSTACLE_CLEARANCE = 0.50      # Clearance buffer

        # --- Debug Storage ---
        self.debug_vectors = []             

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

        # B. Detour Logic
        self.waypoints, self.wp_types, self.wp_normals = self._add_detour_logic(
            self.waypoints, self.wp_types, self.wp_normals
        )

        # C. Obstacle Avoidance
        if len(self.obstacles_pos) > 0:
            self.waypoints, self.wp_types, self.wp_normals = self._apply_obstacle_avoidance(
                self.waypoints, self.wp_types, self.wp_normals
            )

        # D. Compute Tangents
        self.tangents = self._compute_hermite_tangents()

        # E. Generate Spline
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        if self.total_length < 1e-3: self.total_length = 0.1
        
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # F. Generate Frame & Corridor
        num_frame_points = int(max(10, self.total_length * 100)) 
        self.pt_frame = self._generate_parallel_transport_frame(num_points=num_frame_points)
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
        new_wps = [wps[0]]; new_types = [types[0]]; new_normals = [normals[0]]
        for i in range(len(wps) - 1):
            curr_p = wps[i]; next_p = wps[i+1]; curr_type = types[i]
            
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
                        detour_dir = self.gate_z[gate_idx] if np.linalg.norm(proj) < 1e-3 else proj / np.linalg.norm(proj)
                        detour_pos = curr_p + (detour_dir * self.DETOUR_RADIUS) + (gate_norm * 1.5)
                        new_wps.append(detour_pos); new_types.append(2); new_normals.append(np.zeros(3)) 

            new_wps.append(next_p); new_types.append(types[i+1]); new_normals.append(normals[i+1])
        return np.array(new_wps), np.array(new_types), np.array(new_normals)

    def _apply_obstacle_avoidance(self, wps, types, normals):
        print("[Geometry] Scanning for obstacles...")
        dists = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        if np.any(dists < 1e-4): wps += np.random.normal(0, 1e-5, wps.shape)
        
        s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        temp_spline = CubicSpline(s_knots, wps)
        
        total_len = s_knots[-1]
        if total_len < 1e-3: return wps, types, normals

        sample_s = np.linspace(0, total_len, int(total_len * 50))
        sample_pts = temp_spline(sample_s)
        
        final_wps = list(wps); final_types = list(types); final_normals = list(normals)
        points_added = 0
        
        for obs in self.obstacles_pos:
            obs_xy = obs[:2]
            dists_xy = np.linalg.norm(sample_pts[:, :2] - obs_xy, axis=1)
            inside_mask = dists_xy < self.OBSTACLE_CLEARANCE
            
            if np.any(inside_mask):
                indices = np.where(inside_mask)[0]
                entry_pt = sample_pts[indices[0]]
                exit_pt = sample_pts[indices[-1]]
                
                v_entry = entry_pt[:2] - obs_xy
                v_exit = exit_pt[:2] - obs_xy
                push_vec = v_entry + v_exit
                norm_push = np.linalg.norm(push_vec)
                
                if norm_push < 1e-6: push_vec = np.array([0.0, 1.0])
                else: push_vec /= norm_push
                
                avoid_xy = obs_xy + (push_vec * self.OBSTACLE_CLEARANCE * 1.2)
                avoid_z = (entry_pt[2] + exit_pt[2]) / 2
                new_wp = np.array([avoid_xy[0], avoid_xy[1], avoid_z])
                
                collision_s = sample_s[int((indices[0] + indices[-1]) / 2)]
                insert_idx = np.searchsorted(s_knots, collision_s) + points_added
                
                print(f"[Avoidance] Hit Obs at {obs[:2]}. Inserting WP at index {insert_idx}")
                final_wps.insert(insert_idx, new_wp)
                final_types.insert(insert_idx, 3)
                final_normals.insert(insert_idx, np.zeros(3))
                points_added += 1

        return np.array(final_wps), np.array(final_types), np.array(final_normals)

    def _compute_hermite_tangents(self):
        num_pts = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)

        for i in range(num_pts):
            dist_prev = np.linalg.norm(self.waypoints[i] - self.waypoints[i-1]) if i > 0 else 0
            dist_next = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i]) if i < num_pts - 1 else 0
            base_scale = max(1e-3, min(dist_prev if dist_prev > 0 else dist_next, dist_next if dist_next > 0 else dist_prev))
            scale = base_scale * self.TANGENT_SCALE_FACTOR

            if self.wp_types[i] == 1: 
                normal = self.wp_normals[i].copy()
                if i > 0 and i < num_pts - 1:
                    flow_vec = self.waypoints[i+1] - self.waypoints[i-1]
                    if np.dot(normal, flow_vec) < 0: normal = -normal
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
        
        n2_0 = np.cross(t0, np.array([1, 0, 0])) if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3 else np.cross(t0, np.cross(g_vec, t0))
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)
        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            next_t = self.spline(s_eval[i+1], 1) if i < len(s_eval) - 1 else curr_t
            next_t /= np.linalg.norm(next_t)
            axis = np.cross(curr_t, next_t)
            sin_phi = np.linalg.norm(axis)
            cos_phi = np.dot(curr_t, next_t)
            
            next_n1, next_n2 = curr_n1, curr_n2
            if sin_phi > 1e-6:
                axis /= sin_phi
                rot = R.from_rotvec(axis * np.arctan2(sin_phi, cos_phi))
                next_n1 = rot.apply(curr_n1)
                next_n2 = rot.apply(curr_n2)

            frames["pos"].append(pos); frames["t"].append(curr_t)
            frames["n1"].append(curr_n1); frames["n2"].append(curr_n2)
            curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        for k in frames: frames[k] = np.array(frames[k])
        return frames

    def _generate_static_corridor(self) -> Dict[str, NDArray]:
        print(f"[Geometry] Generating bounds (2D). Safety Radius: {self.SAFETY_RADIUS}")
        num_pts = len(self.pt_frame['s'])
        lb_w1 = np.full(num_pts, -self.MAX_LATERAL_WIDTH)
        ub_w1 = np.full(num_pts, self.MAX_LATERAL_WIDTH)
        
        if len(self.obstacles_pos) == 0: return {"lb_w1": lb_w1, "ub_w1": ub_w1}

        for i in range(num_pts):
            frame_pos = self.pt_frame['pos'][i]
            frame_t = self.pt_frame['t'][i]
            pos_2d = np.array([frame_pos[0], frame_pos[1], 0.0])
            t_2d = np.array([frame_t[0], frame_t[1], 0.0])
            if np.linalg.norm(t_2d) < 1e-3: continue
            t_2d /= np.linalg.norm(t_2d)
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
                    if safe_edge < ub_w1[i]:
                        self.debug_vectors.append((frame_pos, obs))
                        ub_w1[i] = safe_edge
                else:
                    safe_edge = w1_obs + self.SAFETY_RADIUS
                    if safe_edge > lb_w1[i]:
                        self.debug_vectors.append((frame_pos, obs))
                        lb_w1[i] = safe_edge

        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05
        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    def plot(self):
        """Visualizes Path, Gates, Obstacles, and Corridor using Plotly."""
        print("[Geometry] Generating interactive Plotly visualization...")
        fig = go.Figure()

        # --- 1. Plot Flight Path ---
        path = self.pt_frame["pos"]
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines', line=dict(color='black', width=4), name='Centerline'
        ))

        # --- 2. Plot Waypoints ---
        color_map = {0: 'green', 1: 'blue', 2: 'red', 3: 'purple'}
        colors = [color_map.get(t, 'black') for t in self.wp_types]
        fig.add_trace(go.Scatter3d(
            x=self.waypoints[:, 0], y=self.waypoints[:, 1], z=self.waypoints[:, 2],
            mode='markers', marker=dict(size=6, color=colors),
            name='Waypoints'
        ))

        # --- 3. [NEW] Plot Gates (Rectangular Frames) ---
        # Using a single trace with "None" separators for performance
        gate_x, gate_y_list, gate_z_list = [], [], []
        
        # Half dimensions
        hw = self.GATE_WIDTH / 2.0
        hh = self.GATE_HEIGHT / 2.0
        
        for i in range(len(self.gates_pos)):
            center = self.gates_pos[i]
            # Get orientation vectors
            y_vec = self.gate_y[i]
            z_vec = self.gate_z[i]
            
            # Calculate 4 corners
            # c1: Top-Left, c2: Top-Right, c3: Bot-Right, c4: Bot-Left
            c1 = center + (y_vec * hw) + (z_vec * hh)
            c2 = center - (y_vec * hw) + (z_vec * hh)
            c3 = center - (y_vec * hw) - (z_vec * hh)
            c4 = center + (y_vec * hw) - (z_vec * hh)
            
            # Append loop: 1->2->3->4->1 -> None
            for p in [c1, c2, c3, c4, c1]:
                gate_x.append(p[0])
                gate_y_list.append(p[1])
                gate_z_list.append(p[2])
            
            # Add separator to break the line between gates
            gate_x.append(None)
            gate_y_list.append(None)
            gate_z_list.append(None)

        fig.add_trace(go.Scatter3d(
            x=gate_x, y=gate_y_list, z=gate_z_list,
            mode='lines',
            line=dict(color='blue', width=5),
            name='Gates'
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
                fig.add_trace(go.Surface(
                    x=X_pole, y=Y_pole, z=Z_pole,
                    colorscale=[[0, 'red'], [1, 'red']], opacity=0.6,
                    showscale=False, name='Obstacle', showlegend=first_obs 
                ))
                first_obs = False

        # --- 5. Plot Corridor Bounds ---
        step = 5
        p_vis = path[::step]; n1_vis = self.pt_frame["n1"][::step]; idx = np.arange(0, len(path), step)
        lb = self.corridor_map["lb_w1"][idx]; ub = self.corridor_map["ub_w1"][idx]
        
        wall_left = p_vis + (n1_vis * ub[:, np.newaxis])
        wall_right = p_vis + (n1_vis * lb[:, np.newaxis])
        
        fig.add_trace(go.Scatter3d(
            x=wall_left[:, 0], y=wall_left[:, 1], z=wall_left[:, 2],
            mode='lines', line=dict(color='orange', width=2), name='Bound L'
        ))
        fig.add_trace(go.Scatter3d(
            x=wall_right[:, 0], y=wall_right[:, 1], z=wall_right[:, 2],
            mode='lines', line=dict(color='orange', width=2), name='Bound R'
        ))

        fig.update_layout(
            title="Interactive Flight Corridor (With Gates & Avoidance)",
            scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()

# --- Config Loader ---
def load_from_toml(filepath: str):
    print(f"Loading config from: {filepath}")
    with open(filepath, "r") as f: data = toml.load(f)
    gates_raw = data["env"]["track"]["gates"]
    gates_pos = np.array([g["pos"] for g in gates_raw], dtype=np.float64)
    gates_rpy = np.array([g.get("rpy", [0, 0, 0]) for g in gates_raw], dtype=np.float64)
    rot = R.from_euler("xyz", gates_rpy, degrees=False)
    matrices = rot.as_matrix()
    gates_normals = matrices[:, :, 0]; gates_y = matrices[:, :, 1]; gates_z = matrices[:, :, 2]
    obs_raw = data["env"]["track"].get("obstacles", [])
    obstacles_pos = np.array([o["pos"] for o in obs_raw], dtype=np.float64) if obs_raw else np.empty((0, 3))
    start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=np.float64)
    return gates_pos, gates_normals, gates_y, gates_z, obstacles_pos, start_pos

if __name__ == "__main__":
    toml_path = "config/level1.toml"
    s_pos = [0, 0, 1]
    g_pos = [[10, 0, 1], [15, 5, 1]]
    g_norm = [[1, 0, 0], [0, 1, 0]]
    g_y = [[0, 1, 0], [-1, 0, 0]] # Rotated 90 deg for 2nd gate
    g_z = [[0, 0, 1], [0, 0, 1]]
    obs_pos = [[5.0, 0.05, 0]] 

    if os.path.exists(toml_path):
        try: g_pos, g_norm, g_y, g_z, obs_pos, s_pos = load_from_toml(toml_path)
        except Exception as e: print(f"Failed to load TOML: {e}")

    geom = GeometryEngine(g_pos, g_norm, g_y, g_z, obs_pos, s_pos)
    geom.plot()