import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import toml
import os

class GeometryEngine:
    def __init__(self, gates_pos, gates_normals, gates_y, gates_z, obstacles_pos, start_pos):
        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.gate_y = np.asarray(gates_y, dtype=np.float64)
        self.gate_z = np.asarray(gates_z, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)

        # --- Configuration ---
        self.tube_length_ratio = 0.3    
        self.tube_min_len = 0.5         
        self.tube_max_len = 1.0         
        self.detour_radius = 1.0        
        self.detour_angle_threshold = 0.0 # Degrees deviation allowed

        # 1. Determine Flow
        self.gate_signs = self._determine_gate_orientations()

        # 2. Generate Path
        self.aug_waypoints, self.aug_tangents, self.detour_info = self._generate_augmented_path()

        # 3. Spline
        dists = np.linalg.norm(np.diff(self.aug_waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        self.spline = CubicHermiteSpline(self.s_knots, self.aug_waypoints, self.aug_tangents)
        
        # 4. Frames
        self.pt_frame = self._generate_parallel_transport_frame(num_points=2000)

    def _determine_gate_orientations(self):
        signs = []
        rough_path = np.vstack([self.start_pos, self.gates_pos])
        for i in range(len(self.gates_pos)):
            path_idx = i + 1 
            prev_p = rough_path[path_idx - 1]
            curr_p = rough_path[path_idx]
            
            if path_idx < len(rough_path) - 1:
                next_p = rough_path[path_idx + 1]
                v_out = next_p - curr_p
            else:
                v_out = curr_p - prev_p

            v_in = curr_p - prev_p
            flow = v_in + v_out
            if np.linalg.norm(flow) > 1e-6: flow /= np.linalg.norm(flow)
            
            normal = self.gate_normals[i]
            alignment = np.dot(flow, normal)
            signs.append(1.0 if alignment >= 0 else -1.0)
        return np.array(signs)

    def _determine_detour_direction(self, v_proj, y_axis, z_axis):
        v_proj_norm = np.linalg.norm(v_proj)
        if v_proj_norm < 1e-6: return y_axis 
        v_proj_y = np.dot(v_proj, y_axis)
        v_proj_z = np.dot(v_proj, z_axis)
        proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi

        if -45 <= proj_angle_deg < 45: return y_axis
        elif 45 <= proj_angle_deg < 135: return z_axis
        elif -135 <= proj_angle_deg < -45: return -z_axis
        else: return -y_axis

    def _generate_augmented_path(self):
        wps = [self.start_pos]
        
        # Initial Tangent
        g0_normal = self.gate_normals[0] * self.gate_signs[0]
        t_start = (self.gates_pos[0] - g0_normal) - self.start_pos
        if np.linalg.norm(t_start) > 1e-6: t_start /= np.linalg.norm(t_start)
        else: t_start = g0_normal
        tans = [t_start]
        
        detour_info = [] 
        num_gates = len(self.gates_pos)

        for i in range(num_gates):
            # --- Current Gate Config ---
            pos = self.gates_pos[i]
            n_raw = self.gate_normals[i]
            sign = self.gate_signs[i]
            n_eff = n_raw * sign # Effective Normal (Direction of Flight)
            
            y_ax = self.gate_y[i]
            z_ax = self.gate_z[i]
            
            prev_pt = wps[-1]

            # --- Calculate Current Gate Geometry ---
            # 1. Incoming Distance
            dist_in = np.linalg.norm(pos - prev_pt)
            tube_len_in = np.clip(dist_in * self.tube_length_ratio, self.tube_min_len, self.tube_max_len)
            p_entry = pos - (n_eff * tube_len_in)

            # 2. Outgoing Distance & Exit Point
            if i < num_gates - 1: 
                dist_out = np.linalg.norm(self.gates_pos[i+1] - pos)
            else: 
                dist_out = 5.0 
            
            tube_len_out = np.clip(dist_out * self.tube_length_ratio, self.tube_min_len, self.tube_max_len)
            p_exit = pos + (n_eff * tube_len_out)

            # --- NEW ANGLE LOGIC: Gate Normal vs (Exit -> Next Entry) ---
            vec_check_dir = n_eff # Default if no deviation
            check_origin = p_exit # For visualization

            if i < num_gates - 1:
                # Retrieve Next Gate Info to find its Entry Point
                pos_next = self.gates_pos[i+1]
                n_eff_next = self.gate_normals[i+1] * self.gate_signs[i+1]
                
                # Distance for next gate's input tube (approx distance between gates)
                dist_connector = np.linalg.norm(pos_next - pos)
                tube_len_in_next = np.clip(dist_connector * self.tube_length_ratio, self.tube_min_len, self.tube_max_len)
                
                # The Entry point of the NEXT gate
                p_next_entry = pos_next - (n_eff_next * tube_len_in_next)
                
                # Vector from Current Exit -> Next Entry
                vec_departure = p_next_entry - p_exit
                if np.linalg.norm(vec_departure) > 1e-6:
                    vec_check_dir = vec_departure / np.linalg.norm(vec_departure)
            else:
                # Last gate: No next gate to check against. 
                # Assume straight exit (0 deviation)
                vec_check_dir = n_eff

            # Measure angle between Current Gate Normal and Departure Vector
            alignment = np.dot(vec_check_dir, n_eff)
            alignment = np.clip(alignment, -1.0, 1.0)
            angle_deviation = np.arccos(alignment) * 180.0 / np.pi

            # --- Detour Logic ---
            if angle_deviation > self.detour_angle_threshold:
                print(f"[Geometry] Gate {i} DETOUR TRIGGERED")
                print(f"   Departure Vector (Exit->NextEntry): {vec_check_dir}")
                print(f"   Gate Normal (Effective):             {n_eff}")
                print(f"   Angle: {angle_deviation:.2f} deg")
                print(f" threshold: {self.detour_angle_threshold} deg")

                # Project deviation to find best side (Left/Right/Top/Bottom)
                v_proj = vec_check_dir - np.dot(vec_check_dir, n_eff) * n_eff
                detour_dir = self._determine_detour_direction(v_proj, y_ax, z_ax)

                # Add Detour Point (Still added at ENTRY to set up the turn)
                p_detour = p_entry - (n_eff * 0.5) + (detour_dir * self.detour_radius)
                t_detour = p_entry - p_detour
                t_detour /= np.linalg.norm(t_detour)

                wps.append(p_detour)
                tans.append(t_detour)
                
                detour_info.append({
                    'index': len(wps) - 1, # Index of detour point in wps list
                    'angle': angle_deviation,
                    'vec_path': vec_check_dir,
                    'vec_norm': n_eff,
                    'pt_origin': check_origin # Draw arrows starting at Exit
                })

            # Add Safety Tube Waypoints
            wps.extend([p_entry, pos, p_exit])
            tans.extend([n_eff, n_eff, n_eff])

        return np.array(wps), np.array(tans), detour_info

    def _generate_parallel_transport_frame(self, num_points=1000):
        s_eval = np.linspace(0, self.total_length, num_points)
        frames = { "pos": [] }
        for s in s_eval:
            frames["pos"].append(self.spline(s))
        frames["pos"] = np.array(frames["pos"])
        return frames

    def plot(self):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        path = self.pt_frame["pos"]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='cyan', linewidth=2, label="Drone Path")
        ax.scatter(self.aug_waypoints[:,0], self.aug_waypoints[:,1], self.aug_waypoints[:,2], c='k', s=10)

        # Plot Detour Logic
        for info in self.detour_info:
            idx = info['index']
            pt_detour = self.aug_waypoints[idx]
            
            # Draw Detour Point
            ax.scatter(pt_detour[0], pt_detour[1], pt_detour[2], c='orange', marker='X', s=150, zorder=10)
            ax.text(pt_detour[0], pt_detour[1], pt_detour[2], f"{info['angle']:.1f}°", color='orange', fontsize=12, fontweight='bold')

            # Draw Comparison Vectors
            origin = info['pt_origin']
            v_path = info['vec_path']
            v_norm = info['vec_norm']
            
            # Blue Arrow = Departure Path
            ax.quiver(origin[0], origin[1], origin[2], 
                      v_path[0], v_path[1], v_path[2], 
                      color='blue', arrow_length_ratio=0.2, linewidth=3, label='Departure Path')

            # Red Arrow = Gate Normal
            ax.quiver(origin[0], origin[1], origin[2], 
                      v_norm[0], v_norm[1], v_norm[2], 
                      color='red', arrow_length_ratio=0.2, linewidth=3, label='Gate Normal')

        # Plot Gates
        for i, (p, n, y, z) in enumerate(zip(self.gates_pos, self.gate_normals, self.gate_y, self.gate_z)):
            s = 0.45 
            corners = np.array([p + s*(y + z), p + s*(y - z), p + s*(-y - z), p + s*(-y + z), p + s*(y + z)])
            ax.plot(corners[:,0], corners[:,1], corners[:,2], color='magenta', linewidth=3)
            ax.text(p[0], p[1], p[2] + 0.6, f"G{i}", color='black', fontsize=12)

        ax.scatter(self.start_pos[0], self.start_pos[1], self.start_pos[2], c='green', s=100, label='Start')

        all_pts = np.vstack([path, self.gates_pos])
        mid_x, mid_y, mid_z = np.mean(all_pts, axis=0)
        r = np.max(np.linalg.norm(all_pts - np.array([mid_x, mid_y, mid_z]), axis=1))
        
        ax.set_xlim(mid_x - r, mid_x + r)
        ax.set_ylim(mid_y - r, mid_y + r)
        ax.set_zlim(mid_z - r, mid_z + r)
        ax.set_title(f"Visualizing Path vs Normal Deviation")
        plt.show()

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
    
    # Sharp Turn Test Case
    s_pos = [0, -4, 0]
    # Gate 0 points X+, Gate 1 is at (2,2) 
    # Logic check: Exit G0 is ~[0.3, 0, 0]. Next Entry is ~[2, 2, 0].
    # Vector is diagonal. Normal is X. Angle should be ~45 deg.
    g_pos = [[0, 0, 0], [2, 2, 0]]
    g_norm = [[1, 0, 0], [0, 1, 0]] 
    g_y = [[0, 1, 0], [-1, 0, 0]]
    g_z = [[0, 0, 1], [0, 0, 1]]
    obs_pos = []

    if os.path.exists(toml_path):
        try:
            g_pos, g_norm, g_y, g_z, obs_pos, s_pos = load_from_toml(toml_path)
        except Exception:
            pass

    geom = GeometryEngine(g_pos, g_norm, g_y, g_z, obs_pos, s_pos)
    geom.plot()