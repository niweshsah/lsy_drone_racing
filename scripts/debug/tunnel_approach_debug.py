import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
import toml
import os

# ==============================================================================
# 1. TRAJECTORY GENERATOR (Exact logic from your Controller)
# ==============================================================================
class TrajectoryGenerator:
    """
    Replicates the logic of MPCSplineController to generate the path
    based on static config data (Gates defined by RPY, not Quats).
    """
    FLIGHT_DURATION = 25.0
    OBSTACLE_CLEARANCE = 0.3 # Slightly larger for visualization safety
    
    def __init__(self, gates_pos, gates_rpy, obstacles_pos, start_pos):
        self.gates_pos = gates_pos
        self.obstacles_pos = obstacles_pos
        self.start_pos = start_pos
        
        # Convert Euler RPY (from TOML) to Rotation Matrices
        # Standard convention in these envs is usually XYZ
        rotations = R.from_euler('xyz', gates_rpy, degrees=False)
        rot_matrices = rotations.as_matrix()
        
        # Extract frames: x=Normal, y=Left, z=Up
        self.gate_normals = rot_matrices[:, :, 0]
        self.gate_y_axes = rot_matrices[:, :, 1]
        self.gate_z_axes = rot_matrices[:, :, 2]

    def generate_spline(self):
        # 1. Approach Points
        path_points = self._generate_gate_approach_points(
            self.start_pos, self.gates_pos, self.gate_normals
        )
        
        # 2. Detours (The complex logic from your controller)
        path_points = self._add_detour_logic(
            path_points, self.gates_pos, self.gate_normals, 
            self.gate_y_axes, self.gate_z_axes
        )
        
        # 3. Obstacles
        time_knots, path_points = self._insert_obstacle_avoidance_points(
            path_points, self.obstacles_pos, self.OBSTACLE_CLEARANCE
        )
        
        # 4. Spline
        spline = self._compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points, custom_time_knots=time_knots
        )
        return spline

    def _generate_gate_approach_points(self, initial_pos, gate_pos, gate_norm, approach_dist=0.5, num_pts=5):
        offsets = np.linspace(-approach_dist, approach_dist, num_pts)
        gate_pos_exp = gate_pos[:, np.newaxis, :]
        gate_norm_exp = gate_norm[:, np.newaxis, :]
        offsets_exp = offsets[np.newaxis, :, np.newaxis]
        waypoints_matrix = gate_pos_exp + offsets_exp * gate_norm_exp
        flat_waypoints = waypoints_matrix.reshape(-1, 3)
        return np.vstack([initial_pos, flat_waypoints])

    def _determine_detour_direction(self, v_proj, v_proj_norm, y_axis, z_axis):
        if v_proj_norm < 1e-6:
            return y_axis
        
        v_proj_y = np.dot(v_proj, y_axis)
        v_proj_z = np.dot(v_proj, z_axis)
        angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi

        if -90 <= angle_deg < 45: return y_axis
        elif 45 <= angle_deg < 135: return z_axis
        else: return -y_axis

    def _add_detour_logic(self, path_points, g_pos, g_norm, g_y, g_z, num_pts=5, angle_deg=120.0, rad=0.65):
        num_gates = g_pos.shape[0]
        pts_list = list(path_points)
        inserts = 0

        for i in range(num_gates - 1):
            last_idx = 1 + (i + 1) * num_pts - 1 + inserts
            first_idx_next = 1 + (i + 1) * num_pts + inserts
            
            if first_idx_next >= len(pts_list): break

            p1 = pts_list[last_idx]
            p2 = pts_list[first_idx_next]
            vec = p2 - p1
            norm = np.linalg.norm(vec)

            if norm < 1e-6: continue

            cos_a = np.dot(vec, g_norm[i]) / norm
            # Check angle
            if np.arccos(np.clip(cos_a, -1, 1)) * 180 / np.pi > angle_deg:
                v_proj = vec - np.dot(vec, g_norm[i]) * g_norm[i]
                detour_vec = self._determine_detour_direction(v_proj, np.linalg.norm(v_proj), g_y[i], g_z[i])
                
                detour_pt = g_pos[i] + rad * detour_vec
                pts_list.insert(last_idx + 1, detour_pt)
                inserts += 1
        return np.array(pts_list)

    def _process_single_obstacle(self, obs_center, sampled_points, sampled_times, clearance):
        collision_free_times = []
        collision_free_points = []
        is_inside = False
        entry_idx = None
        obs_xy = obs_center[:2]

        for i, point in enumerate(sampled_points):
            dist_xy = np.linalg.norm(obs_xy - point[:2])

            if dist_xy < clearance:
                if not is_inside:
                    is_inside = True
                    entry_idx = i
            elif is_inside:
                # Exiting zone
                is_inside = False
                exit_idx = i
                
                entry_pt = sampled_points[entry_idx]
                exit_pt = sampled_points[exit_idx]
                
                # Bisector avoidance
                entry_vec = entry_pt[:2] - obs_xy
                exit_vec = exit_pt[:2] - obs_xy
                avoid_vec = entry_vec + exit_vec
                norm_v = np.linalg.norm(avoid_vec)
                if norm_v > 0: avoid_vec /= norm_v
                
                new_pos_xy = obs_xy + avoid_vec * clearance
                new_pos_z = (entry_pt[2] + exit_pt[2]) / 2
                new_wp = np.concatenate([new_pos_xy, [new_pos_z]])
                
                avg_time = (sampled_times[entry_idx] + sampled_times[exit_idx]) / 2
                collision_free_times.append(avg_time)
                collision_free_points.append(new_wp)
            else:
                collision_free_times.append(sampled_times[i])
                collision_free_points.append(point)
        
        return np.array(collision_free_times), np.array(collision_free_points)

    def _insert_obstacle_avoidance_points(self, path_points, obstacle_centers, clearance):
        temp_spline = self._compute_trajectory_spline(self.FLIGHT_DURATION, path_points)
        num_samples = int(100 * self.FLIGHT_DURATION) 
        sampled_times = np.linspace(0, self.FLIGHT_DURATION, num_samples)
        sampled_points = temp_spline(sampled_times)

        for obs_center in obstacle_centers:
            sampled_times, sampled_points = self._process_single_obstacle(
                obs_center, sampled_points, sampled_times, clearance
            )
        return sampled_times, sampled_points

    def _compute_trajectory_spline(self, total_time, path_points, custom_time_knots=None):
        if custom_time_knots is not None:
            return CubicSpline(custom_time_knots, path_points)
        
        path_segments = np.diff(path_points, axis=0)
        segment_distances = np.linalg.norm(path_segments, axis=1)
        cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)])
        
        if cumulative_distance[-1] == 0:
             return CubicSpline([0, total_time], np.vstack([path_points[0], path_points[0]]))

        time_knots = cumulative_distance / cumulative_distance[-1] * total_time
        return CubicSpline(time_knots, path_points)

# ==============================================================================
# 2. TOML PARSER & PLOTTER
# ==============================================================================

def load_from_toml(filepath):
    """Loads environment data from level2.toml"""
    with open(filepath, 'r') as f:
        data = toml.load(f)
    
    # Gates
    gates_data = data["env"]["track"]["gates"]
    gates_pos = np.array([g["pos"] for g in gates_data])
    gates_rpy = np.array([g["rpy"] for g in gates_data]) # Euler angles
    
    # Obstacles
    obs_data = data["env"]["track"].get("obstacles", [])
    obstacles_pos = np.array([o["pos"] for o in obs_data]) if obs_data else np.empty((0, 3))
    
    # Drone Start (Use first drone)
    drones_data = data["env"]["track"]["drones"]
    start_pos = np.array(drones_data[0]["pos"])
    
    return gates_pos, gates_rpy, obstacles_pos, start_pos

def plot_gate_3d(ax, center, rpy, size=0.5):
    """Draws a gate as a wireframe square with direction arrow."""
    rot = R.from_euler('xyz', rpy, degrees=False).as_matrix()
    
    # Gate Frame: X is normal (flight direction), Y is width, Z is height
    # Corners in local frame (Y-Z plane)
    w = size
    h = size
    # 4 Corners
    local_corners = np.array([
        [0, -w, -h], # Bottom Left
        [0, -w,  h], # Top Left
        [0,  w,  h], # Top Right
        [0,  w, -h]  # Bottom Right
    ])
    
    # Transform to global
    global_corners = (rot @ local_corners.T).T + center
    
    # Draw Square
    poly = Poly3DCollection([global_corners], alpha=0.2, facecolors='cyan', edgecolors='blue', linewidths=2)
    ax.add_collection3d(poly)
    
    # Draw Normal Arrow (Direction)
    normal = rot[:, 0]
    ax.quiver(center[0], center[1], center[2], 
              normal[0], normal[1], normal[2], 
              length=0.5, color='black', arrow_length_ratio=0.3)

def plot_obstacle_3d(ax, center, radius=0.15, height=1.5):
    """Draws a cylinder for an obstacle."""
    # Cylinder mesh
    z = np.linspace(0, height, 10)
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.6, color='red')

def main():
    config_file = "config/level2.toml"
    
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found in current directory.")
        return

    print(f"Loading {config_file}...")
    gates_pos, gates_rpy, obstacles_pos, start_pos = load_from_toml(config_file)
    
    print(f"Found {len(gates_pos)} gates, {len(obstacles_pos)} obstacles.")
    print(f"Start Pos: {start_pos}")

    # --- Generate Trajectory ---
    gen = TrajectoryGenerator(gates_pos, gates_rpy, obstacles_pos, start_pos)
    spline = gen.generate_spline()
    
    # Sample Path
    t_eval = np.linspace(0, gen.FLIGHT_DURATION, 1000)
    path_points = spline(t_eval)

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Start
    ax.scatter(*start_pos, color='green', s=100, label='Start', marker='o')
    
    # 2. Plot Gates
    for i, (pos, rpy) in enumerate(zip(gates_pos, gates_rpy)):
        plot_gate_3d(ax, pos, rpy)
        ax.text(pos[0], pos[1], pos[2] + 0.6, f"G{i}", color='blue')

    # 3. Plot Obstacles
    for pos in obstacles_pos:
        plot_obstacle_3d(ax, pos)

    # 4. Plot Path
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 
            color='lime', linewidth=2, label='Spline Trajectory')

    # Settings
    ax.set_title("Level 2 Config - Generated MPC Spline Path")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    # Set limits based on config safety limits
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(0, 2.0)
    
    # Aspect Ratio Fix (Approximate)
    ax.set_box_aspect([1, 0.8, 0.4]) # Scaled roughly to match env dimensions

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()