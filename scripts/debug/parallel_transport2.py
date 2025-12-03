import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import toml

class GeometryEngine:
    def __init__(self, gates_pos, gates_rpy, obstacles_pos, start_pos):
        self.gates_pos = np.asarray(gates_pos)
        self.gates_rpy = np.asarray(gates_rpy)
        self.obstacles_pos = np.asarray(obstacles_pos)
        self.start_pos = np.asarray(start_pos)

        # 1. Gate Orientations
        rot = R.from_euler("xyz", self.gates_rpy)
        self.Rm = rot.as_matrix()
        self.gate_normals = self.Rm[:, :, 0] 
        self.gate_y = self.Rm[:, :, 1]
        self.gate_z = self.Rm[:, :, 2]

        # 2. Waypoints & Tangents
        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        start_tan = self.gates_pos[0] - self.start_pos
        start_tan = start_tan / np.linalg.norm(start_tan)
        self.tangents = np.vstack((start_tan, self.gate_normals))

        # 3. Spline Generation
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # 4. Compute Frame
        self.pt_frame = self._generate_parallel_transport_frame(num_points=1000)

    def _generate_parallel_transport_frame(self, num_points=1000):
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]

        frames = {
            "s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []
        }

        # Init Frame
        t0 = self.spline(0, 1); t0 /= np.linalg.norm(t0)
        g_vec = np.array([0, 0, -1]) 
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = g_vec - np.dot(g_vec, t0) * t0
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            k_vec = self.spline(s, 2)
            
            k1 = -np.dot(k_vec, curr_n1)
            k2 = -np.dot(k_vec, curr_n2)

            next_n1 = curr_n1 + (k1 * curr_t) * ds
            next_n2 = curr_n2 + (k2 * curr_t) * ds

            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i+1], 1); next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

            next_n1 = next_n1 - np.dot(next_n1, next_t) * next_t
            next_n1 /= np.linalg.norm(next_n1)
            next_n2 = np.cross(next_t, next_n1)

            frames["pos"].append(pos); frames["t"].append(curr_t)
            frames["n1"].append(curr_n1); frames["n2"].append(curr_n2)
            frames["k1"].append(k1); frames["k2"].append(k2)

            curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        for k in frames: frames[k] = np.array(frames[k])
        return frames

    def check_curvature_limits(self, desired_radius):
        """
        Returns True if the tunnel is valid, False if it intersects itself.
        """
        k_mag = np.sqrt(self.pt_frame['k1']**2 + self.pt_frame['k2']**2)
        k_max = np.max(k_mag)
        
        # Minimum physical bend radius the path allows
        min_path_radius = 1.0 / k_max if k_max > 0 else np.inf
        
        print(f"-"*40)
        print(f"GEOMETRY CHECK:")
        print(f"  Max Curvature (k):  {k_max:.4f} m^-1")
        print(f"  Min Path Radius:    {min_path_radius:.4f} m")
        print(f"  Desired Tunnel Rad: {desired_radius:.4f} m")
        
        if desired_radius >= min_path_radius:
            print(f"  [FAIL] Tunnel Radius is too large! The tube will collapse.")
            print(f"         Reduce radius below {min_path_radius:.4f} m")
            return False, k_mag
        else:
            safety_margin = (min_path_radius - desired_radius)
            print(f"  [PASS] Geometry Valid. Margin: {safety_margin:.4f} m")
            return True, k_mag

    def plot(self, tunnel_radius=0.35):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        # Run Check
        valid, k_mag = self.check_curvature_limits(tunnel_radius)

        path = self.pt_frame["pos"]
        n1 = self.pt_frame["n1"]
        n2 = self.pt_frame["n2"]

        # --- 1. Plot Tunnel ---
        tube_res = 12
        tube_step = 5
        theta = np.linspace(0, 2 * np.pi, tube_res)
        theta_grid, _ = np.meshgrid(theta, np.arange(0, len(path), tube_step))
        theta_grid = theta_grid.T 

        p_sub = path[::tube_step]   
        n1_sub = n1[::tube_step]    
        n2_sub = n2[::tube_step]    

        cos_t = np.cos(theta)[None, :]
        sin_t = np.sin(theta)[None, :]

        X = p_sub[:, 0, None] + tunnel_radius * (n1_sub[:, 0, None] * cos_t + n2_sub[:, 0, None] * sin_t)
        Y = p_sub[:, 1, None] + tunnel_radius * (n1_sub[:, 1, None] * cos_t + n2_sub[:, 1, None] * sin_t)
        Z = p_sub[:, 2, None] + tunnel_radius * (n1_sub[:, 2, None] * cos_t + n2_sub[:, 2, None] * sin_t)

        # Color logic: Turn tube RED if curvature limit is breached
        if not valid:
            color = 'red'
            alpha = 0.2
        else:
            color = 'cyan'
            alpha = 0.15
            
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, antialiased=False)
        ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.1, rstride=5, cstride=100)

        # --- 2. Visualize Failure Points ---
        # Find points where Radius * Curvature >= 1.0 (Singularity)
        # Using 0.95 as a "Danger Zone" threshold
        danger_idx = np.where(k_mag * tunnel_radius > 0.95)[0]
        if len(danger_idx) > 0:
            p_fail = path[danger_idx]
            ax.scatter(p_fail[:,0], p_fail[:,1], p_fail[:,2], c='red', s=50, marker='x', label='Singularity/Collision')

        # --- 3. Normals & Path ---
        vec_step = 40 
        idx = np.arange(0, len(path), vec_step)
        p_v, n1_v, n2_v = path[idx], n1[idx], n2[idx]

        ax.quiver(p_v[:,0], p_v[:,1], p_v[:,2], n1_v[:,0], n1_v[:,1], n1_v[:,2], color='green', length=0.3, label='n1')
        ax.quiver(p_v[:,0], p_v[:,1], p_v[:,2], n2_v[:,0], n2_v[:,1], n2_v[:,2], color='blue', length=0.3, label='n2')
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'k-', linewidth=2, label="Centerline")

        # --- 4. Gates ---
        for i, (p, n, y, z) in enumerate(zip(self.gates_pos, self.gate_normals, self.gate_y, self.gate_z)):
            s = 0.25 
            corners = np.array([p+s*(y+z), p+s*(y-z), p+s*(-y-z), p+s*(-y+z), p+s*(y+z)])
            ax.plot(corners[:,0], corners[:,1], corners[:,2], 'm-', linewidth=3)
            ax.text(p[0], p[1], p[2], f"G{i}", color='m')

        # --- Settings ---
        ax.set_title(f"Tunnel (r={tunnel_radius}m) | Valid: {valid}")
        max_range = np.array([np.ptp(path[:,0]), np.ptp(path[:,1]), np.ptp(path[:,2])]).max() / 2.0
        mid_x, mid_y, mid_z = np.mean(path[:,0]), np.mean(path[:,1]), np.mean(path[:,2])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.legend()
        plt.savefig("tunnel_plot.png", dpi=300)

        plt.show()

# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------
def load_from_toml(filepath: str):
    with open(filepath, "r") as f:
        data = toml.load(f)
    gates_raw = data["env"]["track"]["gates"]
    gates_pos = np.array([g["pos"] for g in gates_raw], dtype=float)
    gates_rpy = np.array([g.get("rpy", [0, 0, 0]) for g in gates_raw], dtype=float)
    obs_raw = data["env"]["track"].get("obstacles", [])
    obstacles_pos = np.array([o["pos"] for o in obs_raw], dtype=float) if obs_raw else np.empty((0, 3))
    start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=float)
    return gates_pos, gates_rpy, obstacles_pos, start_pos

if __name__ == "__main__":
    try:
        # gates_pos, gates_rpy, obstacles_pos, start_pos = load_from_toml("config/level2.toml")
        gates_pos, gates_rpy, obstacles_pos, start_pos = load_from_toml("config/level2_noObstacle.toml")
    except:
        print("Using dummy data")
        # A very tight turn to demonstrate failure
        gates_pos = [[1,0,1], [1.5, 0.5, 1], [1, 1, 1]] 
        gates_rpy = [[0,0,0], [0,0,1.57], [0,0,3.14]]
        obstacles_pos = []
        start_pos = [0,0,1]

    geom = GeometryEngine(gates_pos, gates_rpy, obstacles_pos, start_pos)
    
    # ---------------------------------------------------
    # TUNNEL RADIUS CHECK
    # ---------------------------------------------------
    # Change this value to test limits
    DESIRED_TUNNEL_RADIUS = 0.14 
    
    geom.plot(tunnel_radius=DESIRED_TUNNEL_RADIUS)