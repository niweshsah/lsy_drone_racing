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

        # 1. Calculate Gate Orientations
        rot = R.from_euler("xyz", self.gates_rpy)
        self.Rm = rot.as_matrix()
        self.gate_normals = self.Rm[:, :, 0] # The X-axis is the "Forward" direction through the gate
        self.gate_y = self.Rm[:, :, 1]
        self.gate_z = self.Rm[:, :, 2]

        # 2. Prepare Waypoints (Positions)
        # Stack Start Position + Gate Positions
        self.waypoints = np.vstack((self.start_pos, self.gates_pos))

        # 3. Prepare Tangents (Derivatives)
        # The Hermite Spline takes explicit derivatives (directions) for every point.
        
        # A. Start Tangent: We estimate this as the direction towards the first gate
        start_tan = self.gates_pos[0] - self.start_pos
        start_tan = start_tan / np.linalg.norm(start_tan)
        
        # B. Gate Tangents: These are exactly the Gate Normals
        # Stack: [Start_Tangent, Gate_Normal_0, Gate_Normal_1, ...]
        self.tangents = np.vstack((start_tan, self.gate_normals))

        # 4. Parameterize Path (Knots)
        # We use Euclidean distance between points as the knot vector 's'
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]

        # 5. Create Cubic Hermite Spline
        # This guarantees P(s_i) = Waypoint_i AND P'(s_i) = Tangent_i
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # 6. Compute Dense Parallel Transport Frame for MPC
        self.pt_frame = self._generate_parallel_transport_frame(num_points=1000)

    def _generate_parallel_transport_frame(self, num_points=1000):
        """
        Generates the Bishop Frame (t, n1, n2) and Curvatures (k1, k2).
        """
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]

        frames = {
            "s": s_eval, 
            "pos": [], "t": [], "n1": [], "n2": [], 
            "k1": [], "k2": []
        }

        # --- Initialization ---
        # 1. Get initial tangent from Spline (exact)
        t0 = self.spline(0, 1) 
        t0 /= np.linalg.norm(t0)

        # 2. Initial Normal (Gram-Schmidt against Gravity)
        g_vec = np.array([0, 0, -1]) 
        # Handle Singularity if taking off straight up
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = g_vec - np.dot(g_vec, t0) * t0
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        # --- Propagation Loop ---
        for i, s in enumerate(s_eval):
            # Pos
            pos = self.spline(s)
            
            # Curvature Vector (2nd derivative)
            k_vec = self.spline(s, 2)
            
            # Project Curvature onto Frame (MPC Inputs)
            k1 = -np.dot(k_vec, curr_n1)
            k2 = -np.dot(k_vec, curr_n2)

            # Update Frame (Euler Integration)
            next_n1 = curr_n1 + (k1 * curr_t) * ds
            next_n2 = curr_n2 + (k2 * curr_t) * ds

            # Correction Step (Gram-Schmidt against exact tangent)
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i+1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

            # Enforce Orthogonality
            next_n1 = next_n1 - np.dot(next_n1, next_t) * next_t
            next_n1 /= np.linalg.norm(next_n1)
            next_n2 = np.cross(next_t, next_n1)

            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            frames["k1"].append(k1)
            frames["k2"].append(k2)

            curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        for k in frames: frames[k] = np.array(frames[k])
        return frames

    def plot(self):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        # 1. Plot Spline Path
        path = self.pt_frame["pos"]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', linewidth=2, label="Hermite Path")
        
        # 2. Plot Knots (Gates + Start)
        ax.scatter(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,2], c='k', marker='o', s=50)

        # 3. Plot Gates with Orientation
        for i, (p, n, y, z) in enumerate(zip(self.gates_pos, self.gate_normals, self.gate_y, self.gate_z)):
            ax.text(p[0], p[1], p[2]+0.5, f"G{i}", color='m', fontsize=12)
            
            # Gate Box (Visualizing the physical gate)
            s = 0.25 # Half-width
            corners = np.array([
                p+s*(y+z), p+s*(y-z), p+s*(-y-z), p+s*(-y+z), p+s*(y+z)
            ])
            ax.plot(corners[:,0], corners[:,1], corners[:,2], 'm-', linewidth=2)
            
            # Gate Normal (Desired Velocity Direction)
            ax.quiver(*p, *n, color='r', length=0.5, arrow_length_ratio=0.3, label="Gate Normal" if i==0 else "")

        # 4. Highlight High Curvature (Safety Check)
        k_mag = np.sqrt(self.pt_frame['k1']**2 + self.pt_frame['k2']**2)
        
        # Determine strictness based on track length
        limit = 4.0 if self.total_length < 10 else 2.0
        idx_high = np.where(k_mag > limit)[0]
        
        if len(idx_high) > 0:
            spikes = path[idx_high]
            ax.scatter(spikes[:,0], spikes[:,1], spikes[:,2], c='red', marker='x', s=50, label=f"Curvature > {limit}")

        # 5. Graph Settings
        ax.set_title(f"Hermite Spline Geometry (Max K: {np.max(k_mag):.2f})")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        
        # Attempt Equal Aspect Ratio
        try:
            ax.set_aspect('equal')
        except:
            max_range = np.array([path[:,0].max()-path[:,0].min(), 
                                  path[:,1].max()-path[:,1].min(), 
                                  path[:,2].max()-path[:,2].min()]).max() / 2.0
            mid_x = (path[:,0].max()+path[:,0].min()) * 0.5
            mid_y = (path[:,1].max()+path[:,1].min()) * 0.5
            mid_z = (path[:,2].max()+path[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
        plt.legend()
        plt.show()

# ---------------------------------------------------------
# Loader Utility
# ---------------------------------------------------------
def load_from_toml(filepath: str):
    """Load config."""
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
        print("Loaded track from config/level2.toml")
    except:
        print("Using dummy data (config not found)")
        gates_pos = [[1,0,1], [2,1,1]]
        gates_rpy = [[0,0,0], [0,0,1.57]]
        obstacles_pos = []
        start_pos = [0,0,1]

    geom = GeometryEngine(gates_pos, gates_rpy, obstacles_pos, start_pos)
    
    # Analyze Curvature
    k_vec_mag = np.sqrt(geom.pt_frame['k1']**2 + geom.pt_frame['k2']**2)
    k_max = np.max(k_vec_mag)
    
    print(f"Track Length: {geom.total_length:.2f} m")
    print(f"Max Curvature: {k_max:.2f}")
    
    # 4.0 is acceptable for very aggressive racing on tight tracks
    if k_max > 5.0:
        print("WARNING: Curvature is excessively high (Impossible turn).")
    else:
        print("SUCCESS: Path is valid and aligned with gates.")

    geom.plot()