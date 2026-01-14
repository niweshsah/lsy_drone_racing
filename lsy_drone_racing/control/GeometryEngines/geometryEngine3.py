from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# Assuming imports work in your local environment
from lsy_drone_racing.control.common_functions.yaml_import import load_yaml
CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")

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
        initial_obs: dict[str, NDArray[np.floating]] = {}, 
        info: dict = {}, 
        sim_config: dict = {}
    ):
        """Initializes the geometry engine and generates the Gate-Traversing path."""
        # --- 1. Configuration Constants ---
        self.MAX_LATERAL_WIDTH = CONSTANTS["max_lateral_width"]
        self.safety_radius = CONSTANTS["safety_radius"]
        self.OBSTACLE_CLEARANCE = 0.5 # Default buffer if not in constants
        
        # Debug / Visualization containers
        self.debug_vectors = []

        # --- 2. State Initialization ---
        self.__initialize_planner_state(initial_obs, sim_config)
        
        # --- 3. Trajectory Generation (Spline based on S-knots) ---
        self.__plan_initial_trajectory()

        # --- 4. Frame and Corridor Generation ---
        # Note: self.total_length and self.spline are set in __plan_initial_trajectory
        num_frame_points = int(self.total_length * 100)
        self.pt_frame = self._generate_parallel_transport_frame(num_points=num_frame_points)
        self.corridor_map = self._generate_static_corridor()
        
    def __initialize_planner_state(self, initial_obs, sim_config):
        self.__current_step = 0
        self.__control_freq = sim_config.env.freq if hasattr(sim_config, 'env') else 60.0

        # Geometry Data
        self.__gate_positions = initial_obs.get("gates_pos", np.array([]))
        self.__obstacle_positions = initial_obs.get("obstacles_pos", np.array([]))
        self.__start_position = initial_obs.get("pos", np.zeros(3))
        
        # Extract Frames
        if "gates_quat" in initial_obs:
            self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
                initial_obs["gates_quat"]
            )
        else:
            # Fallback if quaternions are not provided directly
            self.__gate_normals = np.array([[1, 0, 0]]) 
            self.__gate_y_axes = np.array([[0, 1, 0]])
            self.__gate_z_axes = np.array([[0, 0, 1]])

        self.spline = None
        self.total_length = 0.0

    def __plan_initial_trajectory(self):
        """Generates the path points and fits an arc-length parameterized Hermite spline."""
        # 1. Generate waypoints through gates (Entry -> Center -> Exit)
        path_points = self.__generate_gate_approach_points(
            self.__start_position, self.__gate_positions, self.__gate_normals
        )
        
        # 2. Add detour logic (Optional placeholder based on original code)
        path_points = self.__add_detour_logic(
            path_points,
            self.__gate_positions,
            self.__gate_normals,
            self.__gate_y_axes,
            self.__gate_z_axes,
        )
        
        # 3. Obstacle Avoidance (Insert points around obstacles)
        # Note: __insert_obstacle_avoidance_points implementation was missing in snippet, 
        # assuming it returns modified path_points.
        # path_points = self.__insert_obstacle_avoidance_points(path_points, ...)

        # 4. Generate Spline
        self.spline, self.total_length = self.__compute_trajectory_spline(path_points)

    def __compute_trajectory_spline(self, path_points: NDArray) -> Tuple[CubicHermiteSpline, float]:
        """
        Creates a CubicHermiteSpline parameterized by arc length (s).
        """
        # 1. Calculate Euclidean distances between consecutive points
        diffs = np.diff(path_points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        
        # Ensure no zero-length segments to prevent division errors
        dists = np.maximum(dists, 1e-6)

        # 2. Compute s_knots (cumulative distance)
        s_knots = np.concatenate([[0.0], np.cumsum(dists)])
        total_length = s_knots[-1]

        # 3. Compute Tangents (Derivatives w.r.t s)
        # For arc-length parameterization, the tangent magnitude must be 1.0 (Unit Vectors).
        # We estimate directions using finite differences.
        num_points = len(path_points)
        tangents = np.zeros((num_points, 3))

        # Start Tangent: Direction to second point
        tangents[0] = (path_points[1] - path_points[0]) / dists[0]

        # End Tangent: Direction from second-to-last point
        tangents[-1] = (path_points[-1] - path_points[-2]) / dists[-1]

        # Internal Tangents: Normalized vector between prev and next (Central Difference)
        # Or simpler: (P_next - P_curr).normalized + (P_curr - P_prev).normalized
        for i in range(1, num_points - 1):
            v_prev = (path_points[i] - path_points[i-1]) / dists[i-1]
            v_next = (path_points[i+1] - path_points[i]) / dists[i]
            
            # Average direction
            t_avg = v_prev + v_next
            norm_t = np.linalg.norm(t_avg)
            
            if norm_t > 1e-6:
                tangents[i] = t_avg / norm_t
            else:
                tangents[i] = v_next # Fallback

        # 4. Create Spline
        # scipy.interpolate.CubicHermiteSpline(x, y, dydx)
        spline = CubicHermiteSpline(s_knots, path_points, tangents)
        
        return spline, total_length

    def __generate_gate_approach_points(
        self, initial_pos, gate_pos, gate_norm, approach_dist=0.5, num_pts=3
    ):
        """Generates points before, at, and after the gate to ensure straight entry."""
        # Using 3 points (Entry, Center, Exit) is usually sufficient for Hermite splines
        # to enforce direction.
        offsets = np.linspace(-approach_dist, approach_dist, num_pts)
        
        gate_pos_exp = gate_pos[:, np.newaxis, :]   # (N_gates, 1, 3)
        gate_norm_exp = gate_norm[:, np.newaxis, :] # (N_gates, 1, 3)
        offsets_exp = offsets[np.newaxis, :, np.newaxis] # (1, num_pts, 1)
        
        # Broadcast to create waypoints
        waypoints_matrix = gate_pos_exp + offsets_exp * gate_norm_exp
        flat_waypoints = waypoints_matrix.reshape(-1, 3)
        
        return np.vstack([initial_pos, flat_waypoints])

    def __add_detour_logic(self, path_points, *args):
        # Placeholder for actual detour logic from your original codebase
        return path_points

    def __extract_gate_frames(self, gates_quaternions):
        rotations = R.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        normals = rotation_matrices[:, :, 0]
        y_axes = rotation_matrices[:, :, 1]
        z_axes = rotation_matrices[:, :, 2]
        return normals, y_axes, z_axes

    def _generate_parallel_transport_frame(self, num_points=3000):
        # Evaluate along arc length s
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]
        
        frames = {
            "s": s_eval,
            "pos": [], "t": [], "n1": [], "n2": [],
            "k1": [], "k2": [], "dk1": [], "dk2": [],
        }

        # Initial Frame Setup
        # 1st derivative of Hermite Spline w.r.t s is the tangent
        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)
        
        g_vec = np.array([0, 0, -1]) # Gravity reference
        
        # Handle case where t0 is parallel to gravity
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = g_vec - np.dot(g_vec, t0) * t0
            
        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)
        
        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        k1_list, k2_list = [], []
        
        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            
            # Curvature vector (2nd derivative)
            k_vec = self.spline(s, 2) 
            
            k1 = np.dot(k_vec, curr_n1)
            k2 = np.dot(k_vec, curr_n2)
            
            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            k1_list.append(k1)
            k2_list.append(k2)
            
            # Bishop Frame Propagation (Parallel Transport)
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t_norm = np.linalg.norm(next_t)
                if next_t_norm > 1e-6:
                    next_t /= next_t_norm
                
                # Rotation from curr_t to next_t
                axis = np.cross(curr_t, next_t)
                dot_prod = np.clip(np.dot(curr_t, next_t), -1.0, 1.0)
                angle = np.arccos(dot_prod)
                
                if np.linalg.norm(axis) > 1e-6:
                    axis /= np.linalg.norm(axis)
                    r_vec = R.from_rotvec(axis * angle)
                    next_n1 = r_vec.apply(curr_n1)
                    next_n2 = r_vec.apply(curr_n2)
                else:
                    next_n1, next_n2 = curr_n1, curr_n2
                
                curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        frames["k1"] = np.array(k1_list)
        frames["k2"] = np.array(k2_list)
        frames["dk1"] = np.gradient(frames["k1"], ds)
        frames["dk2"] = np.gradient(frames["k2"], ds)
        
        # Convert lists to arrays
        for k in frames:
            if isinstance(frames[k], list):
                frames[k] = np.array(frames[k])
        return frames

    def _generate_static_corridor(self):
        """Generates corridor bounds based on obstacle positions."""
        print(f"[Geometry] Pre-computing static corridor. Safety Radius: {self.safety_radius}")
        num_pts = len(self.pt_frame["s"])

        w_max = self.MAX_LATERAL_WIDTH
        lb_w1 = np.full(num_pts, -w_max)
        ub_w1 = np.full(num_pts, w_max)

        if len(self.__obstacle_positions) == 0:
            return {"lb_w1": lb_w1, "ub_w1": ub_w1}

        # Iterate through path points
        for i in range(num_pts):
            frame_pos = self.pt_frame["pos"][i]
            frame_t = self.pt_frame["t"][i]

            # 2D Ground Projection Setup
            pos_2d = np.array([frame_pos[0], frame_pos[1], 0.0])
            t_2d = np.array([frame_t[0], frame_t[1], 0.0])
            
            norm_t = np.linalg.norm(t_2d)
            if norm_t < 1e-3: continue
            t_2d /= norm_t
            
            n1_2d = np.array([-t_2d[1], t_2d[0], 0.0]) # Left Normal

            for obs in self.__obstacle_positions:
                obs_2d = np.array([obs[0], obs[1], 0.0])
                r_vec_2d = obs_2d - pos_2d

                # Longitudinal Check
                d_long = np.dot(r_vec_2d, t_2d)
                if abs(d_long) > self.safety_radius + 0.4:
                    continue

                # Lateral Check
                w1_obs = np.dot(r_vec_2d, n1_2d)
                if abs(w1_obs) > (w_max + self.safety_radius + 0.5):
                    continue

                # Apply Constraints
                if w1_obs > 0: # Left side obstacle
                    safe_edge = w1_obs - self.safety_radius
                    if safe_edge < ub_w1[i]:
                        self.debug_vectors.append((frame_pos, obs))
                        ub_w1[i] = safe_edge
                else: # Right side obstacle
                    safe_edge = w1_obs + self.safety_radius
                    if safe_edge > lb_w1[i]:
                        self.debug_vectors.append((frame_pos, obs))
                        lb_w1[i] = safe_edge

        # Cleanup Collapsed Corridor
        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            print(f"[Geometry] WARNING: Corridor collapsed at {np.sum(collapsed)} points.")
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05

        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    def get_frame(self, s_query):
        idx = np.searchsorted(self.pt_frame["s"], s_query) - 1
        idx = np.clip(idx, 0, len(self.pt_frame["s"]) - 1)
        return {k: self.pt_frame[k][idx] for k in self.pt_frame if k != "s"}
    
    def get_closest_s(self, pos_query, s_guess=0.0, window=5.0):
        mask = (self.pt_frame["s"] >= s_guess - 1.0) & (self.pt_frame["s"] <= s_guess + window)
        if not np.any(mask):
            candidates_pos, candidates_s = self.pt_frame["pos"], self.pt_frame["s"]
        else:
            candidates_pos, candidates_s = self.pt_frame["pos"][mask], self.pt_frame["s"][mask]
        dists = np.linalg.norm(candidates_pos - pos_query, axis=1)
        return candidates_s[np.argmin(dists)]

    def get_static_bounds(self, s_query):
        """Lookup pre-computed bounds for a given s."""
        idx = np.searchsorted(self.pt_frame["s"], s_query)
        idx = np.clip(idx, 0, len(self.pt_frame["s"]) - 1)
        return self.corridor_map["lb_w1"][idx], self.corridor_map["ub_w1"][idx]