from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.common_functions.yaml_import import load_yaml

CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")
from scipy.optimize import minimize  # noqa: E402


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
        self.MAX_LATERAL_WIDTH = CONSTANTS["max_lateral_width"]
        self.safety_radius = CONSTANTS["safety_radius"] 
        
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
    
    def _generate_gate_approach_points(
        self, initial_pos, gate_pos, gate_norm, approach_dist=0.25, num_pts=2
    ):
        offsets = np.linspace(-approach_dist, approach_dist, num_pts)
        gate_pos_exp = gate_pos[:, np.newaxis, :]
        gate_norm_exp = gate_norm[:, np.newaxis, :]
        offsets_exp = offsets[np.newaxis, :, np.newaxis]
        waypoints_matrix = gate_pos_exp + offsets_exp * gate_norm_exp
        flat_waypoints = waypoints_matrix.reshape(-1, 3)
        return np.vstack([initial_pos, flat_waypoints])

    

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

    def _generate_parallel_transport_frame(self, num_points=3000):
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]
        frames = {
            "s": s_eval,
            "pos": [],
            "t": [],
            "n1": [],
            "n2": [],
            "k1": [],
            "k2": [],
            "dk1": [],
            "dk2": [],
        }

        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)
        g_vec = np.array([0, 0, -1])
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
            k_vec = self.spline(s, 2)
            k1 = np.dot(k_vec, curr_n1)
            k2 = np.dot(k_vec, curr_n2)
            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            k1_list.append(k1)
            k2_list.append(k2)
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t /= np.linalg.norm(next_t)
                axis = np.cross(curr_t, next_t)
                angle = np.arccos(np.clip(np.dot(curr_t, next_t), -1.0, 1.0))
                if np.linalg.norm(axis) > 1e-6:
                    axis /= np.linalg.norm(axis)
                    r_vec = R.from_rotvec(axis * angle)
                    next_n1, next_n2 = r_vec.apply(curr_n1), r_vec.apply(curr_n2)
                else:
                    next_n1, next_n2 = curr_n1, curr_n2
                curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        frames["k1"] = np.array(k1_list)
        frames["k2"] = np.array(k2_list)
        frames["dk1"] = np.gradient(frames["k1"], ds)
        frames["dk2"] = np.gradient(frames["k2"], ds)
        for k in frames:
            if isinstance(frames[k], list):
                frames[k] = np.array(frames[k])
        return frames

    def _generate_static_corridor(self):
        """OFFLINE CALCULATIONS:
        Generates lb_w1 and ub_w1 arrays using 2D GROUND PROJECTION logic.
        Checks if obstacles' 2D footprint intersects the path's 2D footprint.
        """
        print(
            f"[Geometry] Pre-computing static corridor bounds (2D PROJECTION). Safety Radius: {self.safety_radius}"
        )
        num_pts = len(self.pt_frame["s"])

        # Initialize with max corridor width
        w_max = self.MAX_LATERAL_WIDTH
        lb_w1 = np.full(num_pts, -w_max)
        ub_w1 = np.full(num_pts, w_max)

        if len(self.obstacles_pos) == 0:
            return {"lb_w1": lb_w1, "ub_w1": ub_w1}

        # Iterate through every point on the path
        for i in range(num_pts):
            frame_pos = self.pt_frame["pos"][i]
            frame_t = self.pt_frame["t"][i]

            # --- GROUND PROJECTION (Flatten Z to 0) ---
            # 1. Project Path Position to Ground
            pos_2d = np.array([frame_pos[0], frame_pos[1], 0.0])

            # 2. Project Tangent to Ground & Normalize
            # This gives us the "Forward" direction on the map
            t_2d = np.array([frame_t[0], frame_t[1], 0.0])
            norm_t = np.linalg.norm(t_2d)
            if norm_t < 1e-3:
                # Vertical flight? Skip, no lateral definition on ground
                continue
            t_2d /= norm_t

            # 3. Compute 2D Normal (Rotate 90 deg around Z)
            # If t_2d = [x, y], then normal is [-y, x] (Standard 2D left normal)
            n1_2d = np.array([-t_2d[1], t_2d[0], 0.0])

            for obs in self.obstacles_pos:
                # 4. Project Obstacle to Ground
                obs_2d = np.array([obs[0], obs[1], 0.0])

                # Vector from Path (2D) to Obstacle (2D)
                r_vec_2d = obs_2d - pos_2d

                # --- A. Longitudinal Check (2D) ---
                d_long = np.dot(r_vec_2d, t_2d)

                # STRICT FILTER: Is the obstacle shadow 'here' along the track?
                # If d_long > radius, the obstacle is ahead/behind, not 'here'.
                if abs(d_long) > self.safety_radius + 0.4:
                    continue

                # --- B. Lateral Check (2D) ---
                # How far left/right is the obstacle shadow?
                w1_obs = np.dot(r_vec_2d, n1_2d)

                # Optimization: Ignore far obstacles
                if abs(w1_obs) > (w_max + self.safety_radius + 0.5):
                    continue

                # --- C. Apply Constraints ---
                # Check Dominant Side based on 2D footprint

                # Left side obstacle (limits Upper Bound)
                if w1_obs > 0:
                    safe_edge = w1_obs - self.safety_radius
                    if safe_edge < ub_w1[i]:
                        # Store the REAL 3D vector for visualization
                        self.debug_vectors.append((frame_pos, obs))
                        ub_w1[i] = safe_edge

                # Right side obstacle (limits Lower Bound)
                else:
                    safe_edge = w1_obs + self.safety_radius
                    if safe_edge > lb_w1[i]:
                        # Store the REAL 3D vector for visualization
                        self.debug_vectors.append((frame_pos, obs))
                        lb_w1[i] = safe_edge

        # Cleanup: Ensure bounds are valid (lb < ub). If not, path is blocked.
        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            print(f"[Geometry] WARNING: Corridor collapsed at {np.sum(collapsed)} points.")
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05

        print("[Geometry] Corridor generation complete.")
        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    def get_static_bounds(self, s_query):
        """Lookup pre-computed bounds for a given s."""
        idx = np.searchsorted(self.pt_frame["s"], s_query)
        idx = np.clip(idx, 0, len(self.pt_frame["s"]) - 1)
        return self.corridor_map["lb_w1"][idx], self.corridor_map["ub_w1"][idx]

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
