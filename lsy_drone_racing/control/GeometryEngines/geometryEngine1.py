import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.common_functions.yaml_import import load_yaml

CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")


class GeometryEngine:
    def __init__(self, gates_pos, gates_normals, start_pos, obstacles_pos, safety_radius):
        self.DETOUR_ANGLE_THRESHOLD = 60.0
        self.DETOUR_RADIUS = 0.3
        self.TANGENT_SCALE_FACTOR = 1.0

        # Store obstacles for offline generation
        self.obstacles_pos = np.asarray(obstacles_pos)
        self.safety_radius = safety_radius

        # --- DEBUG STORAGE ---
        self.debug_vectors = []

        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)

        self.waypoints, self.wp_types, self.wp_normals = self._initialize_waypoints()
        self.waypoints, self.wp_types, self.wp_normals = self._add_detour_logic(
            self.waypoints, self.wp_types, self.wp_normals
        )
        self.tangents = self._compute_hermite_tangents()

        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # Initialize PT frame with high resolution for offline bounds checking
        self.pt_frame = self._generate_parallel_transport_frame(
            num_points=int(self.total_length * 100)
        )  # ~1cm resolution

        # --- NEW: Generate Bounds Offline ---
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
            next_p = wps[i + 1]
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
                        detour_dir = (
                            proj / np.linalg.norm(proj)
                            if np.linalg.norm(proj) > 1e-3
                            else np.array([0, 0, 1])
                        )
                        detour_pos = curr_p + (detour_dir * self.DETOUR_RADIUS) + (gate_norm * 1.5)
                        new_wps.append(detour_pos)
                        new_types.append(2)
                        new_normals.append(np.zeros(3))
            new_wps.append(next_p)
            new_types.append(types[i + 1])
            new_normals.append(normals[i + 1])
        return np.array(new_wps), np.array(new_types), np.array(new_normals)

    def _compute_hermite_tangents(self):
        num_pts = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)
        for i in range(num_pts):
            dist_prev = np.linalg.norm(self.waypoints[i] - self.waypoints[i - 1]) if i > 0 else 0
            dist_next = (
                np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i]) if i < num_pts - 1 else 0
            )
            base_scale = min(
                dist_prev if dist_prev > 0 else dist_next, dist_next if dist_next > 0 else dist_prev
            )
            scale = base_scale * self.TANGENT_SCALE_FACTOR
            if self.wp_types[i] == 1:
                normal = self.wp_normals[i].copy()
                if i > 0 and i < num_pts - 1:
                    flow_vec = self.waypoints[i + 1] - self.waypoints[i - 1]
                    if np.dot(normal, flow_vec) < 0:
                        normal = -normal
                tangents[i] = normal * scale
            else:
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
        w_max = CONSTANTS["max_lateral_width"]
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
