from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import plotly.graph_objects as go


# Assuming imports work in your local environment
from lsy_drone_racing.control.common_functions.yaml_import import load_yaml

CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")


class GeometryEngine:
    def __init__(
        self,
        gates_pos: List[List[float]],
        gates_normal: List[List[float]],
        gates_y: List[List[float]],
        gates_z: List[List[float]],
        gate_size: float = 0.5,
        obstacles_pos: List[List[float]] = [],
        start_pos: List[float] = [-1.5, 0.75, 0.01],
        start_orient: List[float] = [0, 0, 0],
        obs: dict[str, NDArray[np.floating]] = {},
        info: dict = {},
        sim_config: dict = {},
    ):
        self.gates_pos = np.array(gates_pos)
        self.gates_normal = np.array(gates_normal)
        self.gates_y = np.array(gates_y)
        self.gates_z = np.array(gates_z)
        self.gate_size = gate_size
        # self.obstacles_pos = obstacles_pos
        self.start_pos = np.array(start_pos)
        self.start_orient = R.from_euler(
            "xyz", start_orient
        ).as_matrix()  # Convert to rotation matrix
        self.obs = obs
        self.info = info
        self.sim_config = sim_config
        self.POLE_HEIGHT = 3.0  # Meters
        self.SAFETY_RADIUS = 0.1  # Meters
        self.MAX_LATERAL_WIDTH = 0.2  # Meters
        self.CONTRACTION_LEN = 0.3  # Meters
        self.CLEARANCE_RADIUS = 0.25  # Meters for obstacle avoidance
        self.GATE_CONTRACTION_LEN = 0.3  # Meters
        self.U_turn_extension = 0.25  # Meters
        self.U_turn_radius = 0.35  # Meters
        self.gate_approach_dist = 0.5  # Meters
        self.debug_dicts = []
        
        # self.obstacles_pos = self.add_virtual_obstacle(obstacles_pos)
        self.obstacles_pos = np.array(obstacles_pos)
        
        self.gate_vectors = self.gate_to_gate_vectors()
        
        

        self.waypoints = self.__initialize_waypoints()
        self.waypoints = self.__insert_obstacle_avoidance_waypoints(self.waypoints, clearance_radius=self.CLEARANCE_RADIUS)

        self.spline = self.__get_spline(self.waypoints)
        
        
        
        
        
        num_frame_points = int(max(10, self.total_length * 100))
        self.pt_frame = self._generate_parallel_transport_frame(num_points=num_frame_points)

        # self.corridor_map = self.__generate_static_corridor()
        
        num_pts = len(self.pt_frame["s"])
        lb_w1 = np.full(
            num_pts, -self.MAX_LATERAL_WIDTH
        )  # left bound which is filled with -max width
        ub_w1 = np.full(num_pts, self.MAX_LATERAL_WIDTH)
        

        self.corridor_map = {"lb_w1": lb_w1, "ub_w1": ub_w1}

        # self.__print_debug_vectors()
        
    def gate_to_gate_vectors(self) -> List[NDArray[np.floating]]:
        gate_vectors = []
        for i in range(1, len(self.gates_pos)):
            vec = self.gates_pos[i] - self.gates_pos[i - 1]
            gate_vectors.append(vec / np.linalg.norm(vec))
            
        
        return gate_vectors
        
    def add_virtual_obstacle(self, obstacle_pos):
        
        new_obstacles: List[List[float]] = obstacle_pos.copy()
        
        for idx, gate_pos in enumerate(self.gates_pos):
            gate_y = self.gates_y[idx]
            
            virtual_obs_pos = gate_pos - gate_y * (self.gate_size / 2)
            new_obstacles.append(virtual_obs_pos)
            
            virtual_obs_pos = gate_pos + gate_y * (self.gate_size / 2)
            new_obstacles.append(virtual_obs_pos)
            
        return np.array(new_obstacles)
            
        
        
    def __process_single_obstacle_avoidance(self, sampled_points: NDArray[np.floating], clearance_radius: float, obstacle_center: NDArray[np.floating]) -> NDArray[np.floating]:
        
        
        collision_free_points = []

        is_inside_obstacle_zone = False
        entry_index = None

        obstacle_xy = obstacle_center[:2]

        for i, point in enumerate(sampled_points):
            point_xy = point[:2]
            distance_xy = np.linalg.norm(obstacle_xy - point_xy)

            if distance_xy < clearance_radius:
                if not is_inside_obstacle_zone:
                    # Just entered the collision zone
                    is_inside_obstacle_zone = True
                    print(f"Entering obstacle zone at index {i}, point {point}")
                    entry_index = i

            elif is_inside_obstacle_zone:
                # Just exited the collision zone
                is_inside_obstacle_zone = False
                exit_index = i

                # --- Avoidance Calculation ---
                entry_point = sampled_points[entry_index]
                exit_point = sampled_points[exit_index]

                # Vectors from obstacle center to entry/exit
                entry_vec = entry_point[:2] - obstacle_xy
                exit_vec = exit_point[:2] - obstacle_xy

                # Bisector vector determines the direction to push the path
                avoid_vec = entry_vec + exit_vec
                avoid_vec /= np.linalg.norm(avoid_vec) + 1e-6

                # Calculate new waypoint
                new_pos_xy = obstacle_xy + avoid_vec * clearance_radius
                new_pos_z = (entry_point[2] + exit_point[2]) / 2  # Maintain average altitude
                new_avoid_waypoint = np.concatenate([new_pos_xy, [new_pos_z]])

                # Insert waypoint at average time
                # avg_time = (sampled_times[entry_index] + sampled_times[exit_index]) / 2
                # collision_free_times.append(avg_time)
                collision_free_points.append(new_avoid_waypoint)

            else:
                # Point is safe, keep it
                # collision_free_times.append(sampled_times[i])
                collision_free_points.append(point)

        return  np.array(collision_free_points)
        
    def __insert_obstacle_avoidance_waypoints(self, waypoints: NDArray[np.floating], clearance_radius: float, num_points = 1000) -> NDArray[np.floating]:
        temp_spline = self.__get_spline(waypoints)
        s_eval = np.linspace(0, self.total_length, num_points)
        sampled_points = temp_spline(s_eval)
        
        for obs in self.obstacles_pos:
            new_waypoints = self.__process_single_obstacle_avoidance(sampled_points, clearance_radius, obs)
            
        return new_waypoints
            

    def __print_debug_vectors(self):
        for i, debug_dict in enumerate(self.debug_dicts):
            print(
                f"Debug {i}: Frame Pos: {debug_dict['frame_pos']}, Obstacle: {debug_dict['obs']}, w1_obs: {debug_dict['w1_obs']}, reduced_lb: {debug_dict.get('reduced_lb', 'NA')}, reduced_ub: {debug_dict.get('reduced_ub', 'NA')}, s_knots: {debug_dict['s_knots']} , lb_current: {debug_dict.get('lb_current', 'NA')}, ub_current: {debug_dict.get('ub_current', 'NA')}"
            )
            pass
        
    def angle_between_vectors(self, v1: NDArray[np.floating], v2: NDArray[np.floating]) -> float:
        """Calculate the angle in radians between two vectors."""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = np.arccos(dot_product)
        return angle

    def __initialize_waypoints(self) -> NDArray[np.floating]:
        """Initialize waypoints with special handling for 180-degree reversals."""
        waypoints = [self.start_pos]
        
        # Tuning parameters for the geometry
        
        EXTENSION_DIST = self.U_turn_extension  # How far to fly straight out after a reversal gate
        TURN_RADIUS = self.U_turn_radius   # How wide the U-turn should be
        APPROACH_DIST = self.gate_approach_dist  # Distance for pre/post gate guidance

        for idx, gate_pos in enumerate(self.gates_pos):
            
            # 1. Define Standard Approach (Pre-Gate)
            # Aligns the drone with the normal before entering
            before_gate = gate_pos - self.gates_normal[idx] * APPROACH_DIST
            waypoints.append(before_gate)
            
            # 2. Add Gate Center
            waypoints.append(gate_pos)

            # Check angle for reversal detection
            # We assume angle > 120 degrees implies a sharp turn/reversal
            is_reversal = False
            if idx < len(self.gates_pos) - 1:
                # Calculate vector to next gate to check turn sharpness
                vec_to_next = self.gates_pos[idx+1] - gate_pos
                angle = self.angle_between_vectors(vec_to_next, self.gates_normal[idx])
                
                # If angle is large, the next gate is "behind" the current normal
                if np.degrees(angle) > 120:
                    is_reversal = True

            if is_reversal:
                print(f"[Geometry] Generating Reversal Balloon at Gate {idx}")

                # 3. The "Extension" (Fly Out)
                # Force the drone to fly straight OUT of the gate first. 
                # This prevents it from snapping 180 immediately inside the gate.
                extension_point = gate_pos + self.gates_normal[idx] * EXTENSION_DIST
                waypoints.append(extension_point)

                # 4. The "Balloon" Turn (Lateral Offset)
                # To come back, we must turn Left or Right. We use the Gate's Y-axis.
                # Heuristic: Check which side the next gate is on relative to the current gate's Y axis
                vec_to_next = self.gates_pos[idx+1] - gate_pos
                
                # Dot product determines if next gate is to the Left (+Y) or Right (-Y)
                # If vectors are orthogonal or zero, default to +Y (Left)
                side_sign = np.sign(np.dot(vec_to_next, self.gates_y[idx]))
                if side_sign == 0: 
                    side_sign = 1.0
                
                # Create a waypoint that pulls the spline into a wide U-turn
                # This point is: Extended out + Shifted sideways
                turn_point = extension_point + (self.gates_y[idx] * TURN_RADIUS * side_sign)
                waypoints.append(turn_point)

            else:
                # Standard Exit (Just follow the normal out)
                after_gate = gate_pos + self.gates_normal[idx] * APPROACH_DIST
                waypoints.append(after_gate)

        return np.array(waypoints)

    def __create_sknots(self, points: NDArray[np.floating], num_points=3000) -> CubicHermiteSpline:
        """Create a cubic Hermite spline from given points and tangents."""

        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        dists = np.maximum(dists, 1e-6)  # Prevent division by zero

        s_knots = np.concatenate(([0], np.cumsum(dists)))  # Cumulative distance along the points

        self.total_length = s_knots[-1]

        return s_knots

    def __add_debug_statement(
        self,
        i: int,
        frame_pos: NDArray[np.floating],
        obs: NDArray[np.floating],
        w1_obs: float,
        proposed_lb: float,
        proposed_ub: float,
        lb_current: float = None,
        ub_current: float = None,
    ):
        self.debug_dicts.append(
            {
                "frame_pos": f"{frame_pos[0]:.2f}, {frame_pos[1]:.2f}, {frame_pos[2]:.2f}",
                "obs": obs,
                "w1_obs": f"{w1_obs}",
                "reduced_lb": f"{proposed_lb:.2f} " if proposed_lb is not None else "None",
                "reduced_ub": f"{proposed_ub:.2f} " if proposed_ub is not None else "None",
                "s_knots": f"{self.pt_frame['s'][i]:.2f}",
                "lb_current": f"{lb_current:.2f}" if lb_current is not None else "None",
                "ub_current": f"{ub_current:.2f}" if ub_current is not None else "None",
            }
        )
        
            
        
               

        
    def __generate_static_corridor(self) -> Dict[str, NDArray]:
        # print(f"[Geometry] Generating bounds (2D). Safety Radius: {self.SAFETY_RADIUS}")
        num_pts = len(self.pt_frame["s"])
        lb_w1 = np.full(
            num_pts, -self.MAX_LATERAL_WIDTH
        )  # left bound which is filled with -max width
        ub_w1 = np.full(num_pts, self.MAX_LATERAL_WIDTH)
        lb_w2 = np.full(
            num_pts, -self.MAX_LATERAL_WIDTH
        )  # left bound which is filled with -max width
        ub_w2 = np.full(num_pts, self.MAX_LATERAL_WIDTH)

        if len(self.obstacles_pos) == 0:
            return {"lb_w1": lb_w1, "ub_w1": ub_w1}

        for i in range(num_pts):
            frame_pos = self.pt_frame["pos"][i]  # the position at frame i
            frame_t = self.pt_frame["t"][i]  # the tangent at frame i
            n1 = self.pt_frame["n1"][i]  # the normal at frame i
            n2 = self.pt_frame["n2"][i]  # the binormal at frame i
            
                # --- [NEW] Contract bounds near gates ---
            # self.__contract_for_gates(i, frame_pos, frame_t, {'lb_w1': lb_w1, 'ub_w1': ub_w1})

            pos_2d = np.array([frame_pos[0], frame_pos[1], 0.0])  # project to 2D by making z=0
            t_2d = np.array([frame_t[0], frame_t[1], 0.0])  # project to 2D by making z=0

            if np.linalg.norm(t_2d) < 1e-3:
                continue
            
            
            

            t_2d /= np.linalg.norm(t_2d)  # normalize
            n1_2d = np.array([n1[0], n1[1], 0.0])  # normal vector in 2D
            n2_2d = np.array([n2[0], n2[1], 0.0])  # binormal vector in 2D
            
            
            for gate_idx, gate_pos in enumerate(self.gates_pos):
                gate_pos_2d = np.array([gate_pos[0], gate_pos[1], 0.0])
                r_vec_2d = gate_pos_2d - pos_2d
                
                d = np.linalg.norm(r_vec_2d)
                d_long = np.dot(r_vec_2d, t_2d)  # longitudinal distance along the tangent
                
                if abs(d_long) > self.GATE_CONTRACTION_LEN:
                    # print(f"[Geometry] Gate {gate_idx} too far from frame {i} for contraction: d_long = {d_long:.2f} m")
                    continue
                                
                if abs(d) > self.GATE_CONTRACTION_LEN:
                    # print(f"[Geometry] Gate {gate_idx} at {gate_pos_2d} too far from frame {i} for contraction: d = {d:.2f} m")
                    continue
                
                # print(f"[Geometry] Contracting bounds near Gate {gate_idx} at frame {i}: d = {d:.2f} m, d_long = {d_long:.2f} m")
                # Within contraction length, tighten bounds
                new_bound = self.gate_size/2 - 0.1
                
                ub_w1[i] = new_bound  
                lb_w1[i] = -new_bound
                
                self.__add_debug_statement(i, frame_pos, gate_pos, None, -new_bound, new_bound)
                

            for obs in self.obstacles_pos:
                obs_2d = np.array([obs[0], obs[1], 0.0])  # project obstacle to 2D
                # Compute relative vector

                r_vec_2d = obs_2d - pos_2d  # vector from frame pos to obstacle pos
                d = np.linalg.norm(r_vec_2d)
                d_long = np.dot(r_vec_2d, t_2d)  # longitudinal distance along the tangent
                if abs(d_long) > self.CONTRACTION_LEN:  # if obstacle is too far ahead or behind,
                    continue

                w1_obs = np.dot(r_vec_2d, n1_2d)  # lateral distance along the normal
                if abs(d) > self.CONTRACTION_LEN:
                    continue

                if abs(w1_obs) > (self.MAX_LATERAL_WIDTH + self.SAFETY_RADIUS):  # too far laterally
                    continue

                # If we are in the "Danger Zone" where geometry is ambiguous
                if abs(w1_obs) < 0.1:
                    # Check which constraint leaves us more room or is closer to the previous point's decision?
                    # Simple heuristic: Which side allows for a wider corridor?

                    proposed_ub = w1_obs - self.SAFETY_RADIUS
                    proposed_lb = w1_obs + self.SAFETY_RADIUS

                    current_width_if_pass_right = (
                        ub_w1[i] - proposed_lb
                    )  # If we treat obs as Right wall
                    current_width_if_pass_left = (
                        proposed_ub - lb_w1[i]
                    )  # If we treat obs as Left wall

                    # Pick the side that leaves the corridor more open
                    if current_width_if_pass_left > current_width_if_pass_right:
                        # Treat as Left Obstacle (Pass Right)
                        if proposed_ub < ub_w1[i]:
                            ub_w1[i] = proposed_ub
                    else:
                        # Treat as Right Obstacle (Pass Left)
                        if proposed_lb > lb_w1[i]:
                            lb_w1[i] = proposed_lb

                    self.__add_debug_statement(i, frame_pos, obs, w1_obs, proposed_lb, proposed_ub, lb_w1[i] , ub_w1[i])

                else:
                    if w1_obs >= 0:
                        safe_edge = w1_obs - self.SAFETY_RADIUS
                        if safe_edge < ub_w1[i]:
                            self.__add_debug_statement(i, frame_pos, obs, w1_obs, None, safe_edge, lb_w1[i], ub_w1[i])
                            ub_w1[i] = safe_edge
                    else:
                        safe_edge = w1_obs + self.SAFETY_RADIUS
                        if safe_edge > lb_w1[i]:
                            self.__add_debug_statement(i, frame_pos, obs, w1_obs, safe_edge, None, lb_w1[i], ub_w1[i])
                            lb_w1[i] = safe_edge

        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05
            
            print(f"[Geometry] Warning: Collapsed corridor at {np.sum(collapsed)} points. Adjusted to minimal width.")
        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    def __get_spline(self, points: NDArray[np.floating]) -> CubicHermiteSpline:
        """Generate a cubic Hermite spline from points and tangents."""
        s_knots = self.__create_sknots(points)
        spline = CubicSpline(s_knots, points)
        return spline

    def _generate_parallel_transport_frame(self, num_points=3000):
        # Evaluate along arc length s
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

        # Initial Frame Setup
        # 1st derivative of Hermite Spline w.r.t s is the tangent
        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)

        g_vec = np.array([0, 0, -1])  # Gravity reference

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

            k1 = np.dot(k_vec, curr_n1)  # Curvature in n1 direction
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

    def plot(self):
        """Visualizes Path, Gates, Obstacles, and Corridor using Plotly."""
        # print("[Geometry] Generating interactive Plotly visualization...")
        fig = go.Figure()

        # --- 1. Plot Flight Path ---
        path = self.pt_frame["pos"]
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                line=dict(color="black", width=4),
                name="Centerline",
            )
        )

        # --- 2. Plot Waypoints ---
        # color_map = {0: "green", 1: "blue", 2: "red", 3: "purple"}
        # colors = ["green", "blue", "yellow", "green", "green", "green"]  # Start point
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=self.waypoints[:, 0],
        #         y=self.waypoints[:, 1],
        #         z=self.waypoints[:, 2],
        #         mode="markers",
        #         marker=dict(size=6, color=colors),
        #         name="Waypoints",
        #     )
        # )

        # --- 3. [NEW] Plot Gates (Rectangular Frames) ---
        # Using a single trace with "None" separators for performance
        gate_x, gate_y_list, gate_z_list = [], [], []

        # Half dimensions
        hw = self.gate_size / 2.0
        hh = self.gate_size / 2.0

        for i in range(len(self.gates_pos)):
            center = self.gates_pos[i]
            # Get orientation vectors
            y_vec = self.gates_y[i]
            z_vec = self.gates_z[i]

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

        fig.add_trace(
            go.Scatter3d(
                x=gate_x,
                y=gate_y_list,
                z=gate_z_list,
                mode="lines",
                line=dict(color="blue", width=5),
                name="Gates",
            )
        )

        # --- 4. Plot Obstacles ---
        if len(self.obstacles_pos) > 0:
            u = np.linspace(0, 2 * np.pi, 25)
            z_pole = np.linspace(0, self.POLE_HEIGHT, 2)
            U, Z_pole = np.meshgrid(u, z_pole)

            first_obs = True
            for obs in self.obstacles_pos:
                X_pole = self.SAFETY_RADIUS * np.cos(U) + obs[0]
                Y_pole = self.SAFETY_RADIUS * np.sin(U) + obs[1]
                fig.add_trace(
                    go.Surface(
                        x=X_pole,
                        y=Y_pole,
                        z=Z_pole,
                        colorscale=[[0, "red"], [1, "red"]],
                        opacity=0.6,
                        showscale=False,
                        name="Obstacle",
                        showlegend=first_obs,
                    )
                )
                first_obs = False

        # --- 5. Plot Corridor Bounds ---
        step = 5
        p_vis = path[::step]
        n1_vis = self.pt_frame["n1"][::step]
        idx = np.arange(0, len(path), step)
        lb = self.corridor_map["lb_w1"][idx]
        ub = self.corridor_map["ub_w1"][idx]

        wall_left = p_vis + (n1_vis * ub[:, np.newaxis])
        wall_right = p_vis + (n1_vis * lb[:, np.newaxis])

        fig.add_trace(
            go.Scatter3d(
                x=wall_left[:, 0],
                y=wall_left[:, 1],
                z=wall_left[:, 2],
                mode="lines",
                line=dict(color="red", width=2),
                name="Bound L",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=wall_right[:, 0],
                y=wall_right[:, 1],
                z=wall_right[:, 2],
                mode="lines",
                line=dict(color="red", width=2),
                name="Bound R",
            )
        )

        fig.update_layout(
            title="Interactive Flight Corridor (With Gates & Avoidance)",
            scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()
        
        
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

