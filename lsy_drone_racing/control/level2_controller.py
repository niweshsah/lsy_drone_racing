"""WOrking explanation:
1) you only put a waypoint at the center of a gate, the drone might approach it from a sharp angle and hit the frame. To prevent this, the code creates a virtual "tube" through the gate.

Stage 2: The Detour Logic (The "Racing Line")
Code: __add_detour_logic

In drone racing, flying a straight line between two gates often results in a turn that is too sharp for the drone's physics to handle. This logic artificially widens the turn.

Stage 3: Obstacle Avoidance (Reactive Patching)
Code: __insert_obstacle_avoidance_points & __process_single_obstacle

This acts as a safety layer on top of the path.

Simulation: It temporarily generates a spline from the points created in Stages 1 & 2.

Scanning: It samples this path densely (every few centimeters).

Intersection: If the path enters an obstacle's "danger zone" (radius), it records the Entry Point and Exit Point.

Correction: It calculates a "Bisector Vector" (the average direction between entering and exiting) and pushes a new waypoint outwards along that vector. This bends the spline around the obstacle.


"""  # noqa: D205

from __future__ import annotations  # noqa: I001

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from scipy.interpolate import CubicSpline  # type: ignore
from scipy.spatial.transform import Rotation  # pyright: ignore[reportMissingImports]

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):
    """A Spline-based Controller for drone racing.

    This controller generates a Cubic Spline trajectory through gates,
    adds detours for sharp turns, and adjusts the path to avoid obstacles.
    """

    # --- Configuration Constants ---
    FLIGHT_DURATION = 20.0
    STATE_SIZE = 13
    OBSTACLE_CLEARANCE = 0.3
    VIZ_SAMPLES = 100
    LOG_FREQ = 100

    # Trajectory Generation Parameters
    APPROACH_DISTANCE = 0.5  # Distance before/after gate to place waypoints
    NUM_INTERMEDIATE_POINTS = 5  # Number of points to generate through a gate
    DETOUR_ANGLE_THRESHOLD = 120.0  # Angle (degrees) triggering a detour insertion
    DETOUR_RADIUS = 0.65  # Distance to offset detour points

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], info: dict, sim_config: dict):
        """Initializes the controller, sets up state, and plans the first trajectory.

        Args:
            initial_obs: Dictionary containing initial simulation state (pos, gates, obstacles).
            info: Additional simulation info.
            sim_config: Configuration object containing environment frequency.
        """
        super().__init__(initial_obs, info, sim_config)

        self.__initialize_state(initial_obs, sim_config)
        self.__plan_initial_trajectory(initial_obs)

    def __initialize_state(self, initial_obs: dict[str, NDArray[np.floating]], sim_config: dict):
        """Sets up the controller's internal state variables."""
        self.__current_step = 0
        self.__control_freq = sim_config.env.freq
        self.__is_complete = False

        # Flags for environment change detection
        self.__last_gate_flags = None
        self.__last_obstacle_flags = None

        # Debugging artifacts (useful for analyzing path logic)
        self.__debug_detour_log = []
        self.__debug_detour_stats = {}
        self.__debug_detour_points_added = []
        self.__debug_initial_wps = None
        self.__debug_post_detour_wps = None
        self.__debug_final_wps = None
        
        print("observation:",initial_obs)

        # Extract Environment geometry
        self.__gate_positions = initial_obs["gates_pos"]
        self.__obstacle_positions = initial_obs["obstacles_pos"]
        self.__start_position = initial_obs["pos"]

        # Extract gate orientation frames
        self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
            initial_obs["gates_quat"]
        )

        self.__trajectory_spline: Optional[CubicSpline] = None
        self.visualization = False
        self.fig = None
        self.ax = None

    def __plan_initial_trajectory(self, initial_obs: dict[str, NDArray[np.floating]]):
        """Generates the full trajectory pipeline from the initial observation.

        Pipeline:
        1. Generate straight-line approach points through gates.
        2. Detect sharp turns and insert 'detour' points to widen the turn.
        3. Scan for obstacles and shift points to avoid collisions.
        4. Fit a CubicSpline through the final points.
        """
        # 1. Generate waypoints through gates
        path_points = self.__generate_gate_approach_points(
            self.__start_position,
            self.__gate_positions,
            self.__gate_normals,
            approach_distance=self.APPROACH_DISTANCE,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS,
        )
        self.__debug_initial_wps = path_points.copy()

        # 2. Add sharp-turn detours
        path_points = self.__add_detour_logic(
            path_points,
            self.__gate_positions,
            self.__gate_normals,
            self.__gate_y_axes,
            self.__gate_z_axes,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS,
            detour_angle_degrees=self.DETOUR_ANGLE_THRESHOLD,
            detour_radius=self.DETOUR_RADIUS,
        )
        self.__debug_post_detour_wps = path_points.copy()

        # 3. Adjust waypoints to avoid obstacles
        time_knots, path_points = self.__insert_obstacle_avoidance_points(
            path_points, self.__obstacle_positions, self.OBSTACLE_CLEARANCE
        )
        self.__debug_final_wps = path_points.copy()

        # 4. Generate the final spline
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points, custom_time_knots=time_knots
        )

    def __extract_gate_frames(
        self, gates_quaternions: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Extracts the local coordinate axes (Normal, Y, Z) from gate quaternions.

        Args:
            gates_quaternions: Array of shape (N, 4).

        Returns:
            Tuple containing (Normals/X-axis, Y-axis, Z-axis), each of shape (N, 3).
        """
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()

        normals = rotation_matrices[:, :, 0]  # X-axis (normal)
        y_axes = rotation_matrices[:, :, 1]  # Y-axis
        z_axes = rotation_matrices[:, :, 2]  # Z-axis

        return normals, y_axes, z_axes

    def __generate_gate_approach_points(
        self,
        initial_position: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        approach_distance: float,
        num_intermediate_points: int,
    ) -> NDArray[np.floating]:
        """Calculates a sequence of waypoints passing through every gate.

        It generates points along the normal vector of the gate, centered at the gate position.

        Args:
            initial_position: Starting coordinate (3,).
            gate_positions: Centers of gates (N, 3).
            gate_normals: Normal vectors of gates (N, 3).
            approach_distance: How far from the gate center to start/end the points.
            num_intermediate_points: How many points to generate per gate.

        Returns:
            Array of waypoints (Start + N * num_intermediate, 3).
        """
        # Create offsets (e.g., -0.5, -0.25, 0, 0.25, 0.5)
        offsets = np.linspace(-approach_distance, approach_distance, num_intermediate_points)

        # Use NumPy broadcasting to vectorize point generation
        # Dimensions:
        # gate_positions: (N, 3) -> (N, 1, 3)
        # offsets:        (M,)   -> (1, M, 1)
        gate_pos_exp = gate_positions[:, np.newaxis, :]
        gate_norm_exp = gate_normals[:, np.newaxis, :]
        offsets_exp = offsets[np.newaxis, :, np.newaxis]

        # Result: (N, M, 3)
        waypoints_matrix = gate_pos_exp + offsets_exp * gate_norm_exp

        # Flatten to (N*M, 3) and prepend start position
        flat_waypoints = waypoints_matrix.reshape(-1, 3)
        return np.vstack([initial_position, flat_waypoints])

    def __compute_trajectory_spline(
        self,
        total_time: float,
        path_points: NDArray[np.floating],
        custom_time_knots: Optional[NDArray[np.floating]] = None,
    ) -> CubicSpline:
        """Generates a 3D cubic spline trajectory through the given points.

        Args:
            total_time: Desired total flight duration in seconds.
            path_points: Array of 3D coordinates (N, 3).
            custom_time_knots: Optional specific timestamps for each point.
                               If None, timestamps are generated based on arc length.

        Returns:
            A Scipy CubicSpline object.
        """
        if custom_time_knots is not None:
            # Use pre-computed time knots (e.g. modified by collision avoidance)
            return CubicSpline(custom_time_knots, path_points)

        # Parameterize by arc length to ensure constant average speed
        path_segments = np.diff(path_points, axis=0)
        segment_distances = np.linalg.norm(path_segments, axis=1)

        # Cumulative distance starting at 0
        cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)])

        # Normalize (0 to 1) and scale by total flight duration
        time_knots = cumulative_distance / cumulative_distance[-1] * total_time

        return CubicSpline(time_knots, path_points)

    def __process_single_obstacle(
        self,
        obstacle_center: NDArray[np.floating],
        sampled_points: NDArray[np.floating],
        sampled_times: NDArray[np.floating],
        clearance_radius: float,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Scans a trajectory for collisions with a single obstacle and inserts an avoidance waypoint.

        Logic:
        1. Identify the "entry" and "exit" indices where the drone is inside the obstacle radius.
        2. Calculate a vector that pushes the path away from the obstacle (bisector logic).
        3. Insert a new waypoint at the midpoint time, shifted by clearance_radius.
        """  # noqa: D212
        collision_free_times = []
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
                avg_time = (sampled_times[entry_index] + sampled_times[exit_index]) / 2
                collision_free_times.append(avg_time)
                collision_free_points.append(new_avoid_waypoint)

            else:
                # Point is safe, keep it
                collision_free_times.append(sampled_times[i])
                collision_free_points.append(point)

        return np.array(collision_free_times), np.array(collision_free_points)

    def __insert_obstacle_avoidance_points(
        self,
        path_points: NDArray[np.floating],
        obstacle_centers: NDArray[np.floating],
        clearance_radius: float,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Iteratively checks for and mitigates collisions with all known obstacles.

        Returns:
            Tuple of (time_knots, waypoints) suitable for Spline generation.
        """
        # Generate a temporary spline to sample dense points for collision checking
        temp_spline = self.__compute_trajectory_spline(self.FLIGHT_DURATION, path_points)

        num_samples = int(self.__control_freq * self.FLIGHT_DURATION)
        sampled_times = np.linspace(0, self.FLIGHT_DURATION, num_samples)
        sampled_points = temp_spline(sampled_times)

        # Check every obstacle against the sampled path
        for obstacle_center in obstacle_centers:
            sampled_times, sampled_points = self.__process_single_obstacle(
                obstacle_center, sampled_points, sampled_times, clearance_radius
            )

        return sampled_times, sampled_points

    def __check_for_env_update(self, current_obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Detects if the environment state has changed (e.g., a new gate was passed).

        Returns:
            True if a gate or obstacle has been newly flagged as visited/hit.
        """
        current_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
        current_obstacle_flags = np.array(current_obs["obstacles_visited"], dtype=bool)

        if self.__last_gate_flags is None:
            self.__last_gate_flags = current_gate_flags
            self.__last_obstacle_flags = current_obstacle_flags
            return False  # First tick, no change to report

        # Check for rising edges (False -> True transitions)
        gate_newly_hit = np.any((~self.__last_gate_flags) & current_gate_flags)
        obstacle_newly_hit = np.any((~self.__last_obstacle_flags) & current_obstacle_flags)

        self.__last_gate_flags = current_gate_flags
        self.__last_obstacle_flags = current_obstacle_flags

        return gate_newly_hit or obstacle_newly_hit

    def __regenerate_flight_plan(
        self, current_obs: dict[str, NDArray[np.floating]], simulation_time: float
    ) -> None:
        """Re-plans the entire trajectory based on updated environment observations."""
        # Update local knowledge of geometry
        self.__gate_positions = current_obs["gates_pos"]
        self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
            current_obs["gates_quat"]
        )

        # Re-run pipeline
        path_points = self.__generate_gate_approach_points(
            self.__start_position,
            self.__gate_positions,
            self.__gate_normals,
            approach_distance=self.APPROACH_DISTANCE,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS,
        )

        path_points = self.__add_detour_logic(
            path_points,
            self.__gate_positions,
            self.__gate_normals,
            self.__gate_y_axes,
            self.__gate_z_axes,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS,
            detour_angle_degrees=self.DETOUR_ANGLE_THRESHOLD,
            detour_radius=self.DETOUR_RADIUS,
        )

        time_knots, path_points = self.__insert_obstacle_avoidance_points(
            path_points, current_obs["obstacles_pos"], self.OBSTACLE_CLEARANCE
        )

        # Note: This completely replaces the old spline
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points, custom_time_knots=time_knots
        )

    def __determine_detour_direction(
        self,
        v_proj: NDArray[np.floating],
        v_proj_norm: float,
        y_axis: NDArray[np.floating],
        z_axis: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], str, float]:
        """Calculates the optimal "sideways" direction (Right, Top, Left) to widen a turn.

        Args:
            v_proj: The trajectory vector projected onto the gate plane.
            v_proj_norm: Length of v_proj.
            y_axis: Gate's local Y axis.
            z_axis: Gate's local Z axis.

        Returns:
            Tuple of (Direction Vector, Direction Name, Projection Angle).
        """
        if v_proj_norm < 1e-6:
            return y_axis, "right (+y_axis) [default]", 0.0

        # Decompose vector into Gate Frame components
        v_proj_y = np.dot(v_proj, y_axis)
        v_proj_z = np.dot(v_proj, z_axis)

        # Find angle in the Y-Z plane
        proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi

        # Quantize direction to cardinal axes based on the angle
        if -90 <= proj_angle_deg < 45:
            detour_vector = y_axis
            direction_name = "right (+y_axis)"
        elif 45 <= proj_angle_deg < 135:
            detour_vector = z_axis
            direction_name = "top (+z_axis)"
        else:
            detour_vector = -y_axis
            direction_name = "left (-y_axis)"

        return detour_vector, direction_name, proj_angle_deg

    def __add_detour_logic(
        self,
        path_points: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: int,
        detour_angle_degrees: float,
        detour_radius: float,
    ) -> NDArray[np.floating]:
        """Analyzes gate-to-gate transitions. If a turn is too sharp (angle > threshold),
        it inserts an extra waypoint perpendicular to the path to smoothen the turn.
        """  # noqa: D205
        num_gates = gate_positions.shape[0]
        path_points_list = list(path_points)

        detour_insert_count = 0
        self.__debug_detour_log = []
        self.__debug_detour_points_added = []

        # Iterate through consecutive gates
        for i in range(num_gates - 1):
            debug_info = {"gate_i": i, "gate_i_plus_1": i + 1}

            # Calculate indices in the flat waypoint list for Gate(i) exit and Gate(i+1) entry
            # We shift indices by detour_insert_count because the list grows as we add points.
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + detour_insert_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + detour_insert_count

            p1_exit_wp = path_points_list[last_idx_gate_i]
            p2_entry_wp = path_points_list[first_idx_gate_i_plus_1]

            # Vector between the two gates
            transition_vector = p2_entry_wp - p1_exit_wp
            vector_norm = np.linalg.norm(transition_vector)

            if vector_norm < 1e-6:
                debug_info.update({"skipped": True, "reason": "vector_too_short"})
                self.__debug_detour_log.append(debug_info)
                continue

            # Check angle between transition vector and the current gate's normal
            normal_i = gate_normals[i]
            cos_angle = np.dot(transition_vector, normal_i) / vector_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.arccos(cos_angle) * 180 / np.pi

            if angle_deg > detour_angle_degrees:
                # -- Sharp Turn Detected: Plan Detour --
                gate_center = gate_positions[i]
                y_axis = gate_y_axes[i]
                z_axis = gate_z_axes[i]

                # Project transition vector onto the gate's Y-Z plane
                # This tells us which way the drone is "trying" to turn
                v_proj = transition_vector - np.dot(transition_vector, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)

                # Determine optimal direction to widen the turn
                detour_vec, direction_name, proj_angle_deg = self.__determine_detour_direction(
                    v_proj, v_proj_norm, y_axis, z_axis
                )

                # Create new waypoint
                detour_waypoint = gate_center + detour_radius * detour_vec
                self.__debug_detour_points_added.append(detour_waypoint.copy())

                # Insert point *after* the last point of the current gate
                insert_pos = last_idx_gate_i + 1
                path_points_list.insert(insert_pos, detour_waypoint)
                detour_insert_count += 1

                debug_info.update(
                    {
                        "needs_detour": True,
                        "angle_deg": angle_deg,
                        "inserted_at": insert_pos,
                        "direction": direction_name,
                    }
                )
            else:
                debug_info.update({"needs_detour": False, "angle_deg": angle_deg})

            self.__debug_detour_log.append(debug_info)

        self.__debug_detour_stats = {
            "detours_added": detour_insert_count,
            "final_wp_count": len(path_points_list),
        }

        return np.array(path_points_list)

    # --- Public API Methods ---

    def compute_control(
        self, current_obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Calculates the control action (Target Position) for the current time step.

        Returns:
            A 13-dim array: [x, y, z, vx, vy, vz, ax, ay, az, yaw, rate_x, rate_y, rate_z].
            Here, only position (x, y, z) is populated from the spline; others are 0.
        """
        # Determine current position in the trajectory
        elapsed_time = min(self.__current_step / self.__control_freq, self.FLIGHT_DURATION)

        # 1. Get target position from the spline
        reference_position = self.__trajectory_spline(elapsed_time)

        # 2. Check triggers for re-planning
        if self.__check_for_env_update(current_obs):
            self.__regenerate_flight_plan(current_obs, elapsed_time)

        if elapsed_time >= self.FLIGHT_DURATION:
            self.__is_complete = True

        # 3. Visualization (Draw trajectory line in Sim)
        try:
            draw_line(
                self.env,
                self.__trajectory_spline(self.__trajectory_spline.x),
                rgba=np.array([1.0, 1.0, 1.0, 0.2]),
            )
        except (AttributeError, TypeError):
            pass  # Env or spline not ready

        # Return full state vector (Position populated, others zeroed)
        return np.concatenate((reference_position, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        current_obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Called by the simulation loop after every step.

        Returns:
            bool: True if the episode should end (completion or failure).
        """
        self.__current_step += 1
        return self.__is_complete

    def get_spline_function(self) -> CubicSpline:
        """Returns the raw CubicSpline object for external analysis."""
        return self.__trajectory_spline

    def get_sampled_trajectory(self) -> NDArray[np.floating]:
        """Returns a densely sampled array of points from the trajectory (visualization)."""
        num_samples = int(self.__control_freq * self.FLIGHT_DURATION)
        time_samples = np.linspace(0, self.FLIGHT_DURATION, num_samples)
        return self.__trajectory_spline(time_samples)

    def update_current_step(self, step: int) -> None:
        """Manually sets the controller's internal time step."""
        self.__current_step = step
