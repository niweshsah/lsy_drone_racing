"""This is a controller for level 3 drone racing environment.
It is very similar to level2 controller but with added logic to replan the trajectory when it is inside the replan radius of any gate or obstacle.
"""  # noqa: D205


from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):  # noqa: D101
    # --- Constants ---
    FLIGHT_DURATION = 25.0
    STATE_SIZE = 13
    REPLAN_RADIUS = 0.5  # New safety margin for triggering re-plan
    OBSTACLE_CLEARANCE = 0.2
    VIZ_SAMPLES = 100
    LOG_FREQ = 100

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], info: dict, sim_config: dict):
        """Initializes the controller, sets up state, and plans the first trajectory."""
        # Call base constructor using super()
        super().__init__(initial_obs, info, sim_config)

        # These are the private methods that is why they have double underscores
        # #actuallly python has no true private methods, this is just a naming convention
        # This does name mangling to prevent accidental access from outside the class
        # obj.__initialize_state()   # ❌ AttributeError
        # obj._MyController__initialize_state()   # ✔️ works

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

        # Debugging artifacts
        self.__debug_detour_log = []
        self.__debug_detour_stats = {}
        self.__debug_detour_points_added = []
        self.__debug_initial_wps = None
        self.__debug_post_detour_wps = None
        self.__debug_final_wps = None

        # Environment geometry
        self.__gate_positions = initial_obs["gates_pos"]
        self.__obstacle_positions = initial_obs["obstacles_pos"]
        self.__start_position = initial_obs["pos"]
        self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
            initial_obs["gates_quat"]
        )

        self.__trajectory_spline = None
        self.visualization = False  # Keep public for external tools if needed
        self.fig = None
        self.ax = None

    def __plan_initial_trajectory(self, initial_obs: dict[str, NDArray[np.floating]]):
        """Generates the full trajectory from the initial observation."""
        # 1. Generate waypoints through gates
        path_points = self.__generate_gate_approach_points(
            self.__start_position,
            self.__gate_positions,
            self.__gate_normals,
            approach_distance=0.5,
            num_intermediate_points=5,
        )
        self.__debug_initial_wps = path_points.copy()

        # 2. Add sharp-turn detours
        path_points = self.__add_detour_logic(
            path_points,
            
            self.__gate_positions,
            self.__gate_normals,
            self.__gate_y_axes,
            self.__gate_z_axes,
            num_intermediate_points=5,
            detour_angle_degrees=120.0,
            detour_radius=0.65,
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
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Extracts the X (normal), Y, and Z axes from gate quaternions."""
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
        approach_distance: float = 0.5,
        num_intermediate_points: int = 5,
    ) -> NDArray[np.floating]:
        """Calculates waypoints before, at, and after each gate center."""
        # Create the offsets for intermediate points (e.g., -0.5, -0.25, 0, 0.25, 0.5)
        offsets = np.linspace(-approach_distance, approach_distance, num_intermediate_points)

        # Use broadcasting to create all waypoints at once
        # gate_positions: (N, 3) -> (N, 1, 3)
        # gate_normals:   (N, 3) -> (N, 1, 3)
        # offsets:        (M,)   -> (1, M, 1)
        gate_pos_exp = gate_positions[:, np.newaxis, :]

        gate_norm_exp = gate_normals[:, np.newaxis, :]

        offsets_exp = offsets[np.newaxis, :, np.newaxis]

        # Calculate waypoints: (N, 1, 3) + (1, M, 1) * (N, 1, 3) -> (N, M, 3)
        waypoints_matrix = gate_pos_exp + offsets_exp * gate_norm_exp
        # POINT = ORIGIN + SCALAR DISTANCE * DIRECTION_VECTOR

        # Reshape to (N*M, 3)
        flat_waypoints = waypoints_matrix.reshape(-1, 3)

        # Prepend the initial position
        return np.vstack([initial_position, flat_waypoints])

    def __compute_trajectory_spline(
        self,
        total_time: float,
        path_points: NDArray[np.floating],
        custom_time_knots: NDArray[np.floating] | None = None,
    ) -> CubicSpline:
        """Generates a 3D cubic spline trajectory through the given points."""
        if custom_time_knots is not None:
            # Use pre-computed time knots from collision avoidance
            return CubicSpline(custom_time_knots, path_points)

        # If no custom knots, parameterize by arc length
        path_segments = np.diff(path_points, axis=0)
        segment_distances = np.linalg.norm(path_segments, axis=1)

        cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)])

        # Normalize by total length and scale by duration
        time_knots = cumulative_distance / cumulative_distance[-1] * total_time

        return CubicSpline(time_knots, path_points)

    def __process_single_obstacle(
        self,
        obstacle_center: NDArray[np.floating],
        sampled_points: NDArray[np.floating],
        sampled_times: NDArray[np.floating],
        clearance_radius: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Scans a trajectory for collision with a single obstacle and inserts
        an avoidance waypoint.
        """  # noqa: D202, D205

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
                exit_index = i
                is_inside_obstacle_zone = False

                # Get entry and exit points of the conflict
                entry_point = sampled_points[entry_index]
                exit_point = sampled_points[exit_index]

                # Calculate an avoidance direction (bisector of entry/exit vectors)
                entry_vec = entry_point[:2] - obstacle_xy
                exit_vec = exit_point[:2] - obstacle_xy
                avoid_vec = entry_vec + exit_vec
                avoid_vec /= np.linalg.norm(avoid_vec)  # Normalize

                # Define new waypoint at the clearance radius along the avoid_vec
                new_pos_xy = obstacle_xy + avoid_vec * clearance_radius
                new_pos_z = (entry_point[2] + exit_point[2]) / 2  # Midpoint Z
                new_avoid_waypoint = np.concatenate([new_pos_xy, [new_pos_z]])

                # Add the new waypoint at the average time
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
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Iteratively checks for and mitigates collisions with all obstacles."""
        # Generate a temporary spline and sample it densely
        temp_spline = self.__compute_trajectory_spline(self.FLIGHT_DURATION, path_points)

        num_samples = int(self.__control_freq * self.FLIGHT_DURATION)
        sampled_times = np.linspace(0, self.FLIGHT_DURATION, num_samples)
        sampled_points = temp_spline(sampled_times)

        # Process each obstacle, modifying the time and point arrays
        for obstacle_center in obstacle_centers:
            sampled_times, sampled_points = self.__process_single_obstacle(
                obstacle_center, sampled_points, sampled_times, clearance_radius
            )

        # Return the modified time knots and waypoints
        return sampled_times, sampled_points

    # def __check_for_env_update(self, current_obs: dict[str, NDArray[np.bool_]]) -> bool:
    #     """Detects if a new gate or obstacle has been visited."""
    #     if self.__last_gate_flags is None:
    #         self.__last_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
    #         self.__last_obstacle_flags = np.array(current_obs["obstacles_visited"], dtype=bool)
    #         return False  # No change on the first check

    #     current_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
    #     current_obstacle_flags = np.array(current_obs["obstacles_visited"], dtype=bool)

    #     gate_newly_hit = np.any((~self.__last_gate_flags) & current_gate_flags)
    #     obstacle_newly_hit = np.any((~self.__last_obstacle_flags) & current_obstacle_flags)

    #     self.__last_gate_flags = current_gate_flags
    #     self.__last_obstacle_flags = current_obstacle_flags

    #     return gate_newly_hit or obstacle_newly_hit
    
    
    def __check_for_env_update(self, current_obs: dict[str, NDArray[np.bool_]]) -> bool:
            """Detects environment updates. Triggers a full re-plan if:
            1. A new gate or obstacle has been formally visited/hit (State Transition).
            2. The drone is dangerously close to ANY gate or obstacle (RWI Proximity Alert).
            """  # noqa: D205
            # --- 1. State Transition Check (Check for newly logged visits) ---
            
            # Initialize flags on the first run
            if self.__last_gate_flags is None:
                self.__last_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
                self.__last_obstacle_flags = np.array(current_obs["obstacles_visited"], dtype=bool)
                return False  # No change yet

            # Update current flags
            current_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
            current_obstacle_flags = np.array(current_obs["obstacles_visited"], dtype=bool)

            # Check for any new True value in the flags array
            gate_newly_hit = np.any((~self.__last_gate_flags) & current_gate_flags)
            obstacle_newly_hit = np.any((~self.__last_obstacle_flags) & current_obstacle_flags)

            # Update the stored flags for the next iteration
            self.__last_gate_flags = current_gate_flags
            self.__last_obstacle_flags = current_obstacle_flags
            
            # --- 2. Reactive Proximity Check (RWI) ---
            
            drone_pos = current_obs["pos"]
            
            # A. Check Proximity to ALL Gates (3D Distance)
            gate_positions_3d = current_obs["gates_pos"]
            gate_distances = np.linalg.norm(gate_positions_3d - drone_pos, axis=1)
            # Trigger if any gate, visited or unvisited, is too close
            gate_proximity_alert = np.any(gate_distances < self.REPLAN_RADIUS)

            # B. Check Proximity to ALL Obstacles (2D Distance, matching avoidance logic)
            obstacle_positions_3d = current_obs["obstacles_pos"]
            drone_pos_xy = drone_pos[:2]
            obstacle_positions_xy = obstacle_positions_3d[:, :2]
            
            obstacle_distances = np.linalg.norm(obstacle_positions_xy - drone_pos_xy, axis=1)
            # Trigger if any obstacle is too close
            obstacle_proximity_alert = np.any(obstacle_distances < self.REPLAN_RADIUS)
            
            # --- 3. Final Trigger Decision ---
            
            # The planner is triggered if a state change occurred OR if the drone is in the danger zone.
            return (
                gate_newly_hit 
                or obstacle_newly_hit 
                or gate_proximity_alert 
                or obstacle_proximity_alert
            )
        
        
    def __regenerate_flight_plan(
        self, current_obs: dict[str, NDArray[np.floating]], simulation_time: float
    ) -> None:
        """Re-plans the entire trajectory, typically after an environment change."""
        # Update environment geometry
        self.__gate_positions = current_obs["gates_pos"]
        self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
            current_obs["gates_quat"]
        )

        # 1. Generate waypoints
        path_points = self.__generate_gate_approach_points(
            self.__start_position,  # Re-plan from the original start
            self.__gate_positions,
            self.__gate_normals,
            approach_distance=0.5,
            num_intermediate_points=5,
        )

        # 2. Add detours
        path_points = self.__add_detour_logic(
            path_points,
            self.__gate_positions,
            self.__gate_normals,
            self.__gate_y_axes,
            self.__gate_z_axes,
            num_intermediate_points=5,
            detour_angle_degrees=120.0,
            detour_radius=0.65,
        )

        # 3. Avoid obstacles
        time_knots, path_points = self.__insert_obstacle_avoidance_points(
            path_points, current_obs["obstacles_pos"], self.OBSTACLE_CLEARANCE
        )

        # 4. Generate new spline
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points, custom_time_knots=time_knots
        )

    def __determine_detour_direction(
        self,
        v_proj: NDArray[np.floating],
        v_proj_norm: float,
        y_axis: NDArray[np.floating],
        z_axis: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], str, float]:
        """Calculates the best "sideways" direction to add a detour waypoint."""
        if v_proj_norm < 1e-6:  # prevent division by zero
            # Default to right if projection is zero
            return y_axis, "right (+y_axis) [default]", 0.0

        # Project onto gate's Y and Z axes
        v_proj_y = np.dot(v_proj, y_axis)
        v_proj_z = np.dot(v_proj, z_axis)

        # Find angle in the Y-Z plane
        proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi

        # Choose the cardinal direction (right, top, left)
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
        num_intermediate_points: int = 5,
        detour_angle_degrees: float = 120.0,
        detour_radius: float = 0.65,
    ) -> NDArray[np.floating]:
        """Analyzes transitions between gates and inserts extra waypoints to
        widen sharp turns.
        """  # noqa: D205
        num_gates = gate_positions.shape[0]
        path_points_list = list(path_points)

        detour_insert_count = 0  # Count of detour waypoints added, so that indices can be adjusted
        self.__debug_detour_log = []  # Reset log
        self.__debug_detour_points_added = []

        for i in range(num_gates - 1):
            debug_info = {"gate_i": i, "gate_i_plus_1": i + 1}

            # Index of the last waypoint associated with gate i
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + detour_insert_count

            # Index of the first waypoint associated with gate i+1
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + detour_insert_count

            debug_info["last_idx_gate_i"] = last_idx_gate_i
            debug_info["first_idx_gate_i_plus_1"] = first_idx_gate_i_plus_1

            # Transition points
            p1_exit_wp = path_points_list[last_idx_gate_i]
            p2_entry_wp = path_points_list[first_idx_gate_i_plus_1]

            debug_info["p1_last_of_gate_i"] = p1_exit_wp.copy()
            debug_info["p2_first_of_gate_i_plus_1"] = p2_entry_wp.copy()

            transition_vector = p2_entry_wp - p1_exit_wp
            vector_norm = np.linalg.norm(transition_vector)

            debug_info["vector_p1_to_p2"] = transition_vector.copy()
            debug_info["vector_norm"] = vector_norm

            if vector_norm < 1e-6:  # Prevent division by zero
                debug_info["skipped"] = True
                debug_info["skip_reason"] = "vector_too_short"
                self.__debug_detour_log.append(debug_info)
                continue

            # Check angle between transition vector and the exit gate's normal
            normal_i = gate_normals[i]

            cos_angle = (
                np.dot(transition_vector, normal_i) / vector_norm
            )  # finds the cosine of the angle between the two vectors

            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip to valid range for arccos
            angle_deg = np.arccos(cos_angle) * 180 / np.pi  # Convert to degrees

            debug_info["gate_i_normal"] = normal_i.copy()
            debug_info["cos_angle"] = cos_angle
            debug_info["angle_degrees"] = angle_deg
            debug_info["angle_threshold"] = detour_angle_degrees

            if angle_deg > detour_angle_degrees:
                # Angle is too sharp; a detour is needed
                debug_info["needs_detour"] = True

                gate_center = gate_positions[i]
                y_axis = gate_y_axes[i]  # left-right axis of the gate
                z_axis = gate_z_axes[i]  # up-down axis of the gate

                debug_info["gate_i_center"] = gate_center.copy()
                debug_info["gate_i_y_axis"] = y_axis.copy()
                debug_info["gate_i_z_axis"] = z_axis.copy()
                debug_info["detour_distance"] = detour_radius

                # Project transition vector onto the gate's Y-Z plane
                # v_proj is the component of the transition vector that is parallel to the gate plane
                v_proj = transition_vector - np.dot(transition_vector, normal_i) * normal_i

                v_proj_norm = np.linalg.norm(v_proj)

                debug_info["v_projection_on_gate_plane"] = v_proj.copy()
                debug_info["v_projection_norm"] = v_proj_norm

                # Find the best direction for the detour
                detour_direction_vector, detour_direction_name, proj_angle_deg = (
                    self.__determine_detour_direction(v_proj, v_proj_norm, y_axis, z_axis)
                )

                debug_info["detour_direction_vector"] = detour_direction_vector.copy()
                debug_info["detour_direction_name"] = detour_direction_name
                debug_info["projection_angle_degrees"] = proj_angle_deg

                # Create the new detour waypoint
                detour_waypoint = gate_center + detour_radius * detour_direction_vector

                debug_info["detour_waypoint"] = detour_waypoint.copy()
                debug_info["detour_direction"] = detour_direction_name

                # Insert the waypoint *after* the last point of gate i
                insert_position = last_idx_gate_i + 1
                path_points_list.insert(insert_position, detour_waypoint)
                detour_insert_count += 1

                debug_info["insert_position"] = insert_position
                debug_info["inserted"] = True
                self.__debug_detour_points_added.append(detour_waypoint.copy())

            else:
                debug_info["needs_detour"] = False
                debug_info["inserted"] = False

            debug_info["total_inserted_so_far"] = detour_insert_count
            self.__debug_detour_log.append(debug_info)

        self.__debug_detour_stats = {
            "total_detours_added": detour_insert_count,
            "original_waypoint_count": len(path_points),
            "final_waypoint_count": len(path_points_list),
            "num_gate_pairs_checked": num_gates - 1,
            "detour_waypoints": self.__debug_detour_points_added,
        }

        return np.array(path_points_list)

    # --- Public API Methods ---

    # This function is called at every time step to compute the desired state
    """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """

    def compute_control(
        self, current_obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Calculates the control action for the current time step."""
        # Determine current position in the trajectory
        elapsed_time = min(self.__current_step / self.__control_freq, self.FLIGHT_DURATION)

        # Get target position from the spline
        # trajectory is a CubicSpline object so calling it with time returns position
        if self.__check_for_env_update(current_obs):
            # This regenerates the entire spline based on new gate positions
            self.__regenerate_flight_plan(current_obs, elapsed_time) # NOTE: elapsed_time must be defined first
        # self.__regenerate_flight_plan(current_obs, elapsed_time)

        reference_position = self.__trajectory_spline(elapsed_time, nu=0)

        # 2. Get target velocity (1st derivative)
        reference_velocity = self.__trajectory_spline(elapsed_time, nu=1)

        # 3. Get target acceleration (2nd derivative)
        reference_acceleration = self.__trajectory_spline(elapsed_time, nu=2)

        # we do this as there are no gates till the first one is reached

        # Check if the environment has changed (e.g., gate passed)
        if self.__check_for_env_update(current_obs):
            self.__regenerate_flight_plan(current_obs, elapsed_time)

        if elapsed_time >= self.FLIGHT_DURATION:
            self.__is_complete = True

        # Visualization
        try:
            draw_line(
                self.env,
                self.__trajectory_spline(self.__trajectory_spline.x),
                rgba=np.array([1.0, 1.0, 1.0, 0.2]),
            )
        except (AttributeError, TypeError):
            pass  # env not available or spline not ready

        # Return target position (rest of state is zeroed, no pid control here)
        # return np.concatenate((reference_position, np.zeros(10)), dtype=np.float32)

        p_v_a_ref = np.concatenate((reference_position, reference_velocity, reference_acceleration))

        # Fill the remaining 4 components (typically jerk and attitude/yaw) with zeros.
        # The drone's internal controller uses p, v, and a to generate the necessary control forces.
        zero_pad = np.zeros(self.STATE_SIZE - len(p_v_a_ref))  # 13 - 9 = 4 zeros

        return np.concatenate((p_v_a_ref, zero_pad), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        current_obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Called after each simulation step. Increments time."""
        self.__current_step += 1
        return self.__is_complete

    def get_spline_function(self) -> CubicSpline:
        """Returns the raw CubicSpline object."""
        return self.__trajectory_spline

    def get_sampled_trajectory(self) -> NDArray[np.floating]:
        """Returns a densely sampled array of points from the trajectory."""
        num_samples = int(self.__control_freq * self.FLIGHT_DURATION)
        time_samples = np.linspace(0, self.FLIGHT_DURATION, num_samples)
        return self.__trajectory_spline(time_samples)

    def update_current_step(self, step: int) -> None:
        """Manually sets the controller's internal time step."""
        self.__current_step = step
