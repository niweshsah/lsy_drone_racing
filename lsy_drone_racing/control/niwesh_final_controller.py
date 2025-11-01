"""
Spline Trajectory Controller for Drone Racing

Description:
  This module defines a controller for autonomous drone racing. It generates
  a smooth, 3D trajectory through a series of gates using cubic spline 
  interpolation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

# Attempt to import matplotlib for optional 3D visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # Visualization will be disabled if matplotlib is not available

# Type hint for numpy arrays, visible only during static analysis
if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplineTrajectoryController(Controller):

    # === Controller Configuration Constants ===

    # Total time allotted to complete the entire trajectory (in seconds).
    TRAJECTORY_DURATION = 17.0
    
    # Expected dimension of the state vector (pos, vel, acc, yaw, rates).
    STATE_DIMENSION = 13
    
    # Minimum safe horizontal distance to maintain from obstacles (in meters).
    OBSTACLE_SAFETY_DISTANCE = 0.3
    
    # Number of points to use when drawing the trajectory in the 3D plot.
    VISUALIZATION_SAMPLES = 100
    
    # Frequency to print debug info (every N control steps). Not used in this version.
    LOG_INTERVAL = 100

    def __init__(
        self, 
        obs: dict[str, NDArray[np.floating]], 
        info: dict, 
        config: dict
    ):
        """
        Initializes the controller by building the first trajectory.

        Args:
            obs: The initial observation dictionary from the environment,
                 containing drone state, gate positions, and obstacle positions.
            info: The initial info dictionary from the environment.
            config: The environment configuration dictionary, containing
                    settings like control frequency.
        """
        super().__init__(obs, info, config)

        # --- Internal Controller State ---
        self._time_step = 0  # Counter for current control step
        self._control_frequency = config.env.freq  # Control frequency (Hz)
        self._is_finished = False  # Flag to signal trajectory completion

        # --- Environment Change Detection ---
        self._last_gate_flags = None  # Stores 'gates_visited' from last step
        self._last_obstacle_flags = None  # Stores 'obstacles_visited' from last step

        # --- Debugging Attributes ---
        # These store intermediate results from the planning pipeline for analysis
        self._debug_detour_analysis = []
        self._debug_detour_summary = {}
        self._debug_detour_waypoints_added = []
        self._debug_waypoints_initial = None
        self._debug_waypoints_after_detour = None
        self._debug_waypoints_final = None

        # --- Store Initial Environment Data ---
        self.gate_positions = obs['gates_pos']
        self.gate_normals, self.gate_y_axes, self.gate_z_axes = \
            self._get_gate_coordinate_frames(obs['gates_quat'])
        self.obstacle_positions = obs['obstacles_pos']
        self.initial_position = obs['pos']

        # --- Visualization Setup ---
        self.visualization = False  # Set to True to enable 3D matplotlib plot
        self.fig = None  # Figure handle
        self.ax = None  # 3D Axes handle

        # === Initial Trajectory Planning Pipeline ===

        # Step 1: Generate basic waypoints (e.g., 5 points per gate)
        waypoints = self._calculate_initial_waypoints(
            self.initial_position,
            self.gate_positions,
            self.gate_normals,
            approach_distance=0.5,
            num_intermediate_points=5
        )
        self._debug_waypoints_initial = waypoints.copy()

        # Step 2: Analyze angles between gates and insert detour waypoints
        # to prevent sharp, unrealistic "backtracking" turns.
        waypoints = self._insert_detour_waypoints(
            waypoints,
            self.gate_positions,
            self.gate_normals,
            self.gate_y_axes,
            self.gate_z_axes,
            num_intermediate_points=5,
            angle_threshold=120.0,
            detour_distance=0.65
        )
        self._debug_waypoints_after_detour = waypoints.copy()

        # Step 3: Check trajectory against obstacles and insert avoidance points
        time_params, waypoints = self._adjust_waypoints_for_obstacle_avoidance(
            waypoints, 
            self.obstacle_positions,
            self.OBSTACLE_SAFETY_DISTANCE
        )
        self._debug_waypoints_final = waypoints.copy()

        # Step 4: Generate the final, smooth spline from the processed waypoints
        self.trajectory = self._generate_spline_trajectory(
            self.TRAJECTORY_DURATION, 
            waypoints
        )

        # Initialize 3D plot if visualization is enabled
        if self.visualization:
            self._update_3d_visualization(
                self.gate_positions,
                self.gate_normals,
                obstacle_positions=self.obstacle_positions,
                trajectory=self.trajectory,
                waypoints=waypoints,
                drone_position=obs['pos']
            )

    def _get_gate_normals(
        self, 
        gates_quaternions: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Extracts only the normal vector (local X-axis) for each gate.

        Note: This is a subset of `_get_gate_coordinate_frames`.
        
        Args:
            gates_quaternions: Array of gate orientations as quaternions [w, x, y, z].

        Returns:
            Array of (N, 3) gate normal vectors.
        """
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        # The normal vector (direction of passage) is the local X-axis.
        return rotation_matrices[:, :, 0]

    def _calculate_initial_waypoints(
        self,
        initial_position: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        approach_distance: float = 0.5,
        num_intermediate_points: int = 5
    ) -> NDArray[np.floating]:
        """
        Generates a set of waypoints centered around each gate.

        For each gate, it creates points along its normal vector, 
        from -approach_distance (entry) to +approach_distance (exit).
        

        Args:
            initial_position: The drone's (3,) starting position.
            gate_positions: (N, 3) positions of all gates.
            gate_normals: (N, 3) normal vectors of all gates.
            approach_distance: Distance before/after gate center.
            num_intermediate_points: Number of points to generate per gate.

        Returns:
            Array of (1 + N*num_intermediate_points, 3) waypoints, 
            starting with the initial_position.
        """
        num_gates = gate_positions.shape[0]

        waypoints_per_gate = []
        for i in range(num_intermediate_points):
            # Linearly interpolate offset from -approach_distance to +approach_distance
            offset = -approach_distance + (i / (num_intermediate_points - 1)) * 2 * approach_distance
            # Calculate this waypoint for all gates
            waypoints_per_gate.append(gate_positions + offset * gate_normals)

        # Reshape the list of (N,3) arrays into (N * num_intermediate_points, 3)
        waypoints = np.concatenate(waypoints_per_gate, axis=1)
        waypoints = waypoints.reshape(num_gates, num_intermediate_points, 3).reshape(-1, 3)

        # Prepend the drone's starting position
        waypoints = np.vstack([initial_position, waypoints])

        return waypoints

    def _generate_spline_trajectory(
        self, 
        duration: float, 
        waypoints: NDArray[np.floating]
    ) -> CubicSpline:
        """
        Generates a 3D Cubic Spline trajectory through the given waypoints.
        
        Uses arc-length parameterization to distribute time steps more
        evenly along the path's actual length, preventing the drone from
        speeding up excessively on long segments and slowing down on short ones.
        

        Args:
            duration: The total time the trajectory should take.
            waypoints: An array of (K, 3) waypoints.

        Returns:
            A `scipy.interpolate.CubicSpline` object that maps time `t`
            in [0, duration] to a 3D position.
        """
        # Calculate the Euclidean distance for each segment
        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)

        # Calculate cumulative distance (arc length) at each waypoint
        cumulative_arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])

        # Normalize cumulative distance to [0, 1] and scale by total duration
        # This maps time 't' proportionally to the distance traveled
        total_length = cumulative_arc_length[-1]
        if total_length < 1e-6:
             # Handle edge case: no movement (or single waypoint)
            time_parameters = np.linspace(0, duration, len(waypoints))
        else:
            time_parameters = (cumulative_arc_length / total_length) * duration

        # Create the spline (time -> 3D position)
        # `axis=0` treats each dimension (x, y, z) independently
        return CubicSpline(time_parameters, waypoints, axis=0)

    def _adjust_waypoints_for_obstacle_avoidance(
        self,
        waypoints: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating],
        safety_distance: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Modifies a trajectory to avoid cylindrical obstacles.

        This method samples the trajectory and inserts new avoidance
        waypoints where the path violates the horizontal safety distance.
        

        Args:
            waypoints: The original (K, 3) waypoints.
            obstacle_positions: (M, 3) positions of obstacles.
            safety_distance: The minimum horizontal (XY) distance to maintain.

        Returns:
            A tuple of (time_parameters, modified_waypoints_array).
        """
        # Generate a temporary trajectory from the current waypoints
        trajectory = self._generate_spline_trajectory(
            self.TRAJECTORY_DURATION, 
            waypoints
        )

        # Sample the trajectory at high resolution
        time_samples = np.linspace(0, self.TRAJECTORY_DURATION, 
                                   int(self._control_frequency * self.TRAJECTORY_DURATION))
        trajectory_points = trajectory(time_samples)

        # Iterate over each obstacle and adjust the trajectory points
        for obstacle_position in obstacle_positions:
            collision_free_times = []
            collision_free_waypoints = []

            is_inside_obstacle = False
            entry_index = None

            for i, point in enumerate(trajectory_points):
                # Check horizontal (XY) distance only, ignoring Z
                distance_xy = np.linalg.norm(obstacle_position[:2] - point[:2])

                if distance_xy < safety_distance:
                    # Point is inside the safety radius
                    if not is_inside_obstacle:
                        # This is the first point inside: mark the entry
                        is_inside_obstacle = True
                        entry_index = i
                
                elif is_inside_obstacle:
                    # We were inside, but this point is now outside: mark the exit
                    exit_index = i
                    is_inside_obstacle = False

                    # Get the points where the path entered and exited
                    entry_point = trajectory_points[entry_index]
                    exit_point = trajectory_points[exit_index]

                    # Calculate a detour direction vector in the XY plane
                    entry_direction = entry_point[:2] - obstacle_position[:2]
                    exit_direction = exit_point[:2] - obstacle_position[:2]
                    avoidance_direction = entry_direction + exit_direction
                    avoidance_direction /= np.linalg.norm(avoidance_direction)

                    # Create a new waypoint at the safety distance
                    new_position_xy = obstacle_position[:2] + avoidance_direction * safety_distance
                    # Set Z to the average Z of the entry/exit points
                    new_position_z = (entry_point[2] + exit_point[2]) / 2
                    new_waypoint = np.concatenate([new_position_xy, [new_position_z]])

                    # Insert this new waypoint (and its time) into our list
                    collision_free_times.append((time_samples[entry_index] + time_samples[exit_index]) / 2)
                    collision_free_waypoints.append(new_waypoint)

                else:
                    # This point is safe, so we keep it
                    collision_free_times.append(time_samples[i])
                    collision_free_waypoints.append(point)

            # Update the trajectory points for the next obstacle check
            time_samples = np.array(collision_free_times)
            trajectory_points = np.array(collision_free_waypoints)

        # Return the new set of waypoints (and their associated times)
        return time_samples, trajectory_points

    def _update_3d_visualization(
        self,
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating] = None,
        trajectory: CubicSpline = None,
        waypoints: NDArray[np.floating] = None,
        drone_position: NDArray[np.floating] = None
    ) -> None:
        """
        Updates the 3D matplotlib visualization, if enabled.

        Args:
            gate_positions: (N, 3) array of gate positions.
            gate_normals: (N, 3) array of gate normal vectors.
            obstacle_positions: Optional (M, 3) array of obstacle positions.
            trajectory: Optional CubicSpline trajectory object.
            waypoints: Optional (K, 3) array of discrete waypoints.
            drone_position: Optional (3,) array of the drone's current position.
        """
        if plt is None:
            # Matplotlib not imported, cannot visualize
            return

        # Initialize figure and 3D axes on first call
        if self.fig is None:
            plt.ion()  # Turn on interactive mode
            self.fig = plt.figure(num=1, figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')

        # Clear the axes for the new frame
        self.ax.cla()

        # Plot discrete waypoints (blue dashed line)
        if waypoints is not None:
            self.ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                        marker='.', linestyle='--', color='blue', 
                        label='Waypoints', linewidth=1)

        # Plot the smooth spline trajectory (orange line)
        if trajectory is not None:
            t_samples = np.linspace(0, self.TRAJECTORY_DURATION, self.VISUALIZATION_SAMPLES)
            traj_points = trajectory(t_samples)
            self.ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2],
                        marker='x', linestyle='-', color='orange',
                        label='Trajectory', markersize=3, linewidth=2)

        # Plot gates as green arrows (representing their normal vector)
        for pos, normal in zip(gate_positions, gate_normals):
            self.ax.quiver(pos[0], pos[1], pos[2],
                          normal[0], normal[1], normal[2],
                          length=0.5, color='green', linewidth=1.5, arrow_length_ratio=0.3)

        # Plot obstacles as vertical grey cylinders
        if obstacle_positions is not None:
            for obs_pos in obstacle_positions:
                self.ax.plot([obs_pos[0], obs_pos[0]], 
                           [obs_pos[1], obs_pos[1]], 
                           [0, 1.4],  # Assume height of 1.4m
                           color='grey', linewidth=5, alpha=0.6)

        # Plot the drone's current position (red 'X')
        if drone_position is not None:
            self.ax.scatter(drone_position[0], drone_position[1], drone_position[2],
                          marker='x', s=200, color='red', linewidths=3,
                          label='Drone')

        # Configure plot labels and appearance
        self.ax.set_title("Planned Trajectory Visualization", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("X (m)", fontsize=11)
        self.ax.set_ylabel("Y (m)", fontsize=11)
        self.ax.set_zlabel("Z (m)", fontsize=11)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)

        # Redraw the canvas
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Pause to allow GUI to update

    def _check_for_environment_change(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """
        Detects if any gate or obstacle has been newly visited.
        
        This serves as a simple proxy for detecting environment changes
        in scenarios where gate/obstacle positions are randomized upon being
        visited.

        Args:
            obs: The current observation dictionary containing 'visited' flags.

        Returns:
            True if a new "visited" flag is set, False otherwise.
        """
        # Initialize flags on the first call
        if self._last_gate_flags is None:
            self._last_gate_flags = np.array(obs['gates_visited'], dtype=bool)
            self._last_obstacle_flags = np.array(obs['obstacles_visited'], dtype=bool)
            return False  # No change on the first step

        # Get current boolean flags
        current_gate_flags = np.array(obs['gates_visited'], dtype=bool)
        current_obstacle_flags = np.array(obs['obstacles_visited'], dtype=bool)

        # Check if any flag has changed from False to True
        gate_newly_visited = np.any((~self._last_gate_flags) & current_gate_flags)
        obstacle_newly_visited = np.any((~self._last_obstacle_flags) & current_obstacle_flags)

        # Update stored flags for the next step's comparison
        self._last_gate_flags = current_gate_flags
        self._last_obstacle_flags = current_obstacle_flags

        # Return True if any new item was visited
        return gate_newly_visited or obstacle_newly_visited

    def _execute_replanning_pipeline(
        self, 
        obs: dict[str, NDArray[np.floating]], 
        current_time: float
    ) -> None:
        """
        Executes the full trajectory replanning pipeline using new obs data.

        This re-runs the same logic as `__init__` but with updated
        gate and obstacle positions from the latest observation.

        Args:
            obs: The current observation (containing new positions).
            current_time: The current simulation time (for logging, etc.).
        """
        # Step 1: Update all environment information
        self.gate_positions = obs['gates_pos']
        self.gate_normals, self.gate_y_axes, self.gate_z_axes = \
            self._get_gate_coordinate_frames(obs['gates_quat'])

        # Step 2: Generate basic waypoints from the new gate positions
        waypoints = self._calculate_initial_waypoints(
            self.initial_position,  # NOTE: Still plans from the original start
            self.gate_positions,
            self.gate_normals,
            approach_distance=0.5,
            num_intermediate_points=5
        )

        # Step 3: Insert detour waypoints for the new path
        waypoints = self._insert_detour_waypoints(
            waypoints,
            self.gate_positions,
            self.gate_normals,
            self.gate_y_axes,
            self.gate_z_axes,
            num_intermediate_points=5,
            angle_threshold=120.0,
            detour_distance=0.65
        )

        # Step 4: Adjust for new obstacle positions
        _, waypoints = self._adjust_waypoints_for_obstacle_avoidance(
            waypoints,
            obs['obstacles_pos'],
            self.OBSTACLE_SAFETY_DISTANCE
        )

        # Step 5: Generate and store the new trajectory
        self.trajectory = self._generate_spline_trajectory(
            self.TRAJECTORY_DURATION, 
            waypoints
        )

    def _get_gate_coordinate_frames(
        self, 
        gates_quaternions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Extracts the complete local coordinate frame (X, Y, Z axes) for each gate.
        

        Args:
            gates_quaternions: Array of (N, 4) gate orientations [w, x, y, z].

        Returns:
            A tuple of three (N, 3) arrays:
            - normals (Local X-axis: direction of passage)
            - y_axes (Local Y-axis: "right" direction / width)
            - z_axes (Local Z-axis: "up" direction / height)
        """
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()

        normals = rotation_matrices[:, :, 0]  # First column: X-axis
        y_axes = rotation_matrices[:, :, 1]   # Second column: Y-axis
        z_axes = rotation_matrices[:, :, 2]   # Third column: Z-axis

        return normals, y_axes, z_axes

    def _insert_detour_waypoints(
        self,
        waypoints: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: int = 5,
        angle_threshold: float = 120.0,
        detour_distance: float = 0.65
    ) -> NDArray[np.floating]:
        """
        Inserts "detour" waypoints to handle sharp, backtracking turns.

        It checks the angle between the exit of one gate and the entry of
        the next. If the angle is too large (i.e., the drone must fly
        "backwards"), it inserts a detour waypoint to the side (left, right,
        or top) of the first gate to create a smoother, wider turn.
        

        Args:
            waypoints: The current (K, 3) waypoint array.
            gate_positions: (N, 3) positions of all gates.
            gate_normals: (N, 3) normal vectors (X-axes).
            gate_y_axes: (N, 3) Y-axes (width).
            gate_z_axes: (N, 3) Z-axes (height).
            num_intermediate_points: Number of waypoints per gate (must
                                     match `_calculate_initial_waypoints`).
            angle_threshold: Angle (in degrees) to trigger detour insertion.
            detour_distance: How far from the gate center to place the detour.

        Returns:
            A new (K+M, 3) waypoint array, where M is the number of
            detours inserted.
        """
        num_gates = gate_positions.shape[0]
        # Convert to list for efficient `insert` operations
        waypoints_list = list(waypoints)

        inserted_count = 0  # Keep track of insertions to correct indices

        # Clear debug info for this run
        self._debug_detour_analysis = []
        self._debug_detour_waypoints_added = []

        # Check the transition between each pair of consecutive gates
        for i in range(num_gates - 1):
            debug_info = {'gate_i': i, 'gate_i_plus_1': i + 1}

            # Calculate indices for the exit/entry points,
            # accounting for waypoints we've already inserted.
            # `1 + ...`: +1 to skip the drone's initial position
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + inserted_count
            debug_info['last_idx_gate_i'] = last_idx_gate_i
            debug_info['first_idx_gate_i_plus_1'] = first_idx_gate_i_plus_1

            # p1 = Exit waypoint of gate `i`
            # p2 = Entry waypoint of gate `i+1`
            p1 = waypoints_list[last_idx_gate_i]
            p2 = waypoints_list[first_idx_gate_i_plus_1]
            debug_info['p1_last_of_gate_i'] = p1.copy()
            debug_info['p2_first_of_gate_i_plus_1'] = p2.copy()

            # Vector connecting the two waypoints
            v = p2 - p1
            v_norm = np.linalg.norm(v)
            debug_info['vector_p1_to_p2'] = v.copy()
            debug_info['vector_norm'] = v_norm

            if v_norm < 1e-6:
                # Points are identical, skip this pair
                debug_info['skipped'] = True
                debug_info['skip_reason'] = 'vector_too_short'
                self._debug_detour_analysis.append(debug_info)
                continue

            # Check angle between transition vector `v` and gate `i`'s normal
            normal_i = gate_normals[i]
            cos_angle = np.dot(v, normal_i) / v_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # For numerical stability
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            debug_info['gate_i_normal'] = normal_i.copy()
            debug_info['cos_angle'] = cos_angle
            debug_info['angle_degrees'] = angle_deg
            debug_info['angle_threshold'] = angle_threshold

            # If angle > threshold, we are "backtracking"
            if angle_deg > angle_threshold:
                debug_info['needs_detour'] = True
                gate_center = gate_positions[i]
                y_axis = gate_y_axes[i]
                z_axis = gate_z_axes[i]
                debug_info['gate_i_center'] = gate_center.copy()
                debug_info['gate_i_y_axis'] = y_axis.copy()
                debug_info['gate_i_z_axis'] = z_axis.copy()

                # --- Determine Detour Direction (Left/Right/Top) ---
                
                # 1. Project the transition vector `v` onto gate `i`'s plane
                #    (by removing the component along its normal)
                v_proj = v - np.dot(v, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)
                debug_info['v_projection_on_gate_plane'] = v_proj.copy()

                if v_proj_norm < 1e-6:
                    # Projection is zero (e.g., flying straight backwards)
                    # Default to a "right" detour
                    detour_direction_vector = y_axis
                    detour_direction_name = 'right (+y_axis) [default]'
                    proj_angle_deg = 0.0
                else:
                    # 2. Get components of the projected vector in the gate's
                    #    local YZ (width/height) coordinate system
                    v_proj_y = np.dot(v_proj, y_axis)  # Left/Right component
                    v_proj_z = np.dot(v_proj, z_axis)  # Up/Down component
                    debug_info['v_proj_y_component'] = v_proj_y
                    debug_info['v_proj_z_component'] = v_proj_z

                    # 3. Calculate angle in the gate's YZ plane
                    #    (0=Right, 90=Top, 180=Left, -90=Bottom)
                    proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi
                    debug_info['projection_angle_degrees'] = proj_angle_deg

                    # 4. Choose the best detour direction (Right, Top, or Left)
                    if -90 <= proj_angle_deg < 45:
                        detour_direction_vector = y_axis
                        detour_direction_name = 'right (+y_axis)'
                    elif 45 <= proj_angle_deg < 135:
                        detour_direction_vector = z_axis
                        detour_direction_name = 'top (+z_axis)'
                    else:  # angle >= 135 or angle < -90
                        detour_direction_vector = -y_axis
                        detour_direction_name = 'left (-y_axis)'
                
                debug_info['detour_direction_name'] = detour_direction_name

                # 5. Calculate the final detour waypoint
                detour_waypoint = gate_center + detour_distance * detour_direction_vector
                debug_info['detour_waypoint'] = detour_waypoint.copy()

                # 6. Insert the new waypoint into the list *after* gate `i`
                insert_position = last_idx_gate_i + 1
                waypoints_list.insert(insert_position, detour_waypoint)
                inserted_count += 1
                debug_info['insert_position'] = insert_position
                debug_info['inserted'] = True
            
            else:
                # Angle is fine, no detour needed
                debug_info['needs_detour'] = False
                debug_info['inserted'] = False

            debug_info['total_inserted_so_far'] = inserted_count
            self._debug_detour_analysis.append(debug_info)

        # Store summary debug information
        self._debug_detour_summary = {
            'total_detours_added': inserted_count,
            'original_waypoint_count': len(waypoints),
            'final_waypoint_count': len(waypoints_list),
            'num_gate_pairs_checked': num_gates - 1,
            'detour_waypoints': self._debug_detour_waypoints_added
        }

        # Convert list back to numpy array
        return np.array(waypoints_list)

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """
        Calculates the desired state for the next control step.

        This is the main "tick" method of the controller.

        Args:
            obs: Current observation from the environment.
            info: Optional info dictionary.

        Returns:
            A 13-element desired state vector:
            [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
            Only the 3D position (x, y, z) is set; the rest are zeros.
        """
        # Calculate current time along the trajectory
        # `min` ensures we don't sample past the end time
        current_time = min(self._time_step / self._control_frequency, 
                           self.TRAJECTORY_DURATION)

        # Sample the target 3D position from our spline
        target_position = self.trajectory(current_time)

        # Check for environment changes (e.g., passed a gate)
        if self._check_for_environment_change(obs):
            # If so, trigger a full replan of the trajectory
            self._execute_replanning_pipeline(obs, current_time)
        
        # Update the 3D plot, if enabled
        if self.visualization:
            self._update_3d_visualization(
                self.gate_positions,
                self.gate_normals,
                obstacle_positions=obs['obstacles_pos'],
                trajectory=self.trajectory,
                drone_position=obs['pos']  # Show current drone position
            )
        
        # Check if we have completed the trajectory
        if current_time >= self.TRAJECTORY_DURATION:
            self._is_finished = True

        # Try to draw the trajectory in the simulation (if available)
        try:
            draw_line(self.env, self.trajectory(self.trajectory.x), 
                     rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        except (AttributeError, TypeError):
            # `self.env` might not be set, or `draw_line` not supported
            pass

        # Return the 13-element state vector
        # The low-level controller will handle velocity, acceleration, etc.
        # We only need to provide the target position.
        return np.concatenate((target_position, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict
    ) -> bool:
        """
        Called after each environment step.

        Args:
            action: The action taken by the low-level controller.
            obs: The resulting observation.
            reward: The reward received.
            terminated: Whether the episode terminated.
            truncated: Whether the episode was truncated.
            info: Additional information.

        Returns:
            True if the controller is finished, False otherwise.
        """
        # Increment our internal time step
        self._time_step += 1
        # Signal to the environment if our trajectory is complete
        return self._is_finished

    # === Public Helper Methods ===

    def get_trajectory_function(self) -> CubicSpline:
        """
        Public accessor for the trajectory spline.

        Returns:
            The `CubicSpline` object representing the trajectory.
        """
        return self.trajectory

    def get_trajectory_waypoints(self) -> NDArray[np.floating]:
        """
        Public accessor for the full, time-sampled trajectory.

        Returns:
            An (N, 3) array of 3D points, sampled at the
            control frequency for the entire trajectory duration.
        """
        time_samples = np.linspace(0, self.TRAJECTORY_DURATION,
                                   int(self._control_frequency * self.TRAJECTORY_DURATION))
        return self.trajectory(time_samples)

    def set_time_step(self, time_step: int) -> None:
        """
        Public method to manually set the controller's internal time.
        (Useful for testing or debugging)

        Args:
            time_step: The new time step value.
        """
        self._time_step = time_step