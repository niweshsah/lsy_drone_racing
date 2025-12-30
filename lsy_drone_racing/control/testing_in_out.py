"""
TESTING MODE:
1) Targeted only at Gate 0.
2) Generates a path through Gate 0, stops, and reverses back to start.
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

    MODIFIED FOR TESTING:
    This version only processes the FIRST gate. It generates a trajectory
    to fly through the gate and immediately return to the start position
    along the same path.
    """

    # --- Configuration Constants ---
    FLIGHT_DURATION = 15.0  # Reduced time for the short test
    STATE_SIZE = 13
    OBSTACLE_CLEARANCE = 0.3
    
    # Trajectory Generation Parameters
    APPROACH_DISTANCE = 0.5  # Distance before/after gate to place waypoints
    NUM_INTERMEDIATE_POINTS = 5  # Number of points to generate through a gate
    
    # Detour constants are kept but logic is bypassed for this specific test
    DETOUR_ANGLE_THRESHOLD = 120.0  
    DETOUR_RADIUS = 0.65  

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], info: dict, sim_config: dict, env=None):
        """Initializes the controller, sets up state, and plans the first trajectory."""
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
        
        print("observation:", initial_obs)

        # --- TESTING MODIFICATION: ONLY KEEP FIRST GATE ---
        # We slice [:1] to ensure we only look at the first gate
        self.__gate_positions = initial_obs["gates_pos"][:1] 
        self.__obstacle_positions = initial_obs["obstacles_pos"]
        self.__start_position = initial_obs["pos"]

        # Extract gate orientation frames (Only for the first gate)
        self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
            initial_obs["gates_quat"][:1]
        )

        self.__trajectory_spline: Optional[CubicSpline] = None
        self.visualization = False
        self.fig = None
        self.ax = None

    def __plan_initial_trajectory(self, initial_obs: dict[str, NDArray[np.floating]]):
        """Generates the Out-and-Back trajectory."""
        
        # 1. Generate waypoints through the first gate (Forward Path)
        forward_path = self.__generate_gate_approach_points(
            self.__start_position,
            self.__gate_positions,
            self.__gate_normals,
            approach_distance=self.APPROACH_DISTANCE,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS,
        )

        # 2. Generate Return Path (Reverse the forward path)
        # We slice [::-1] to reverse. 
        # We also slice [1:] to avoid duplicating the turnaround point exactly, 
        # though CubicSpline handles duplicates by assuming 0 velocity usually.
        return_path = forward_path[::-1]
        
        # Combine: Start -> Gate -> Turnaround -> Gate -> Start
        full_path_points = np.vstack([forward_path, return_path[1:]])

        # 3. Adjust waypoints to avoid obstacles (Applied to the full path)
        time_knots, path_points = self.__insert_obstacle_avoidance_points(
            full_path_points, self.__obstacle_positions, self.OBSTACLE_CLEARANCE
        )

        # 4. Generate the final spline
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points, custom_time_knots=time_knots
        )

    def __extract_gate_frames(
        self, gates_quaternions: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Extracts the local coordinate axes (Normal, Y, Z) from gate quaternions."""
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
        """Calculates a sequence of waypoints passing through the gate."""
        offsets = np.linspace(-approach_distance, approach_distance, num_intermediate_points)

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
        """Generates a 3D cubic spline trajectory."""
        if custom_time_knots is not None:
            return CubicSpline(custom_time_knots, path_points)

        path_segments = np.diff(path_points, axis=0)
        segment_distances = np.linalg.norm(path_segments, axis=1)

        cumulative_distance = np.concatenate([[0], np.cumsum(segment_distances)])
        time_knots = cumulative_distance / cumulative_distance[-1] * total_time

        return CubicSpline(time_knots, path_points)

    def __process_single_obstacle(
        self,
        obstacle_center: NDArray[np.floating],
        sampled_points: NDArray[np.floating],
        sampled_times: NDArray[np.floating],
        clearance_radius: float,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Scans a trajectory for collisions and inserts avoidance waypoints."""
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
                    is_inside_obstacle_zone = True
                    entry_index = i

            elif is_inside_obstacle_zone:
                is_inside_obstacle_zone = False
                exit_index = i

                entry_point = sampled_points[entry_index]
                exit_point = sampled_points[exit_index]

                entry_vec = entry_point[:2] - obstacle_xy
                exit_vec = exit_point[:2] - obstacle_xy

                avoid_vec = entry_vec + exit_vec
                avoid_vec /= np.linalg.norm(avoid_vec) + 1e-6

                new_pos_xy = obstacle_xy + avoid_vec * clearance_radius
                new_pos_z = (entry_point[2] + exit_point[2]) / 2
                new_avoid_waypoint = np.concatenate([new_pos_xy, [new_pos_z]])

                avg_time = (sampled_times[entry_index] + sampled_times[exit_index]) / 2
                collision_free_times.append(avg_time)
                collision_free_points.append(new_avoid_waypoint)

            else:
                collision_free_times.append(sampled_times[i])
                collision_free_points.append(point)

        return np.array(collision_free_times), np.array(collision_free_points)

    def __insert_obstacle_avoidance_points(
        self,
        path_points: NDArray[np.floating],
        obstacle_centers: NDArray[np.floating],
        clearance_radius: float,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Iteratively checks for and mitigates collisions."""
        temp_spline = self.__compute_trajectory_spline(self.FLIGHT_DURATION, path_points)

        num_samples = int(self.__control_freq * self.FLIGHT_DURATION)
        sampled_times = np.linspace(0, self.FLIGHT_DURATION, num_samples)
        sampled_points = temp_spline(sampled_times)

        for obstacle_center in obstacle_centers:
            sampled_times, sampled_points = self.__process_single_obstacle(
                obstacle_center, sampled_points, sampled_times, clearance_radius
            )

        return sampled_times, sampled_points

    def __check_for_env_update(self, current_obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Detects if the environment state has changed."""
        current_gate_flags = np.array(current_obs["gates_visited"], dtype=bool)
        current_obstacle_flags = np.array(current_obs["obstacles_visited"], dtype=bool)

        if self.__last_gate_flags is None:
            self.__last_gate_flags = current_gate_flags
            self.__last_obstacle_flags = current_obstacle_flags
            return False

        gate_newly_hit = np.any((~self.__last_gate_flags) & current_gate_flags)
        obstacle_newly_hit = np.any((~self.__last_obstacle_flags) & current_obstacle_flags)

        self.__last_gate_flags = current_gate_flags
        self.__last_obstacle_flags = current_obstacle_flags

        return gate_newly_hit or obstacle_newly_hit

    def __regenerate_flight_plan(
        self, current_obs: dict[str, NDArray[np.floating]], simulation_time: float
    ) -> None:
        """Re-plans the Out-and-Back trajectory based on updated state."""
        
        # --- TESTING MODIFICATION: Only grab the FIRST gate ---
        self.__gate_positions = current_obs["gates_pos"][:1]
        self.__gate_normals, self.__gate_y_axes, self.__gate_z_axes = self.__extract_gate_frames(
            current_obs["gates_quat"][:1]
        )

        # 1. Forward Path
        forward_path = self.__generate_gate_approach_points(
            self.__start_position,
            self.__gate_positions,
            self.__gate_normals,
            approach_distance=self.APPROACH_DISTANCE,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS,
        )

        # 2. Backward Path
        return_path = forward_path[::-1]
        full_path_points = np.vstack([forward_path, return_path[1:]])

        # 3. Obstacles
        time_knots, path_points = self.__insert_obstacle_avoidance_points(
            full_path_points, current_obs["obstacles_pos"], self.OBSTACLE_CLEARANCE
        )

        # 4. Spline
        self.__trajectory_spline = self.__compute_trajectory_spline(
            self.FLIGHT_DURATION, path_points, custom_time_knots=time_knots
        )

    # --- Public API Methods ---

    def compute_control(
        self, current_obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Calculates the control action (Target Position) for the current time step."""
        elapsed_time = min(self.__current_step / self.__control_freq, self.FLIGHT_DURATION)

        reference_position = self.__trajectory_spline(elapsed_time)

        if self.__check_for_env_update(current_obs):
            self.__regenerate_flight_plan(current_obs, elapsed_time)

        if elapsed_time >= self.FLIGHT_DURATION:
            self.__is_complete = True

        try:
            draw_line(
                self.env,
                self.__trajectory_spline(self.__trajectory_spline.x),
                rgba=np.array([1.0, 1.0, 1.0, 0.2]),
            )
        except (AttributeError, TypeError):
            pass

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
        """Called by the simulation loop after every step."""
        self.__current_step += 1
        return self.__is_complete

    def get_spline_function(self) -> CubicSpline:
        return self.__trajectory_spline

    def get_sampled_trajectory(self) -> NDArray[np.floating]:
        num_samples = int(self.__control_freq * self.FLIGHT_DURATION)
        time_samples = np.linspace(0, self.FLIGHT_DURATION, num_samples)
        return self.__trajectory_spline(time_samples)

    def update_current_step(self, step: int) -> None:
        self.__current_step = step