from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SmartPathController(Controller):
    """SmartPathController:
    A drone controller that generates smooth, collision-free trajectories
    through a sequence of gates while dynamically adapting to environmental changes.
    """

    # --- Controller constants ---
    TRAJECTORY_DURATION = 25.0  # seconds
    STATE_DIM = 13  # total drone state dimension
    OBSTACLE_BUFFER = 0.3  # safety margin around obstacles (in meters)
    TRAJECTORY_SAMPLES = 100  # number of samples for visualization
    LOG_INTERVAL = 100  # step interval for logging/debug

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller with environment observations, configuration, and state."""
        super().__init__(obs, info, config)

        # Simulation parameters
        self._time_step = 0
        self._control_freq = config.env.freq
        self._finished = False

        # Track last environment states to detect changes
        self._prev_gate_flags = None
        self._prev_obstacle_flags = None

        # Extract environment data
        self.gate_positions = obs["gates_pos"]
        self.gate_normals = self._compute_gate_normals(obs["gates_quat"])
        self.obstacle_positions = obs["obstacles_pos"]
        self.start_position = obs["pos"]

        # Generate base waypoints from gates
        waypoints = self._create_gate_waypoints(
            start_pos=self.start_position,
            gate_positions=self.gate_positions,
            gate_normals=self.gate_normals,
            approach_offset=0.5,
            interp_points=5,
        )

        # Apply obstacle avoidance correction
        _, collision_free_points = self._adjust_for_obstacles(
            waypoints=waypoints,
            obstacles=self.obstacle_positions,
            safety_radius=self.OBSTACLE_BUFFER,
        )

        # Create a smooth trajectory spline
        self.trajectory = self._build_cubic_trajectory(
            total_time=self.TRAJECTORY_DURATION, waypoints=collision_free_points
        )

    # -------------------------------------------------------------------------
    # --- Trajectory Generation Utilities ---
    # -------------------------------------------------------------------------

    def _compute_gate_normals(self, gate_quaternions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert gate orientations (quaternions) to forward-facing normal vectors."""
        rotations = Rotation.from_quat(gate_quaternions)
        return rotations.as_matrix()[:, :, 0]  # Extract the X-axis direction

    def _create_gate_waypoints(
        self,
        start_pos: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        approach_offset: float = 0.5,
        interp_points: int = 5,
    ) -> NDArray[np.floating]:
        """Generate a series of intermediate waypoints through each gate to ensure smooth transitions."""
        num_gates = gate_positions.shape[0]
        interpolated_points = []

        for i in range(interp_points):
            offset = -approach_offset + (i / (interp_points - 1)) * 2 * approach_offset
            interpolated_points.append(gate_positions + offset * gate_normals)

        # Flatten and concatenate
        waypoints = np.concatenate(interpolated_points, axis=1)
        waypoints = waypoints.reshape(num_gates, interp_points, 3).reshape(-1, 3)
        waypoints = np.vstack([start_pos, waypoints])

        return waypoints

    def _build_cubic_trajectory(
        self, total_time: float, waypoints: NDArray[np.floating]
    ) -> CubicSpline:
        """Fit a cubic spline trajectory through the provided waypoints,
        parametrized by arc length mapped to total flight time.
        """
        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        time_params = arc_lengths / arc_lengths[-1] * total_time

        return CubicSpline(time_params, waypoints)

    def _adjust_for_obstacles(
        self, waypoints: NDArray[np.floating], obstacles: NDArray[np.floating], safety_radius: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Adjust trajectory waypoints to maintain a safe distance from obstacles
        by introducing detours if necessary.
        """
        trajectory = self._build_cubic_trajectory(self.TRAJECTORY_DURATION, waypoints)
        time_samples = np.linspace(
            0, self.TRAJECTORY_DURATION, int(self._control_freq * self.TRAJECTORY_DURATION)
        )
        points = trajectory(time_samples)

        # For each obstacle, check if any path points violate safety distance
        for obs_pos in obstacles:
            safe_times, safe_points = [], []
            in_collision = False
            entry_idx = None

            for i, pt in enumerate(points):
                dist_xy = np.linalg.norm(obs_pos[:2] - pt[:2])

                # Entering obstacle zone
                if dist_xy < safety_radius:
                    if not in_collision:
                        in_collision = True
                        entry_idx = i
                # Exiting obstacle zone
                elif in_collision:
                    in_collision = False
                    exit_idx = i

                    entry_pt, exit_pt = points[entry_idx], points[exit_idx]
                    entry_dir = entry_pt[:2] - obs_pos[:2]
                    exit_dir = exit_pt[:2] - obs_pos[:2]
                    avoidance_vec = entry_dir + exit_dir
                    avoidance_vec /= np.linalg.norm(avoidance_vec)

                    new_xy = obs_pos[:2] + avoidance_vec * safety_radius
                    new_z = (entry_pt[2] + exit_pt[2]) / 2
                    new_wp = np.concatenate([new_xy, [new_z]])

                    safe_times.append((time_samples[entry_idx] + time_samples[exit_idx]) / 2)
                    safe_points.append(new_wp)
                else:
                    safe_times.append(time_samples[i])
                    safe_points.append(pt)

            time_samples = np.array(safe_times)
            points = np.array(safe_points)

        return time_samples, points

    # -------------------------------------------------------------------------
    # --- Environment Awareness & Replanning ---
    # -------------------------------------------------------------------------

    def _environment_changed(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Detect changes in gate or obstacle visitation status."""
        if self._prev_gate_flags is None:
            self._prev_gate_flags = np.array(obs["gates_visited"], dtype=bool)
            self._prev_obstacle_flags = np.array(obs["obstacles_visited"], dtype=bool)
            return False

        gate_flags = np.array(obs["gates_visited"], dtype=bool)
        obstacle_flags = np.array(obs["obstacles_visited"], dtype=bool)

        new_gate = np.any((~self._prev_gate_flags) & gate_flags)
        new_obstacle = np.any((~self._prev_obstacle_flags) & obstacle_flags)

        self._prev_gate_flags = gate_flags
        self._prev_obstacle_flags = obstacle_flags

        return new_gate or new_obstacle

    def _update_trajectory(self, obs: dict[str, NDArray[np.floating]], current_time: float) -> None:
        """Replan trajectory dynamically if the environment has changed."""
        self.gate_normals = self._compute_gate_normals(obs["gates_quat"])
        self.gate_positions = obs["gates_pos"]

        new_waypoints = self._create_gate_waypoints(
            start_pos=self.start_position,
            gate_positions=self.gate_positions,
            gate_normals=self.gate_normals,
            approach_offset=0.5,
            interp_points=5,
        )

        _, safe_points = self._adjust_for_obstacles(
            waypoints=new_waypoints,
            obstacles=obs["obstacles_pos"],
            safety_radius=self.OBSTACLE_BUFFER,
        )

        self.trajectory = self._build_cubic_trajectory(self.TRAJECTORY_DURATION, safe_points)

    # -------------------------------------------------------------------------
    # --- Core Control Logic ---
    # -------------------------------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the target state for the drone at the current simulation time step."""
        current_time = min(self._time_step / self._control_freq, self.TRAJECTORY_DURATION)
        target_pos = self.trajectory(current_time)

        # Log periodically
        if self._time_step % self.LOG_INTERVAL == 0:
            pass  # Add custom logging here if needed

        # Detect environment changes and replan if required
        if self._environment_changed(obs):
            self._update_trajectory(obs, current_time)

        # Check if trajectory completed
        if current_time >= self.TRAJECTORY_DURATION:
            self._finished = True

        # Visualization
        try:
            draw_line(
                self.env, self.trajectory(self.trajectory.x), rgba=np.array([1.0, 1.0, 1.0, 0.2])
            )
        except (AttributeError, TypeError):
            pass

        # Return control action: position + zeros for unused state dimensions
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Called after each environment step; updates time and checks completion."""
        self._time_step += 1
        return self._finished

    # -------------------------------------------------------------------------
    # --- Helper Accessors ---
    # -------------------------------------------------------------------------

    def get_trajectory(self) -> CubicSpline:
        """Return the current cubic trajectory function."""
        return self.trajectory

    def get_trajectory_points(self) -> NDArray[np.floating]:
        """Return sampled trajectory waypoints."""
        times = np.linspace(
            0, self.TRAJECTORY_DURATION, int(self._control_freq * self.TRAJECTORY_DURATION)
        )
        return self.trajectory(times)

    def set_time_step(self, t: int) -> None:
        """Manually set the simulation time step."""
        self._time_step = t
