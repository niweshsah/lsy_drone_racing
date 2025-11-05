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
    """SmartPathController
    -------------------
    An adaptive drone controller that generates a continuous, smooth trajectory
    through gates while actively avoiding obstacles and dynamically updating
    its plan when the environment changes.
    """

    # --- Configuration constants ---
    MAX_FLIGHT_TIME = 25
    STATE_DIMENSION = 13
    AVOID_RADIUS = 0.3
    PATH_SAMPLES = 100
    LOG_FREQUENCY = 100

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        # Core simulation setup
        self._step = 0
        self._freq = config.env.freq
        self._complete = False

        # Environment history
        self._last_gate_status = None
        self._last_obstacle_status = None

        # Extract positions and orientations
        self._gates = obs["gates_pos"]
        self._obstacles = obs["obstacles_pos"]
        self._drone_start = obs["pos"]
        self._gate_normals = self._extract_gate_normals(obs["gates_quat"])

        # Initial trajectory generation
        raw_path = self._generate_gate_path(
            start=self._drone_start,
            gates=self._gates,
            normals=self._gate_normals,
            offset=0.5,
            segments=5,
        )

        # Collision-aware refinement
        _, safe_path = self._apply_obstacle_clearance(
            path=raw_path, obstacles=self._obstacles, margin=self.AVOID_RADIUS
        )

        # Smooth cubic spline trajectory
        self._traj = self._fit_cubic_path(duration=self.MAX_FLIGHT_TIME, points=safe_path)

    # =========================================================================
    # === Trajectory Construction Utilities ==================================
    # =========================================================================

    def _extract_gate_normals(self, quaternions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert quaternion orientations into gate-facing unit normals."""
        rot = Rotation.from_quat(quaternions)
        return rot.as_matrix()[:, :, 0]  # Forward x-axis vector

    def _generate_gate_path(
        self,
        start: NDArray[np.floating],
        gates: NDArray[np.floating],
        normals: NDArray[np.floating],
        offset: float = 0.5,
        segments: int = 5,
    ) -> NDArray[np.floating]:
        """Create smooth intermediate waypoints through the gates
        with slight forward/backward offsets to guide the spline.
        """
        steps = np.linspace(-offset, offset, segments)
        interpolated = [gates + s * normals for s in steps]

        path = np.concatenate(interpolated, axis=1)
        path = path.reshape(gates.shape[0], segments, 3).reshape(-1, 3)
        return np.vstack([start, path])

    def _fit_cubic_path(self, duration: float, points: NDArray[np.floating]) -> CubicSpline:
        """Fit a cubic spline along waypoints using cumulative distance as time parameter."""
        diffs = np.diff(points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        arc_length = np.concatenate([[0], np.cumsum(distances)])
        t_values = arc_length / arc_length[-1] * duration
        return CubicSpline(t_values, points)

    def _apply_obstacle_clearance(
        self, path: NDArray[np.floating], obstacles: NDArray[np.floating], margin: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Re-route the path slightly to maintain safe distance from each obstacle."""
        spline = self._fit_cubic_path(self.MAX_FLIGHT_TIME, path)
        times = np.linspace(0, self.MAX_FLIGHT_TIME, int(self._freq * self.MAX_FLIGHT_TIME))
        trajectory_points = spline(times)

        for obstacle in obstacles:
            safe_t, safe_pts = [], []
            inside_zone = False
            entry_idx = None

            for i, pt in enumerate(trajectory_points):
                dist = np.linalg.norm(pt[:2] - obstacle[:2])

                if dist < margin:
                    if not inside_zone:
                        inside_zone = True
                        entry_idx = i
                elif inside_zone:
                    inside_zone = False
                    exit_idx = i

                    entry, exit_ = trajectory_points[entry_idx], trajectory_points[exit_idx]
                    dir_in = entry[:2] - obstacle[:2]
                    dir_out = exit_[:2] - obstacle[:2]
                    deviation = dir_in + dir_out
                    deviation /= np.linalg.norm(deviation)

                    adjusted_xy = obstacle[:2] + deviation * margin
                    adjusted_z = (entry[2] + exit_[2]) / 2
                    new_point = np.array([*adjusted_xy, adjusted_z])

                    safe_t.append((times[entry_idx] + times[exit_idx]) / 2)
                    safe_pts.append(new_point)
                else:
                    safe_t.append(times[i])
                    safe_pts.append(pt)

            times = np.array(safe_t)
            trajectory_points = np.array(safe_pts)

        return times, trajectory_points

    # =========================================================================
    # === Dynamic Replanning ==================================================
    # =========================================================================

    def _has_environment_changed(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Return True if any gate or obstacle visitation flags changed."""
        if self._last_gate_status is None:
            self._last_gate_status = np.array(obs["gates_visited"], dtype=bool)
            self._last_obstacle_status = np.array(obs["obstacles_visited"], dtype=bool)
            return False

        current_gates = np.array(obs["gates_visited"], dtype=bool)
        current_obstacles = np.array(obs["obstacles_visited"], dtype=bool)

        gate_change = np.any((~self._last_gate_status) & current_gates)
        obstacle_change = np.any((~self._last_obstacle_status) & current_obstacles)

        self._last_gate_status = current_gates
        self._last_obstacle_status = current_obstacles
        return gate_change or obstacle_change

    def _replan_path(self, obs: dict[str, NDArray[np.floating]], t_now: float) -> None:
        """Recompute a new path spline if environment conditions changed."""
        self._gates = obs["gates_pos"]
        self._obstacles = obs["obstacles_pos"]
        self._gate_normals = self._extract_gate_normals(obs["gates_quat"])

        new_path = self._generate_gate_path(
            start=self._drone_start, gates=self._gates, normals=self._gate_normals
        )

        _, updated_points = self._apply_obstacle_clearance(
            path=new_path, obstacles=self._obstacles, margin=self.AVOID_RADIUS
        )

        self._traj = self._fit_cubic_path(self.MAX_FLIGHT_TIME, updated_points)

    # =========================================================================
    # === Core Controller =====================================================
    # =========================================================================

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute desired target position for current timestep based on spline."""
        t_now = min(self._step / self._freq, self.MAX_FLIGHT_TIME)
        target_position = self._traj(t_now)

        if self._step % self.LOG_FREQUENCY == 0:
            pass  # optional debug logging here

        if self._has_environment_changed(obs):
            self._replan_path(obs, t_now)

        if t_now >= self.MAX_FLIGHT_TIME:
            self._complete = True

        try:
            draw_line(self.env, self._traj(self._traj.x), rgba=np.array([1.0, 1.0, 1.0, 0.25]))
        except Exception:
            pass

        # Output control vector (position + zeros for unused dimensions)
        return np.concatenate([target_position, np.zeros(10)], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Called each step; updates timestep and returns True if mission done."""
        self._step += 1
        return self._complete

    # =========================================================================
    # === Accessors ===========================================================
    # =========================================================================

    def trajectory(self) -> CubicSpline:
        """Return the current spline function."""
        return self._traj

    def sampled_points(self) -> NDArray[np.floating]:
        """Return uniformly sampled points from the current trajectory."""
        t_vals = np.linspace(0, self.MAX_FLIGHT_TIME, int(self._freq * self.MAX_FLIGHT_TIME))
        return self._traj(t_vals)

    def set_step(self, step: int) -> None:
        """Force the controller to a specific simulation step."""
        self._step = step
