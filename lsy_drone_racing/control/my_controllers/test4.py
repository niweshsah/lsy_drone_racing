"""Controller to follow gates using a cubic Hermite spline trajectory
    that enforces a straight pass-through and avoids obstacles.
    """


from __future__ import annotations

import os
os.environ["SCIPY_ARRAY_API"] = "1"  # Ensure SciPy API is set first

from typing import TYPE_CHECKING
import numpy as np
# --- MODIFIED: Import CubicHermiteSpline and Rotation ---
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
# -------------
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ml_collections import ConfigDict


class GateTrajectoryController(Controller):
    
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        """Initialize the controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._start_phase_ticks = 50  # ticks to reach first gate smoothly

        # Extract gate positions and orientations
        gate_positions = [g['pos'] for g in config.env.track.gates]
        gate_rpys = [g['rpy'] for g in config.env.track.gates]
        gates = np.array(gate_positions)
        if len(gates) < 2:
            raise ValueError("Need at least 2 gates to define a trajectory!")

        # Obstacles positions
        obstacles = np.array([obs['pos'] for obs in getattr(config.env.track, "obstacles", [])])

        self._initial_pos: NDArray[np.floating] | None = None

        # --- Obstacle-aware Hermite spline generation ---
        normals = [Rotation.from_euler('xyz', rpy).apply([1, 0, 0]) for rpy in gate_rpys]
        normals = np.array(normals)

        straight_half_length = 0.5
        smoothness_factor = 2.5
        safe_height = 0.5  # min clearance above obstacles

        spline_points = []
        spline_derivatives = []

        # Calculate magnitudes (derivative lengths)
        mags = np.zeros(len(gates))
        mags[0] = np.abs(np.dot(gates[1] - gates[0], normals[0])) * smoothness_factor
        mags[-1] = np.abs(np.dot(gates[-1] - gates[-2], normals[-1])) * smoothness_factor
        for i in range(1, len(gates)-1):
            mags[i] = np.abs(np.dot((gates[i+1] - gates[i-1])/2.0, normals[i])) * smoothness_factor

        # Build points & derivatives with obstacle avoidance
        for i in range(len(gates)):
            pos = gates[i]
            normal = normals[i]
            mag = mags[i]

            before = pos - normal * straight_half_length
            after = pos + normal * straight_half_length

            # Lift path if near obstacle poles
            for obs in obstacles:
                obs_x, obs_y, obs_z = obs
                radius = 0.3  # horizontal safety radius
                for point in [before, after]:
                    if np.linalg.norm(point[:2] - obs[:2]) < radius:
                        point[2] = max(point[2], obs_z + safe_height)

            spline_points.extend([before, after])
            derivative = normal * mag
            spline_derivatives.extend([derivative, derivative])

        spline_points = np.array(spline_points)
        spline_derivatives = np.array(spline_derivatives)

        # Parameterize along chord length
        distances = np.linalg.norm(np.diff(spline_points, axis=0), axis=1)
        t_spline = np.zeros(len(spline_points))
        t_spline[1:] = np.cumsum(distances)

        # Hermite spline
        self._des_pos_spline = CubicHermiteSpline(t_spline, spline_points, spline_derivatives, axis=0)
        self._spline_t_min = t_spline[0]
        self._spline_t_max = t_spline[-1]
        self._t_total = 15  # seconds to fly entire path
        self._first_spline_point = spline_points[0]

        # Log positions
        self.position_log: list[NDArray[np.floating]] = []

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute next desired drone state along obstacle-aware path."""

        if self._initial_pos is None:
            self._initial_pos = obs["pos"][:3].copy()

        if self._tick <= self._start_phase_ticks:
            alpha = min(self._tick / self._start_phase_ticks, 1.0)
            des_pos = (1 - alpha) * self._initial_pos + alpha * self._first_spline_point
        else:
            spline_elapsed_time = (self._tick - self._start_phase_ticks) / self._freq
            spline_elapsed_time = min(spline_elapsed_time, self._t_total)
            if spline_elapsed_time >= self._t_total:
                self._finished = True

            spline_param_t = (spline_elapsed_time / self._t_total) * (self._spline_t_max - self._spline_t_min) + self._spline_t_min
            des_pos = self._des_pos_spline(spline_param_t)

        self.position_log.append(des_pos.copy())
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
        self._tick += 1
        return action

    def step_callback(
        self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]],
        reward: float, terminated: bool, truncated: bool, info: dict
    ) -> bool:
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False
        self.position_log.clear()
        self._initial_pos = None