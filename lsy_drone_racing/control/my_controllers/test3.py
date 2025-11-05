"""Controller that follows a trajectory passing through all gates defined in the config."""

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
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


class GateTrajectoryController(Controller):
    """Controller to follow gates using a cubic Hermite spline trajectory
    that enforces a straight pass-through.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        """Initialize the controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._start_phase_ticks = 50  # ticks to smoothly reach first gate

        # Extract both gate positions and RPY orientations
        gate_positions = []
        gate_rpys = []
        for gate_info in config.env.track.gates:
            gate_positions.append(gate_info["pos"])
            gate_rpys.append(gate_info["rpy"])

        gates = np.array(gate_positions)
        if len(gates) < 2:
            raise ValueError("Need at least 2 gates to define a trajectory!")

        # We'll store the drone's initial position on the first compute_control call
        self._initial_pos: NDArray[np.floating] | None = None

        # --- MODIFIED: Generate CubicHermiteSpline with straight pass-through ---

        # 1. Calculate gate normals
        normals = []
        for rpy in gate_rpys:
            r = Rotation.from_euler("xyz", rpy)
            normals.append(r.apply([1, 0, 0]))
        normals = np.array(normals)

        # 2. Define "before" and "after" points for each gate
        straight_half_length = 0.5  # Creates a 1.0m straight segment
        smoothness_factor = 2.5

        spline_points = []
        spline_derivatives = []

        # Calculate magnitudes (speed) based on original gate spacing
        mags = np.zeros(len(gates))

        # First point: forward diff
        v_avg = gates[1] - gates[0]
        mags[0] = np.abs(np.dot(v_avg, normals[0])) * smoothness_factor

        # Last point: backward diff
        v_avg = gates[-1] - gates[-2]
        mags[-1] = np.abs(np.dot(v_avg, normals[-1])) * smoothness_factor

        # Intermediate points: central diff
        for i in range(1, len(gates) - 1):
            v_avg = (gates[i + 1] - gates[i - 1]) / 2.0
            mags[i] = np.abs(np.dot(v_avg, normals[i])) * smoothness_factor

        # Build the new points and derivatives (2 for each gate)
        for i in range(len(gates)):
            pos = gates[i]
            normal = normals[i]
            mag = mags[i]

            spline_points.append(pos - normal * straight_half_length)
            spline_points.append(pos + normal * straight_half_length)

            derivative = normal * mag
            spline_derivatives.append(derivative)
            spline_derivatives.append(derivative)

        spline_points = np.array(spline_points)
        spline_derivatives = np.array(spline_derivatives)

        # 3. Create the 't' parameter based on cumulative chordal distance
        distances = np.linalg.norm(np.diff(spline_points, axis=0), axis=1)
        t_spline = np.zeros(len(spline_points))
        t_spline[1:] = np.cumsum(distances)

        # 4. Create the Hermite spline
        self._des_pos_spline = CubicHermiteSpline(
            t_spline, spline_points, spline_derivatives, axis=0
        )

        # 5. Define spline parameter range and total flight time
        self._spline_t_min = t_spline[0]  # Will be 0.0
        self._spline_t_max = t_spline[-1]  # Total path length
        self._t_total = 15  # Total seconds to fly the *entire* spline (can adjust)

        # We need the first *spline* point (point_before_0) for Phase 1
        self._first_spline_point = spline_points[0]
        # --- END MODIFIED SECTION ---

        # Log of drone positions at each step
        self.position_log: list[NDArray[np.floating]] = []

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute next desired drone state and log position."""
        # On the very first tick, record the drone's starting position
        if self._initial_pos is None:
            self._initial_pos = obs["pos"][:3].copy()

        if self._tick <= self._start_phase_ticks:
            # --- Phase 1: Moving to the *start of the spline* (point_before_0) ---
            alpha = min(self._tick / self._start_phase_ticks, 1.0)
            # Linearly interpolate from initial pos to the first spline point
            des_pos = (1 - alpha) * self._initial_pos + alpha * self._first_spline_point
        else:
            # --- Phase 2: Following the Hermite spline ---

            # Calculate elapsed time *since the spline phase started*
            spline_elapsed_time = (self._tick - self._start_phase_ticks) / self._freq

            # Clamp the elapsed time to the total duration
            spline_elapsed_time = min(spline_elapsed_time, self._t_total)

            if spline_elapsed_time >= self._t_total:
                self._finished = True

            # Map the elapsed time [0, _t_total] to the spline parameter [t_min, t_max]
            spline_param_t = (spline_elapsed_time / self._t_total) * (
                self._spline_t_max - self._spline_t_min
            ) + self._spline_t_min

            # Get the desired position from the Hermite spline
            des_pos = self._des_pos_spline(spline_param_t)

        # Log the desired position
        self.position_log.append(des_pos.copy())

        # Full action: position + zeros for remaining states
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)

        self._tick += 1
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Return True if trajectory finished."""
        return self._finished

    def episode_callback(self):
        """Reset internal state at episode end."""
        self._tick = 0
        self._finished = False
        self.position_log.clear()  # reset logged positions

        # Reset initial_pos so it's re-captured on the next episode's first tick
        self._initial_pos = None
