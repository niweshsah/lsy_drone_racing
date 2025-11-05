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
    """Controller to follow gates using a cubic *Hermite* spline trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        """Initialize the controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._start_phase_ticks = 50  # ticks to smoothly reach first gate

        # --- MODIFIED: Extract both gate positions and RPY orientations ---
        gate_positions = []
        gate_rpys = []
        for gate_info in config.env.track.gates:
            gate_positions.append(gate_info["pos"])
            gate_rpys.append(gate_info["rpy"])

        self._waypoints = np.array(gate_positions)
        if len(self._waypoints) < 2:
            raise ValueError("Need at least 2 gates to define a trajectory!")
        # -------------

        # We need the first gate's position for Phase 1
        self._first_gate_pos = self._waypoints[0]
        # We'll store the drone's initial position on the first compute_control call
        self._initial_pos: NDArray[np.floating] | None = None

        # --- MODIFIED: Generate CubicHermiteSpline using logic from draw_3d_lines ---

        gates = self._waypoints
        t = np.arange(len(gates))  # Spline parameter 't'

        # 1. Calculate the normal vector (desired direction) for each gate
        normals = []
        for rpy in gate_rpys:
            r = Rotation.from_euler("xyz", rpy)
            # The normal vector is the gate's local X-axis rotated to world frame
            normals.append(r.apply([1, 0, 0]))
        normals = np.array(normals)

        # 2. Calculate derivatives (dy/dt) for the spline
        derivatives = np.zeros_like(gates)

        # This factor controls the "speed" at the gate, influencing turn radius
        smoothness_factor = 2.5

        # First point: use forward difference
        v_avg = (gates[1] - gates[0]) / (t[1] - t[0])
        mag = np.abs(np.dot(v_avg, normals[0])) * smoothness_factor
        derivatives[0] = normals[0] * mag

        # Last point: use backward difference
        v_avg = (gates[-1] - gates[-2]) / (t[-1] - t[-2])
        mag = np.abs(np.dot(v_avg, normals[-1])) * smoothness_factor
        derivatives[-1] = normals[-1] * mag

        # Intermediate points: use central difference
        for i in range(1, len(gates) - 1):
            v_avg = (gates[i + 1] - gates[i - 1]) / (t[i + 1] - t[i - 1])
            mag = np.abs(np.dot(v_avg, normals[i])) * smoothness_factor
            derivatives[i] = normals[i] * mag

        # 3. Create the Hermite spline
        # This spline passes through each gates[i] with the exact derivative[i]
        self._des_pos_spline = CubicHermiteSpline(t, gates, derivatives, axis=0)

        # 4. Define spline parameter range and total flight time
        self._spline_t_max = len(gates) - 1  # Spline is parameterized from 0 to N-1
        self._t_total = 15  # Total seconds to fly the spline (can adjust)
        # --- END MODIFIED SECTION ---

        # Log of drone positions at each step
        self.position_log: list[NDArray[np.floating]] = []

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute next desired drone state and log position."""
        # On the very first tick of an episode, record the drone's starting position
        if self._initial_pos is None:
            self._initial_pos = obs["pos"][:3].copy()

        if self._tick <= self._start_phase_ticks:
            # --- Phase 1: Moving to the first gate (Unchanged) ---
            alpha = min(self._tick / self._start_phase_ticks, 1.0)
            # Linearly interpolate from initial pos to the first gate
            des_pos = (1 - alpha) * self._initial_pos + alpha * self._first_gate_pos
        else:
            # --- MODIFIED: Phase 2: Following the Hermite spline ---

            # Calculate elapsed time *since the spline phase started*
            spline_elapsed_time = (self._tick - self._start_phase_ticks) / self._freq

            # Clamp the elapsed time to the total duration
            spline_elapsed_time = min(spline_elapsed_time, self._t_total)

            if spline_elapsed_time >= self._t_total:
                self._finished = True

            # Map the elapsed time [0, _t_total] to the spline parameter [0, _spline_t_max]
            spline_param_t = (spline_elapsed_time / self._t_total) * self._spline_t_max

            # Get the desired position from the Hermite spline
            des_pos = self._des_pos_spline(spline_param_t)
        # -------------

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
