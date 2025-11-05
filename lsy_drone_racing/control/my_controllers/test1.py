"""Controller that follows a trajectory passing through all gates defined in the config."""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # Ensure SciPy API is set first

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


class GateTrajectoryController(Controller):
    """Controller to follow gates using a cubic spline trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        """Initialize the controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._start_phase_ticks = 50  # ticks to smoothly reach first gate

        # Extract gate positions from config
        self._waypoints = np.array([gate["pos"] for gate in config.env.track.gates])
        if len(self._waypoints) < 2:
            raise ValueError("Need at least 2 gates to define a trajectory!")

        # --- MODIFIED ---
        # We need the first gate's position for Phase 1
        self._first_gate_pos = self._waypoints[0]
        # We'll store the drone's initial position on the first compute_control call
        self._initial_pos: NDArray[np.floating] | None = None
        # -------------

        # Cubic spline over the gates
        self._t_total = 15  # seconds, can adjust
        t = np.linspace(0, self._t_total, len(self._waypoints))
        self._des_pos_spline = CubicSpline(t, self._waypoints)

        # Log of drone positions at each step
        self.position_log: list[NDArray[np.floating]] = []

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute next desired drone state and log position."""
        # --- MODIFIED ---
        # On the very first tick of an episode, record the drone's starting position
        if self._initial_pos is None:
            self._initial_pos = obs["pos"][:3].copy()

        if self._tick <= self._start_phase_ticks:
            # --- Phase 1: Moving to the first gate ---
            alpha = min(self._tick / self._start_phase_ticks, 1.0)
            # Linearly interpolate from initial pos to the first gate
            des_pos = (1 - alpha) * self._initial_pos + alpha * self._first_gate_pos
        else:
            # --- Phase 2: Following the spline ---
            # Calculate time *since the spline phase started*
            spline_tick = self._tick - self._start_phase_ticks
            t = min(spline_tick / self._freq, self._t_total)

            if t >= self._t_total:
                self._finished = True
                t = self._t_total  # Clamp time to avoid spline error

            des_pos = self._des_pos_spline(t)
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

        # --- MODIFIED ---
        # Reset initial_pos so it's re-captured on the next episode's first tick
        self._initial_pos = None
        # -------------
