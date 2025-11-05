from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Level0Controller(Controller):
    """Simple Level 0 controller that generates a trajectory through the gates in obs."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False

        # Get gate positions from observations
        gate_positions = obs["gates_pos"]
        self._n_gates = gate_positions.shape[0]

        # Total trajectory time in seconds
        self._t_total = 15
        t = np.linspace(0, self._t_total, self._n_gates)

        # Create cubic splines for x, y, z
        self._spline_x = CubicSpline(t, gate_positions[:, 0])
        self._spline_y = CubicSpline(t, gate_positions[:, 1])
        self._spline_z = CubicSpline(t, gate_positions[:, 2])

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True

        # Desired position, velocity, acceleration
        pos = np.array([self._spline_x(t), self._spline_y(t), self._spline_z(t)], dtype=np.float32)
        vel = np.array(
            [
                self._spline_x.derivative()(t),
                self._spline_y.derivative()(t),
                self._spline_z.derivative()(t),
            ],
            dtype=np.float32,
        )
        acc = np.array(
            [
                self._spline_x.derivative(2)(t),
                self._spline_y.derivative(2)(t),
                self._spline_z.derivative(2)(t),
            ],
            dtype=np.float32,
        )

        # Simple yaw and angular rates
        yaw = rrate = prate = yrate = 0.0

        return np.concatenate([pos, vel, acc, [yaw, rrate, prate, yrate]], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0
