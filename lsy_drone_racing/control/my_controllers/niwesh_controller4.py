from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SafeGateFollower(Controller):
    """Drone follows gates with safe height offset to avoid top rim collisions."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._tick = 0
        self._finished = False
        self._gates = obs["gates_pos"]
        self._target_idx = 0

        self._kp = 1.0  # proportional gain
        self._max_vel = 0.3  # max velocity
        self._safe_offset = -0.15  # lower 15 cm below gate center
        self._switch_dist = 0.2  # distance to switch to next gate

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        pos = obs["pos"]

        # Choose target gate
        if self._finished:
            target = self._gates[-1].copy()
        else:
            target = self._gates[self._target_idx].copy()

        # Apply safe offset in Z to avoid top rim
        target[2] += self._safe_offset

        # Compute proportional velocity toward target
        vel_cmd = self._kp * (target - pos)
        vel_norm = np.linalg.norm(vel_cmd)
        if vel_norm > self._max_vel:
            vel_cmd = vel_cmd / vel_norm * self._max_vel

        # Acceleration & yaw placeholders
        acc_cmd = np.zeros(3, dtype=np.float32)
        yaw = rrate = prate = yrate = 0.0

        # Switch gate if close enough
        if not self._finished and np.linalg.norm(target - pos) < self._switch_dist:
            self._target_idx += 1
            if self._target_idx >= len(self._gates):
                self._finished = True

        # Construct action
        action = np.concatenate(
            [pos, vel_cmd, acc_cmd, [yaw, rrate, prate, yrate]], dtype=np.float32
        )
        return action

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
        self._target_idx = 0
        self._finished = False
