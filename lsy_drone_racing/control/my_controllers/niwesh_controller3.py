from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

# class SafeGateFollower(Controller):
#     """Simple Level 0 safe controller: go to gates one by one using P controller."""

#     def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
#         super().__init__(obs, info, config)
#         self._tick = 0
#         self._finished = False

#         # Take gates from observation
#         self._gates = obs['gates_pos']
#         self._target_idx = 0

#         self._kp = 1.0      # proportional gain for velocity
#         self._max_vel = 0.3 # limit velocity to avoid crashes

#     def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
#         pos = obs['pos']
#         target_pos = self._gates[self._target_idx]

#         # Compute velocity toward target gate
#         vel_cmd = self._kp * (target_pos - pos)
#         vel_cmd = np.clip(vel_cmd, -self._max_vel, self._max_vel)

#         # Acceleration and angular rates = 0 for simplicity
#         acc_cmd = np.zeros(3, dtype=np.float32)
#         yaw = rrate = prate = yrate = 0.0

#         # Switch to next gate if close enough
#         if np.linalg.norm(target_pos - pos) < 0.1:
#             self._target_idx += 1
#             if self._target_idx >= len(self._gates):
#                 self._finished = True
#                 self._target_idx = len(self._gates) - 1

#         # Full state
#         action = np.concatenate([pos, vel_cmd, acc_cmd, [yaw, rrate, prate, yrate]], dtype=np.float32)
#         return action

#     def step_callback(self, action: NDArray[np.floating], obs: dict, reward: float,
#                       terminated: bool, truncated: bool, info: dict) -> bool:
#         self._tick += 1
#         return self._finished

#     def episode_callback(self):
#         self._tick = 0
#         self._target_idx = 0
#         self._finished = False


class SafeGateFollower(Controller):
    """Safe gate follower with height offset to avoid top collisions."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._tick = 0
        self._finished = False
        self._gates = obs["gates_pos"]
        self._target_idx = 0
        self._kp = 1.0
        self._max_vel = 0.3
        self._safe_height_offset = -0.1  # 10 cm below gate center

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        pos = obs["pos"]
        target_pos = self._gates[self._target_idx].copy()
        target_pos[2] += self._safe_height_offset  # apply height safety offset

        vel_cmd = self._kp * (target_pos - pos)
        vel_cmd = np.clip(vel_cmd, -self._max_vel, self._max_vel)

        acc_cmd = np.zeros(3, dtype=np.float32)
        yaw = rrate = prate = yrate = 0.0

        if np.linalg.norm(target_pos - pos) < 0.1:
            self._target_idx += 1
            if self._target_idx >= len(self._gates):
                self._finished = True
                self._target_idx = len(self._gates) - 1

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
