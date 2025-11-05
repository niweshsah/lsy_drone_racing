from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # Must be first

import time  # <-- Import system time
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


class MoveToStartController(Controller):
    """Controller that brings drone to the first gate smoothly and hovers
    for a fixed duration using system time.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        super().__init__(obs, info, config)

        self._tick = 0
        self._finished = False

        # --- Phase 1: Move to Start ---
        self._start_phase_ticks = 50  # number of ticks to reach start smoothly
        self._start_pos = np.array(config.env.track.gates[0]["pos"])
        self._initial_pos = obs["pos"][:3].copy()

        # --- Phase 2: Hover ---
        self._hover_duration_sec = 5.0
        # Timer to track hover start, initialized to None
        self._hover_start_time: float | None = None

        self.position_log: list[NDArray[np.floating]] = []

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        if self._tick <= self._start_phase_ticks:
            # --- Phase 1: Moving to the start position ---
            alpha = min(self._tick / self._start_phase_ticks, 1.0)
            des_pos = (1 - alpha) * self._initial_pos + alpha * self._start_pos
            self._tick += 1
        else:
            # --- Phase 2: Hovering at the start position ---
            des_pos = self._start_pos

            # If this is the first tick of the hover phase, record the start time
            if self._hover_start_time is None:
                self._hover_start_time = time.monotonic()

            # Check if 10 seconds have elapsed
            elapsed_time = time.monotonic() - self._hover_start_time
            if elapsed_time >= self._hover_duration_sec:
                self._finished = True

        # Log position
        self.position_log.append(des_pos.copy())

        # Action: desired position + zeros for other states
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)

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
        # Signal that this controller is finished
        return self._finished

    def episode_callback(self):
        # Reset for the next episode
        self._tick = 0
        self._finished = False
        self._hover_start_time = None  # <-- Reset the timer
        self.position_log.clear()
