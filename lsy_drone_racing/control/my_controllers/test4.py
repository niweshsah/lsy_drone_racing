from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # Must be first

# --- MODIFIED: Removed time, added Rotation ---
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation  # <-- Import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


class PathFollowingController(Controller):  # <-- Renamed class
    """Controller that brings drone to the first gate, then follows a pre-computed
    straight-line path through all gate 'before' and 'after' points.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        super().__init__(obs, info, config)

        # --- Common ---
        self._tick = 0
        self._finished = False
        self._initial_pos = obs["pos"][:3].copy()
        self.position_log: list[NDArray[np.floating]] = []

        # --- Path Generation ---
        # Get gate and obstacle info from config
        gate_centers = [g["pos"] for g in config.env.track.gates]
        gate_rpys = [g["rpy"] for g in config.env.track.gates]
        gates = np.array(gate_centers)
        obstacles = np.array([o["pos"] for o in getattr(config.env.track, "obstacles", [])])

        # Get gate normals
        normals = [Rotation.from_euler("xyz", rpy).apply([1, 0, 0]) for rpy in gate_rpys]
        normals = np.array(normals)

        straight_half_length = 0.5
        safe_height = 0.5

        path_points = []
        for i in range(len(gates)):
            pos = gates[i]
            normal = normals[i]
            before = pos - normal * straight_half_length
            after = pos + normal * straight_half_length

            points_to_check = [after]
            if i > 0:
                points_to_check.append(before)

            if len(obstacles) > 0:
                for obs in obstacles:
                    radius = 0.3
                    for point in points_to_check:
                        if np.linalg.norm(point[:2] - obs[:2]) < radius:
                            point[2] = max(point[2], obs[2] + safe_height)

            if i == 0:
                path_points.extend([pos, after])  # Start AT center of first gate
            else:
                path_points.extend([before, after])

        self._waypoints = np.array(path_points)

        # --- Phase 1: Move to Start ---
        self._start_phase_ticks = 100  # Ticks to get to the first waypoint

        # --- Phase 2: Follow Path ---
        self._segment_ticks = 60  # Ticks to travel between each waypoint
        self._segment_tick_counter = 0
        self._target_waypoint_idx = 1  # Start by targeting waypoint 1 (after G1)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        if self._tick <= self._start_phase_ticks:
            # --- Phase 1: Moving to the start position (waypoint 0) ---
            alpha = min(self._tick / self._start_phase_ticks, 1.0)
            des_pos = (1 - alpha) * self._initial_pos + alpha * self._waypoints[0]
            self._tick += 1

            # When phase 1 is over, reset segment counter for phase 2
            if self._tick > self._start_phase_ticks:
                self._segment_tick_counter = 0

        elif self._target_waypoint_idx < len(self._waypoints):
            # --- Phase 2: Following the path segments ---
            start_of_segment = self._waypoints[self._target_waypoint_idx - 1]
            end_of_segment = self._waypoints[self._target_waypoint_idx]

            alpha = min(self._segment_tick_counter / self._segment_ticks, 1.0)
            des_pos = (1 - alpha) * start_of_segment + alpha * end_of_segment

            self._segment_tick_counter += 1

            # If this segment is done, move to the next one
            if self._segment_tick_counter >= self._segment_ticks:
                self._target_waypoint_idx += 1
                self._segment_tick_counter = 0

        else:
            # --- Phase 3: Finished. Hover at the last waypoint ---
            des_pos = self._waypoints[-1]
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
        self._segment_tick_counter = 0
        self._target_waypoint_idx = 1  # Reset target
        self.position_log.clear()
