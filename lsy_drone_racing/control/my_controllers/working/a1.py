"""A reactive state controller that uses the observation data to
dynamically generate a smooth cubic spline trajectory.

This version uses "chordal" parameterization (actual distance)
for the spline 's' variable. This prevents the spline from
"overshooting" the waypoints and makes the path more geometrically
accurate, avoiding collisions with the gate.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Add a small value to prevent division by zero
EPSILON = 1e-6


class StateController(Controller):
    """State controller that dynamically re-plans a distance-based spline path
    using observation data.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller."""
        super().__init__(obs, info, config)
        self._finished = False

        # --- Controller Tuning Knobs ---

        # How many gates to include in the spline calculation.
        self._lookahead_gates = 2

        # How "far ahead" on the spline to set our target (in meters).
        # This is now a fixed distance, which is more intuitive.
        # This acts as a "carrot" for the drone to follow.
        self._lookahead_distance = 0.4  # 40 cm
        # --- End Tuning Knobs ---

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone."""
        # 1. Get all necessary data from the observation
        current_pos = obs["pos"]
        gates_pos = obs["gates_pos"]
        target_idx = obs["target_gate"]
        num_gates = len(gates_pos)

        # 2. Build the list of waypoints for our *new* spline
        waypoints_list = [current_pos]
        gates_to_add = min(self._lookahead_gates, num_gates - target_idx)
        for i in range(gates_to_add):
            waypoints_list.append(gates_pos[target_idx + i])

        waypoints = np.array(waypoints_list)

        # 3. Handle different cases

        # Fallback: If we have no gates left, just aim for the last one
        if len(waypoints) < 2:
            target_pos = gates_pos[-1]

        else:
            # --- THIS IS THE FIX ---
            # Create the 's' parameter based on cumulative distance
            distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
            s = np.zeros(len(waypoints))
            s[1:] = np.cumsum(distances)  # s will be [0, dist_0_1, dist_0_1 + dist_1_2, ...]
            # --- END FIX ---

            # If all points are at the same location, spline will fail
            if s[-1] < EPSILON:
                target_pos = waypoints[-1]
            else:
                # Create the 3D spline
                spline = CubicSpline(s, waypoints, axis=0, bc_type="natural")

                # Get the target position by looking a fixed distance ahead
                # along the spline path.
                # We cap it at the second-to-last knot to avoid flying
                # wildly past the final point.
                lookahead_s = min(self._lookahead_distance, s[-1] - EPSILON)
                target_pos = spline(lookahead_s)

        # 4. Calculate desired yaw to face the *actual* next gate
        vec_to_gate = gates_pos[target_idx] - current_pos
        desired_yaw = np.arctan2(vec_to_gate[1], vec_to_gate[0])

        # 5. Build the final action
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = target_pos
        action[9] = desired_yaw

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
        """Check if the episode is finished."""
        if np.all(obs["gates_visited"]):
            self._finished = True

        return self._finished or terminated or truncated

    def episode_callback(self):
        """Reset the internal state for a new episode."""
        self._finished = False
