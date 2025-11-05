"""A simple state controller that reactively plans a path to the next gate
based on the observation data.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """A simple state controller that flies directly to the next target gate
    provided in the observation dictionary.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The drone's desired state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate].
        """
        # 1. Get the index of the gate we need to fly to
        target_gate_idx = obs["target_gate"]

        # 2. Get the 3D position [x, y, z] of that target gate
        target_pos = obs["gates_pos"][target_gate_idx]

        # 3. (Optional but better) Calculate the desired yaw to face the target
        current_pos = obs["pos"]
        delta_pos = target_pos - current_pos
        # Calculate yaw angle (rotation around Z-axis) to face the target
        desired_yaw = np.arctan2(delta_pos[1], delta_pos[0])

        # 4. Create the 13-element action vector for "state" control
        # Initialize a vector of 13 zeros
        action = np.zeros(13, dtype=np.float32)

        # Set the first 3 elements to our desired [x, y, z] position
        action[0:3] = target_pos

        # Set the 10th element (index 9) to our desired yaw
        action[9] = desired_yaw

        # All other elements (velocities, accelerations, angular rates) remain zero.
        # We are telling the drone: "Go to this position and face this direction,
        # I don't care how you get there."
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
        """Check if the episode is finished.

        This controller is finished once all gates have been visited.
        """
        # Check the 'gates_visited' array in the observation
        if np.all(obs["gates_visited"]):
            self._finished = True

        # The 'terminated' and 'truncated' flags signal if the environment
        # has ended (e.g., due to a crash or timeout)
        return self._finished or terminated or truncated

    def episode_callback(self):
        """Reset the internal state for a new episode."""
        self._finished = False
