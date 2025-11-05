"""A Proportional-Derivative (PD) feedback controller for state control.

This controller actively calculates a desired state (position, velocity, acceleration, yaw)
to navigate the drone to the next target gate.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """PD Feedback Controller.

    Computes a full 13-element state reference (pos, vel, acc, yaw, ang_vel)
    based on PD control logic to guide the drone to the next target gate.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)

        # Store waypoints (gate positions) from the config
        self.waypoints = np.array([g.pos for g in config.env.track.gates])

        # --- PD Controller Gains ---
        # These gains are for calculating the target state, not motor commands.
        # They may need tuning for optimal performance.

        # P-gain for position error -> target velocity
        self.Kp_pos = np.array([4.0, 4.0, 4.0])
        # P-gain for velocity error -> target acceleration (acts as D-gain on position)
        self.Kp_vel = np.array([3.0, 3.0, 3.0])

        # --- Limits ---
        self.max_vel = 5.0  # m/s
        self.max_acc = 8.0  # m/s^2 (excluding gravity)
        self.g = 9.81  # Gravity

        # --- Yaw Initialization ---
        # Initialize target yaw from the drone's starting orientation
        q = obs["state"][6:10]  # Current attitude quaternion
        # Convert quaternion to yaw (Z-axis rotation)
        self.target_yaw = np.arctan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
            as a numpy array.
        """
        # Get current state
        pos = obs["state"][:3]
        vel = obs["state"][3:6]

        # Get target gate index
        target_idx = obs["target_gate"]

        # --------------------------------------------------------------------
        # 1. Determine Target Position
        # --------------------------------------------------------------------
        if target_idx == -1:
            # Race is finished, hover at the last gate's position
            target_pos = self.waypoints[-1]
        else:
            # Target the next gate
            target_pos = self.waypoints[target_idx]

        # --------------------------------------------------------------------
        # 2. Calculate Target Velocity (P-controller on position)
        # --------------------------------------------------------------------
        error_pos = target_pos - pos
        target_vel = self.Kp_pos * error_pos

        # Clip velocity to a maximum
        vel_norm = np.linalg.norm(target_vel)
        if vel_norm > self.max_vel:
            target_vel = target_vel * (self.max_vel / vel_norm)

        # --------------------------------------------------------------------
        # 3. Calculate Target Acceleration (P-controller on velocity)
        # --------------------------------------------------------------------
        error_vel = target_vel - vel
        target_acc = self.Kp_vel * error_vel

        # Add gravity compensation (to command total acceleration in world frame)
        target_acc[2] += self.g

        # Clip acceleration to a maximum
        acc_norm = np.linalg.norm(target_acc)
        if acc_norm > self.max_acc:
            target_acc = target_acc * (self.max_acc / acc_norm)

        # --------------------------------------------------------------------
        # 4. Calculate Target Yaw
        # --------------------------------------------------------------------
        # Point the drone in the direction of the target velocity
        if np.linalg.norm(target_vel[:2]) > 0.2:  # Only update if moving horizontally
            self.target_yaw = np.arctan2(target_vel[1], target_vel[0])

        # --------------------------------------------------------------------
        # 5. Assemble the 13-element action vector
        # --------------------------------------------------------------------
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = target_pos  # Target position (for low-level controller)
        action[3:6] = target_vel  # Target velocity
        action[6:9] = target_acc  # Target acceleration
        action[9] = self.target_yaw  # Target yaw
        action[10:13] = 0.0  # Target angular rates (command zero, let yaw setpoint work)

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
        """Callback function called once after the control step.

        This controller is stateless (beyond target_yaw, which is handled
        in compute_control), so no updates are needed here.

        Returns:
            False, to signal the controller is never "finished" early.
        """
        return False  # Do not terminate the episode from the controller

    def episode_callback(self):
        """Callback function called once after each episode.

        A new controller is initialized for each episode, so no reset
        logic is needed here.
        """
        pass

    def episode_reset(self):
        """Reset the controller's internal state.

        A new controller is initialized for each episode, so no reset
        logic is needed here.
        """
        pass
