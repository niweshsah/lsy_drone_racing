"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.constants import MASS
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.fresssack_controller import FresssackController
from lsy_drone_racing.tools.ext_tools import TrajectoryTool
from lsy_drone_racing.utils.utils import draw_line
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeController(FresssackController):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, env:VecDroneRaceEnv=None):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.env = env
        self.num_envs = env.num_envs
        self.freq = config.env.freq
        self.drone_mass = MASS
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self._i_error = np.zeros((self.num_envs, 3))
        self.g = 9.81
        self._tick = 0

        # Demo waypoints
        waypoints = np.array(
            [
                [1.0, 1.5, 0.05],
                [0.8, 1.0, 0.2],
                [0.55, -0.3, 0.5],
                [0.2, -1.3, 0.65],
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.55],
                [0.0, 0.9, 0.45],
                [0.0, 1.2, 0.45],
                [0.0, 1.2, 0.55],
                [0.0, 1.0, 0.6],
                [0.0, 0.7, 0.75],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
            ]
        )
        self.waypoints = waypoints
        t = np.linspace(0,10,len(waypoints))
        trajectory = CubicSpline(t, waypoints)

        # # pre-planned trajectory
        # t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets.csv")     
        # trajectory = CubicSpline(t, pos)
        # trajectory reparameterization
        self.traj_tool = TrajectoryTool()
        trajectory = self.traj_tool.extend_trajectory(trajectory)
        self.arc_trajectory = self.traj_tool.arclength_reparameterize(trajectory)
        self._theta = np.zeros(self.num_envs) # traveled distance
        self._prev_pos = obs['pos']

        self.gates_theta, _ = self.traj_tool.find_gate_waypoint(self.arc_trajectory, obs['gates_pos'][0])

        self._finished = False
    
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """ Vectorized attitude/thrust reference for VecDroneRaceEnv. """
        N = self.num_envs

        des_pos    = np.empty((N, 3), dtype=float)
        des_vel    = np.zeros((N, 3), dtype=float)
        des_yaw    = 0.0
        for i in range(N):
            curr_theta, _ = self.traj_tool.find_nearest_waypoint(
                self.arc_trajectory,
                obs["pos"][i],
                self.gates_theta[obs["target_gate"][i]],
            )
            self._theta[i] += np.linalg.norm(obs["pos"][i] - self._prev_pos[i]) # integrated theta
            alpha = 1.0 # trust calculated theta
            self._theta[i] = (1 - alpha) * self._theta[i] + alpha * curr_theta
            self._prev_pos[i] = obs["pos"][i]

            des_pos[i] = self.arc_trajectory(self._theta[i] + 0.25)
            # vel = self.arc_trajectory.derivative(1)(self.theta+0.15)
            # des_vel[i] = 0.5 * vel/np.linalg.norm(vel)

        pos_error = des_pos - obs["pos"]                # (N,3)
        vel_error = des_vel - obs["vel"]

        dt = 1.0 / self.freq
        self._i_error += pos_error * dt
        self._i_error  = np.clip(self._i_error, -self.ki_range, self.ki_range)

        target_thrust = (
            self.kp * pos_error +
            self.ki * self._i_error +
            self.kd * vel_error
        )
        target_thrust[:, 2] += self.drone_mass * self.g

        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, :, 2]   # (N,3)

        thrust_desired = np.einsum("ij,ij->i", target_thrust, z_axis)  # (N,)
        thrust_desired = np.clip(
            thrust_desired,
            0.3 * self.drone_mass * self.g,
            1.8 * self.drone_mass * self.g,
        )

        z_axis_desired = target_thrust / np.linalg.norm(target_thrust, axis=1, keepdims=True)
        x_c_des = np.stack([np.cos(des_yaw) * np.ones(N), np.sin(des_yaw) * np.ones(N), np.zeros(N)], axis=1)
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired, axis=1, keepdims=True)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.stack([x_axis_desired, y_axis_desired, z_axis_desired], axis=-1)  # (N,3,3)
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)          # (N,3)

        out = np.concatenate([thrust_desired[:, None], euler_desired], axis=1).astype(np.float32)
        return out

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self.i_error[:] = 0
        self._tick = 0
