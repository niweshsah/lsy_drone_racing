"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from operator import ne
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(Controller):
    """Trajectory controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        # Same waypoints as in the trajectory controller. Determined by trial and error.
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.2],
                [0.3, 0.35, 0.5],
                [1.3, -0.05, 0.65],
                [0.85, 0.85, 1.1],
                [-0.5, -0.05, 0.65],
                [-1.3, -0.1, 0.52],
                [-1.3, -0.1, 1.1],
                [-0.0, -0.65, 1.1],
                [0.5, -0.65, 1.1],
            ]
        )
        self.t_total = 30
        t = np.linspace(0, self.t_total, len(waypoints))
        self.trajectory = CubicSpline(t, waypoints)
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

        self.delta_z = 0.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        i = min(self._tick / self._freq, self.t_total)
        target_pos = self.trajectory(i)
        #### Obstacle avoidance
        min_obstacle_dist = 0.25
        closest_obstacle_pos = np.argmin(np.linalg.norm(obs["obstacles_pos"] - target_pos, axis=-1))
        closest_obstacle_pos = obs["obstacles_pos"][closest_obstacle_pos].copy()
        closest_obstacle_pos[2] = 0  # Only consider x-y plane for obstacle avoidance
        target_pos_xy = target_pos.copy()
        target_pos_xy[2] = 0
        # print(f"{closest_obstacle_pos=}, {np.linalg.norm(closest_obstacle_pos - target_pos_xy)=}")
        distance_to_obstacle = np.linalg.norm(closest_obstacle_pos - target_pos_xy)
        if distance_to_obstacle < min_obstacle_dist:
            print(f"Too close to an obstacle! {target_pos=}, {distance_to_obstacle=}")
            # Move outwards to avoid obstacle
            normal_vector = closest_obstacle_pos - target_pos
            normal_vector[2] = 0  # Only move in x-y plane
            normal_vector /= np.linalg.norm(normal_vector) + 1e-6
            print(f"Normal vector: {normal_vector}")
            target_pos_delta = normal_vector * (distance_to_obstacle - min_obstacle_dist)
            print(f"{target_pos=}, {target_pos + target_pos_delta=}")
            target_pos += target_pos_delta

        #### Gate hit avoidance
        min_gate_dist = 0.2
        closest_gate_id = np.argmin(np.linalg.norm(obs["gates_pos"] - target_pos, axis=-1))
        gates_visited = np.array([*obs["gates_visited"], False])
        print(f"{gates_visited=}")
        next_gate_id = np.where(~gates_visited)[0][0]
        if next_gate_id >= len(obs["gates_pos"]):
            next_gate_id = len(obs["gates_pos"]) - 1
        print(next_gate_id)
        target_pos_xy = target_pos.copy()
        target_pos_xy[2] = 0
        # Compute gate edges from gate_quat and known width
        gate_width = 0.4
        gate_quat = obs["gates_quat"][closest_gate_id]
        gate_pos = obs["gates_pos"][closest_gate_id]

        # Convert quaternion to rotation matrix
        rot = R.from_quat(gate_quat)
        gate_right = rot.apply([1, 0, 0])
        gate_norm = rot.apply([0, 0, 1])

        # Gate edges in world coordinates
        edge_offset = gate_right * (gate_width / 2)
        gate_edge1 = gate_pos + edge_offset
        gate_edge2 = gate_pos - edge_offset
        gate_edge1[2] = 0
        gate_edge2[2] = 0

        # Compute normal vectors from each edge to the gate center (in x-y plane)
        gate_center_xy = gate_pos.copy()
        gate_center_xy[2] = 0
        gate_center_z = gate_pos.copy()
        gate_center_z[2] = 0
        normal_edge1_to_center = gate_edge1 - gate_center_xy
        normal_edge2_to_center = gate_center_xy - gate_edge2
        normal_edge1_to_center /= np.linalg.norm(normal_edge1_to_center) + 1e-6
        normal_edge2_to_center /= np.linalg.norm(normal_edge2_to_center) + 1e-6

        distance_to_gate_edge1 = np.linalg.norm(gate_edge1 - target_pos_xy)
        distance_to_gate_edge2 = np.linalg.norm(gate_edge2 - target_pos_xy)

        # If too close to either gate edge, project target_pos onto the gate center line (between gate_edge1 and gate_edge2)
        if distance_to_gate_edge1 < min_gate_dist or distance_to_gate_edge2 < min_gate_dist:
            print(
                f"Too close to gate edge! {target_pos=}, {distance_to_gate_edge1=}, {distance_to_gate_edge2=}"
            )
            # Move target_pos onto the gate center line using gate normal
            gate_center_line_point = gate_center_xy
            # Project target_pos_xy onto the gate center line (defined by gate_center_xy and gate_norm in x-y)
            gate_norm_xy = gate_norm.copy()
            gate_norm_xy[2] = 0
            gate_norm_xy /= np.linalg.norm(gate_norm_xy) + 1e-6
            vec_to_target = target_pos_xy - gate_center_line_point
            proj_length = np.dot(vec_to_target, gate_norm_xy)
            proj_point = gate_center_line_point + gate_norm_xy * proj_length
            # Set target_pos x-y to projected point, keep z unchanged
            target_pos[:2] = proj_point[:2]
            print(f"Moved target_pos onto gate center line using gate normal: {target_pos}")

        # --- Upper and lower gate bar avoidance ---
        # Assume gate height is known (e.g., 0.4m)
        gate_height = 0.4
        # Gate upper and lower bar positions in world coordinates
        gate_pos = obs["gates_pos"][next_gate_id]
        gate_upper_bar = gate_pos.copy()
        gate_upper_bar[2] += gate_height / 2
        gate_lower_bar = gate_pos.copy()
        gate_lower_bar[2] -= gate_height / 2

        distance_to_upper_bar = abs(target_pos[2] - gate_upper_bar[2])
        distance_to_lower_bar = abs(target_pos[2] - gate_lower_bar[2])

        # If too close to either bar, project target_pos z onto gate center z
        smoothing_factor = 0.95
        if (
            distance_to_upper_bar < min_gate_dist or distance_to_lower_bar < min_gate_dist
        ) and np.linalg.norm(gate_pos - target_pos) < 1.0:
            print(
                f"Too close to gate bar! {target_pos[2]=}, {gate_pos[2]=}, {distance_to_upper_bar=}, {distance_to_lower_bar=}"
            )
            # Move target_pos z to gate center z
            self.delta_z = smoothing_factor * self.delta_z + (1 - smoothing_factor) * (
                gate_pos[2] - target_pos[2]
            )
            target_pos[2] += self.delta_z
            print(f"Moved target_pos onto gate center z: {target_pos}, {self.delta_z=}")
        else:
            self.delta_z *= smoothing_factor

        print(f"{gate_pos=}, {gate_edge1=}, {gate_edge2=}")
        if i >= self.t_total:  # Maximum duration reached
            self._finished = True
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished
        """Reset the time step counter."""
        self._tick = 0
