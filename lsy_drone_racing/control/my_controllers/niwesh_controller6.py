"""
A simple state controller that flies straight through all gates in sequence
without hitting obstacles. It aims directly for the gate centers and moves
in straight lines from one to the next.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

EPSILON = 1e-6


class StraightPathController(Controller):
    """
    Controller that flies through all gates in straight lines.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """
        Initialize the controller.

        Args:
            obs: Initial observation.
            info: Initial environment info.
            config: Race configuration.
        """
        super().__init__(obs, info, config)

        # Distance threshold to consider gate reached
        self._gate_reach_threshold = 0.3  # meters

        # Stats
        self._total_reward = 0.0
        self._terminated = False
        self._truncated = False
        self._last_obs = obs

        print("[StraightPathController] Initialized. Ready for takeoff.")

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """
        Compute the desired state to fly straight through gates.

        Returns:
            np.ndarray: 13-element desired state vector.
        """
        # Current drone position
        current_pos = obs["pos"]
        # Current target gate index
        gate_idx = int(obs["target_gate"])
        # Gate center position
        target_pos = obs["gates_pos"][gate_idx]

        # Vector toward the gate
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction) + EPSILON
        unit_direction = direction / distance

        # Move straight toward the gate center
        aim_point = target_pos.copy()

        # Desired yaw faces toward gate
        desired_yaw = np.arctan2(direction[1], direction[0])

        # Build action vector
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = aim_point
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
        """
        Called after each environment step.
        Tracks progress and determines if race is finished.
        """
        self._total_reward += reward
        self._last_obs = obs
        self._terminated = terminated
        self._truncated = truncated

        gates_visited = obs["gates_visited"]
        all_visited = np.all(gates_visited)

        # Episode ends when all gates are visited or simulation ends
        return all_visited or terminated or truncated

    def episode_callback(self):
        """
        Print a summary after finishing the race.
        """
        print("\n" + "=" * 30)
        print("       STRAIGHT PATH SUMMARY")
        print("=" * 30)
        print(f"Total Reward: {self._total_reward:.2f}")

        if self._terminated:
            print("Outcome: CRASHED (Terminated)")
        elif self._truncated:
            print("Outcome: TIMED OUT (Truncated)")
        elif np.all(self._last_obs["gates_visited"]):
            print("Outcome: SUCCESS (All gates visited)")
        else:
            print("Outcome: UNKNOWN")

        gates_visited = self._last_obs["gates_visited"]
        print(f"Gates Visited: {np.sum(gates_visited)} / {len(gates_visited)}")
        print(f"Gate Status: {gates_visited}")

        final_pos = self._last_obs["pos"]
        print(f"Final Position (x,y,z): "
              f"[{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")
        print("=" * 30 + "\n")

        # Reset for next run
        self._total_reward = 0.0
        self._terminated = False
        self._truncated = False
