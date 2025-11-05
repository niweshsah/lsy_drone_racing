import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

# Set up a logger for this module
logger = logging.getLogger(__name__)


class PathFollowingController(Controller):
    """Controller that brings the drone to the first gate, then follows a refined
    path passing through all gates and avoiding obstacles.

    Phases:
    1.  Move from initial position to the start of the refined path.
    2.  Follow the refined path segment by segment.
    3.  Hover at the final point of the path.
    """

    def __init__(self, obs: Dict[str, NDArray[np.floating]], info: Dict, config: ConfigDict):
        """Initialize the controller, generate the refined path, and set up state."""
        super().__init__(obs, info, config)

        # --- State variables ---
        self._global_tick = 0
        self._segment_tick = 0
        self._finished = False
        self._initial_pos = obs["pos"][:3].copy()
        self.position_log: List[NDArray[np.floating]] = []

        # --- Generate refined path ---
        # This is computed once when the controller is initialized.
        self._refined_path = self.get_refined_path_from_config(config)
        self._num_path_points = len(self._refined_path)

        # --- Phase timing parameters ---
        self._start_phase_ticks = 100  # ticks to reach the first path point
        self._segment_ticks = 2  # ticks to traverse one path segment
        self._path_idx = 0  # current target index in refined path

    @staticmethod
    def refine_path_with_obstacles(
        control_points: np.ndarray,
        obstacles: np.ndarray,
        *,
        avoidance_radius_pole: float = 0.4,
        avoidance_radius_sphere: float = 0.4,
        safety_margin: float = 0.1,
        max_iterations: int = 10,
        dense_samples: int = 250,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Iteratively refine control points such that a spline passing through
        them avoids obstacles.

        Args:
            control_points: (N, 3) array of initial waypoints.
            obstacles: (M, 3) array of obstacle centers (x, y, z).
            avoidance_radius_pole: XY radius for ground-up "pole" obstacles.
            avoidance_radius_sphere: 3D radius for "sphere" obstacles.
            safety_margin: Additional buffer added to radii.
            max_iterations: Max refinement loops.
            dense_samples: Number of points to sample along the spline
                           for collision checking.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - (dense_samples, 3) final smooth path
                - (K, 3) final control points (K >= N)
        """
        # Guard: if no obstacles, just return a single smooth path
        if obstacles.size == 0:
            t = np.linspace(0, 1, len(control_points))
            cs_x = CubicSpline(t, control_points[:, 0])
            cs_y = CubicSpline(t, control_points[:, 1])
            cs_z = CubicSpline(t, control_points[:, 2])
            t_smooth = np.linspace(0, 1, dense_samples)
            smooth_path = np.vstack((cs_x(t_smooth), cs_y(t_smooth), cs_z(t_smooth))).T
            return smooth_path, control_points

        current_points = control_points.copy()
        current_t = np.linspace(0, 1, len(current_points))

        for iteration in range(max_iterations):
            # 1) Build spline and dense samples
            cs_x = CubicSpline(current_t, current_points[:, 0])
            cs_y = CubicSpline(current_t, current_points[:, 1])
            cs_z = CubicSpline(current_t, current_points[:, 2])
            t_smooth = np.linspace(0, 1, dense_samples)
            smooth_path = np.vstack((cs_x(t_smooth), cs_y(t_smooth), cs_z(t_smooth))).T

            # 2) Detect collisions and prepare new control points to add
            points_to_add: List[Tuple[float, np.ndarray]] = []
            epsilon = 1e-8  # For safe normalization

            for i, p in enumerate(smooth_path):
                t_val = t_smooth[i]
                collided = False
                for o in obstacles:
                    dist_xy = np.linalg.norm(p[:2] - o[:2])
                    dist_3d = np.linalg.norm(p - o)

                    # Check for horizontal pole collision (ground-up)
                    if dist_xy < avoidance_radius_pole and p[2] < o[2]:
                        v_xy = p[:2] - o[:2]
                        v_xy_norm = v_xy / (np.linalg.norm(v_xy) + epsilon)
                        push_dist = avoidance_radius_pole + safety_margin
                        p_new = np.array(
                            [o[0] + v_xy_norm[0] * push_dist, o[1] + v_xy_norm[1] * push_dist, p[2]]
                        )
                        points_to_add.append((t_val, p_new))
                        collided = True
                        break

                    # Check for 3D sphere collision
                    if dist_3d < avoidance_radius_sphere + epsilon:
                        v = p - o
                        v_norm = v / (np.linalg.norm(v) + epsilon)
                        push_dist = avoidance_radius_sphere + safety_margin
                        p_new = o + v_norm * push_dist
                        points_to_add.append((t_val, p_new))
                        collided = True
                        break

                if collided:
                    continue  # Skip other obstacles for this point

            # 3) If no collisions found -> done
            if not points_to_add:
                logger.info("Refinement converged (no collisions detected).")
                break

            # 4) Add new control points (efficiently)
            new_control_points = dict(zip(current_t, current_points))
            # Use rounding to create a set for quick, approximate checking
            t_to_check = set(np.round(current_t, 6))

            added_count = 0
            for t_val, p_new in points_to_add:
                t_rounded = round(t_val, 6)
                if t_rounded not in t_to_check:
                    # This is a new t-value. If multiple points_to_add
                    # have the same t_rounded, this logic adds the *first* one.
                    new_control_points[t_val] = p_new
                    t_to_check.add(t_rounded)  # Prevent adding others at this t
                    added_count += 1

            # 5) If no *new* points were added -> done
            if not added_count:
                logger.info("Refinement converged (no new collision points to add).")
                break

            # 6) Sort by t and update for next iteration
            sorted_t_vals = sorted(new_control_points.keys())
            current_t = np.array(sorted_t_vals)
            current_points = np.array([new_control_points[t] for t in sorted_t_vals])

            logger.debug(
                f"Iteration {iteration}: added {added_count} points -> total control points {len(current_points)}"
            )

        else:
            logger.warning("Refinement stopped: reached max iterations.")

        # Final smooth path from last control points
        cs_x = CubicSpline(current_t, current_points[:, 0])
        cs_y = CubicSpline(current_t, current_points[:, 1])
        cs_z = CubicSpline(current_t, current_points[:, 2])
        t_smooth = np.linspace(0, 1, dense_samples)
        smooth_path = np.vstack((cs_x(t_smooth), cs_y(t_smooth), cs_z(t_smooth))).T

        return smooth_path, current_points

    @staticmethod
    def get_refined_path_from_config(config: ConfigDict) -> NDArray[np.floating]:
        """Compute refined path starting at the first gate center, using the
        environment configuration object.
        """
        gate_centers = [np.array(g["pos"]) for g in config.env.track.gates]
        gate_rpys = [np.array(g["rpy"]) for g in config.env.track.gates]

        # Defensively get obstacles, default to empty array
        obstacles_list = getattr(config.env.track, "obstacles", [])
        obstacles = (
            np.array([obs["pos"] for obs in obstacles_list]) if obstacles_list else np.empty((0, 3))
        )

        # Build initial control points
        straight_half_length = 0.6
        path_points = []
        for idx, (pos, rpy) in enumerate(zip(gate_centers, gate_rpys)):
            # Gate normal vector (points "through" the gate)
            normal = Rotation.from_euler("xyz", rpy).apply([1.0, 0.0, 0.0])
            if idx == 0:
                # Start at first gate center
                path_points.append(pos)
                path_points.append(pos + normal * straight_half_length)
            else:
                # Point before, at, and after the gate
                before = pos - normal * straight_half_length
                path_points.extend([before, pos, pos + normal * straight_half_length])

        control_points = np.array(path_points)

        # Refine the path using the static method
        smooth_path, _ = PathFollowingController.refine_path_with_obstacles(
            control_points, obstacles
        )
        return smooth_path

    def compute_control(
        self, obs: Dict[str, NDArray[np.floating]], info: Optional[Dict] = None
    ) -> NDArray[np.floating]:
        """Compute the control action based on the current controller phase.

        Returns:
            (13,) ndarray: [des_pos, 10x zeros]
        """
        self._global_tick += 1

        # Phase 1: Move from initial_pos to the start of the path
        if self._global_tick <= self._start_phase_ticks:
            alpha = self._global_tick / self._start_phase_ticks
            des_pos = (1 - alpha) * self._initial_pos + alpha * self._refined_path[0]

            # On the last tick of phase 1, set up for phase 2
            if self._global_tick == self._start_phase_ticks:
                self._path_idx = 1  # Start by targeting the 2nd point (index 1)
                self._segment_tick = 0

        # Phase 2: Follow the refined path segments
        elif self._path_idx < self._num_path_points:
            self._segment_tick += 1
            start_pt = self._refined_path[self._path_idx - 1]
            end_pt = self._refined_path[self._path_idx]

            # Linearly interpolate between the two path points
            alpha = min(self._segment_tick / self._segment_ticks, 1.0)
            des_pos = (1 - alpha) * start_pt + alpha * end_pt

            # If segment is finished, advance to the next one
            if alpha >= 1.0:
                self._path_idx += 1
                self._segment_tick = 0

        # Phase 3: Finished. Hover at the last point
        else:
            des_pos = self._refined_path[-1]
            self._finished = True

        self.position_log.append(des_pos.copy())

        # Action is desired position [3] + 10 zeros (for vel, rpy, rates, etc.)
        action = np.concatenate((des_pos, np.zeros(10, dtype=np.float32)))
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: Dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict,
    ) -> bool:
        """Returns True if the episode should terminate (path is finished)."""
        return self._finished

    def episode_callback(self):
        """Reset controller state for a new episode.

        Note: Assumes __init__ is called for each new episode to reset
        _initial_pos and re-compute the path if the config changed.
        """
        self._global_tick = 0
        self._segment_tick = 0
        self._finished = False
        self._path_idx = 0
        self.position_log
