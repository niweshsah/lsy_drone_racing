"""
Adaptive Spline Controller for Level 3 Drone Racing.

This module implements a reactive trajectory planner that generates smooth 3D 
cubic splines to navigate a drone through a sequence of gates while actively 
avoiding dynamic obstacles.

Key Features:
1.  Gate-Aligned Waypoints: Generates approach vectors normal to gate planes.
2.  Detour Logic: Detects sharp turns (>120 deg) and inserts intermediate 
    waypoints to ensure flyable trajectories.
3.  Reactive Obstacle Avoidance: Uses a geometric bisector method to modify 
    the spline path in real-time if an obstacle enters the flight path.
4.  Replanning Trigger: Monitors environment state and proximity to trigger 
    trajectory regeneration only when necessary.
    
    

Author: Niwesh Sah
Email: sahniwesh@gmail.com
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AdaptiveSplineController(Controller):
    """
    A reactive controller using Cubic Splines for trajectory generation and tracking.
    
    Attributes:
        FLIGHT_DURATION (float): Total expected time for the race.
        STATE_SIZE (int): Size of the control state vector (13).
        REPLAN_RADIUS (float): Distance threshold (m) to trigger a replan if
                               unexpectedly close to an object.
        OBSTACLE_CLEARANCE (float): Safety margin (m) to maintain around obstacles.
        DETOUR_ANGLE_THRESHOLD (float): Angle (degrees) triggering a detour waypoint.
    """

    # --- Configuration Constants ---
    FLIGHT_DURATION = 25.0
    STATE_SIZE = 13
    REPLAN_RADIUS = 0.5
    OBSTACLE_CLEARANCE = 0.2
    
    # Detour Parameters
    DETOUR_ANGLE_THRESHOLD = 120.0
    DETOUR_RADIUS = 0.65
    
    # Gate Approach Parameters
    APPROACH_DIST = 0.5
    APPROACH_POINTS = 5

    def __init__(self, initial_obs: Dict[str, Any], info: Dict[str, Any], 
                 sim_config: Dict[str, Any], env: Optional[Any] = None):
        """
        Initializes the controller and plans the initial trajectory.

        Args:
            initial_obs: The first observation from the environment.
            info: Additional environment info.
            sim_config: Simulation configuration object.
            env: Reference to the Gym environment (for visualization).
        """
        super().__init__(initial_obs, info, sim_config)
        self.env = env
        
        # Internal State
        self._current_step = 0
        self._control_freq = sim_config.env.freq
        self._is_finished = False
        
        # Environment State Tracking (for Replanning)
        self._last_gate_flags: Optional[NDArray[np.bool_]] = None
        self._last_obs_flags: Optional[NDArray[np.bool_]] = None
        
        # Trajectory Data
        self._trajectory_spline: Optional[CubicSpline] = None
        
        # Initial Planning
        self._plan_trajectory(initial_obs)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def compute_control(self, obs: Dict[str, Any], info: Optional[Dict[str, Any]] = None) -> NDArray[np.float32]:
        """
        Computes the control command for the current time step.

        This method checks if a replan is necessary due to environment changes,
        samples the spline for the current time, and returns the target state.

        Args:
            obs: Current environment observation.
            info: Optional info dictionary.

        Returns:
            NDArray: The target state vector [x, y, z, vx, vy, vz, ax, ay, az, 0, 0, 0, 0].
        """
        # 1. Check for triggers to regenerate the path
        if self._should_replan(obs):
            self._plan_trajectory(obs)

        # 2. Determine simulation time
        t_current = min(self._current_step / self._control_freq, self.FLIGHT_DURATION)
        
        if t_current >= self.FLIGHT_DURATION:
            self._is_finished = True

        # 3. Sample Trajectory (0=Pos, 1=Vel, 2=Acc)
        ref_pos = self._trajectory_spline(t_current, nu=0)
        ref_vel = self._trajectory_spline(t_current, nu=1)
        ref_acc = self._trajectory_spline(t_current, nu=2)

        # 4. Visualization (Optional)
        self._visualize_path()

        # 5. Construct State Vector
        # [pos(3), vel(3), acc(3), jerk(3), yaw(1)] -> Standard 13-state vector
        state_cmd = np.zeros(self.STATE_SIZE, dtype=np.float32)
        state_cmd[0:3] = ref_pos
        state_cmd[3:6] = ref_vel
        state_cmd[6:9] = ref_acc
        # Remaining states (jerk, yaw) left as 0.0

        return state_cmd

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        """Callback executed after every simulation step."""
        self._current_step += 1
        return self._is_finished

    # =========================================================================
    # TRAJECTORY PLANNING (The "Brain")
    # =========================================================================

    def _plan_trajectory(self, obs: Dict[str, Any]) -> None:
        """
        Orchestrates the generation of the flight path based on current observations.
        
        Steps:
        1. Extract geometry (Gate normals, positions).
        2. Generate linear waypoints through gates.
        3. Insert detour waypoints for sharp turns.
        4. Insert avoidance waypoints for obstacles.
        5. Fit a Cubic Spline to the final set of waypoints.
        """
        # Extract Geometry
        gate_pos = obs["gates_pos"]
        obs_pos = obs["obstacles_pos"]
        start_pos = obs["pos"] if self._trajectory_spline else obs["pos"]
        
        gate_normals, gate_y, gate_z = self._extract_gate_frames(obs["gates_quat"])

        # 1. Base Waypoints
        waypoints = self._generate_gate_approach_points(
            start_pos, gate_pos, gate_normals
        )

        # 2. Detour Logic (Handle sharp turns)
        waypoints = self._apply_detour_logic(
            waypoints, gate_pos, gate_normals, gate_y, gate_z
        )

        # 3. Obstacle Avoidance (Handle collisions)
        time_knots, waypoints = self._apply_obstacle_avoidance(
            waypoints, obs_pos
        )

        # 4. Spline Fitting
        self._trajectory_spline = self._compute_spline(
            self.FLIGHT_DURATION, waypoints, time_knots
        )

    def _should_replan(self, obs: Dict[str, Any]) -> bool:
        """
        Determines if the trajectory needs to be regenerated.
        
        Triggers:
        1. **State Transition:** A gate or obstacle has been newly visited.
        2. **Safety Violation:** Drone is inside the 'replan radius' of any object.
        """
        # --- 1. State Transition Check ---
        curr_gate_flags = np.array(obs["gates_visited"], dtype=bool)
        curr_obs_flags = np.array(obs["obstacles_visited"], dtype=bool)

        if self._last_gate_flags is None:
            self._last_gate_flags = curr_gate_flags
            self._last_obs_flags = curr_obs_flags
            return False # No change on first step

        gate_change = np.any((~self._last_gate_flags) & curr_gate_flags)
        obs_change = np.any((~self._last_obs_flags) & curr_obs_flags)

        self._last_gate_flags = curr_gate_flags
        self._last_obs_flags = curr_obs_flags

        # --- 2. Proximity Check (Safety) ---
        drone_pos = obs["pos"]
        
        # Check 3D distance to gates
        dist_gates = np.linalg.norm(obs["gates_pos"] - drone_pos, axis=1)
        proximity_gate = np.any(dist_gates < self.REPLAN_RADIUS)

        # Check 2D distance to obstacles (cylindrical approximation)
        dist_obs = np.linalg.norm(obs["obstacles_pos"][:, :2] - drone_pos[:2], axis=1)
        proximity_obs = np.any(dist_obs < self.REPLAN_RADIUS)

        return gate_change or obs_change or proximity_gate or proximity_obs

    # =========================================================================
    # GEOMETRY & MATH HELPERS
    # =========================================================================

    def _generate_gate_approach_points(self, start_pos: NDArray, gates: NDArray, 
                                     normals: NDArray) -> NDArray:
        """
        Generates a sequence of points passing through each gate.
        
        Creates points before, at, and after the gate center to ensure the 
        drone passes through perpendicularly.
        """
        # Create offsets: e.g. [-0.5, -0.25, 0.0, 0.25, 0.5]
        offsets = np.linspace(-self.APPROACH_DIST, self.APPROACH_DIST, self.APPROACH_POINTS)
        
        # Broadcast: Gate_Pos (N,1,3) + Offset (1,M,1) * Normal (N,1,3)
        wps = (gates[:, None, :] + 
               offsets[None, :, None] * normals[:, None, :])
        
        # Flatten to (N*M, 3) and prepend start position
        wps_flat = wps.reshape(-1, 3)
        return np.vstack([start_pos, wps_flat])

    def _apply_detour_logic(self, waypoints: NDArray, gate_pos: NDArray, 
                           gate_normals: NDArray, gate_y: NDArray, gate_z: NDArray) -> NDArray:
        """
        Inspects the path between gates; if a turn is too sharp (>120 deg),
        inserts a 'detour' waypoint to widen the turn.
        """
        wps_list = list(waypoints)
        n_gates = gate_pos.shape[0]
        inserts_made = 0

        for i in range(n_gates - 1):
            # Calculate indices in the flattened waypoint list
            # The list grows as we insert, so we offset by `inserts_made`
            idx_exit_current = 1 + (i + 1) * self.APPROACH_POINTS - 1 + inserts_made
            idx_entry_next = 1 + (i + 1) * self.APPROACH_POINTS + inserts_made

            p1 = wps_list[idx_exit_current]
            p2 = wps_list[idx_entry_next]
            vec = p2 - p1
            norm = np.linalg.norm(vec)

            if norm < 1e-6:
                continue

            # Calculate angle between flight path and current gate normal
            # If we are flying "backwards" relative to the gate normal, angle is high
            cos_theta = np.clip(np.dot(vec, gate_normals[i]) / norm, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_theta))

            if angle_deg > self.DETOUR_ANGLE_THRESHOLD:
                # 1. Project path onto gate plane to find best detour direction
                v_proj = vec - np.dot(vec, gate_normals[i]) * gate_normals[i]
                
                # 2. Determine if we should go Left, Right, or Top
                detour_dir = self._get_detour_direction(v_proj, gate_y[i], gate_z[i])
                
                # 3. Create and insert the point
                detour_pt = gate_pos[i] + self.DETOUR_RADIUS * detour_dir
                wps_list.insert(idx_exit_current + 1, detour_pt)
                inserts_made += 1

        return np.array(wps_list)

    def _get_detour_direction(self, v_proj: NDArray, y_axis: NDArray, z_axis: NDArray) -> NDArray:
        """Determines the cardinal direction for a detour based on projection."""
        norm = np.linalg.norm(v_proj)
        if norm < 1e-6:
            return y_axis # Default to Right
        
        # Calculate angle in Y-Z plane
        dot_y = np.dot(v_proj, y_axis)
        dot_z = np.dot(v_proj, z_axis)
        angle = np.degrees(np.arctan2(dot_z, dot_y))

        # Map angle to direction
        if -90 <= angle < 45:
            return y_axis    # Right
        if 45 <= angle < 135:
            return z_axis    # Top
        return -y_axis                         # Left

    def _apply_obstacle_avoidance(self, waypoints: NDArray, obstacles: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Samples the current spline trajectory and modifies it if collisions are detected.
        
        Algorithm:
        1. Generate a temporary spline.
        2. Densely sample points along it.
        3. Check intersection with obstacles.
        4. If intersection found: Calculate Entry and Exit points.
        5. Insert a new point at the 'Bisector' of Entry/Exit, pushed to safety radius.
        """
        # Generate temporary spline to check for collisions
        temp_spline = self._compute_spline(self.FLIGHT_DURATION, waypoints)
        
        n_samples = int(self._control_freq * self.FLIGHT_DURATION)
        times = np.linspace(0, self.FLIGHT_DURATION, n_samples)
        points = temp_spline(times)

        # Iteratively clean the path against each obstacle
        for obs_center in obstacles:
            times, points = self._process_single_obstacle(obs_center, points, times)

        return times, points

    def _process_single_obstacle(self, center: NDArray, points: NDArray, times: NDArray) -> Tuple[NDArray, NDArray]:
        """Filters a path against a single obstacle, inserting avoidance points."""
        safe_times, safe_points = [], []
        inside_obs = False
        entry_idx = -1
        center_xy = center[:2]

        for i, pt in enumerate(points):
            dist = np.linalg.norm(pt[:2] - center_xy)
            
            if dist < self.OBSTACLE_CLEARANCE:
                if not inside_obs:
                    inside_obs = True
                    entry_idx = i
            elif inside_obs:
                # Exiting the obstacle zone
                inside_obs = False
                
                # Logic: Find the vector that bisects the entry and exit vectors.
                # This points away from the obstacle center in the direction of travel.
                entry_vec = points[entry_idx][:2] - center_xy
                exit_vec = points[i][:2] - center_xy
                
                avoid_vec = entry_vec + exit_vec
                avoid_norm = np.linalg.norm(avoid_vec)
                
                if avoid_norm > 1e-6:
                    avoid_vec /= avoid_norm
                else:
                    avoid_vec = np.array([1.0, 0.0]) # Fallback
                
                # Create Avoidance Point
                new_xy = center_xy + avoid_vec * self.OBSTACLE_CLEARANCE
                new_z = (points[entry_idx][2] + points[i][2]) / 2.0
                
                safe_points.append(np.array([*new_xy, new_z]))
                
                # Average time (approximating constant speed through turn)
                safe_times.append((times[entry_idx] + times[i]) / 2.0)
            else:
                safe_points.append(pt)
                safe_times.append(times[i])

        return np.array(safe_times), np.array(safe_points)

    def _compute_spline(self, duration: float, points: NDArray, 
                       knots: Optional[NDArray] = None) -> CubicSpline:
        """Fits a Cubic Spline to the points."""
        if knots is not None:
            return CubicSpline(knots, points)

        # Heuristic: Distribute time based on Euclidean distance between points
        dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cum_dist = np.concatenate([[0], np.cumsum(dists)])
        knots = (cum_dist / cum_dist[-1]) * duration
        
        return CubicSpline(knots, points)

    def _extract_gate_frames(self, quats: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """Helper to convert quaternions to local axes (Normal, Y, Z)."""
        mats = Rotation.from_quat(quats).as_matrix()
        return mats[:, :, 0], mats[:, :, 1], mats[:, :, 2]

    def _visualize_path(self):
        """Draws the trajectory in the gym environment if available."""
        if self.env and self._trajectory_spline:
            try:
                # Draw dense line
                ts = np.linspace(0, self.FLIGHT_DURATION, 100)
                pts = self._trajectory_spline(ts)
                draw_line(self.env, pts, rgba=np.array([1.0, 1.0, 1.0, 0.2]))
            except (AttributeError, TypeError):
                pass