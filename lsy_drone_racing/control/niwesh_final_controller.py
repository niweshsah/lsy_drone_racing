from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line


if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):
    
    TRAJECTORY_DURATION = 18.0
    STATE_DIMENSION = 13
    OBSTACLE_SAFETY_DISTANCE = 0.3
    VISUALIZATION_SAMPLES = 100
    LOG_INTERVAL = 100

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict
    ):
        super().__init__(obs, info, config)
        
        self._time_step = 0
        self._control_frequency = config.env.freq
        self._is_finished = False
        
        self._last_gate_flags = None
        self._last_obstacle_flags = None

        self._debug_detour_analysis = []
        self._debug_detour_summary = {}
        self._debug_detour_waypoints_added = []
        self._debug_waypoints_initial = None
        self._debug_waypoints_after_detour = None
        self._debug_waypoints_final = None

        self.gate_positions = obs['gates_pos']
        self.gate_normals, self.gate_y_axes, self.gate_z_axes = \
        self._extract_gate_coordinate_frames(obs['gates_quat'])
        
        
        self.obstacle_positions = obs['obstacles_pos']
        
        self.initial_position = obs['pos']
        
        self.visualization = False

        waypoints = self.calc_waypoints_from_gates(
            self.initial_position,
            self.gate_positions,
            self.gate_normals,
            approach_distance=0.5,
            num_intermediate_points=5
        )
        
        self._debug_waypoints_initial = waypoints.copy()
        
        waypoints = self._add_detour_waypoints(
            waypoints,
            self.gate_positions,
            self.gate_normals,
            self.gate_y_axes,
            self.gate_z_axes,
            num_intermediate_points=5,
            angle_threshold=120.0,
            detour_distance=0.65
        )

        self._debug_waypoints_after_detour = waypoints.copy()
        
        time_params, waypoints = self._avoid_collisions(
            waypoints,
            self.obstacle_positions,
            self.OBSTACLE_SAFETY_DISTANCE
        )

        self._debug_waypoints_final = waypoints.copy()
        
        self.trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, waypoints)
        
        self.fig = None
        self.ax = None


    def _extract_gate_normals(self, gates_quaternions: NDArray[np.floating]) -> NDArray[np.floating]:
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        return rotation_matrices[:, :, 0]

    def calc_waypoints_from_gates(
        self,
        initial_position: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        approach_distance: float = 0.5,
        num_intermediate_points: int = 5
    ) -> NDArray[np.floating]:
        num_gates = gate_positions.shape[0]
        
        waypoints_per_gate = []
        for i in range(num_intermediate_points):
            offset = -approach_distance + (i / (num_intermediate_points - 1)) * 2 * approach_distance
            waypoints_per_gate.append(gate_positions + offset * gate_normals)
        
        waypoints = np.concatenate(waypoints_per_gate, axis=1)
        waypoints = waypoints.reshape(num_gates, num_intermediate_points, 3).reshape(-1, 3)
        
        waypoints = np.vstack([initial_position, waypoints])
        
        return waypoints

    def _generate_trajectory(
        self,
        duration: float,
        waypoints: NDArray[np.floating]
    ) -> CubicSpline:
        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        cumulative_arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        time_parameters = cumulative_arc_length / cumulative_arc_length[-1] * duration
        
        return CubicSpline(time_parameters, waypoints)

    def _avoid_collisions(
        self,
        waypoints: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating],
        safety_distance: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, waypoints)
        
        time_samples = np.linspace(0, self.TRAJECTORY_DURATION,
                                  int(self._control_frequency * self.TRAJECTORY_DURATION))
        trajectory_points = trajectory(time_samples)
        
        for obstacle_position in obstacle_positions:
            collision_free_times = []
            collision_free_waypoints = []
            
            is_inside_obstacle = False
            entry_index = None
            
            for i, point in enumerate(trajectory_points):
                distance_xy = np.linalg.norm(obstacle_position[:2] - point[:2])
                
                if distance_xy < safety_distance:
                    if not is_inside_obstacle:
                        is_inside_obstacle = True
                        entry_index = i
                        
                elif is_inside_obstacle:
                    exit_index = i
                    is_inside_obstacle = False
                    
                    entry_point = trajectory_points[entry_index]
                    exit_point = trajectory_points[exit_index]
                    
                    entry_direction = entry_point[:2] - obstacle_position[:2]
                    exit_direction = exit_point[:2] - obstacle_position[:2]
                    avoidance_direction = entry_direction + exit_direction
                    avoidance_direction /= np.linalg.norm(avoidance_direction)
                    
                    new_position_xy = obstacle_position[:2] + avoidance_direction * safety_distance
                    new_position_z = (entry_point[2] + exit_point[2]) / 2
                    new_waypoint = np.concatenate([new_position_xy, [new_position_z]])
                    
                    collision_free_times.append((time_samples[entry_index] + time_samples[exit_index]) / 2)
                    collision_free_waypoints.append(new_waypoint)
                    
                else:
                    collision_free_times.append(time_samples[i])
                    collision_free_waypoints.append(point)
            
            time_samples = np.array(collision_free_times)
            trajectory_points = np.array(collision_free_waypoints)
        
        return time_samples, trajectory_points

    def _detect_environment_change(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        if self._last_gate_flags is None:
            self._last_gate_flags = np.array(obs['gates_visited'], dtype=bool)
            self._last_obstacle_flags = np.array(obs['obstacles_visited'], dtype=bool)
            return False
        
        current_gate_flags = np.array(obs['gates_visited'], dtype=bool)
        current_obstacle_flags = np.array(obs['obstacles_visited'], dtype=bool)
        
        gate_newly_visited = np.any((~self._last_gate_flags) & current_gate_flags)
        obstacle_newly_visited = np.any((~self._last_obstacle_flags) & current_obstacle_flags)
        
        self._last_gate_flags = current_gate_flags
        self._last_obstacle_flags = current_obstacle_flags
        
        return gate_newly_visited or obstacle_newly_visited

    def _replan_trajectory(self, obs: dict[str, NDArray[np.floating]], current_time: float) -> None:
        
        self.gate_positions = obs['gates_pos']
        self.gate_normals, self.gate_y_axes, self.gate_z_axes = \
            self._extract_gate_coordinate_frames(obs['gates_quat'])
        
        waypoints = self.calc_waypoints_from_gates(
            self.initial_position,
            self.gate_positions,
            self.gate_normals,
            approach_distance=0.5,
            num_intermediate_points=5
        )
        
        waypoints = self._add_detour_waypoints(
            waypoints,
            self.gate_positions,
            self.gate_normals,
            self.gate_y_axes,
            self.gate_z_axes,
            num_intermediate_points=5,
            angle_threshold=120.0,
            detour_distance=0.65
        )
        
        _, waypoints = self._avoid_collisions(
            waypoints,
            obs['obstacles_pos'],
            self.OBSTACLE_SAFETY_DISTANCE
        )
        
        self.trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, waypoints)

    def _extract_gate_coordinate_frames(
        self,
        gates_quaternions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        
        normals = rotation_matrices[:, :, 0]
        y_axes = rotation_matrices[:, :, 1]
        z_axes = rotation_matrices[:, :, 2]
        
        return normals, y_axes, z_axes

    def _add_detour_waypoints(
        self,
        waypoints: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: int = 5,
        angle_threshold: float = 120.0,
        detour_distance: float = 0.65
    ) -> NDArray[np.floating]:
        num_gates = gate_positions.shape[0]
        waypoints_list = list(waypoints)
        
        inserted_count = 0
        
        self._debug_detour_analysis = []
        
        self._debug_detour_waypoints_added = []
        
        
        for i in range(num_gates - 1):
            debug_info = {
                'gate_i': i,
                'gate_i_plus_1': i + 1,
            }
            
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + inserted_count
            
            debug_info['last_idx_gate_i'] = last_idx_gate_i
            debug_info['first_idx_gate_i_plus_1'] = first_idx_gate_i_plus_1
            
            p1 = waypoints_list[last_idx_gate_i]
            p2 = waypoints_list[first_idx_gate_i_plus_1]
            
            debug_info['p1_last_of_gate_i'] = p1.copy()
            debug_info['p2_first_of_gate_i_plus_1'] = p2.copy()
            
            v = p2 - p1
            v_norm = np.linalg.norm(v)
            
            debug_info['vector_p1_to_p2'] = v.copy()
            debug_info['vector_norm'] = v_norm
            
            if v_norm < 1e-6:
                debug_info['skipped'] = True
                debug_info['skip_reason'] = 'vector_too_short'
                self._debug_detour_analysis.append(debug_info)
                continue
            
            normal_i = gate_normals[i]
            cos_angle = np.dot(v, normal_i) / v_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            
            debug_info['gate_i_normal'] = normal_i.copy()
            debug_info['cos_angle'] = cos_angle
            debug_info['angle_degrees'] = angle_deg
            debug_info['angle_threshold'] = angle_threshold
            
            
            if angle_deg > angle_threshold:
                
                debug_info['needs_detour'] = True
                
                gate_center = gate_positions[i]
                y_axis = gate_y_axes[i]
                z_axis = gate_z_axes[i]
                
                debug_info['gate_i_center'] = gate_center.copy()
                debug_info['gate_i_y_axis'] = y_axis.copy()
                debug_info['gate_i_z_axis'] = z_axis.copy()
                debug_info['detour_distance'] = detour_distance
                                
                v_proj = v - np.dot(v, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)
                
                debug_info['v_projection_on_gate_plane'] = v_proj.copy()
                debug_info['v_projection_norm'] = v_proj_norm
                
                if v_proj_norm < 1e-6:
                    detour_direction_vector = y_axis
                    detour_direction_name = 'right (+y_axis) [default]'
                    proj_angle_deg = 0.0
                else:
                    v_proj_y = np.dot(v_proj, y_axis)
                    v_proj_z = np.dot(v_proj, z_axis)
                    
                    debug_info['v_proj_y_component'] = v_proj_y
                    debug_info['v_proj_z_component'] = v_proj_z
                    
                    proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi
                    
                    debug_info['projection_angle_degrees'] = proj_angle_deg
                    
                    if -90 <= proj_angle_deg < 45:
                        detour_direction_vector = y_axis
                        detour_direction_name = 'right (+y_axis)'
                    elif 45 <= proj_angle_deg < 135:
                        detour_direction_vector = z_axis
                        detour_direction_name = 'top (+z_axis)'
                    else:
                        detour_direction_vector = -y_axis
                        detour_direction_name = 'left (-y_axis)'
                    
                
                debug_info['detour_direction_vector'] = detour_direction_vector.copy()
                debug_info['detour_direction_name'] = detour_direction_name
                debug_info['projection_angle_degrees'] = proj_angle_deg
                
                detour_waypoint = gate_center + detour_distance * detour_direction_vector
                
                debug_info['detour_waypoint'] = detour_waypoint.copy()
                debug_info['detour_direction'] = detour_direction_name
                
                insert_position = last_idx_gate_i + 1
                waypoints_list.insert(insert_position, detour_waypoint)
                inserted_count += 1
                
                debug_info['insert_position'] = insert_position
                debug_info['inserted'] = True
                
            else:
                debug_info['needs_detour'] = False
                debug_info['inserted'] = False
            
            debug_info['total_inserted_so_far'] = inserted_count
            
            self._debug_detour_analysis.append(debug_info)
        
        self._debug_detour_summary = {
            'total_detours_added': inserted_count,
            'original_waypoint_count': len(waypoints),
            'final_waypoint_count': len(waypoints_list),
            'num_gate_pairs_checked': num_gates - 1,
            'detour_waypoints': self._debug_detour_waypoints_added
        }
        
        
        return np.array(waypoints_list)
    
    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        current_time = min(self._time_step / self._control_frequency, self.TRAJECTORY_DURATION)
        
        target_position = self.trajectory(current_time)
        
        
        if self._detect_environment_change(obs):
            self._replan_trajectory(obs, current_time)
        
        if current_time >= self.TRAJECTORY_DURATION:
            self._is_finished = True
        
        try:
            draw_line(self.env, self.trajectory(self.trajectory.x),
                      rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        except (AttributeError, TypeError):
            pass
        
        return np.concatenate((target_position, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict
    ) -> bool:
        self._time_step += 1
        return self._is_finished

    
    def get_trajectory_function(self) -> CubicSpline:
        return self.trajectory

    def get_trajectory_waypoints(self) -> NDArray[np.floating]:
        time_samples = np.linspace(0, self.TRAJECTORY_DURATION,
                                  int(self._control_frequency * self.TRAJECTORY_DURATION))
        return self.trajectory(time_samples)

    def set_time_step(self, time_step: int) -> None:
        self._time_step = time_step