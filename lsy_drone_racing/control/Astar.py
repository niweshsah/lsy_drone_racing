from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AStarNode:
    """Helper class for A* nodes, supporting priority queue operations."""
    def __init__(
        self, 
        position: NDArray[np.floating], 
        grid_index: tuple[int, int, int], 
        g_cost: float = np.inf, 
        h_cost: float = np.inf, 
        parent: AStarNode | None = None
    ):
        self.position = position  # World (x, y, z)
        self.grid_index = grid_index  # Grid (i, j, k)
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent  # Parent node for path reconstruction

    def __lt__(self, other: AStarNode) -> bool:
        """Comparison for priority queue (min-heap)."""
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        """Equality check based on grid index."""
        if not isinstance(other, AStarNode):
            return False
        return self.grid_index == other.grid_index

    def __hash__(self) -> int:
        """Hash based on grid index."""
        return hash(self.grid_index)


class MyAStarController(Controller):
    """A* path planning controller for drone racing.
    
    This controller discretizes the environment into a 3D grid, uses A* to 
    find a collision-free path through the gates, simplifies the path, and 
    then generates a smooth CubicSpline trajectory to follow.
    """

    # Class constants
    TRAJECTORY_DURATION = 15.0  # Total trajectory duration in seconds
    STATE_DIMENSION = 13  # [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
    OBSTACLE_SAFETY_DISTANCE = 0.4  # Inflated radius for obstacles
    VISUALIZATION_SAMPLES = 100  # Number of points for trajectory visualization
    LOG_INTERVAL = 100  # Print debug info every N ticks
    
    # A* and Grid Configuration
    GRID_RESOLUTION = 0.25  # Size of each grid voxel (in meters)
    GRID_PADDING = 2.0  # Padding around known obstacles/gates (in meters)
    # Define 26-connected neighbors (3D)
    A_STAR_NEIGHBORS = np.array(
        [(i, j, k) for i in [-1, 0, 1] 
                   for j in [-1, 0, 1] 
                   for k in [-1, 0, 1] 
                   if (i, j, k) != (0, 0, 0)]
    )
    LINE_OF_SIGHT_SAMPLES = 10 # Samples for line-of-sight check

    def __init__(
        self, 
        obs: dict[str, NDArray[np.floating]], 
        info: dict, 
        config: dict
    ):
        """Initialize the controller."""
        super().__init__(obs, info, config)
        
        # Controller state
        self._time_step = 0
        self._control_frequency = config.env.freq
        self._is_finished = False
        
        # Environment state tracking
        self._last_gate_flags = None
        self._last_obstacle_flags = None
        
        # Extract environment features
        self.gate_positions = obs['gates_pos']
        self.gate_normals = self._extract_gate_normals(obs['gates_quat'])
        self.obstacle_positions = obs['obstacles_pos']
        self.initial_position = obs['pos']
        
        # Enable visualization (trajectory plotting)
        self.visualization = False
        self.fig = None
        self.ax = None
        
        # --- A* Planning Initialization ---
        
        # 1. Define grid boundaries
        self._calculate_grid_bounds()
        
        # 2. Create 3D occupancy grid
        self._create_occupancy_grid()
        
        # 3. Plan the full path from start to last gate
        print("Planning initial race path...")
        self.waypoints = self._plan_full_race_path(self.initial_position)
        if not self.waypoints:
            print("!!! A* PLANNING FAILED: No path found. !!!")
            # Fallback: Use a simple, non-collision-aware path
            self.waypoints = np.vstack([self.initial_position, self.gate_positions])
        else:
            print(f"A* path planned with {len(self.waypoints)} waypoints.")

        # 4. Generate smooth trajectory from A* waypoints
        self.trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, self.waypoints)
        
        # 5. Initialize visualization
        if self.visualization:
            self._visualize_trajectory(
                self.gate_positions,
                self.gate_normals,
                obstacle_positions=self.obstacle_positions,
                trajectory=self.trajectory,
                waypoints=self.waypoints,
                drone_position=obs['pos']
            )

    # ========================================================================
    # == A* and Grid Methods
    # ========================================================================

    def _calculate_grid_bounds(self) -> None:
        """Determine the min/max (x, y, z) bounds for the 3D grid."""
        all_points = np.vstack([
            self.initial_position,
            self.gate_positions,
            self.obstacle_positions
        ])
        min_bounds = np.min(all_points, axis=0) - self.GRID_PADDING
        max_bounds = np.max(all_points, axis=0) + self.GRID_PADDING
        
        # Ensure Z bound starts from ground (or slightly below)
        min_bounds[2] = min(min_bounds[2], 0.0)
        
        self.grid_min_bounds = min_bounds
        self.grid_shape = tuple(
            np.ceil((max_bounds - min_bounds) / self.GRID_RESOLUTION).astype(int)
        )
        print(f"Grid Initialized: Shape={self.grid_shape}, Res={self.GRID_RESOLUTION}m")

    def _create_occupancy_grid(self) -> None:
        """Create a 3D boolean grid, marking obstacle cells as True."""
        self.grid = np.zeros(self.grid_shape, dtype=bool)
        
        # Get grid indices for all obstacles
        obstacle_indices = self._world_to_grid(self.obstacle_positions)
        
        # Define the "brush" size for obstacle inflation
        brush_radius = int(np.ceil(self.OBSTACLE_SAFETY_DISTANCE / self.GRID_RESOLUTION))
        
        # Create a coordinate grid for the brush
        x, y, z = np.indices((2 * brush_radius + 1, 
                              2 * brush_radius + 1, 
                              2 * brush_radius + 1))
        brush_indices = np.stack([x, y, z], axis=-1).reshape(-1, 3) - brush_radius
        
        # Calculate distances for cylindrical check
        brush_distances_xy = np.linalg.norm(brush_indices[:, :2], axis=1) * self.GRID_RESOLUTION
        
        # Filter brush to only indices within the safety distance (cylindrical)
        valid_brush_indices = brush_indices[brush_distances_xy < self.OBSTACLE_SAFETY_DISTANCE]

        for obs_idx in obstacle_indices:
            # Apply the brush around each obstacle's grid index
            cells_to_occupy_indices = obs_idx + valid_brush_indices
            
            # Clip indices to be within grid bounds
            valid_mask = np.all((cells_to_occupy_indices >= 0) & 
                                (cells_to_occupy_indices < self.grid_shape), axis=1)
            valid_cells = cells_to_occupy_indices[valid_mask]
            
            # Mark valid cells as occupied
            self.grid[valid_cells[:, 0], valid_cells[:, 1], valid_cells[:, 2]] = True

    def _plan_full_race_path(self, start_pos: NDArray[np.floating]) -> NDArray[np.floating] | None:
        """Plan a complete path from start, through all gates."""
        all_waypoints = [start_pos]
        current_start_pos = start_pos
        
        all_goals = self.gate_positions
        
        for i, goal_pos in enumerate(all_goals):
            print(f"  Planning segment {i}: Start -> Gate {i}")
            
            # 1. Run A* to find a path of grid cells
            path_segment_world = self._a_star_search(current_start_pos, goal_pos)
            
            if path_segment_world is None:
                print(f"  [!] A* failed for segment {i}. Aborting plan.")
                return None  # Failed to find a path
            
            # 2. Simplify the path
            simplified_segment = self._simplify_path(path_segment_world)
            
            # Add all but the first point (to avoid duplicates)
            all_waypoints.extend(simplified_segment[1:])
            
            # Next segment starts from the end of this one
            current_start_pos = simplified_segment[-1]

        return np.array(all_waypoints)

    def _a_star_search(
        self, 
        start_pos_world: NDArray[np.floating], 
        goal_pos_world: NDArray[np.floating]
    ) -> list[NDArray[np.floating]] | None:
        """Find the shortest path from start to goal using A*."""
        
        start_index = self._world_to_grid(start_pos_world)
        goal_index = self._world_to_grid(goal_pos_world)
        
        if self._is_occupied(start_index) or not self._is_valid_grid_index(start_index):
            print("[A* Error] Start position is in an obstacle or out of bounds.")
            return None
        if not self._is_valid_grid_index(goal_index):
            print("[A* Error] Goal position is out of bounds.")
            return None
            
        start_node = AStarNode(
            position=start_pos_world,
            grid_index=start_index,
            g_cost=0.0,
            h_cost=self._heuristic(start_index, goal_index)
        )
        
        open_set: list[AStarNode] = [start_node]  # Min-heap
        closed_set: set[tuple[int, int, int]] = set()
        
        # Map to track the best node to reach a grid index
        g_costs = {start_index: 0.0}

        while open_set:
            current_node = heapq.heappop(open_set)
            
            if current_node.grid_index == goal_index:
                return self._reconstruct_path(current_node)
            
            if current_node.grid_index in closed_set:
                continue
            closed_set.add(current_node.grid_index)
            
            for neighbor_offset in self.A_STAR_NEIGHBORS:
                neighbor_index = tuple(current_node.grid_index + neighbor_offset)
                
                if not self._is_valid_grid_index(neighbor_index) or \
                   self._is_occupied(neighbor_index):
                    continue
                
                new_g_cost = current_node.g_cost + self._cost_between(
                    current_node.grid_index, neighbor_index
                )
                
                if new_g_cost < g_costs.get(neighbor_index, np.inf):
                    g_costs[neighbor_index] = new_g_cost
                    neighbor_pos = self._grid_to_world(neighbor_index)
                    
                    neighbor_node = AStarNode(
                        position=neighbor_pos,
                        grid_index=neighbor_index,
                        g_cost=new_g_cost,
                        h_cost=self._heuristic(neighbor_index, goal_index),
                        parent=current_node
                    )
                    heapq.heappush(open_set, neighbor_node)
                    
        return None  # No path found

    def _reconstruct_path(self, end_node: AStarNode) -> list[NDArray[np.floating]]:
        """Trace parents from end_node to start_node to build the path."""
        path = []
        current = end_node
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Reverse to get start-to-end

    def _simplify_path(self, path: list[NDArray[np.floating]]) -> list[NDArray[np.floating]]:
        """Prune redundant waypoints from a path using line-of-sight."""
        if len(path) < 3:
            return path
            
        simplified_path = [path[0]]
        last_waypoint_node = path[0]
        
        i = 1
        while i < len(path) - 1:
            if not self._check_line_of_sight(last_waypoint_node, path[i+1]):
                # Obstacle in the way, so we *must* go through the previous node
                simplified_path.append(path[i])
                last_waypoint_node = path[i]
            i += 1
            
        simplified_path.append(path[-1])  # Add the final goal
        return simplified_path

    def _check_line_of_sight(
        self, 
        pos1: NDArray[np.floating], 
        pos2: NDArray[np.floating]
    ) -> bool:
        """Check if any grid cell on the line from pos1 to pos2 is occupied."""
        samples = np.linspace(0, 1, self.LINE_OF_SIGHT_SAMPLES)
        points_on_line = pos1 + (pos2 - pos1) * samples[:, np.newaxis]
        
        for point in points_on_line:
            grid_idx = self._world_to_grid(point)
            if not self._is_valid_grid_index(grid_idx) or self._is_occupied(grid_idx):
                return False  # Hit an obstacle or out of bounds
        return True

    # --- A* and Grid Helpers ---

    def _heuristic(self, idx1: tuple[int, int, int], idx2: tuple[int, int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.linalg.norm((np.array(idx1) - np.array(idx2))) * self.GRID_RESOLUTION

    def _cost_between(self, idx1: tuple[int, int, int], idx2: tuple[int, int, int]) -> float:
        """Cost to move between adjacent grid cells (Euclidean distance)."""
        return np.linalg.norm((np.array(idx1) - np.array(idx2))) * self.GRID_RESOLUTION

    def _world_to_grid(self, world_pos: NDArray[np.floating]) -> tuple[int, int, int] | NDArray:
        """Convert (x, y, z) world coordinates to (i, j, k) grid indices."""
        if world_pos.ndim == 1:
            idx = (world_pos - self.grid_min_bounds) / self.GRID_RESOLUTION
            return tuple(idx.astype(int))
        else: # Batch conversion
            idx = (world_pos - self.grid_min_bounds) / self.GRID_RESOLUTION
            return idx.astype(int)

    def _grid_to_world(self, grid_index: tuple[int, int, int]) -> NDArray[np.floating]:
        """Convert (i, j, k) grid index to (x, y, z) world coordinates (cell center)."""
        return (np.array(grid_index) + 0.5) * self.GRID_RESOLUTION + self.grid_min_bounds

    def _is_valid_grid_index(self, grid_index: tuple[int, int, int]) -> bool:
        """Check if a grid index is within the grid's bounds."""
        return (0 <= grid_index[0] < self.grid_shape[0] and
                0 <= grid_index[1] < self.grid_shape[1] and
                0 <= grid_index[2] < self.grid_shape[2])

    def _is_occupied(self, grid_index: tuple[int, int, int]) -> bool:
        """Check if a grid cell is marked as an obstacle."""
        return self.grid[grid_index]

    # ========================================================================
    # == Original Methods (Mostly Unchanged)
    # ========================================================================

    def _extract_gate_normals(self, gates_quaternions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Extract gate normal vectors from quaternions."""
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        return rotation_matrices[:, :, 0]

    def _generate_trajectory(
        self, 
        duration: float, 
        waypoints: NDArray[np.floating]
    ) -> CubicSpline:
        """Generate a cubic spline trajectory through waypoints."""
        # Calculate segment lengths
        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        # Cumulative arc length
        cumulative_arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        # Handle case of single waypoint (or no movement)
        if cumulative_arc_length[-1] == 0:
            cumulative_arc_length = np.linspace(0, 1, len(waypoints))
            
        # Parameterize time by arc length for uniform velocity
        time_parameters = cumulative_arc_length / cumulative_arc_length[-1] * duration
        
        # Ensure time_parameters is strictly increasing for CubicSpline
        if len(time_parameters) > 1:
            for i in range(1, len(time_parameters)):
                if time_parameters[i] <= time_parameters[i-1]:
                    time_parameters[i] = time_parameters[i-1] + 1e-6
        
        if len(waypoints) < 2:
            # Spline requires at least 2 points
            waypoints = np.vstack([waypoints, waypoints[0] + 1e-6])
            time_parameters = np.array([0.0, duration])

        return CubicSpline(time_parameters, waypoints)

    def _visualize_trajectory(
        self,
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating] = None,
        trajectory: CubicSpline = None,
        waypoints: NDArray[np.floating] = None,
        drone_position: NDArray[np.floating] = None
    ) -> None:
        """Visualize trajectory, gates, obstacles, and drone in 3D."""
        if plt is None:
            return
        
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(num=1, figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.cla()
        
        # Plot waypoints
        if waypoints is not None:
            self.ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                        marker='.', linestyle='--', color='blue', 
                        label=f'A* Waypoints ({len(waypoints)})', linewidth=1)
        
        # Plot smooth trajectory
        if trajectory is not None:
            t_samples = np.linspace(0, self.TRAJECTORY_DURATION, self.VISUALIZATION_SAMPLES)
            traj_points = trajectory(t_samples)
            self.ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2],
                        linestyle='-', color='orange',
                        label='Spline Trajectory', linewidth=2)
        
        # Plot gates
        for pos, normal in zip(gate_positions, gate_normals):
            self.ax.quiver(pos[0], pos[1], pos[2],
                          normal[0], normal[1], normal[2],
                          length=0.5, color='green', linewidth=1.5, arrow_length_ratio=0.3)
        
        # Plot obstacles
        if obstacle_positions is not None:
            for obs_pos in obstacle_positions:
                self.ax.plot([obs_pos[0], obs_pos[0]], 
                           [obs_pos[1], obs_pos[1]], 
                           [0, 1.4], # Assuming 1.4m height
                           color='grey', linewidth=5, alpha=0.6)
        
        # Plot drone
        if drone_position is not None:
            self.ax.scatter(drone_position[0], drone_position[1], drone_position[2],
                          marker='x', s=200, color='red', linewidths=3,
                          label='Drone')
        
        self.ax.set_title("A* Planned Trajectory Visualization", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("X (m)"), self.ax.set_ylabel("Y (m)"), self.ax.set_zlabel("Z (m)")
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        if waypoints is not None:
            all_pts = waypoints
            if drone_position is not None:
                all_pts = np.vstack([all_pts, drone_position])
            max_range = np.array([all_pts[:, 0].max() - all_pts[:, 0].min(),
                                all_pts[:, 1].max() - all_pts[:, 1].min(),
                                all_pts[:, 2].max() - all_pts[:, 2].min()]).max() / 2.0
            
            mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) * 0.5
            mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) * 0.5
            mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) * 0.5
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def _detect_environment_change(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Detect if any gate or obstacle position has changed."""
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
        """Replan trajectory when environment changes."""
        print(f"\n[REPLANNING] Time: {current_time:.2f}s")
        
        # 1. Update environment state
        self.gate_positions = obs['gates_pos']
        self.gate_normals = self._extract_gate_normals(obs['gates_quat'])
        self.obstacle_positions = obs['obstacles_pos']
        current_drone_pos = obs['pos']

        # 2. Re-build the occupancy grid (in case obstacles moved)
        # Note: Grid bounds do not change
        self._create_occupancy_grid()
        
        # 3. Re-plan the full path from the *current position*
        print("Re-planning new A* path...")
        self.waypoints = self._plan_full_race_path(current_drone_pos)
        
        if not self.waypoints:
            print("!!! A* RE-PLANNING FAILED. Using old trajectory. !!!")
            # If replan fails, just keep going on the old path
            return
        
        print(f"New path planned with {len(self.waypoints)} waypoints.")

        # 4. Generate a new trajectory
        self.trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, self.waypoints)
        
        # 5. Reset the time step to start at the beginning of the new trajectory
        self._time_step = 0

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state for the drone."""
        
        # Compute current time along trajectory
        current_time = min(self._time_step / self._control_frequency, self.TRAJECTORY_DURATION)
        
        # Check for environment changes and replan if necessary
        # We must check *before* sampling the trajectory
        if self._detect_environment_change(obs):
            self._replan_trajectory(obs, current_time)
            # After replanning, time is reset
            current_time = min(self._time_step / self._control_frequency, self.TRAJECTORY_DURATION)
        
        # Sample target position from the (potentially new) trajectory
        target_position = self.trajectory(current_time)
        
        # Periodic logging
        if self._time_step % self.LOG_INTERVAL == 0:
            print(f"Time: {current_time:.2f}s | "
                  f"Target: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        # Update visualization
        if self.visualization:
            self._visualize_trajectory(
                self.gate_positions,
                self.gate_normals,
                obstacle_positions=obs['obstacles_pos'],
                trajectory=self.trajectory,
                waypoints=self.waypoints,
                drone_position=obs['pos']
            )
            
        # Check if trajectory is complete
        if current_time >= self.TRAJECTORY_DURATION:
            self._is_finished = True
        
        # Draw trajectory in simulation
        try:
            draw_line(self.env, self.trajectory(self.trajectory.x), 
                     rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        except (AttributeError, TypeError):
            pass
        
        # Return 13D state
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
        """Called after each environment step."""
        self._time_step += 1
        return self._is_finished

    # ==================== Utility Methods for External Use ====================
    
    def get_trajectory_function(self) -> CubicSpline:
        """Get the trajectory spline function."""
        return self.trajectory

    def get_trajectory_waypoints(self) -> NDArray[np.floating]:
        """Get discrete waypoints sampled from trajectory at control frequency."""
        time_samples = np.linspace(0, self.TRAJECTORY_DURATION,
                                   int(self._control_frequency * self.TRAJECTORY_DURATION))
        return self.trajectory(time_samples)

    def set_time_step(self, time_step: int) -> None:
        """Set the current time step (for testing/debugging)."""
        self._time_step = time_step