from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # Must be first

# --- MODIFIED: Added Rotation, added time for planner debugging ---
import time  # For planner timing
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

# ----------------------------------------------------------------------------
# ## 🚀 NEW: RRT* Planner Implementation
# ----------------------------------------------------------------------------


class RRTStarPlanner:
    """Implements the RRT* path planning algorithm in 3D space.

    This planner finds an obstacle-free, optimized path from a start point
    to a goal point within given boundaries.
    """

    class Node:
        """Helper class to represent a node in the RRT* tree."""

        def __init__(
            self,
            pos: NDArray[np.floating],
            parent: "RRTStarPlanner.Node" | None = None,
            cost: float = 0.0,
        ):
            self.pos = pos
            self.parent = parent
            self.cost = cost  # Total cost (distance) from the start node

    def __init__(
        self,
        obstacles: List[dict[str, NDArray[np.floating] | float]],
        bounds: Tuple[NDArray[np.floating], NDArray[np.floating]],
        step_size: float = 0.5,
        search_radius: float = 1.5,
        max_iter: int = 2000,
        goal_bias: float = 0.1,
    ):
        """Initialize the RRT* planner.

        Args:
            obstacles: List of obstacle dicts, e.g., [{'pos': [x,y,z], 'radius': r}, ...]
            bounds: Tuple (min_bounds, max_bounds) defining the 3D search space.
            step_size: Max distance to "steer" from a node to a random point.
            search_radius: Radius for finding "near" nodes for RRT* rewiring.
            max_iter: Maximum number of iterations to run the planning.
            goal_bias: Probability (0-1) of sampling the goal point directly.
        """
        self.obstacles = obstacles
        self.min_bounds, self.max_bounds = bounds
        self.step_size = step_size
        self.search_radius = search_radius
        self.max_iter = max_iter
        self.goal_bias = goal_bias

        # Pre-calculate bounds dimension and range for sampling
        self.dims = len(self.min_bounds)
        self.range = self.max_bounds - self.min_bounds

    def plan(
        self, start: NDArray[np.floating], goal: NDArray[np.floating]
    ) -> List[NDArray[np.floating]]:
        """Finds a path from start to goal."""
        start_time = time.time()
        start_node = self.Node(start, cost=0.0)
        self.nodes = [start_node]

        goal_node = None

        for i in range(self.max_iter):
            # 1. Sample a random point
            rand_point = self._sample_space(goal)

            # 2. Find the nearest node in the tree
            nearest_node = self._find_nearest(rand_point)

            # 3. Steer from nearest node towards the random point
            new_point = self._steer(nearest_node.pos, rand_point)

            # 4. Check for collision
            if self._is_collision(nearest_node.pos, new_point):
                continue

            # 5. RRT* Core: Choose best parent and rewire
            new_node = self._choose_parent_and_rewire(nearest_node, new_point)

            if new_node:
                self.nodes.append(new_node)

                # 6. Check if goal is reachable from the new node
                if goal_node is None or new_node.cost < goal_node.cost - self._get_line_cost(
                    new_node.pos, goal
                ):
                    if self._get_line_cost(
                        new_node.pos, goal
                    ) < self.step_size and not self._is_collision(new_node.pos, goal):
                        # Found a path to the goal!
                        cost_to_goal = new_node.cost + self._get_line_cost(new_node.pos, goal)

                        if goal_node is None or cost_to_goal < goal_node.cost:
                            goal_node = self.Node(goal, parent=new_node, cost=cost_to_goal)
                            self.nodes.append(goal_node)  # Add goal to tree

        # --- Path Reconstruction ---
        if goal_node is None:
            # Fallback: if goal was never reached, find the node closest to it
            print("RRT* WARNING: Goal not directly reached. Finding closest node.")
            closest_node = min(self.nodes, key=lambda n: self._get_line_cost(n.pos, goal))
            if self._is_collision(closest_node.pos, goal):
                # If path to goal is blocked, just use the closest node
                goal_node = closest_node
            else:
                # If path is clear, connect it
                goal_node = self.Node(
                    goal,
                    parent=closest_node,
                    cost=closest_node.cost + self._get_line_cost(closest_node.pos, goal),
                )

        path = self._reconstruct_path(goal_node)
        print(f"RRT*: Planning took {time.time() - start_time:.2f}s, {len(path)} waypoints.")
        return path

    def _sample_space(self, goal: NDArray[np.floating]) -> NDArray[np.floating]:
        """Sample a random point, with a bias towards the goal."""
        if np.random.rand() < self.goal_bias:
            return goal
        return self.min_bounds + np.random.rand(self.dims) * self.range

    def _find_nearest(self, point: NDArray[np.floating]) -> Node:
        """Find the node in the tree closest to the given point."""
        dists = [self._get_line_cost(node.pos, point) for node in self.nodes]
        return self.nodes[np.argmin(dists)]

    def _steer(
        self, from_pos: NDArray[np.floating], to_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Create a new point `step_size` away from `from_pos` towards `to_pos`."""
        v = to_pos - from_pos
        dist = np.linalg.norm(v)
        if dist < self.step_size:
            return to_pos
        else:
            unit_v = v / dist
            return from_pos + unit_v * self.step_size

    def _is_collision(self, p1: NDArray[np.floating], p2: NDArray[np.floating]) -> bool:
        """Check if the line segment (p1, p2) collides with any obstacle.
        Uses a robust line-segment-to-sphere distance check.
        """
        for obs in self.obstacles:
            obs_pos = obs["pos"]
            obs_rad = obs["radius"]

            # Vector from p1 to p2
            v = p2 - p1
            # Vector from obs_pos to p1
            w = p1 - obs_pos

            v_dot_v = np.dot(v, v)
            if v_dot_v < 1e-9:  # p1 and p2 are the same point
                continue

            # Project obs_pos onto the *infinite* line (p1, p2)
            # t = -(w.v) / (v.v)
            t = -np.dot(w, v) / v_dot_v

            # Clamp t to the segment [0, 1]
            t_clamped = max(0, min(1, t))

            # Find the closest point on the segment to the obstacle center
            closest_point_on_segment = p1 + t_clamped * v

            # Check distance
            if np.linalg.norm(closest_point_on_segment - obs_pos) < obs_rad:
                return True  # Collision!

        return False  # No collisions

    def _find_near_nodes(self, point: NDArray[np.floating]) -> List[Node]:
        """Find all nodes within `search_radius` of the point."""
        near_nodes = []
        for node in self.nodes:
            if self._get_line_cost(node.pos, point) < self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def _get_cost(self, node: Node) -> float:
        """Get the cost (distance from start) of a node."""
        return node.cost

    def _get_line_cost(self, p1: NDArray[np.floating], p2: NDArray[np.floating]) -> float:
        """Get the cost (Euclidean distance) of a straight line."""
        return np.linalg.norm(p1 - p2)

    def _choose_parent_and_rewire(
        self, nearest_node: Node, new_point: NDArray[np.floating]
    ) -> Node | None:
        """RRT* Core:
        1. Find all "near" nodes.
        2. Choose the best parent (lowest cost) from "near" nodes.
        3. Create the new node.
        4. Rewire the tree: check if any "near" nodes are now
           cheaper to reach via the new_node.
        """
        # 1. Find near nodes
        near_nodes = self._find_near_nodes(new_point)
        if not near_nodes:
            near_nodes = [nearest_node]  # Fallback if search_radius is too small

        # 2. Choose best parent
        best_parent = nearest_node
        min_cost = self._get_cost(nearest_node) + self._get_line_cost(nearest_node.pos, new_point)

        for near_node in near_nodes:
            cost_via_near = self._get_cost(near_node) + self._get_line_cost(
                near_node.pos, new_point
            )
            if cost_via_near < min_cost and not self._is_collision(near_node.pos, new_point):
                min_cost = cost_via_near
                best_parent = near_node

        # 3. Create the new node
        # Double-check final parent-child connection for collision (can happen if nearest_node wasn't in near_nodes)
        if self._is_collision(best_parent.pos, new_point):
            return None  # Failed to find a valid parent

        new_node = self.Node(new_point, parent=best_parent, cost=min_cost)

        # 4. Rewire
        for near_node in near_nodes:
            if near_node == best_parent:
                continue

            cost_via_new = self._get_cost(new_node) + self._get_line_cost(
                new_node.pos, near_node.pos
            )

            if cost_via_new < self._get_cost(near_node) and not self._is_collision(
                new_node.pos, near_node.pos
            ):
                near_node.parent = new_node
                near_node.cost = cost_via_new

        return new_node

    def _reconstruct_path(self, goal_node: Node) -> List[NDArray[np.floating]]:
        """Trace parents back from goal node to start node."""
        path = []
        current = goal_node
        while current is not None:
            path.append(current.pos)
            current = current.parent
        return path[::-1]  # Reverse to get [start, ..., goal]


# ----------------------------------------------------------------------------
# ## UPDATE: Modified Controller
# ----------------------------------------------------------------------------


class PathFollowingController(Controller):
    """Controller that uses an RRT* planner to generate an obstacle-avoiding
    path through all gate centers, then follows that path.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        super().__init__(obs, info, config)

        # --- Common ---
        self._tick = 0
        self._finished = False
        self._initial_pos = obs["pos"][:3].copy()
        self.position_log: list[NDArray[np.floating]] = []

        # --- Path Generation (RRT*) ---
        print("Initializing RRT* Path Planner...")

        # Get gate and obstacle info from config
        # --- MODIFIED: Convert positions from list to np.array ---
        gate_centers = [np.array(g["pos"]) for g in config.env.track.gates]
        obstacle_positions = [
            np.array(o["pos"]) for o in getattr(config.env.track, "obstacles", [])
        ]
        # ---------------------------------------------------------

        # Define obstacles as spheres for the planner
        self.obstacle_list = []
        obstacle_radius = 0.35  # meters, give a small safety buffer
        for pos in obstacle_positions:
            self.obstacle_list.append({"pos": pos, "radius": obstacle_radius})

        # Define search space boundaries (auto-calculated from points)
        # We can stack _initial_pos (ndarray) with lists of ndarrays now
        all_points = np.vstack([gate_centers, obstacle_positions, self._initial_pos.reshape(1, 3)])
        min_bounds = np.min(all_points, axis=0) - 1.0  # 1m margin
        max_bounds = np.max(all_points, axis=0) + 1.0  # 1m margin
        min_bounds[2] = 0.1  # Don't plan through the floor
        max_bounds[2] = max(max_bounds[2], 3.0)  # Ensure at least 3m high

        # Instantiate planner
        # These parameters may need tuning for performance vs. path quality
        self.planner = RRTStarPlanner(
            obstacles=self.obstacle_list,
            bounds=(min_bounds, max_bounds),
            step_size=0.4,  # RRT* "steer" distance (meters)
            search_radius=1.2,  # RRT* "near" node radius (meters)
            max_iter=1500,  # Iterations *per segment*
            goal_bias=0.1,
        )

        # Define the key waypoints (start, G1, G2, ...)
        key_waypoints = [self._initial_pos] + gate_centers

        # Run the planner sequentially for each segment
        full_path_points = []
        for i in range(len(key_waypoints) - 1):
            start = key_waypoints[i]
            goal = key_waypoints[i + 1]

            print(
                f"RRT*: Planning segment {i + 1}/{len(key_waypoints) - 1} (from {np.round(start, 1)} to {np.round(goal, 1)})..."
            )

            # Run the RRT* planner
            segment_path = self.planner.plan(start, goal)  # Returns [start, ..., goal]

            if not segment_path:
                print(f"FATAL: RRT* failed to find any path for segment {i}.")
                # Fallback: use a straight line (DANGEROUS!)
                segment_path = [start, goal]

            # Add the path, but skip the first point (it was the end of the last segment)
            if i == 0:
                full_path_points.extend(segment_path)
            else:
                full_path_points.extend(segment_path[1:])

        self._waypoints = np.array(full_path_points)
        print(f"RRT*: Total path generated with {len(self._waypoints)} waypoints.")

        # --- MODIFIED: Phase 1 (start) is REMOVED ---
        # The RRT* path starts at _initial_pos, so we just follow it.

        # --- Phase 2: Follow Path ---
        # Ticks to travel between RRT* waypoints.
        # RRT* step_size is 0.4m. Let's aim for ~2 m/s.
        # 0.4m / 2m/s = 0.2s.
        # If sim freq is 100Hz (0.01s/tick), 0.2s = 20 ticks.
        # NOTE: Your sim freq is 500Hz and env freq is 50Hz.
        # The controller runs at the *environment* step freq (50Hz).
        # 0.2s * 50 Hz = 10 ticks.
        self._segment_ticks = 10
        self._segment_tick_counter = 0
        self._target_waypoint_idx = 1  # Start by targeting waypoint 1 (idx 0 is start)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        # --- MODIFIED: Removed Phase 1 ---
        # We start directly at Phase 2, as the RRT path begins at our initial_pos.

        if self._target_waypoint_idx < len(self._waypoints):
            # --- Phase 2: Following the path segments ---
            start_of_segment = self._waypoints[self._target_waypoint_idx - 1]
            end_of_segment = self._waypoints[self._target_waypoint_idx]

            alpha = min(self._segment_tick_counter / self._segment_ticks, 1.0)
            des_pos = (1 - alpha) * start_of_segment + alpha * end_of_segment

            self._segment_tick_counter += 1

            # If this segment is done, move to the next one
            if self._segment_tick_counter >= self._segment_ticks:
                self._target_waypoint_idx += 1
                self._segment_tick_counter = 0

        else:
            # --- Phase 3: Finished. Hover at the last waypoint ---
            des_pos = self._waypoints[-1]
            self._finished = True

        self._tick += 1  # Keep global tick counter

        # Log position
        self.position_log.append(des_pos.copy())

        # Action: desired position + zeros for other states
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)

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
        # Signal that this controller is finished
        return self._finished

    def episode_callback(self):
        # Reset for the next episode
        self._tick = 0
        self._finished = False
        self._segment_tick_counter = 0
        self._target_waypoint_idx = 1  # Reset target
        self.position_log.clear()

        # NOTE: We do NOT re-plan the path. The RRT* plan is created once
        # in __init__ based on the *initial* _initial_pos.
        # This assumes the drone and track don't change between episodes.
        # If the drone's *starting* position changes, __init__ must be
        # run again (which requires a simulator reset).
