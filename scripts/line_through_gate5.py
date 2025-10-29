from __future__ import annotations


import logging
import os
os.environ["SCIPY_ARRAY_API"] = "1"  # Must come first, before any other imports

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import fire
# --- MODIFIED: Import CubicSpline as well ---
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

logger = logging.getLogger(__name__)


# --- NEW HELPER FUNCTION (from controller) ---
def _apply_obstacle_avoidance(
    base_path: NDArray[np.floating],
    t_smooth: NDArray[np.floating],
    t_spline_knots: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    num_gates: int,
    safety_radius: float,
    repulsion_gain: float,
) -> NDArray[np.floating]:
    """
    Refines a base path by applying a 2D repulsive force from obstacles.
    This force is *not* applied to the straight-line segments that pass through gates.
    """
    
    avoid_path = np.copy(base_path)
    obstacles_xy = obstacles_pos[:, :2] # We only care about 2D poles
    
    # Pre-calculate the 't' intervals for all straight segments
    straight_intervals = []
    for i in range(num_gates):
        # t_start is t_spline_knots[2*i]
        # t_end is t_spline_knots[2*i + 1]
        straight_intervals.append((t_spline_knots[2*i], t_spline_knots[2*i + 1]))
    
    # Iterate over every sample point in the base path
    for j, (t, p) in enumerate(zip(t_smooth, base_path)):
        
        # Check if this point's 't' value falls within any straight segment
        is_straight = False
        for t_start, t_end in straight_intervals:
            # Use a small tolerance to ensure we catch the end points
            if (t_start - 1e-6) <= t <= (t_end + 1e-6):
                is_straight = True
                break
        
        # If the point is on a straight segment, DO NOT modify it.
        if is_straight:
            continue 

        # This point is on a curved segment. Apply repulsion.
        p_xy = p[:2]
        repulsion_vec_xy = np.zeros(2, dtype=np.float32)
        
        for obs_xy in obstacles_xy:
            vec_to_obs_xy = p_xy - obs_xy
            dist_xy = np.linalg.norm(vec_to_obs_xy)
            
            # If inside the safety radius, calculate repulsion
            if 0 < dist_xy < safety_radius:
                # Force scales inversely with distance, becoming 0 at the radius boundary
                force_mag = repulsion_gain * (1.0 / dist_xy - 1.0 / safety_radius)
                force_dir = vec_to_obs_xy / dist_xy
                repulsion_vec_xy += force_dir * force_mag
        
        # Apply the summed repulsion to the path point (in XY only)
        avoid_path[j, :2] += repulsion_vec_xy
        
    return avoid_path
# --- END HELPER FUNCTION ---

def draw_3d_lines(config: ConfigDict):
    """Draw gates, obstacles, vertical poles, and smooth 3D path through gates while avoiding poles."""

    # Read gate centers and orientations
    gate_centers = [g['pos'] for g in config.env.track.gates]
    gate_rpys = [g['rpy'] for g in config.env.track.gates]
    gates = np.array(gate_centers)

    # Obstacles positions
    obstacles = np.array([obs['pos'] for obs in getattr(config.env.track, "obstacles", [])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot gates ---
    gate_size = getattr(config.env, "gate_size", 1.0)
    half_size = gate_size / 2.0
    local_verts = np.array([
        [0, -half_size, -half_size],
        [0,  half_size, -half_size],
        [0,  half_size,  half_size],
        [0, -half_size,  half_size],
        [0, -half_size, -half_size]
    ])
    for i, (pos, rpy) in enumerate(zip(gate_centers, gate_rpys)):
        r = Rotation.from_euler('xyz', rpy)
        world_verts = r.apply(local_verts) + pos
        ax.plot(world_verts[:, 0], world_verts[:, 1], world_verts[:, 2],
                c='green', label='Gates' if i == 0 else None)
        ax.scatter(pos[0], pos[1], pos[2], c='cyan', s=15, alpha=0.8)

    # --- Plot obstacles and vertical poles ---
    if len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c='red', s=50, label='Obstacles')
        
        # Draw vertical poles connecting each obstacle to the ground
        for obs in obstacles:
            x, y, z = obs
            ax.plot([x, x], [y, y], [0, z], c='brown', linewidth=2,
                    label='Obstacle Pole' if 'Obstacle Pole' not in [l.get_label() for l in ax.lines] else None)

    # --- Compute spline points through gates ---
    normals = [Rotation.from_euler('xyz', rpy).apply([1, 0, 0]) for rpy in gate_rpys]
    normals = np.array(normals)

    straight_half_length = 0.5
    smoothness_factor = 2.5

    spline_points = []
    spline_derivatives = []

    # Calculate derivative magnitudes
    mags = np.zeros(len(gates))
    mags[0] = np.abs(np.dot(gates[1] - gates[0], normals[0])) * smoothness_factor
    mags[-1] = np.abs(np.dot(gates[-1] - gates[-2], normals[-1])) * smoothness_factor
    for i in range(1, len(gates) - 1):
        mags[i] = np.abs(np.dot((gates[i+1] - gates[i-1])/2.0, normals[i])) * smoothness_factor

    # Build spline points while avoiding obstacle poles
    safe_height = 0.5  # minimum clearance above obstacles
    for i in range(len(gates)):
        pos = gates[i]
        normal = normals[i]
        mag = mags[i]

        # Base points through the gate
        before = pos - normal * straight_half_length
        after = pos + normal * straight_half_length

        # Check obstacles: if obstacle pole intersects this segment, raise Z
        if len(obstacles) > 0:
            for obs in obstacles:
                obs_x, obs_y, obs_z = obs
                radius = 0.3  # safety radius around pole
                for point in [before, after]:
                    if np.linalg.norm(point[:2] - obs[:2]) < radius:
                        point[2] = max(point[2], obs_z + safe_height)

        spline_points.extend([before, after])
        derivative = normal * mag
        spline_derivatives.extend([derivative, derivative])

    spline_points = np.array(spline_points)
    spline_derivatives = np.array(spline_derivatives)

    # Parameterization and Hermite spline
    distances = np.linalg.norm(np.diff(spline_points, axis=0), axis=1)
    t_spline = np.zeros(len(spline_points))
    t_spline[1:] = np.cumsum(distances)
    spline = CubicHermiteSpline(t_spline, spline_points, spline_derivatives, axis=0)

    t_smooth = np.linspace(t_spline[0], t_spline[-1], 200)
    smooth_path = spline(t_smooth)

    ax.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2],
            c='blue', linewidth=2, label='Smooth Path')

    # Arrows along path
    derivs = spline.derivative()(t_smooth)
    for i in np.linspace(0, len(t_smooth)-1, 10, dtype=int):
        d = derivs[i]
        if np.linalg.norm(d) > 1e-6:
            d /= np.linalg.norm(d)
        ax.quiver(*smooth_path[i], *d, length=0.2, color='orange', arrow_length_ratio=0.5, zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Path Through Gates (Obstacle-Aware)')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.show()



def simulate(config: str = "level0.toml"):
    """Load the config and draw gates/obstacles in 3D."""
    
    try:
        cfg_path = Path(__file__).parents[1] / "config" / config
        if not cfg_path.exists():
            cfg_path = Path.cwd() / "config" / config
    except NameError:
        cfg_path = Path.cwd() / "config" / config
        if not cfg_path.exists():
             cfg_path = Path.cwd().parents[0] / "config" / config

    if not cfg_path.exists():
        logger.error(f"Could not find config file: {cfg_path}")
        logger.error("Please ensure 'level0.toml' is in a 'config' directory within the project structure.")
        return

    config = load_config(cfg_path)
    print(f"Config '{getattr(config.env, 'id', 'unknown')}' loaded successfully from {cfg_path}")

    draw_3d_lines(config)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate)