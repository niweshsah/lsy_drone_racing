from __future__ import annotations

import logging
import os
os.environ["SCIPY_ARRAY_API"] = "1"  # Must come first

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import fire
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# --- HELPER: obstacle avoidance (optional enhancement) ---
# ----------------------------------------------------------------------
def _apply_obstacle_avoidance(
    base_path: NDArray[np.floating],
    t_smooth: NDArray[np.floating],
    t_spline_knots: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    num_gates: int,
    safety_radius: float = 1.0,
    repulsion_gain: float = 0.3,
) -> NDArray[np.floating]:
    """
    Apply a simple 2D repulsion from obstacles on curved sections of the path.
    """
    avoid_path = np.copy(base_path)
    obstacles_xy = obstacles_pos[:, :2]

    # straight intervals correspond to (before, center, after) -> straight between before->after
    straight_intervals = []
    for i in range(num_gates):
        t_start = t_spline_knots[3 * i]     # before
        t_end = t_spline_knots[3 * i + 2]   # after
        straight_intervals.append((t_start, t_end))

    for j, (t, p) in enumerate(zip(t_smooth, base_path)):
        # Check if this t lies inside any straight gate interval
        is_straight = any((t_start - 1e-6) <= t <= (t_end + 1e-6)
                          for (t_start, t_end) in straight_intervals)
        if is_straight:
            continue

        p_xy = p[:2]
        repulsion_vec = np.zeros(2, dtype=np.float32)

        for obs_xy in obstacles_xy:
            vec = p_xy - obs_xy
            dist = np.linalg.norm(vec)
            if 0 < dist < safety_radius:
                force_mag = repulsion_gain * (1.0 / dist - 1.0 / safety_radius)
                repulsion_vec += (vec / dist) * force_mag

        avoid_path[j, :2] += repulsion_vec
    return avoid_path


# ----------------------------------------------------------------------
# --- MAIN: Draw gates, obstacles, and perpendicular spline path ---
# ----------------------------------------------------------------------
def draw_3d_lines(config: ConfigDict):
    """Draw 3D gates, obstacles, and a path that passes through each gate center perpendicularly."""

    gate_centers = [g['pos'] for g in config.env.track.gates]
    gate_rpys = [g['rpy'] for g in config.env.track.gates]
    gates = np.array(gate_centers)

    obstacles = np.array([obs['pos'] for obs in getattr(config.env.track, "obstacles", [])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot Gates ---
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
                c='green', label='Gate' if i == 0 else None)
        ax.scatter(pos[0], pos[1], pos[2], c='cyan', s=20, alpha=0.9)

    # --- Plot Obstacles ---
    if len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2],
                   c='red', s=50, label='Obstacle')
        for obs in obstacles:
            x, y, z = obs
            ax.plot([x, x], [y, y], [0, z], c='brown', linewidth=2,
                    label='Obstacle Pole' if 'Obstacle Pole' not in [l.get_label() for l in ax.lines] else None)

    # --- Compute spline path ---
    normals = [Rotation.from_euler('xyz', rpy).apply([1, 0, 0]) for rpy in gate_rpys]
    normals = np.array(normals)

    straight_half_length = 0.5
    smoothness_factor = 2.5

    spline_points = []
    spline_derivatives = []

    mags = np.zeros(len(gates))
    if len(gates) >= 2:
        mags[0] = np.abs(np.dot(gates[1] - gates[0], normals[0])) * smoothness_factor
        mags[-1] = np.abs(np.dot(gates[-1] - gates[-2], normals[-1])) * smoothness_factor
    for i in range(1, len(gates) - 1):
        mags[i] = np.abs(np.dot((gates[i+1] - gates[i-1]) / 2.0, normals[i])) * smoothness_factor

    safe_height = 0.5
    pole_radius = 0.3

    # Build spline points: before, center, after for each gate
    for i in range(len(gates)):
        pos = gates[i].copy()
        normal = normals[i].copy()
        mag = mags[i]

        before = pos - normal * straight_half_length
        center = pos.copy()
        after = pos + normal * straight_half_length

        if len(obstacles) > 0:
            for obs in obstacles:
                for point in (before, center, after):
                    if np.linalg.norm(point[:2] - obs[:2]) < pole_radius:
                        point[2] = max(point[2], obs[2] + safe_height)

        spline_points.extend([before, center, after])
        deriv = normal * mag
        spline_derivatives.extend([deriv, deriv, deriv])

    spline_points = np.array(spline_points)
    spline_derivatives = np.array(spline_derivatives)

    # --- Hermite Spline Construction ---
    distances = np.linalg.norm(np.diff(spline_points, axis=0), axis=1)
    t_spline = np.zeros(len(spline_points))
    t_spline[1:] = np.cumsum(distances)
    spline = CubicHermiteSpline(t_spline, spline_points, spline_derivatives, axis=0)

    t_smooth = np.linspace(t_spline[0], t_spline[-1], 300)
    smooth_path = spline(t_smooth)

    # --- Apply obstacle avoidance on curved segments ---
    if len(obstacles) > 0:
        smooth_path = _apply_obstacle_avoidance(
            smooth_path, t_smooth, t_spline, obstacles, len(gates),
            safety_radius=1.0, repulsion_gain=0.3
        )

    # --- Plot path and direction arrows ---
    ax.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2],
            c='blue', linewidth=2, label='Smooth Path')

    derivs = spline.derivative()(t_smooth)
    indices = np.linspace(0, len(t_smooth) - 1, 10, dtype=int)
    for i in indices:
        d = derivs[i]
        if np.linalg.norm(d) > 1e-6:
            d = d / np.linalg.norm(d)
        ax.quiver(*smooth_path[i], *d, length=0.2, color='orange', arrow_length_ratio=0.5, zorder=5)

    # --- Final plot setup ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Path Through Gates (Perpendicular at Centers)')
    ax.legend()
    ax.set_box_aspect([1, 1, 0.5])  # better visual proportions
    plt.show()


# ----------------------------------------------------------------------
# --- ENTRY POINT ---
# ----------------------------------------------------------------------
def simulate(config: str = "level0.toml"):
    """Load the config and visualize 3D path."""
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
        return

    config = load_config(cfg_path)
    print(f"Config '{getattr(config.env, 'id', 'unknown')}' loaded successfully from {cfg_path}")

    draw_3d_lines(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(simulate)
