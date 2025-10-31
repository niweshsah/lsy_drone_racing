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
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from ml_collections import ConfigDict

# Suppress noisy warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

logger = logging.getLogger(__name__)


def refine_path_with_obstacles(
    control_points: np.ndarray,
    obstacles: np.ndarray,
    *,
    avoidance_radius_pole: float = 0.4,
    avoidance_radius_sphere: float = 0.4,
    safety_margin: float = 0.1,
    max_iterations: int = 10,
    dense_samples: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively refine control points such that a spline passing through them
    avoids obstacles. Returns (final_smooth_path, final_control_points).

    control_points : (N,3) array of initial waypoints (before/through/after gates)
    obstacles      : (M,3) array of obstacle centers (x,y,z)
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
        # 1) build spline and dense samples
        cs_x = CubicSpline(current_t, current_points[:, 0])
        cs_y = CubicSpline(current_t, current_points[:, 1])
        cs_z = CubicSpline(current_t, current_points[:, 2])
        t_smooth = np.linspace(0, 1, dense_samples)
        smooth_path = np.vstack((cs_x(t_smooth), cs_y(t_smooth), cs_z(t_smooth))).T

        # 2) detect collisions and prepare new control points to add
        points_to_add: list[tuple[float, np.ndarray]] = []

        for i, p in enumerate(smooth_path):
            t_val = t_smooth[i]
            # check each obstacle
            collided = False
            for o in obstacles:
                # horizontal pole check (xy distance) but only when below obstacle height
                dist_xy = np.linalg.norm(p[:2] - o[:2])
                dist_3d = np.linalg.norm(p - o)

                if dist_xy < avoidance_radius_pole and p[2] < o[2]:
                    # push horizontally away from pole center, keep same z
                    v_xy = p[:2] - o[:2]
                    v_xy_norm = v_xy / (np.linalg.norm(v_xy) + 1e-8)
                    push_dist = avoidance_radius_pole + safety_margin
                    p_new = np.array([o[0] + v_xy_norm[0] * push_dist,
                                      o[1] + v_xy_norm[1] * push_dist,
                                      p[2]])
                    points_to_add.append((t_val, p_new))
                    collided = True
                    break

                if dist_3d < avoidance_radius_sphere + 1e-8:
                    # push in full 3D away from center
                    v = p - o
                    v_norm = v / (np.linalg.norm(v) + 1e-8)
                    push_dist = avoidance_radius_sphere + safety_margin
                    p_new = o + v_norm * push_dist
                    points_to_add.append((t_val, p_new))
                    collided = True
                    break

            if collided:
                # skip checking other obstacles for this sample point
                continue

        # 3) if no collisions found -> done
        if not points_to_add:
            logger.info("Refinement converged (no collisions detected).")
            break

        # 4) add new control points (avoid duplicates in t)
        existing_t = set(current_t.tolist())
        new_t_list = current_t.tolist()
        new_points_list = current_points.tolist()

        for t_val, p_new in points_to_add:
            # small tolerance for equality of t
            if not any(abs(t_val - t_exist) < 1e-6 for t_exist in new_t_list):
                new_t_list.append(t_val)
                new_points_list.append(p_new.tolist())

        # sort by t
        sort_idx = np.argsort(new_t_list)
        current_t = np.array(new_t_list)[sort_idx]
        current_points = np.array(new_points_list)[sort_idx]

        logger.debug(f"Iteration {iteration}: added {len(points_to_add)} points -> total control points {len(current_points)}")

    else:
        logger.warning("Refinement stopped: reached max iterations.")

    # final smooth path from last control points
    cs_x = CubicSpline(current_t, current_points[:, 0])
    cs_y = CubicSpline(current_t, current_points[:, 1])
    cs_z = CubicSpline(current_t, current_points[:, 2])
    t_smooth = np.linspace(0, 1, dense_samples)
    smooth_path = np.vstack((cs_x(t_smooth), cs_y(t_smooth), cs_z(t_smooth))).T

    return smooth_path, current_points


def draw_3d_lines(config: ConfigDict):
    """Draw gates, obstacles, vertical poles, and a smooth path passing through all gates."""

    gate_centers = [np.array(g["pos"]) for g in config.env.track.gates]
    gate_rpys = [np.array(g["rpy"]) for g in config.env.track.gates]
    obstacles_list = getattr(config.env.track, "obstacles", [])
    obstacles = np.array([obs["pos"] for obs in obstacles_list]) if obstacles_list else np.empty((0, 3))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # --- Plot gates ---
    gate_size = float(getattr(config.env, "gate_size", 1.0))
    half_size = gate_size / 2.0
    local_verts = np.array([
        [0, -half_size, -half_size],
        [0,  half_size, -half_size],
        [0,  half_size,  half_size],
        [0, -half_size,  half_size],
        [0, -half_size, -half_size],
    ])

    for i, (pos, rpy) in enumerate(zip(gate_centers, gate_rpys)):
        r = Rotation.from_euler("xyz", rpy)
        world_verts = r.apply(local_verts) + pos
        ax.plot(world_verts[:, 0], world_verts[:, 1], world_verts[:, 2],
                c="green", label="Gates" if i == 0 else None)
        ax.scatter(pos[0], pos[1], pos[2], c="cyan", s=30, alpha=0.9,
                   label="Gate Centers" if i == 0 else None)

    # --- Plot obstacles and poles ---
    if obstacles.shape[0] > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2],
                   c="red", s=50, label="Obstacles")
        for obs in obstacles:
            x, y, z = obs
            # vertical pole from ground to obstacle z
            ax.plot([x, x], [y, y], [0, z], c="brown", linewidth=2,
                    label="Obstacle Pole" if "Obstacle Pole" not in [l.get_label() for l in ax.lines] else None)

    # --- Build initial control points (before, center, after each gate) ---
    straight_half_length = 0.6
    path_points = []
    for pos, rpy in zip(gate_centers, gate_rpys):
        normal = Rotation.from_euler("xyz", rpy).apply([1.0, 0.0, 0.0])
        before = pos - normal * straight_half_length
        after = pos + normal * straight_half_length
        path_points.extend([before, pos, after])
    control_points = np.array(path_points)

    # --- Iteratively refine control points to avoid obstacles and produce smooth path ---
    smooth_path, refined_control_points = refine_path_with_obstacles(control_points, obstacles)

    # --- Plot final path and refined control points ---
    ax.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2],
            c="blue", linewidth=2.5, label="Refined Path")
    ax.scatter(refined_control_points[:, 0], refined_control_points[:, 1], refined_control_points[:, 2],
               c="purple", s=10, alpha=0.6, label="Refined Control Points")

    # --- Labels and title ---
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Path Passing Through Gates (With Obstacle Avoidance)")
    ax.legend()

    # --- Enforce equal-like aspect by setting axis limits based on data range ---
    all_pts = np.vstack((smooth_path, refined_control_points)) if refined_control_points.size else smooth_path
    x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
    y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
    z_min, z_max = np.min(all_pts[:, 2]), np.max(all_pts[:, 2])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def simulate(config: str = "level0.toml"):
    """Load the config and draw the scene."""
    try:
        cfg_path = Path(__file__).parents[1] / "config" / config
        if not cfg_path.exists():
            cfg_path = Path.cwd() / "config" / config
    except NameError:
        cfg_path = Path.cwd() / "config" / config
        if not cfg_path.exists():
            cfg_path = Path.cwd().parents[0] / "config" / config

    if not cfg_path.exists():
        logger.error("Could not find config file: %s", cfg_path)
        logger.error("Please ensure the config is located in a 'config' directory.")
        return

    cfg = load_config(cfg_path)
    print(f"Config '{getattr(cfg.env, 'id', 'unknown')}' loaded from {cfg_path}")
    draw_3d_lines(cfg)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate)
