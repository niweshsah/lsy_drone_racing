from __future__ import annotations

import logging
import os

os.environ["SCIPY_ARRAY_API"] = "1"  # Must come first, before any other imports

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from ml_collections import ConfigDict


# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

logger = logging.getLogger(__name__)


def draw_3d_lines(config: ConfigDict):
    """Draw gates, obstacles, vertical poles, and a straight-line path through gates."""
    # Read gate centers and orientations
    gate_centers = [g["pos"] for g in config.env.track.gates]
    gate_rpys = [g["rpy"] for g in config.env.track.gates]
    gates = np.array(gate_centers)

    # Obstacles positions
    obstacles = np.array([obs["pos"] for obs in getattr(config.env.track, "obstacles", [])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # --- Plot gates ---
    gate_size = getattr(config.env, "gate_size", 1.0)
    half_size = gate_size / 2.0
    local_verts = np.array(
        [
            [0, -half_size, -half_size],
            [0, half_size, -half_size],
            [0, half_size, half_size],
            [0, -half_size, half_size],
            [0, -half_size, -half_size],
        ]
    )
    for i, (pos, rpy) in enumerate(zip(gate_centers, gate_rpys)):
        r = Rotation.from_euler("xyz", rpy)
        world_verts = r.apply(local_verts) + pos
        ax.plot(
            world_verts[:, 0],
            world_verts[:, 1],
            world_verts[:, 2],
            c="green",
            label="Gates" if i == 0 else None,
        )
        ax.scatter(
            pos[0],
            pos[1],
            pos[2],
            c="cyan",
            s=15,
            alpha=0.8,
            label="Gate Centers" if i == 0 else None,
        )

    # --- Plot obstacles and vertical poles ---
    if len(obstacles) > 0:
        ax.scatter(
            obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c="red", s=50, label="Obstacles"
        )

        for obs in obstacles:
            x, y, z = obs
            ax.plot(
                [x, x],
                [y, y],
                [0, z],
                c="brown",
                linewidth=2,
                label="Obstacle Pole"
                if "Obstacle Pole" not in [l.get_label() for l in ax.lines]
                else None,
            )

    # --- MODIFIED: Compute straight path points passing THROUGH gates ---

    # Get the normal vector (x-axis) for each gate
    normals = [Rotation.from_euler("xyz", rpy).apply([1, 0, 0]) for rpy in gate_rpys]
    normals = np.array(normals)

    straight_half_length = 0.5  # How far before/after the gate center to set points
    path_points = []

    # Build path points while avoiding obstacle poles
    safe_height = 0.5  # minimum clearance above obstacles

    for i in range(len(gates)):
        pos = gates[i]
        normal = normals[i]

        # Base points through the gate
        before = pos - normal * straight_half_length
        after = pos + normal * straight_half_length

        # Check obstacles: if obstacle pole intersects this segment, raise Z
        if len(obstacles) > 0:
            for obs in obstacles:
                # obs_x, obs_y, obs_z = obs
                radius = 0.3  # safety radius around pole
                for point in [before, after]:
                    if np.linalg.norm(point[:2] - obs[:2]) < radius:
                        point[2] = max(point[2], obs[2] + safe_height)

        path_points.extend([before, after])

    # Convert to numpy array for plotting
    path_points = np.array(path_points)

    # Plot the straight-line path connecting all 'before' and 'after' points
    if len(path_points) > 0:
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            c="blue",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Straight Path",
        )
    # --- End of modification ---

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Straight Path Through Gates (Before/After Points)")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
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
        logger.error(
            "Please ensure 'level0.toml' is in a 'config' directory within the project structure."
        )
        return

    config = load_config(cfg_path)
    print(f"Config '{getattr(config.env, 'id', 'unknown')}' loaded successfully from {cfg_path}")

    draw_3d_lines(config)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate)
