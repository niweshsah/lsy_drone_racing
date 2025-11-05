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
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from ml_collections import ConfigDict

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

logger = logging.getLogger(__name__)


def draw_3d_lines(config: ConfigDict):
    """Draw gates, obstacles, and smooth 3D path through gates with arrows."""
    # Read gate centers (pos) and orientations (rpy)
    gate_centers = []
    gate_rpys = []
    for gate_info in config.env.track.gates:
        gate_centers.append(gate_info["pos"])
        gate_rpys.append(gate_info["rpy"])

    gates = np.array(gate_centers)  # Center positions for the spline

    # Obstacles positions
    obstacles = np.array([obs["pos"] for obs in getattr(config.env.track, "obstacles", [])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot gates as frames
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
        ax.scatter(pos[0], pos[1], pos[2], c="cyan", s=15, alpha=0.8)

    # Plot obstacles
    if len(obstacles) > 0:
        ax.scatter(
            obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c="red", s=50, label="Obstacles"
        )

    # Interpolate a smooth 3D path through the gates
    t = np.arange(len(gates))
    cs_x = CubicSpline(t, gates[:, 0])
    cs_y = CubicSpline(t, gates[:, 1])
    cs_z = CubicSpline(t, gates[:, 2])

    t_smooth = np.linspace(0, len(gates) - 1, 200)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    z_smooth = cs_z(t_smooth)

    ax.plot(x_smooth, y_smooth, z_smooth, c="blue", linewidth=2, label="Smooth Path")

    # --- NEW SECTION: Add arrows to the path ---
    # Choose how many arrows to draw
    num_arrows = 10
    # Select evenly spaced points along the smooth path for arrows
    arrow_indices = np.linspace(0, len(t_smooth) - 2, num_arrows, dtype=int)

    for i in arrow_indices:
        # Get current point (tail of the arrow)
        p_curr = np.array([x_smooth[i], y_smooth[i], z_smooth[i]])
        # Get next point (to determine direction)
        p_next = np.array([x_smooth[i + 1], y_smooth[i + 1], z_smooth[i + 1]])

        # Calculate direction vector (normalized)
        direction = p_next - p_curr
        # Avoid division by zero if points are identical (unlikely with smooth spline)
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
        else:
            continue  # Skip if no clear direction

        # Scale the arrow length for better visualization
        arrow_length = 0.2
        # Endpoint of the arrow vector
        p_end = p_curr + direction * arrow_length

        # Plot the arrow using quiver
        ax.quiver(
            p_curr[0],
            p_curr[1],
            p_curr[2],  # Starting point (x, y, z)
            direction[0],
            direction[1],
            direction[2],  # Direction vector (dx, dy, dz)
            length=arrow_length,
            normalize=False,  # Length already normalized by direction
            color="orange",
            arrow_length_ratio=0.5,  # Adjust arrow head size
            zorder=5,
        )  # Ensure arrows are drawn on top
    # --- END NEW SECTION ---

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Path Through Gates and Obstacles with Direction")
    ax.legend()
    ax.set_aspect(
        "equal", adjustable="box"
    )  # Use 'box' for older matplotlib versions if 'equal' with 'adjustable' fails
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
