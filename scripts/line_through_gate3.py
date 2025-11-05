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
from scipy.interpolate import CubicHermiteSpline
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

    # --- UPDATED SECTION: Use CubicHermiteSpline to control direction ---

    # 't' is our parameter (like time)
    t = np.arange(len(gates))

    # 1. Calculate the normal vector (desired direction) for each gate
    normals = []
    for rpy in gate_rpys:
        r = Rotation.from_euler("xyz", rpy)
        # The normal vector is the gate's local X-axis rotated to world frame
        normals.append(r.apply([1, 0, 0]))
    normals = np.array(normals)

    # 2. Calculate derivatives (dy/dt) for the spline
    # We use the normal for direction and estimate the "speed" (magnitude)
    # based on the distance to adjacent points.
    derivatives = np.zeros_like(gates)

    # A factor > 1.0 will make turns wider (less sharp)
    # A factor < 1.0 will make turns tighter (sharper)
    smoothness_factor = 2.5

    # First point: use forward difference
    v_avg = (gates[1] - gates[0]) / (t[1] - t[0])
    mag = np.abs(np.dot(v_avg, normals[0])) * smoothness_factor  # <-- Use abs() and factor
    derivatives[0] = normals[0] * mag

    # Last point: use backward difference
    v_avg = (gates[-1] - gates[-2]) / (t[-1] - t[-2])
    mag = np.abs(np.dot(v_avg, normals[-1])) * smoothness_factor  # <-- Use abs() and factor
    derivatives[-1] = normals[-1] * mag

    # Intermediate points: use central difference
    for i in range(1, len(gates) - 1):
        v_avg = (gates[i + 1] - gates[i - 1]) / (t[i + 1] - t[i - 1])
        mag = np.abs(np.dot(v_avg, normals[i])) * smoothness_factor  # <-- Use abs() and factor
        derivatives[i] = normals[i] * mag

    # 3. Create the Hermite spline
    # This spline now passes through each gate[i] with the exact derivative[i]
    spline = CubicHermiteSpline(t, gates, derivatives, axis=0)

    # 4. Generate the smooth path
    t_smooth = np.linspace(0, len(gates) - 1, 200)
    smooth_path = spline(t_smooth)  # This is an (N, 3) array of points
    x_smooth = smooth_path[:, 0]
    y_smooth = smooth_path[:, 1]
    z_smooth = smooth_path[:, 2]

    ax.plot(x_smooth, y_smooth, z_smooth, c="blue", linewidth=2, label="Smooth Path")
    # --- END UPDATED SECTION ---

    # --- UPDATED SECTION: Add arrows using the spline's derivative ---
    # We can get the *exact* derivative from the spline itself
    spline_deriv = spline.derivative()(t_smooth)  # (N, 3) array of derivatives

    num_arrows = 10
    arrow_indices = np.linspace(0, len(t_smooth) - 1, num_arrows, dtype=int)

    for i in arrow_indices:
        # Get current point (tail of the arrow)
        p_curr = smooth_path[i]
        # Get the derivative (direction vector) at that point
        direction = spline_deriv[i]

        # Avoid division by zero
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)  # Normalize to get pure direction
        else:
            continue  # Skip if no clear direction

        # Scale the arrow length for better visualization
        arrow_length = 0.2

        # Plot the arrow using quiver
        ax.quiver(
            p_curr[0],
            p_curr[1],
            p_curr[2],  # Starting point (x, y, z)
            direction[0],
            direction[1],
            direction[2],  # Direction vector (dx, dy, dz)
            length=arrow_length,
            normalize=False,  # We already normalized
            color="orange",
            arrow_length_ratio=0.5,  # Adjust arrow head size
            zorder=5,
        )  # Ensure arrows are drawn on top
    # --- END UPDATED SECTION ---

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Path Through Gates and Obstacles (Perpendicular Pass)")
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
