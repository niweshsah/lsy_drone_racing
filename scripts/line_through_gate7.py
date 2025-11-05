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
    """Draw gates, obstacles, vertical poles, and a smooth path passing through all gates."""
    gate_centers = [np.array(g["pos"]) for g in config.env.track.gates]
    gate_rpys = [np.array(g["rpy"]) for g in config.env.track.gates]
    gates = np.array(gate_centers)

    obstacles = np.array([obs["pos"] for obs in getattr(config.env.track, "obstacles", [])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # --- Plot gates ---
    gate_size = getattr(
        config.env, "gate_size", 1.0
    )  # as there is no gate_size in config file, so we set a default value

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
        r = Rotation.from_euler("xyz", rpy)  # converts roll-pitch-yaw to rotation matrix

        world_verts = r.apply(local_verts) + pos  # rotate and translate to world coordinates

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
            s=20,
            alpha=0.9,
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

    # --- Path that passes through each gate ---
    straight_half_length = 0.6  # meters before/after the gate
    path_points = []

    for pos, rpy in zip(gate_centers, gate_rpys):
        normal = Rotation.from_euler("xyz", rpy).apply(
            [1, 0, 0]
        )  # x-axis direction wrt gate orientation

        before = pos - normal * straight_half_length  # point before the gate
        after = pos + normal * straight_half_length  # point after the gate

        path_points.extend([before, pos, after])  # ensures path passes *through* the gate

    path_points = np.array(path_points)

    # --- Optional: Smooth the full path ---
    from scipy.interpolate import CubicSpline

    t = np.linspace(0, 1, len(path_points))
    cs_x = CubicSpline(t, path_points[:, 0])
    cs_y = CubicSpline(t, path_points[:, 1])
    cs_z = CubicSpline(t, path_points[:, 2])

    t_smooth = np.linspace(0, 1, 200)
    smooth_path = np.vstack((cs_x(t_smooth), cs_y(t_smooth), cs_z(t_smooth))).T

    # --- Plot path ---
    ax.plot(
        smooth_path[:, 0],
        smooth_path[:, 1],
        smooth_path[:, 2],
        c="blue",
        linewidth=2.5,
        label="Path Through Gates",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Path Passing Through All Gates (Aligned with Gate Orientation)")
    ax.legend()
    ax.set_aspect("auto")
    plt.tight_layout()
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
