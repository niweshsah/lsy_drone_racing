from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # Must come first, before any other imports

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import matplotlib.pyplot as plt
import numpy as np

from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from ml_collections import ConfigDict


# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/dev/null"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

logger = logging.getLogger(__name__)


def draw_3d_lines(config: ConfigDict):
    """Draw gates and obstacles from the loaded config in 3D using Matplotlib."""
    # print("tracks:", config.env.track)
    # print("gate types:", type(config.env.track.gates))
    # print("gates:", config.env.track.gates)

    print("gates: ", [gate for gate in config.env.track.gates])

    # Get gates
    gates = np.array([gate["pos"] for gate in config.env.track.gates])

    # Get obstacles
    obstacles = np.array([obs["pos"] for obs in getattr(config.env.track, "obstacles", [])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot gates
    if len(gates) > 0:
        ax.scatter(gates[:, 0], gates[:, 1], gates[:, 2], c="green", s=50, label="Gates")

    # Plot obstacles
    if len(obstacles) > 0:
        ax.scatter(
            obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c="red", s=50, label="Obstacles"
        )

    # Draw lines connecting gates
    for i in range(len(gates) - 1):
        x = [gates[i][0], gates[i + 1][0]]
        y = [gates[i][1], gates[i + 1][1]]
        z = [gates[i][2], gates[i + 1][2]]
        ax.plot(x, y, z, c="blue", linewidth=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("3D Path Through Gates and Obstacles")
    plt.show()


def simulate(config: str = "level0.toml"):
    """Load the config and draw gates/obstacles in 3D."""
    cfg_path = Path(__file__).parents[1] / "config" / config
    config = load_config(cfg_path)
    print("Config loaded successfully.")

    draw_3d_lines(config)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate)
