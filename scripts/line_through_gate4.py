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
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from ml_collections import ConfigDict

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

logger = logging.getLogger(__name__)


def draw_3d_lines(config: ConfigDict):
    """Draw gates, obstacles, and smooth 3D path through gates with arrows."""

    # Read gate centers (pos) and orientations (rpy)
    gate_centers = []
    gate_rpys = []
    for gate_info in config.env.track.gates:
        gate_centers.append(gate_info['pos'])
        gate_rpys.append(gate_info['rpy'])

    gates = np.array(gate_centers) # Center positions for the spline
    
    # Obstacles positions
    obstacles = np.array([obs['pos'] for obs in getattr(config.env.track, "obstacles", [])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot gates as frames
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

    # Plot obstacles
    if len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c='red', s=50, label='Obstacles')

    # --- UPDATED SECTION: Create "tunnel" points for straight pass-through ---
    
    # 1. Calculate gate normals (same as before)
    normals = []
    for rpy in gate_rpys:
        r = Rotation.from_euler('xyz', rpy)
        normals.append(r.apply([1, 0, 0]))
    normals = np.array(normals)

    # 2. Define "before" and "after" points for each gate
    # We create two points for each gate: one just before, one just after.
    # The spline will be forced to be straight between these two points.
    straight_half_length = 0.5  # Creates a 1.0m straight segment through the gate
    smoothness_factor = 2.5 # Controls "speed" and thus turn radius

    spline_points = []
    spline_derivatives = []
    
    # Calculate magnitudes (speed) based on original gate spacing
    mags = np.zeros(len(gates))
    
    # First point: forward diff
    v_avg = (gates[1] - gates[0])
    mags[0] = np.abs(np.dot(v_avg, normals[0])) * smoothness_factor
    
    # Last point: backward diff
    v_avg = (gates[-1] - gates[-2])
    mags[-1] = np.abs(np.dot(v_avg, normals[-1])) * smoothness_factor

    # Intermediate points: central diff
    for i in range(1, len(gates) - 1):
        v_avg = (gates[i+1] - gates[i-1]) / 2.0 # Central difference
        mags[i] = np.abs(np.dot(v_avg, normals[i])) * smoothness_factor

    # Build the new points and derivatives (2 for each gate)
    for i in range(len(gates)):
        pos = gates[i]
        normal = normals[i]
        mag = mags[i]
        
        # Point just before the gate center
        spline_points.append(pos - normal * straight_half_length)
        # Point just after the gate center
        spline_points.append(pos + normal * straight_half_length)
        
        # The derivative at *both* points is the same.
        # This forces the spline segment between them to be a straight line.
        derivative = normal * mag
        spline_derivatives.append(derivative)
        spline_derivatives.append(derivative)

    spline_points = np.array(spline_points)
    spline_derivatives = np.array(spline_derivatives)

    # 3. Create the 't' parameter based on cumulative chordal distance
    # This makes the spline parameterization more natural (t ~ distance)
    distances = np.linalg.norm(np.diff(spline_points, axis=0), axis=1)
    t_spline = np.zeros(len(spline_points))
    t_spline[1:] = np.cumsum(distances) # t = [0, dist_0_1, dist_0_1+dist_1_2, ...]

    # 4. Create the Hermite spline
    # This spline now passes through each *pair* of points
    spline = CubicHermiteSpline(t_spline, spline_points, spline_derivatives, axis=0)

    # 5. Generate the smooth path
    t_min = t_spline[0]
    t_max = t_spline[-1]
    t_smooth = np.linspace(t_min, t_max, 200)
    smooth_path = spline(t_smooth) # This is an (N, 3) array of points
    # --- END UPDATED SECTION ---

    x_smooth = smooth_path[:, 0]
    y_smooth = smooth_path[:, 1]
    z_smooth = smooth_path[:, 2]

    ax.plot(x_smooth, y_smooth, z_smooth, c='blue', linewidth=2, label='Smooth Path')

    # --- Add arrows using the spline's derivative (Unchanged) ---
    spline_deriv = spline.derivative()(t_smooth) # (N, 3) array of derivatives
    
    num_arrows = 10 
    arrow_indices = np.linspace(0, len(t_smooth) - 1, num_arrows, dtype=int)

    for i in arrow_indices:
        p_curr = smooth_path[i]
        direction = spline_deriv[i]
        
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
        else:
            continue

        arrow_length = 0.2 
        ax.quiver(p_curr[0], p_curr[1], p_curr[2],
                  direction[0], direction[1], direction[2],
                  length=arrow_length, normalize=False,
                  color='orange', arrow_length_ratio=0.5,
                  zorder=5)
    # --- END ARROW SECTION ---

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Path Through Gates (Straight Pass-Through)')
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