"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
import os  # <-- 1. Import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

# --- Suppress C++ DLPack warnings ---
# 2. Add this BEFORE importing jax, gymnasium, etc.
# This filters out C++ level WARNING messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Forces JAX to use CPU
# ------------------------------------

# --- Force JAX to CPU (avoid GPU warning if CUDA jaxlib is not installed) ---
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/dev/null'

# --- Suppress RuntimeWarnings (like overflow) from JAX ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)

def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
    camera_view: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # <-- New argument
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file.
        controller: Controller file name or None.
        n_runs: Number of episodes.
        render: Enable/disable rendering.
        camera_view: Optional camera settings [distance, azimuth, elevation, lookat_x, lookat_y, lookat_z].
    """
    # Load configuration
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render

    # Load the controller
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)

    # Create the environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    # Set camera view dynamically
    if camera_view is not None:
        try:
            env.sim.set_camera_view(*camera_view)  # This function depends on your sim backend
        except AttributeError:
            print(
                "[Warning] Camera view could not be set dynamically. Make sure your simulator supports `set_camera_view`."
            )

    ep_times = []
    for _ in range(n_runs):
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq
            action = controller.compute_control(obs, info)
            action = np.asarray(jp.asarray(action), copy=True)

            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                break

            if config.sim.render:
                if ((i * fps) % config.env.freq) < fps:
                    env.render()

            i += 1

        controller.episode_callback()
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    env.close()
    return ep_times



def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)

    # print("drone position: ", controller_cls.position_log)
