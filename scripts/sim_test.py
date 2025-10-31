import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/dev/null'
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils import load_config, load_controller  # <-- added load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

logger = logging.getLogger(__name__)


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,  # CLI argument for controller
    n_runs: int = 1,
    render: bool | None = None,
    camera_view: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
) -> list[float]:
    """Run simulation with a specified controller via CLI."""

    # Load configuration
    config_path = Path(__file__).parents[1] / "config" / config
    config: ConfigDict = load_config(config_path)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render

    # Load controller dynamically
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_file = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_file)

    # Create environment
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

    # Camera view
    if camera_view is not None:
        try:
            env.sim.set_camera_view(*camera_view)
        except AttributeError:
            print("[Warning] Camera view could not be set dynamically.")

    ep_times = []
    for run_idx in range(n_runs):
        print(f"\n--- Starting Episode {run_idx + 1} ---")
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
                reason = (
                    getattr(controller, "_stop_reason", "Controller finished all waypoints")
                    if controller_finished else
                    "Environment terminated" if terminated else
                    "Environment truncated"
                )
                print(f"[Episode {run_idx + 1}] Stopped at t={curr_time:.2f}s. Reason: {reason}")
                break

            if config.sim.render and ((i * fps) % config.env.freq) < fps:
                env.render()

            i += 1

        # Log path
        # print(f"[Episode {run_idx + 1}] Drone path positions:")
        # for pos in controller.position_log:
        #     print(pos)

        controller.episode_callback()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    env.close()
    return ep_times


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
