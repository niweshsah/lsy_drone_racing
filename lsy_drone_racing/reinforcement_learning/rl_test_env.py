import gymnasium
import re
from pathlib import Path
import numpy as np
try:
    import keyboard
    KEYBOARD=True
except:
    KEYBOARD=False
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.reinforcement_learning.rl_drone_race import RLDroneRaceEnv, RLDroneHoverEnv

def make_env(seed):
    def _init():
        # config = load_config(Path(__file__).parents[2] / "config/levelrl_single_gate.toml")
        # env = RLDroneHoverEnv(
        config = load_config(Path(__file__).parents[2] / "config/levelrl.toml")
        env = RLDroneRaceEnv(
            freq=config.env.freq,
            sim_config=config.sim,
            track=config.env.track,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=seed,
        )
        return env
    return _init

def get_latest_model_path(log_dir: str, lesson: int, idx: int = None) -> tuple[Path, int]:
    log_path = Path(log_dir)
    pattern = re.compile(rf"ppo_final_model_{lesson}_(\d+)\.zip")

    model_files = [(f, int(m.group(1))) for f in log_path.glob(f"ppo_final_model_{lesson}_*.zip")
                   if (m := pattern.match(f.name))]

    if not model_files:
        raise FileNotFoundError(f"No model found for lesson {lesson} in {log_path}")
    
    latest_file, max_idx = max(model_files, key=lambda x: x[1])

    if idx is not None:
        for f, n in model_files:
            if n == idx:
                return f, n
        raise FileNotFoundError(f"Model ppo_final_model_{lesson}_{idx}.zip not found in {log_path}")
    return latest_file, max_idx

def test_models(model_paths, num_episodes=999, render=True):
    env_fns = [make_env(seed=0)]
    env = DummyVecEnv(env_fns)

    models = []
    for path in model_paths:
        model = PPO.load(path, env=env, device="cpu")
        models.append(model)

    current_model_idx = 0
    print(f"current model: {model_paths[current_model_idx].name}")

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step = 0
        vel_record = []
        max_vel = 0.0
        while not done:
            if KEYBOARD:
                if keyboard.is_pressed('right'):
                    current_model_idx = (current_model_idx + 1) % len(models)
                    print(f"switch to model: {model_paths[current_model_idx].name}")
                    time.sleep(0.3)

                elif keyboard.is_pressed('left'):
                    current_model_idx = (current_model_idx - 1) % len(models)
                    print(f"switch to model: {model_paths[current_model_idx].name}")
                    time.sleep(0.3)

                elif keyboard.is_pressed('q'):
                    env.close()
                    return

            model = models[current_model_idx]
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            vel_record.append(np.linalg.norm(obs[0,3:6]))

            if render:
                fps = 60
                env.envs[0].render()
        vel_record = np.array(vel_record)
        gate_pass = info[0]['target_gate']
        print(
            f"Episode {ep + 1}: reward={episode_reward}, steps={step}, gate pass={4 if gate_pass < 0 else gate_pass}, "
            f"lap time={step/env.envs[0].freq}, max vel={np.max(vel_record):.2f}, avg vel={np.mean(vel_record):.2f}"
        )

    env.close()
    

if __name__ == "__main__":
    lesson = 4
    latest_model_path, lesson_train_idx = get_latest_model_path(Path(__file__).parent / "log3", lesson, idx=None)
    print(f"Testing Lesson {lesson}.{lesson_train_idx}")
    model_paths = [
        latest_model_path,
        Path(__file__).parent / "log3/ppo_final_model_4_30",
        Path(__file__).parent / "log3/ppo_final_model_4_29",
        Path(__file__).parent / "log3/ppo_final_model_4_28",
        Path(__file__).parent / "log3/ppo_final_model_4_25",
        Path(__file__).parent / "log3/ppo_final_model_4_20",
        Path(__file__).parent / "log3/ppo_final_model_4_16",
        # Path(__file__).parent / "log2/best_model",
    ]
    test_models(model_paths)





# import gymnasium
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.vec_env import DummyVecEnv
# from torch import nn
# from datetime import datetime
# import os
# from pathlib import Path

# from docs import conf
# from lsy_drone_racing.utils import load_config
# from lsy_drone_racing.reinforcement_learning.rl_drone_race import RLDroneRaceEnv, RenderCallback

# config = load_config(Path(__file__).parents[2] / "config" / "levelrl.toml")

# env = RLDroneRaceEnv = gymnasium.make(
#         config.env.id,
#         freq=config.env.freq,
#         sim_config=config.sim,
#         sensor_range=config.env.sensor_range,
#         control_mode=config.env.control_mode,
#         track=config.env.track,
#         disturbances=config.env.get("disturbances"),
#         randomizations=config.env.get("randomizations"),
#         seed=config.env.seed,
#     )

# obs, info = env.reset()
# i = 0
# fps = 60

# while True:
#     curr_time = i / config.env.freq

#     action = None
#     obs, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         break
#     # Synchronize the GUI.
#     if config.sim.gui:
#         if ((i * fps) % config.env.freq) < fps:
#             env.render()
#     i += 1

# # Close the environment
# env.close()