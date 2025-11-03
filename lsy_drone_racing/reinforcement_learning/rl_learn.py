import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn
from datetime import datetime
import os
import re
from pathlib import Path
import time
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.reinforcement_learning.rl_drone_race import RLDroneRaceEnv, RLDroneHoverEnv, RenderCallback
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# === 1. 创建训练环境 ===

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
                return f, max_idx+1
        raise FileNotFoundError(f"Model ppo_final_model_{lesson}_{idx}.zip not found in {log_path}")
    return latest_file, max_idx+1

def main():
    num_envs = 20
    start_time = time.time()
    env_fns = [make_env(seed=i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # === 2. 设置训练保存目录和回调 ===
    log_dir = Path(__file__).parent / "log3"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=log_dir, name_prefix=f"model_{timestamp}")
    eval_callback = EvalCallback(vec_env, best_model_save_path=log_dir, eval_freq=10000, n_eval_episodes=5)

    # === 3. 初始化 PPO 模型 ===
    policy_kwargs = dict(
        net_arch=[128, 128],         # 两层，每层 128
        activation_fn=nn.ReLU        # 激活函数（默认是 Tanh，可以改为 ReLU）
    )
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        device="cpu",
    )
    # 加载模型
    lesson = 4
    lesson_train_idx = None # default None, use if need reset to earlier model
    latest_model_path, lesson_train_idx = get_latest_model_path(log_dir, lesson, idx=lesson_train_idx)
    model = PPO.load(latest_model_path, env=vec_env, device="cpu")
    print(f"Learning Lesson {lesson}.{lesson_train_idx}")

    # === 4. 启动训练 ===
    if num_envs > 1:
        model.learn(total_timesteps=2*1_000_000, callback=[checkpoint_callback, eval_callback])
    else: # for visualization
        render_callback = RenderCallback(render_freq=1)
        model.learn(total_timesteps=10000, callback=[render_callback])

    # === 5. 保存最终模型 ===
    model.save(f"{log_dir}/ppo_final_model_{lesson}_{lesson_train_idx}")
    end_time = time.time()
    print(f"✅ 训练完成，耗时{end_time - start_time}s ，模型已保存至 ppo_final_model_{lesson}_{lesson_train_idx}")


if __name__ == "__main__":
    main()
