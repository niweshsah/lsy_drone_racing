# test_ppo_drone.py
import re
import time
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy
from gymnasium.wrappers.vector import RecordEpisodeStatistics
import numpy as np
import torch
from torch.distributions.normal import Normal

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.reinforcement_learning.rl_env_wrapper import RLDroneRacingWrapper
from lsy_drone_racing.utils import load_config
from rl_train_ppo import load_latest_model, layer_init, make_env, Agent, Args
    
# env
def make_eval_env(num_envs=1, device="cpu"):
    cfg = load_config(Path(__file__).parents[2] / "config/trainrl.toml")

    env = VecDroneRaceEnv(
        num_envs       = num_envs,
        freq           = cfg.env.freq,
        sim_config     = cfg.sim,
        track          = cfg.env.track,
        sensor_range   = cfg.env.sensor_range,
        control_mode   = cfg.env.control_mode,
        disturbances   = cfg.env.get("disturbances"),
        randomizations = cfg.env.get("randomizations"),
        seed           = cfg.env.seed,
        device         = device,
    )
    env   = JaxToNumpy(env)
    env = RLDroneRacingWrapper(
        env,
        k_alive   = Args.k_alive,
        k_alive_anneal  = Args.k_alive_anneal,
        k_obst    = Args.k_obst,
        k_obst_d  = Args.k_obst_d,
        k_gates   = Args.k_gates,
        k_center  = Args.k_center,
        k_vel     = Args.k_vel,
        k_act     = Args.k_act,
        k_act_d   = Args.k_act_d,
        k_yaw     = Args.k_yaw,
        k_crash   = Args.k_crash,
        k_success = Args.k_success,
        k_finish  = Args.k_finish,
        k_imit    = Args.k_imit,
    ) # my custom wrapper
    env   = RecordEpisodeStatistics(env)
    return env

def main():
    log_dir = Path(__file__).parent / "log4"
    model_path = load_latest_model(log_dir)
    model_path = Path(__file__).parent / "log4" / "rl_drone_racing_iter_15.pth"

    env = make_eval_env(num_envs=1)
    agent    = Agent(env).to("cpu")
    agent.load_state_dict(torch.load(model_path))
    # agent.eval()

    EPISODES = 20
    for ep in range(EPISODES):
        obs, _ = env.reset()
        obs    = torch.Tensor(obs)
        done   = False
        while not done:
            # act = agent.act(obs, deterministic=False)
            with torch.no_grad():
                act, _, _, _ = agent.get_action_and_value(obs)
            obs, reward, terminated, truncated, info = env.step(act.cpu().numpy())
            obs  = torch.tensor(obs, dtype=torch.float32)
            done = np.logical_or(terminated, truncated).item()
            env.render()
            
            if "episode" in info:
                ep_return = np.sum(info['episode']['r'][info['_episode']])
                ep_length = np.sum(info['episode']['l'][info['_episode']])
                print(f"[Episode {ep+1}] return = {ep_return:.2f} | length = {ep_length}")

    env.close()

if __name__ == "__main__":
    main()
