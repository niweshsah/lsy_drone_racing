from posixpath import relpath
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.jax_to_numpy import jax_to_numpy
from lsy_drone_racing.envs.drone_race import DroneRaceEnv, VecDroneRaceEnv
from jax import Array
import jax.numpy as jp
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim.physics import ang_vel2rpy_rates
from lsy_drone_racing.utils import draw_line

from lsy_drone_racing.tools import race_objects
from lsy_drone_racing.envs.race_core import RaceCoreEnv, build_action_space, build_observation_space

from jax import Array
from ml_collections import ConfigDict
from typing import Dict, Tuple

IMMITATION_LEARNING = True
if IMMITATION_LEARNING:
    from pathlib import Path
    from lsy_drone_racing.utils import load_config
    from rl_teacher_policy_att_pid import AttitudeController
RAND_INIT = True

class RLDroneRacingWrapper(gymnasium.vector.VectorWrapper):
    def __init__(self, 
                 env: VecDroneRaceEnv,
                 k_alive = 0.5,
                 k_alive_anneal = 0.1,
                 k_obst = 0.2,
                 k_obst_d = 0.5,
                 k_gates = 2.0,
                 k_center = 0.3,
                 k_center_d = 0.1,
                 k_vel = +0.04,
                 k_act = 0.01,
                 k_act_d = 0.001,
                 k_yaw = 0.1,
                 k_crash = 25,
                 k_success = 40,
                 k_finish = 60,
                 k_imit = 0.4):
        super().__init__(env)
        # turn off autoreset
        env.unwrapped.autoreset = False
        self.marked_for_reset = np.zeros(env.num_envs, dtype=bool)
        # create action & observation spaces
        self._num_envs = env.num_envs
        self.action_space = env.action_space
        state_dim = 36
        lim = np.full(state_dim, np.inf, dtype=np.float32) # set to infinite for now
        self.single_observation_space = spaces.Box(-lim, lim, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self._num_envs)
        # initialize internal state saving variables
        self.obs_env = None
        self._d_safe = 1.0
        self._act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        self._prev_act = np.repeat(self._act_bias[None, :], self._num_envs, axis=0)
        self._prev_gate = np.zeros(self._num_envs, dtype=int)
        self._steps = np.zeros(self._num_envs, dtype=int)
        # fetch properties and methods from core env
        self.sim = self.find_attr(env, 'sim')
        self._reset_env_data = self.find_attr(env, '_reset_env_data')
        self.obs = self.find_attr(env, 'obs')
        self.info = self.find_attr(env, 'info')
        # region Param
        """REWARD PARAMETERS"""
        self.k_alive = k_alive
        self.k_alive_anneal = k_alive_anneal
        self.k_obst = k_obst
        self.k_obst_d = k_obst_d
        self.k_gates = k_gates
        self.k_center = k_center
        self.k_center_d = k_center_d
        self.k_vel = k_vel
        self.k_act = k_act
        self.k_act_d = k_act_d
        self.k_yaw = k_yaw
        self.k_crash = k_crash
        self.k_success = k_success
        self.k_finish = k_finish
        self.k_imit = k_imit

    # region Reset
    @staticmethod
    def find_attr(env, attr_name):
        while hasattr(env, 'unwrapped'):
            if hasattr(env, attr_name):
                return getattr(env, attr_name)
            env = env.unwrapped
        raise AttributeError(f"Attribute '{attr_name}' not found.")

    def _reset(self, seed=None, options=None, mask=None):
        # random initialization
        if seed is not None:
            self.sim.seed(seed)
        self.sim.reset(mask=mask)

        if RAND_INIT:
            mask = mask if mask is not None else jp.ones(self.unwrapped.unwrapped.data.steps.shape, dtype=bool)
            num_reset = mask.sum()
            # manually recorded init points
            self.rand_init_list = [
                # {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0}, # emphasize takeoff point
                # {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0},
                # {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0},
                # {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0},
                {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0},
                {'pos': jp.array([0.9081, 1.1422, 0.2201]), 'vel': jp.array([-0.2142, -0.7419, 0.2087]), 'quat': jp.array([0.1611, -0.0436, 0.0031, 0.9860]), 'f_thrust': 0.3179, 'target_gate': 0},
                {'pos': jp.array([0.7550, 0.6635, 0.3080]), 'vel': jp.array([-0.2109, -0.7631, 0.1146]), 'quat': jp.array([0.0452, 0.0307, -0.0066, 0.9985]), 'f_thrust': 0.2883, 'target_gate': 0},
                {'pos': jp.array([0.2309, -1.1061, 1.0188]), 'vel': jp.array([0.1798, -0.5673, 0.4537]), 'quat': jp.array([-0.0357, 0.0800, 0.0031, 0.9965]), 'f_thrust': 0.2255, 'target_gate': 1},
                {'pos': jp.array([0.5624, -1.2678, 1.1197]), 'vel': jp.array([1.0049, 0.1084, 0.1169]), 'quat': jp.array([-0.0709, 0.0366, -0.0009, 0.9968]), 'f_thrust': 0.2705, 'target_gate': 1},
                {'pos': jp.array([1.1311, -0.8747, 1.1062]), 'vel': jp.array([0.0588, 1.0162, -0.1100]), 'quat': jp.array([-0.0605, -0.1642, -0.0146, 0.9845]), 'f_thrust': 0.2624, 'target_gate': 2},
                {'pos': jp.array([0.6138, -0.0001, 0.8368]), 'vel': jp.array([-0.5123, 0.6669, -0.3205]), 'quat': jp.array([-0.0417, -0.0282, 0.0048, 0.9987]), 'f_thrust': 0.2299, 'target_gate': 2},
                {'pos': jp.array([0.0045, 0.9539, 0.4696]), 'vel': jp.array([-0.1742, 0.8196, -0.0696]), 'quat': jp.array([0.1123, 0.0797, -0.0008, 0.9905]), 'f_thrust': 0.2878, 'target_gate': 2},
                {'pos': jp.array([-0.0996, 0.9104, 0.5883]), 'vel': jp.array([-0.3977, -0.8926, 0.0738]), 'quat': jp.array([0.0938, -0.0228, -0.0006, 0.9953]), 'f_thrust': 0.2662, 'target_gate': 3},
                {'pos': jp.array([-0.2380, 0.5384, 0.7121]), 'vel': jp.array([-0.3728, -1.1330, 0.7930]), 'quat': jp.array([-0.0511, -0.0051, -0.0002, 0.9987]), 'f_thrust': 0.3220, 'target_gate': 3},
            ]
            # randomly pick one init points for envs to be reset
            rand_indices = np.random.randint(len(self.rand_init_list), size=int(num_reset))
            init_pos = jp.stack([self.rand_init_list[i]['pos'] for i in rand_indices])
            init_vel = jp.stack([self.rand_init_list[i]['vel'] for i in rand_indices])
            init_quat = jp.stack([self.rand_init_list[i]['quat'] for i in rand_indices])
            target_gate = jp.array([self.rand_init_list[i]['target_gate'] for i in rand_indices])
            
            self.sim.data = self.sim.data.replace(
                states=self.sim.data.states.replace(
                    pos=self.sim.data.states.pos.at[mask].set(init_pos[:,None,:]),
                    vel=self.sim.data.states.vel.at[mask].set(init_vel[:,None,:]),
                    quat=self.sim.data.states.quat.at[mask].set(init_quat[:,None,:]),
                )
            )
        self.unwrapped.unwrapped.data = self._reset_env_data(self.unwrapped.unwrapped.data, self.sim.data.states.pos, mask) # NOTE: self.unwrapped.unwrapped.data and self.sim.data are different
        
        if RAND_INIT:
            # correct self.unwrapped.unwrapped.data after _reset_env_data()
            self.unwrapped.unwrapped.data = self.unwrapped.unwrapped.data.replace(
                target_gate=self.unwrapped.unwrapped.data.target_gate.at[mask].set(target_gate[:,None])
            )
            pass

        return self.obs(), self.info()


    def reset(self, *, seed: int | None = None, options: dict | None = None, mask: Array | None = None) -> Tuple[np.ndarray, Dict]:
        # call lower level reset
        # obs, info = self.env.unwrapped._reset(seed=seed, mask=mask)
        obs, info = self._reset(seed=seed, mask=mask)
        self.obs_env = {k: jax_to_numpy(v[:, 0]) for k, v in obs.items()}
        info = {k: jax_to_numpy(v[:, 0]) for k, v in info.items()}
        state = self._obs_to_state(self.obs_env,
                                   np.repeat(self._act_bias[None, :], self._num_envs, axis=0))
        # reset storage
        self._prev_drone_pos = self.obs_env['pos']
        self._prev_obst_xy = self.obs_env['obstacles_pos'][:, 0, :2]
        self._prev_act = np.repeat(self._act_bias[None, :], self._num_envs, axis=0)
        self._prev_gate = np.zeros(self._num_envs, dtype=int)
        self._prev_gate_pos = self.obs_env['gates_pos'][:, 0]
        self._steps[mask] = int(0)
        if mask is None or mask[0]: # if the first world is reset
            self.traj_record = self.obs_env['pos'][0, :] # debug trajectory
        # setup teacher policy
        if IMMITATION_LEARNING:
            config = load_config(Path(__file__).parents[2] / "config/level0.toml")
            self.teacher_controller = AttitudeController(self.obs_env, info, config, self)
        return state, info

    # region Step
    def step(self, action: np.ndarray):
        if IMMITATION_LEARNING: # test teacher policy
            action = self.teacher_controller.compute_control(self.obs_env, None) - self._act_bias
        action_exec = action + self._act_bias
        self.obs_env, _, terminated, truncated, info = self.env.step(action_exec)
        state = self._obs_to_state(self.obs_env, action)
        reward = self._reward(self.obs_env, state, action)
        # self handle autoreset
        if self.marked_for_reset.any():
            # add crash & finish reward
            r_crash = -self.k_crash * (self.marked_for_reset & (self.obs_env["target_gate"] >= 0))
            r_finish = self.k_finish * (self.marked_for_reset & (self.obs_env["target_gate"] < 0))
            reward += r_crash + r_finish
            # reset specific world
            state, info = self.reset(mask=self.marked_for_reset)
            terminated = terminated & ~self.marked_for_reset
            truncated = truncated & ~self.marked_for_reset
        done = terminated | truncated
        self.marked_for_reset = done # update mask after reset

        self.traj_record = np.vstack([self.traj_record, self.obs_env['pos'][0, :]]) # debug trajectory
        self._steps += 1
        try:
            draw_line(self, self.traj_record[0:-1:5], rgba=np.array([0.0, 1.0, 0.0, 0.2]))
        except:
            pass

        return state, reward, done, truncated, info

    # region OBS
    def _obs_to_state(self, obs: dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        """
        Args:
            obs    : Dict[str, np.ndarray]
                    pos            (N, 3)
                    vel            (N, 3)
                    quat           (N, 4)
                    ang_vel        (N, 3)
                    gates_pos      (N, n_gates, 3)
                    gates_quat     (N, n_gates, 4)
                    obstacles_pos  (N, n_obst, 3)
            action : np.ndarray, shape (N, 4)
        Returns:
            state  : np.ndarray, shape (N, 36)
        """
        pos      = obs["pos"]           # (N, 3)
        vel      = obs["vel"]           # (N, 3)
        quat     = obs["quat"]          # (N, 4)
        ang_vel  = obs["ang_vel"]       # (N, 3)
        N        = pos.shape[0]

        curr_gate_idx = obs['target_gate']
        gate_quat = obs["gates_quat"][np.arange(N), curr_gate_idx]   # (N, 4)
        gate_pos  = obs["gates_pos"][np.arange(N), curr_gate_idx]    # (N, 3)
        gate_rot_mat = R.from_quat(gate_quat).as_matrix()

        half_w, half_h = 0.2, 0.2
        corners_local = np.array([
            [-half_w, 0.0,  half_h],
            [ half_w, 0.0,  half_h],
            [-half_w, 0.0, -half_h],
            [ half_w, 0.0, -half_h],
        ])
        gate_corners = (gate_rot_mat @ corners_local.T).transpose(0, 2, 1) + gate_pos[:, None, :]
        draw_line(self, np.stack([gate_corners[0, 0], pos[0]]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 1], pos[0]]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 2], pos[0]]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 3], pos[0]]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        rel_pos_gate = gate_corners - pos[:, None, :]   # (N, 4, 3)

        obst_rel_xy = obs["obstacles_pos"][:, :, :2] - pos[:, None, :2]       # (N, n_obst, 2)
        obst_dists  = np.linalg.norm(obst_rel_xy, axis=-1)                    # (N, n_obst)
        closest_idx = obst_dists.argmin(axis=-1)                              # (N,)
        rel_xy_obst = obst_rel_xy[np.arange(N), closest_idx]                  # (N, 2)
        dist        = obst_dists[np.arange(N), closest_idx]                   # (N,)

        rel_xy_obst_gaus = rel_xy_obst * np.exp(-(dist / (0.5 * self._d_safe))**2)[:, None] \
                        / (dist[:, None] + 1e-6)                              # (N, 2)


        rot_mat  = R.from_quat(quat).as_matrix().reshape(N, -1)               # (N, 9)
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)                          # (N, 3)

        state = np.concatenate([
            pos,                        # (N, 3)
            vel,                        # (N, 3)
            rot_mat,                    # (N, 9)
            rpy_rates,                  # (N, 3)
            rel_pos_gate.reshape(N, -1),# (N, 12)
            rel_xy_obst_gaus,           # (N, 2)
            action                      # (N, 4)
        ], axis=-1).astype(np.float32)  # => (N, 36)

        # save to self just in case
        self.rel_pos_gate = rel_pos_gate     # (N, 4, 3)
        self.rel_xy_obst  = rel_xy_obst      # (N, 2)

        return state
    
    # region Reward
    def _reward(self, obs: dict, obs_rl: np.ndarray, act: np.ndarray) -> np.ndarray:
        """
        Args:
            obs    : dict[str, np.ndarray]
            obs_rl : np.ndarray, shape (N, 36)
            act    : np.ndarray, shape (N, 4)

        Returns:
            rewards: np.ndarray, shape (N,)
        """
        N = act.shape[0]
        rewards = np.full(N, self.k_alive, dtype=float)
        rewards = rewards * (self.k_alive_anneal ** self._steps)
        # data preparation
        curr_gate   = obs["target_gate"].astype(int)                # (N,)
        drone_pos   = obs["pos"]                                    # (N, 3)
        drone_vel   = obs["vel"]                                    # (N, 3)
        gates_pos   = obs["gates_pos"]                              # (N, n_gates, 3)
        gates_quat  = obs["gates_quat"]                             # (N, n_gates, 4)
        rel_xy_obst_gaus = obs_rl[:, -6:-4]                         # (N, 2)
        obst_xy     = self.rel_xy_obst + drone_pos[:, :2]           # (N, 2)
        gate_pos = gates_pos[np.arange(N), curr_gate]               # (N, 3)
        gate_quat = gates_quat[np.arange(N), curr_gate]             # (N, 4)
        gates_norm = R.from_quat(gate_quat).as_matrix()[:, :, 1]    # (N, 3)
        rel_gate   = gate_pos - drone_pos                           # (N, 3)
        # reward: success passing gates | handle gate switching
        prev_gate_delta = (curr_gate != self._prev_gate)   # (N,) bool
        rewards[prev_gate_delta] += self.k_success         # gate pass reward
        self._prev_gate_pos[prev_gate_delta] = gate_pos[prev_gate_delta] # update changed gates position
        # reward: obstacle distance P&D
        r_obst   = -self.k_obst * np.linalg.norm(rel_xy_obst_gaus, axis=1)
        dist_now   = np.linalg.norm(obst_xy - drone_pos[:, :2], axis=1)
        dist_prev  = np.linalg.norm(self._prev_obst_xy - self._prev_drone_pos[:, :2], axis=1)
        r_obst_d = -self.k_obst_d * np.linalg.norm(rel_xy_obst_gaus, axis=1) * (dist_prev - dist_now)
        # reward: gates distance D
        prev_gate_delta_pos = np.linalg.norm(self._prev_gate_pos - self._prev_drone_pos, axis=1)
        curr_gate_delta_pos = np.linalg.norm(rel_gate, axis=1)
        r_gates = self.k_gates * (prev_gate_delta_pos - curr_gate_delta_pos)
        # reward: deviation from gate center line
        dot_rg = (rel_gate * gates_norm).sum(axis=1)
        vec_center = rel_gate - gates_norm * dot_rg[:, None]
        r_center = -self.k_center * np.linalg.norm(vec_center, axis=1) / (np.linalg.norm(rel_gate, axis=1) + 1)
        r_center_d = self.k_center_d * (drone_vel * vec_center).sum(axis=1)
        # reward: smooth action
        r_act = -self.k_act * np.linalg.norm(act, axis=1) \
                -self.k_act_d * np.linalg.norm(act - self._prev_act, axis=1)
        # reward: velocity related
        r_vel = self.k_vel * (1 + np.linalg.norm(rel_xy_obst_gaus, axis=1) - r_obst_d) \
                * np.linalg.norm(drone_vel, axis=1)
        # reward: yaw angle
        yaw = np.sum(np.abs(R.from_quat(obs["quat"]).as_euler('zyx', degrees=False)), axis=-1)
        r_yaw = -self.k_yaw * yaw

        # sum up
        rewards += r_obst + r_obst_d + r_gates + r_center + r_center_d + r_act + r_vel + r_yaw

        # reward: immitation learning
        if IMMITATION_LEARNING:
            demo_action = self.teacher_controller.compute_control(self.obs_env, None) - self._act_bias
            r_imit = -self.k_imit * np.linalg.norm(demo_action - act, axis=1)
            rewards += r_imit

        # reward debug
        # i = 0
        # print(
        #     f"alive:{self.k_alive * self.k_alive_anneal ** self._steps[i]:+.3f} | obst:{r_obst[i]:+.3f} | obst_d:{r_obst_d[i]:+.3f} | "
        #     f"gates:{r_gates[i]:+.3f} | center:{r_center[i]:+.3f} | center_d:{r_center_d[i]:+.3f}  | pass:{(self.k_success if prev_gate_delta[i] else 0.0):+.3f} | "
        #     f"act:{r_act[i]:+.3f} | vel:{r_vel[i]:+.3f} | yaw:{r_yaw[i]:+.3f} | "
        #     f"imit:{r_imit[i]:+.3f} | "
        #     f"total:{rewards[i]:+.3f}"
        # )
        
        # update saving
        self._prev_act        = act
        self._prev_gate       = curr_gate
        self._prev_gate_pos   = gate_pos
        self._prev_obst_xy    = obst_xy
        self._prev_drone_pos  = drone_pos

        return rewards.astype(np.float32)
