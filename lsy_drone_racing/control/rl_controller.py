"""Controller that follows RL policy.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

# utils
from scipy.spatial.transform import Rotation as R
from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim.physics import ang_vel2rpy_rates
from lsy_drone_racing.utils.utils import draw_line
import re
from pathlib import Path
# RL
from stable_baselines3 import PPO

class RLController(Controller):
    """Trajectory controller following RL policy."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, env=None):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self.env = env # just for visualization
        
        # create model & load parameter
        log_dir = Path(__file__).parent.parent / "reinforcement_learning/log5"
        # lesson = 4
        # model_path, model_idx = self.get_latest_model_path(log_dir, lesson)
        model_path = log_dir / "ppo_final_model_stablest_5s.zip"
        print(f"[RLController] Loaded model: {model_path.name}")
        self.model = PPO.load(model_path, device="cpu")

        # init parameters
        self.act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32) # action bias to add to policy output
        self.action = np.zeros(4) # for first prev_action input to rl agent
        self.d_safe = 1.0
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False


    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        # run _obs_to_state() to get rl type states
        state = self._obs_to_state(obs, self.action)

        # fetch action from model
        input_obs = state[None, :]  # add batch dimension
        action, _ = self.model.predict(input_obs, deterministic=True)

        # return action
        self.action = action.squeeze() + self.act_bias

        #region Visualization
        if not hasattr(self, "trajectory_record"):
            self.trajectory_record = []
        self.trajectory_record.append(obs['pos'])
        try:
            draw_line(self.env, np.array(self.trajectory_record), rgba=self.hex2rgba("#2BFF00F9")) # recorded trajectory
        except:
            pass


        return self.action

    # region obs
    def _obs_to_state(self, obs: dict[str, NDArray], action: NDArray) -> NDArray:
        # define rl input states: [pos(3), vel(3), rot_mat(9), rpy_rates(3), rel_pos_gate(4*3), rel_xy_obst(2), prev_act(4)] (dim=36)
        pos = obs["pos"].squeeze()
        vel = obs["vel"].squeeze()
        quat = obs["quat"].squeeze()
        ang_vel = obs["ang_vel"].squeeze()

        # calc vectors pointing to four gate corners
        curr_gate = obs['target_gate']
        self.gates_size = [0.4, 0.4] # [width, height]
        self.gate_rot_mat = np.array(R.from_quat(obs['gates_quat'][curr_gate]).as_matrix())
        half_w, half_h = self.gates_size[0] / 2, self.gates_size[1] / 2
        corners_local = np.array([
            [-half_w, 0.0,  half_h],
            [ half_w, 0.0,  half_h],
            [-half_w, 0.0, -half_h],
            [ half_w, 0.0, -half_h],
        ])
        gate_corners_pos = (self.gate_rot_mat @ corners_local.T).T + obs['gates_pos'][curr_gate]  # shape: (4, 3)
        draw_line(self.env, np.stack([gate_corners_pos[0], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self.env, np.stack([gate_corners_pos[1], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self.env, np.stack([gate_corners_pos[2], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self.env, np.stack([gate_corners_pos[3], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        
        self.rel_pos_gate = gate_corners_pos - pos[None, :]  # shape: (4, 3)
        # self.rel_pos_gate_norm = self.rel_pos_gate / np.linalg.norm(self.rel_pos_gate, axis=1, keepdims=True) # normalize

        obst_rel_xy_list = np.array(obs['obstacles_pos'])[:,:2] - pos[:2]
        obst_dists = np.linalg.norm(obst_rel_xy_list, axis=-1)
        closest_obst_idx = np.argmin(obst_dists)
        self.rel_xy_obst = obst_rel_xy_list[closest_obst_idx]
        dist = obst_dists[closest_obst_idx]
        self.rel_xy_obst_gaus = self.rel_xy_obst * np.exp(-(dist/(0.5*self.d_safe))**2) / (dist+1e-6)
        try:
            draw_line(self, self.traj_record[0:-1:5], rgba=np.array([0.0, 1.0, 0.0, 0.2]))
            draw_line(self, np.stack([np.concatenate([self.rel_xy_obst_gaus, np.array([0])])+pos, pos]), rgba=np.array([1.0, 0.0, 1.0, 0.5]))
        except:
            pass
        
        # calc euler
        rot_mat = R.from_quat(quat).as_matrix().reshape(-1)
        
        # ang_vel to rpy_rates
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)
        
        state = np.concatenate([pos, vel, rot_mat, rpy_rates, self.rel_pos_gate.reshape(-1), self.rel_xy_obst_gaus, action])
        # state = np.concatenate([pos, vel, rot_mat, rpy_rates, self.rel_pos_gate.reshape(-1), action]) # old version
        return state
    
    @staticmethod
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
    
    @staticmethod
    def hex2rgba(hex="#FFFFFFFF"):
        hex = hex.lstrip('#')
        r, g, b, a = int(hex[0:2], 16), int(hex[2:4], 16), int(hex[4:6], 16), int(hex[6:8], 16)
        rgba = np.array([r/255, g/255, b/255, a/255])
        return rgba

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished
        """Reset the time step counter."""
        self._tick = 0
