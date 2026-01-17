from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# Assuming imports work in your local environment
from lsy_drone_racing.control.common_functions.yaml_import import load_yaml

CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")

class GeometryEngine:
    
    def __init__(self, gates_pos : List[List[float]], gates_normal : List[List[float]], gate_size : float, obstacles: List[Dict[str, NDArray]], start_pos: List[float], start_orient: List[float] = [0,0,0], obs: dict[str, NDArray[np.floating]] = {}, 
        info: dict = {}, 
        sim_config: dict = {}):
        
        self.gates_pos = np.array(gates_pos)
        self.gates_normal = np.array(gates_normal)
        self.gate_size = gate_size
        self.obstacles = obstacles
        self.start_pos = np.array(start_pos)
        self.start_orient = R.from_euler('xyz', start_orient).as_matrix() # Convert to rotation matrix
        self.obs = obs
        self.info = info
        self.sim_config = sim_config    