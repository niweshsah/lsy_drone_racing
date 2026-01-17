import json  # noqa: D100
import os
import sys
from datetime import datetime
from pprint import pprint
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from lsy_drone_racing.control.common_functions.yaml_import import load_yaml
    from lsy_drone_racing.control.GeometryEngines.geometryEngine3 import GeometryEngine
    from lsy_drone_racing.control.model_dynamics.mpc1 import SpatialMPC, get_drone_params
    
    print("✅ All modules imported successfully!")
    print(f"GeometryEngine location: {GeometryEngine.__module__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nCurrent Python Path:")
    pprint(sys.path)
from scipy.spatial.transform import Rotation as R

# Simulation Environment Imports
try:
    from drone_models.core import load_params
    from drone_models.utils.rotation import ang_vel2rpy_rates

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.utils.utils import draw_line
except ImportError:
    print("Warning: Simulation specific modules not found.")
    
    
matplotlib.use("Agg") # Use a non-interactive backend for matplotlib

CONSTANTS = load_yaml("lsy_drone_racing/control/constants.yaml")



