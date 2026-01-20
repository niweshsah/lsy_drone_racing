from lsy_drone_racing.control.GeometryEngines.final_geometryEngine import GeometryEngine
import toml
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_from_toml(filepath: str):
    print(f"Loading config from: {filepath}")
    with open(filepath, "r") as f:
        data = toml.load(f)
    gates_raw = data["env"]["track"]["gates"]
    gates_pos = np.array([g["pos"] for g in gates_raw], dtype=np.float64)
    gates_rpy = np.array([g.get("rpy", [0, 0, 0]) for g in gates_raw], dtype=np.float64)
    rot = R.from_euler("xyz", gates_rpy, degrees=False)
    matrices = rot.as_matrix()
    gates_normals = matrices[:, :, 0]
    gates_y = matrices[:, :, 1]
    gates_z = matrices[:, :, 2]
    obs_raw = data["env"]["track"].get("obstacles", [])
    obstacles_pos = (
        np.array([o["pos"] for o in obs_raw], dtype=np.float64) if obs_raw else np.empty((0, 3))
    )
    start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=np.float64)
    return gates_pos, gates_normals, gates_y, gates_z, obstacles_pos, start_pos

if __name__ == "__main__":
    toml_path = "config/level1.toml"

    gates_pos, gates_normals, gates_y, gates_z, obstacles_pos, start_pos = load_from_toml(toml_path)

    gate_size = 0.5

    geometry_engine = GeometryEngine(
        gates_pos=gates_pos.tolist(),
        gates_normal=gates_normals.tolist(),
        gates_y=gates_y.tolist(),
        gates_z=gates_z.tolist(),
        gate_size=gate_size,
        obstacles_pos=obstacles_pos.tolist()
    )
   
    geometry_engine.plot()

    # Now you can use geometry_engine for further processing or visualization
    print("GeometryEngine initialized successfully.")