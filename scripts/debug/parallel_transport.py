import os

import matplotlib.pyplot as plt
import numpy as np
import toml
from scipy.spatial.transform import Rotation as R


class GeometryEngine:
    def __init__(self, gates_pos, gates_normals, gates_y, gates_z, obstacles_pos, start_pos):
        # --- FIX: Force float64 to avoid integer division errors ---
        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.gate_y = np.asarray(gates_y, dtype=np.float64)
        self.gate_z = np.asarray(gates_z, dtype=np.float64)
        self.obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        # -----------------------------------------------------------

        # 1. Setup Waypoints
        # self.waypoints = np.vstack((self.start_pos, self.gates_pos))

        # # 2. Compute "Smart" Tangents
        # # (Fixes loops and sharp turns by blending normals)
        # self.tangents = self._compute_smoothed_tangents(blend_strength=0.9)

        # # 3. Spline Generation (Chord-length parameterization)
        # dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        # self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        # self.total_length = self.s_knots[-1]

        # self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # 4. Compute Parallel Transport Frame
        self.pt_frame = self._generate_parallel_transport_frame(num_points=1000)

    def _compute_smoothed_tangents(self, blend_strength=0.9):
        n_points = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)

        # --- A. Start Tangent ---
        start_dir = self.waypoints[1] - self.waypoints[0]
        tangents[0] = start_dir / np.linalg.norm(start_dir)

        # --- B. Gate Tangents ---
        for i in range(1, n_points):
            curr_p = self.waypoints[i]
            prev_p = self.waypoints[i - 1]

            # Vector arriving at current gate
            v_in = curr_p - prev_p
            if np.linalg.norm(v_in) > 1e-6:
                v_in /= np.linalg.norm(v_in)

            # Vector leaving current gate
            if i < n_points - 1:
                next_p = self.waypoints[i + 1]
                v_out = next_p - curr_p
                if np.linalg.norm(v_out) > 1e-6:
                    v_out /= np.linalg.norm(v_out)
            else:
                v_out = v_in

            # 1. Compute "Natural" Flow
            t_natural = v_in + v_out
            if np.linalg.norm(t_natural) > 1e-6:
                t_natural /= np.linalg.norm(t_natural)
            else:
                t_natural = v_in

            # 2. Get Strict Gate Normal
            gate_idx = i - 1
            t_strict = self.gate_normals[gate_idx].copy()

            # 3. AUTO-FLIP Check
            if np.dot(t_strict, t_natural) < 0:
                t_strict = -t_strict
                self.gate_normals[gate_idx] = t_strict

            # 4. Blend
            t_final = (blend_strength * t_strict) + ((1 - blend_strength) * t_natural)
            t_final /= np.linalg.norm(t_final)
            tangents[i] = t_final

        return tangents

    def _generate_parallel_transport_frame(self, num_points=1000):
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]

        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []}

        # Initial Frame Setup
        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)

        g_vec = np.array([0, 0, -1])
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = g_vec - np.dot(g_vec, t0) * t0

        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            k_vec = self.spline(s, 2)

            k1 = -np.dot(k_vec, curr_n1)
            k2 = -np.dot(k_vec, curr_n2)

            next_n1 = curr_n1 + (k1 * curr_t) * ds
            next_n2 = curr_n2 + (k2 * curr_t) * ds

            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

            next_n1 = next_n1 - np.dot(next_n1, next_t) * next_t
            next_n1 /= np.linalg.norm(next_n1)
            next_n2 = np.cross(next_t, next_n1)

            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            frames["k1"].append(k1)
            frames["k2"].append(k2)

            curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        for k in frames:
            frames[k] = np.array(frames[k])
        return frames

    def check_curvature_limits(self, desired_radius):
        k_mag = np.sqrt(self.pt_frame["k1"] ** 2 + self.pt_frame["k2"] ** 2)
        k_max = np.max(k_mag)
        min_path_radius = 1.0 / k_max if k_max > 1e-6 else np.inf

        print("-" * 40)
        print("GEOMETRY CHECK:")
        print(f"  Max Curvature (k):  {k_max:.4f} m^-1")
        print(f"  Min Path Radius:    {min_path_radius:.4f} m")

        if desired_radius >= min_path_radius:
            print(
                f"  [FAIL] Tunnel Radius ({desired_radius}m) > Min Path Radius ({min_path_radius:.4f}m)."
            )
            return False, k_mag
        else:
            print(
                f"  [PASS] Geometry Valid. Safety Margin: {(min_path_radius - desired_radius):.4f} m"
            )
            return True, k_mag

    def plot(self, tunnel_radius=0.35):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        valid, k_mag = self.check_curvature_limits(tunnel_radius)

        path = self.pt_frame["pos"]
        n1 = self.pt_frame["n1"]
        n2 = self.pt_frame["n2"]

        # --- Plot Tunnel ---
        tube_res = 12
        tube_step = 5
        theta = np.linspace(0, 2 * np.pi, tube_res)
        p_sub = path[::tube_step]
        n1_sub = n1[::tube_step]
        n2_sub = n2[::tube_step]

        cos_t = np.cos(theta)[None, :]
        sin_t = np.sin(theta)[None, :]

        X = p_sub[:, 0, None] + tunnel_radius * (
            n1_sub[:, 0, None] * cos_t + n2_sub[:, 0, None] * sin_t
        )
        Y = p_sub[:, 1, None] + tunnel_radius * (
            n1_sub[:, 1, None] * cos_t + n2_sub[:, 1, None] * sin_t
        )
        Z = p_sub[:, 2, None] + tunnel_radius * (
            n1_sub[:, 2, None] * cos_t + n2_sub[:, 2, None] * sin_t
        )

        color = "cyan" if valid else "red"
        ax.plot_surface(X, Y, Z, color=color, alpha=0.15, linewidth=0)
        ax.plot_wireframe(X, Y, Z, color="gray", alpha=0.1, rstride=5, cstride=100)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "k-", linewidth=2, label="Drone Path")

        # --- Plot Gates ---
        for i, (p, n, y, z) in enumerate(
            zip(self.gates_pos, self.gate_normals, self.gate_y, self.gate_z)
        ):
            s = 0.45
            corners = np.array(
                [
                    p + s * (y + z),
                    p + s * (y - z),
                    p + s * (-y - z),
                    p + s * (-y + z),
                    p + s * (y + z),
                ]
            )
            ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="magenta", linewidth=3)
            ax.text(
                p[0], p[1], p[2] + 0.6, f"G{i}", color="magenta", fontsize=12, fontweight="bold"
            )
            ax.quiver(p[0], p[1], p[2], n[0], n[1], n[2], length=0.6, color="magenta", alpha=0.5)

        ax.scatter(
            self.start_pos[0], self.start_pos[1], self.start_pos[2], c="green", s=100, label="Start"
        )

        mid_x, mid_y, mid_z = np.mean(path[:, 0]), np.mean(path[:, 1]), np.mean(path[:, 2])
        r = 3.0
        ax.set_xlim(mid_x - r, mid_x + r)
        ax.set_ylim(mid_y - r, mid_y + r)
        ax.set_zlim(mid_z - r, mid_z + r)
        plt.legend()
        plt.show()


# ---------------------------------------------------------
# Loader Logic
# ---------------------------------------------------------
def load_from_toml(filepath: str):
    """Parses the TOML file. Handles RPY (Roll-Pitch-Yaw) if present."""
    print(f"Loading config from: {filepath}")
    with open(filepath, "r") as f:
        data = toml.load(f)

    # 1. Parse Gates
    gates_raw = data["env"]["track"]["gates"]
    print(f"Found {len(gates_raw)} gates.")

    # Extract Positions
    gates_pos = np.array([g["pos"] for g in gates_raw], dtype=np.float64)

    # Extract Orientations (RPY)
    # Default to [0,0,0] if not found
    gates_rpy = np.array([g.get("rpy", [0, 0, 0]) for g in gates_raw], dtype=np.float64)

    # Convert RPY -> Rotation Matrices -> Normals/Axes
    # Assuming standard "xyz" Euler order (common in robotics)
    rot = R.from_euler(
        "xyz", gates_rpy, degrees=False
    )  # Change degrees=True if your TOML uses degrees
    matrices = rot.as_matrix()

    # In body frame: X=Normal, Y=Left, Z=Up (Standard for many drone setups)
    gates_normals = matrices[:, :, 0]
    gates_y = matrices[:, :, 1]
    gates_z = matrices[:, :, 2]

    # 2. Parse Obstacles
    obs_raw = data["env"]["track"].get("obstacles", [])
    obstacles_pos = (
        np.array([o["pos"] for o in obs_raw], dtype=np.float64) if obs_raw else np.empty((0, 3))
    )

    # 3. Parse Start Pos
    start_pos = np.array(data["env"]["track"]["drones"][0]["pos"], dtype=np.float64)

    return gates_pos, gates_normals, gates_y, gates_z, obstacles_pos, start_pos


if __name__ == "__main__":
    # 1. Try to load file, otherwise use dummy data
    toml_path = "config/level1_noObstacle.toml"

    if os.path.exists(toml_path):
        try:
            g_pos, g_norm, g_y, g_z, obs_pos, s_pos = load_from_toml(toml_path)
            print("Loaded TOML successfully.")
        except Exception as e:
            print(f"Error loading TOML: {e}")
            # Fallback if file exists but parsing fails
            exit(1)
    else:
        print(f"TOML file '{toml_path}' not found. Using DUMMY test data.")
        s_pos = [0, 0, 1]
        g_pos = [[2, 0, 1], [4, 2, 1], [6, 0, 1], [8, 2, 2]]
        g_norm = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 1]]
        g_norm = [np.array(n) / np.linalg.norm(n) for n in g_norm]
        g_y = [[0, 1, 0], [-1, 0, 0], [0, 1, 0], [-1, 0, 0]]
        g_z = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        obs_pos = []

    # 2. Init Engine
    geom = GeometryEngine(g_pos, g_norm, g_y, g_z, obs_pos, s_pos)

    # 3. Plot
    geom.plot(tunnel_radius=0.25)
