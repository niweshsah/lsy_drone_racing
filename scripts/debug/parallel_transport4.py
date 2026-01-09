import os

# Type hinting for clarity
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import toml
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R


class GeometryEngine:
    def __init__(
        self,
        gates_pos: List[List[float]],
        gates_normals: List[List[float]],
        gates_y: List[List[float]],
        gates_z: List[List[float]],
        obstacles_pos: List[List[float]],
        start_pos: List[float],
    ):
        """Initializes the geometry engine and generates the safe flight path."""
        # --- 1. Configuration Constants ---
        self.DETOUR_ANGLE_THRESHOLD = 60.0  # Degrees. If turn > this, add detour.
        self.DETOUR_RADIUS = 1.0  # Meters. How far to swing out for detours.
        self.TANGENT_SCALE_FACTOR = 1.0  # Controls how "aggressive" the curves are.

        # --- 2. Data Ingestion ---
        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.gate_y = np.asarray(gates_y, dtype=np.float64)
        self.gate_z = np.asarray(gates_z, dtype=np.float64)
        self.obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)

        # --- 3. Pipeline Execution ---

        # A. Initialize Waypoints (Start + Gates)
        # We track 'types': 0=Start, 1=Gate, 2=Detour
        self.waypoints, self.wp_types, self.wp_normals = self._initialize_waypoints()

        # B. Insert Detour Points for Sharp Turns
        self.waypoints, self.wp_types, self.wp_normals = self._add_detour_logic(
            self.waypoints, self.wp_types, self.wp_normals
        )

        # C. Compute Tangents (Strict constraints for Gates, Smooth for others)
        self.tangents = self._compute_hermite_tangents()

        # D. Generate Cubic Hermite Spline
        # Parameterize by cumulative Euclidean distance (Arc Length approximation)
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]

        # P(s) -> Returns (x,y,z)
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)

        # E. Generate Parallel Transport Frame (for visualization/physics)
        self.pt_frame = self._generate_parallel_transport_frame(num_points=1000)

    def _initialize_waypoints(self):
        """Creates the initial ordered list of waypoints starting with Start Pos."""
        wps = [self.start_pos]
        types = [0]  # 0 = Start
        normals = [np.zeros(3)]  # Start has no forced orientation

        for i in range(len(self.gates_pos)):
            wps.append(self.gates_pos[i])
            types.append(1)  # 1 = Gate
            normals.append(self.gate_normals[i])

        return np.array(wps), np.array(types), np.array(normals)

    def _add_detour_logic(self, wps, types, normals):
        """Analyzes consecutive waypoints. If the angle required to hit the next point
        is too sharp relative to the current gate's normal, inserts a detour point.
        """
        new_wps = [wps[0]]
        new_types = [types[0]]
        new_normals = [normals[0]]

        for i in range(len(wps) - 1):
            curr_p = wps[i]
            next_p = wps[i + 1]
            curr_type = types[i]

            # Only apply detour logic if we are LEAVING a GATE (Type 1)
            if curr_type == 1:
                # Identify which gate index this corresponds to in original arrays
                # (Since wps includes start, gate index is i-1)
                gate_idx = i - 1
                gate_norm = self.gate_normals[gate_idx]

                vec_to_next = next_p - curr_p
                dist = np.linalg.norm(vec_to_next)

                if dist > 1e-6:
                    vec_to_next /= dist

                    # Dot product: 1.0 (Straight), 0.0 (90 deg), -1.0 (180 deg)
                    alignment = np.dot(gate_norm, vec_to_next)
                    # Convert to angle
                    angle_deg = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))

                    if angle_deg > self.DETOUR_ANGLE_THRESHOLD:
                        # --- Create Detour ---
                        # Project vector onto the Gate's Plane (Y-Z plane) to find "sideways" direction
                        # Formula: v_proj = v - (v . n) * n
                        proj = vec_to_next - (np.dot(vec_to_next, gate_norm) * gate_norm)

                        if np.linalg.norm(proj) < 1e-3:
                            # Perfectly backwards (180 deg). Default to "Up" relative to gate
                            detour_dir = self.gate_z[gate_idx]
                        else:
                            detour_dir = proj / np.linalg.norm(proj)

                        # Place detour point:
                        # 1. Start at gate center
                        # 2. Move 'out' by radius
                        # 3. Move 'forward' slightly along normal so we don't clip the frame
                        detour_pos = curr_p + (detour_dir * self.DETOUR_RADIUS) + (gate_norm * 1.5)

                        new_wps.append(detour_pos)
                        new_types.append(2)  # 2 = Detour
                        new_normals.append(np.zeros(3))  # No forced normal

            # Always add the target point
            new_wps.append(next_p)
            new_types.append(types[i + 1])
            new_normals.append(normals[i + 1])

        return np.array(new_wps), np.array(new_types), np.array(new_normals)

    def _compute_hermite_tangents(self):
        """Calculates the tangent (velocity) vectors for the Cubic Hermite Spline.
        - Gates: Tangent MUST be aligned with Gate Normal.
        - Detours/Start: Tangent is heuristic (Catmull-Rom / Finite Difference).
        """
        num_pts = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)

        for i in range(num_pts):
            # 1. Determine Scale (Speed) based on segment lengths
            #    We want the drone to move faster on long segments, slower on short ones.
            dist_prev = np.linalg.norm(self.waypoints[i] - self.waypoints[i - 1]) if i > 0 else 0
            dist_next = (
                np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i]) if i < num_pts - 1 else 0
            )

            # Use minimum neighbor distance to prevent loops/overshoot
            base_scale = min(
                dist_prev if dist_prev > 0 else dist_next, dist_next if dist_next > 0 else dist_prev
            )

            scale = base_scale * self.TANGENT_SCALE_FACTOR

            if self.wp_types[i] == 1:
                # --- GATE: Strict Alignment ---
                normal = self.wp_normals[i].copy()

                # Auto-Flip: If the natural path flow opposes the normal, flip the normal
                # This handles gates defined "backwards" in the config
                if i > 0 and i < num_pts - 1:
                    flow_vec = self.waypoints[i + 1] - self.waypoints[i - 1]
                    if np.dot(normal, flow_vec) < 0:
                        normal = -normal

                tangents[i] = normal * scale

            else:
                # --- START / DETOUR: Smooth Curve ---
                # Use Catmull-Rom style (vector between prev and next)
                if i == 0:
                    t = self.waypoints[i + 1] - self.waypoints[i]
                elif i == num_pts - 1:
                    t = self.waypoints[i] - self.waypoints[i - 1]
                else:
                    t = self.waypoints[i + 1] - self.waypoints[i - 1]

                if np.linalg.norm(t) > 1e-6:
                    t = t / np.linalg.norm(t)

                tangents[i] = t * scale

        return tangents

    def _generate_parallel_transport_frame(self, num_points=1000):
        """Generates coordinate frames along the spline using Parallel Transport (Bishop Frame).
        This ensures the frame doesn't twist unnecessarily around the curve.
        """
        s_eval = np.linspace(0, self.total_length, num_points)

        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": []}

        # 1. Initial Frame at s=0
        t0 = self.spline(0, 1)  # First derivative
        t0 /= np.linalg.norm(t0)

        # Arbitrary guide vector (Gravity) to fix the first frame
        g_vec = np.array([0, 0, -1])

        # n2 is horizontal-ish (perp to Tangent and Gravity)
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            # If diving straight down, use X axis
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = np.cross(t0, np.cross(g_vec, t0))

        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        # 2. Propagate Frame
        for i, s in enumerate(s_eval):
            pos = self.spline(s)

            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t

            # Rotation from curr_t to next_t
            axis = np.cross(curr_t, next_t)
            sin_phi = np.linalg.norm(axis)
            cos_phi = np.dot(curr_t, next_t)

            next_n1 = curr_n1
            next_n2 = curr_n2

            # Apply minimal rotation (Parallel Transport)
            if sin_phi > 1e-6:
                axis /= sin_phi
                phi = np.arctan2(sin_phi, cos_phi)
                rot = R.from_rotvec(axis * phi)
                next_n1 = rot.apply(curr_n1)
                next_n2 = rot.apply(curr_n2)

            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)

            curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        for k in frames:
            frames[k] = np.array(frames[k])
        return frames

    def plot(self, tunnel_radius=0.2):
        """Visualizes the path, safety tube, and gates in 3D."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # A. Plot Path Tube
        path = self.pt_frame["pos"]
        step = 5
        theta = np.linspace(0, 2 * np.pi, 12)

        # Downsample for mesh generation
        p = path[::step]
        n1 = self.pt_frame["n1"][::step]
        n2 = self.pt_frame["n2"][::step]

        cos_t = np.cos(theta)[None, :]
        sin_t = np.sin(theta)[None, :]

        X = p[:, 0, None] + tunnel_radius * (n1[:, 0, None] * cos_t + n2[:, 0, None] * sin_t)
        Y = p[:, 1, None] + tunnel_radius * (n1[:, 1, None] * cos_t + n2[:, 1, None] * sin_t)
        Z = p[:, 2, None] + tunnel_radius * (n1[:, 2, None] * cos_t + n2[:, 2, None] * sin_t)

        ax.plot_surface(X, Y, Z, color="cyan", alpha=0.15, linewidth=0)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "k-", linewidth=1.5, label="Flight Path")

        # B. Plot Waypoints & Tangents
        # Gates = Blue, Detours = Red
        colors = ["green" if t == 0 else "blue" if t == 1 else "red" for t in self.wp_types]
        ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], c=colors, s=50)

        # Draw tangents
        ax.quiver(
            self.waypoints[:, 0],
            self.waypoints[:, 1],
            self.waypoints[:, 2],
            self.tangents[:, 0],
            self.tangents[:, 1],
            self.tangents[:, 2],
            length=0.5,
            color="orange",
            label="Tangents",
        )

        # C. Plot Gates (Square Frames)
        for i, (pos, norm, gy, gz) in enumerate(
            zip(self.gates_pos, self.gate_normals, self.gate_y, self.gate_z)
        ):
            w, h = 1.0, 1.0  # Gate dimensions
            corners = [
                pos + (gy * w / 2) + (gz * h / 2),
                pos - (gy * w / 2) + (gz * h / 2),
                pos - (gy * w / 2) - (gz * h / 2),
                pos + (gy * w / 2) - (gz * h / 2),
                pos + (gy * w / 2) + (gz * h / 2),
            ]
            c = np.array(corners)
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color="magenta", linewidth=3)
            ax.quiver(pos[0], pos[1], pos[2], norm[0], norm[1], norm[2], length=0.8, color="red")
            ax.text(pos[0], pos[1], pos[2] + 0.6, f"G{i}", fontsize=10)

        # Start Position
        ax.scatter(
            self.start_pos[0], self.start_pos[1], self.start_pos[2], c="green", s=100, label="Start"
        )

        # Scaling and Labels
        mid = np.mean(path, axis=0)
        r = 5
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()


# --- Config Loader ---
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

    # Sharp Turn Test Case
    s_pos = [0, -4, 0]
    # Gate 0 points X+, Gate 1 is at (2,2)
    # Logic check: Exit G0 is ~[0.3, 0, 0]. Next Entry is ~[2, 2, 0].
    # Vector is diagonal. Normal is X. Angle should be ~45 deg.
    g_pos = [[0, 0, 0], [2, 2, 0]]
    g_norm = [[1, 0, 0], [0, 1, 0]]
    g_y = [[0, 1, 0], [-1, 0, 0]]
    g_z = [[0, 0, 1], [0, 0, 1]]
    obs_pos = []

    if os.path.exists(toml_path):
        try:
            g_pos, g_norm, g_y, g_z, obs_pos, s_pos = load_from_toml(toml_path)
        except Exception:
            pass

    geom = GeometryEngine(g_pos, g_norm, g_y, g_z, obs_pos, s_pos)
    geom.plot()
