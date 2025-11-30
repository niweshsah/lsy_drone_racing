import matplotlib.pyplot as plt  # noqa: D100
import numpy as np
from scipy.interpolate import CubicSpline


class GeometryEngine:  # noqa: D101
    def __init__(self):  # noqa: D107
        # 1. Define Waypoints (Same as in your controller)
        
        # Waypoints in 3D space (x, y, z)
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )

        # 2. Create Spline parameterized by approximate Arc Length (s)
        # Calculate cumulative distance
        # dists is the distance between consecutive waypoints
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        
        # Insert cumulative distances to get s values
        # s = [0, d1, d1+d2, ..., total_length]
        self.s_path = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_path[-1]

        # CubicSpline: s -> (x,y,z)
        self.spline = CubicSpline(self.s_path, waypoints)

        # 3. Precompute Frame
        self.pt_frame = self._generate_parallel_transport_frame()



    def _generate_parallel_transport_frame(self, num_points=1000):  # noqa: ANN001, ANN202
        """Generates the Parallel Transport frame using the paper's update law.
        Eq (7): [t, n1, n2]' = Matrix * [t, n1, n2].
        """  # noqa: D205
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0] # ds is the step size in s

        frames = {"s": s_eval, "pos": [], "t": [], "n1": [], "n2": [], "k1": [], "k2": []}

        # --- Initialization (Section 2.4.1) ---
        # "n2 is pointing to the direction of gravity" roughly
        # t0 is the tangent at s=0
        self.spline(0)
        t_0 = self.spline(0, 1)  # First derivative
        t_0 /= np.linalg.norm(t_0) # make it unit vector

        # Initial Normal (Gram-Schmidt against Gravity)
        g_vec = np.array([0, 0, -1])  # Down vector (gravity)

        # If tangent is perfectly vertical, handle singularity
        if np.linalg.norm(np.cross(t_0, g_vec)) < 1e-3:
            dummy = np.array([1, 0, 0])
            n1_0 = np.cross(t_0, dummy)
        else:
            # Paper Eq 20 strategy: n2 is roughly g projected
            # n2 = g - (g.t)t
            # remove the projection onto t to get vector orthogonal to t
            n2_0 = g_vec - np.dot(g_vec, t_0) * t_0
            n2_0 /= np.linalg.norm(n2_0)
            
            
            # n1 completes the triad
            n1_0 = np.cross(n2_0, t_0)

        # Storage
        curr_t = t_0
        curr_n1 = n1_0
        curr_n2 = n2_0

        for i, s in enumerate(s_eval):
            # 1. Exact Position & Tangent from Spline
            pos = self.spline(s)

            # 2. Calculate Curvature Vector (dt/ds)
            # Second derivative of position w.r.t s is curvature vector (approx)
            curvature_vec = self.spline(s, 2)

            # 3. Paper Logic: Find k1, k2 by projecting curvature
            # The paper defines dot(t) = -k1*n1 - k2*n2 (Eq 7 implies this structure)
            # Therefore:
            # k1 = - dot(curvature_vec, curr_n1)
            # k2 = - dot(curvature_vec, curr_n2)
            k1 = -np.dot(curvature_vec, curr_n1)
            k2 = -np.dot(curvature_vec, curr_n2)

            # 4. Propagate Frame (Integration)
            # Update t is handled by the spline geometry naturally,
            # we mainly need to rotate n1 and n2 to stay twist-free.

            # Eq 7: dot(n1) = k1 * t
            # Eq 7: dot(n2) = k2 * t

            next_n1 = curr_n1 + (k1 * curr_t) * ds
            next_n2 = curr_n2 + (k2 * curr_t) * ds

            # 5. Re-align with new tangent (Gram-Schmidt)
            # Get next tangent from spline geometry to be exact
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t /= np.linalg.norm(next_t)
            else:
                next_t = curr_t  # End of path

            # Orthogonalize n1 against new t
            next_n1 = next_n1 - np.dot(next_n1, next_t) * next_t
            next_n1 /= np.linalg.norm(next_n1)

            # Recompute n2 to ensure perfect triad
            next_n2 = np.cross(next_t, next_n1)

            # Store
            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            frames["k1"].append(k1)
            frames["k2"].append(k2)

            # Update
            curr_t = next_t
            curr_n1 = next_n1
            curr_n2 = next_n2

        # Convert lists to numpy arrays
        for key in frames:
            frames[key] = np.array(frames[key])

        return frames

    def verify_mathematically(self):
        """Runs numerical assertions to ensure the frame is valid."""
        print("\n--- Running Mathematical Verification ---")

        t = self.pt_frame["t"]
        n1 = self.pt_frame["n1"]
        n2 = self.pt_frame["n2"]
        s = self.pt_frame["s"]
        ds = s[1] - s[0]

        # 1. Orthonormality Check
        # t.t=1, n1.n1=1, n2.n2=1
        norm_err = np.max(np.abs(np.linalg.norm(t, axis=1) - 1.0))
        # t.n1=0, t.n2=0, n1.n2=0
        orth_err_1 = np.max(np.abs(np.einsum("ij,ij->i", t, n1)))
        orth_err_2 = np.max(np.abs(np.einsum("ij,ij", t, n2)))
        orth_err_3 = np.max(np.abs(np.einsum("ij,ij", n1, n2)))

        print(f"[Check 1] Unit Norm Error: {norm_err:.2e}")
        print(f"[Check 2] Orthogonality Error: {max(orth_err_1, orth_err_2, orth_err_3):.2e}")

        assert norm_err < 1e-6, "Vectors are not unit length!"
        assert orth_err_1 < 1e-6, "Vectors are not orthogonal!"

        # 2. Tangency Check
        # Does 't' match the analytical derivative of the spline?
        # Note: Spline derivative isn't normalized by default.
        spline_t = self.spline(s, 1)
        spline_t = spline_t / np.linalg.norm(spline_t, axis=1)[:, None]
        tangency_err = np.max(np.linalg.norm(t - spline_t, axis=1))

        print(f"[Check 3] Tangency Mismatch: {tangency_err:.2e}")
        assert tangency_err < 1e-3, "Frame tangent diverges from Spline tangent!"

        # 3. Minimal Twist Check (The Parallel Transport Property)
        # Definition: The derivative of n1 along s should have NO component in n2 direction.
        # i.e., n1_dot . n2 ~= 0

        # Numerical differentiation of n1
        n1_dot = np.gradient(n1, ds, axis=0)

        # Project n1_dot onto n2
        twist_component = np.einsum("ij,ij->i", n1_dot, n2)
        max_twist = np.max(np.abs(twist_component))
        avg_twist = np.mean(np.abs(twist_component))

        print(f"[Check 4] Twist Error (Max): {max_twist:.2e} rad/m")
        print(f"[Check 4] Twist Error (Avg): {avg_twist:.2e} rad/m")

        # Tolerance Note: Discrete integration always yields some error proportional to ds^2
        if max_twist < 0.1:
            print(">> SUCCESS: Frame is Twist-Minimal.")
        else:
            print(">> FAILURE: Frame is rotating around tangent!")

    def plot_frame(self):
        """Visualizes the Spline and PT Frame."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot Path
        pos = self.pt_frame["pos"]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "k-", linewidth=2, label="Path")

        # Subsample for quiver plot (too many arrows looks messy)
        step = 30
        idx = np.arange(0, len(pos), step)

        xyz = pos[idx]
        t = self.pt_frame["t"][idx]
        n1 = self.pt_frame["n1"][idx]
        n2 = self.pt_frame["n2"][idx]

        # Plot Frame Vectors
        # Tangent (Red)
        ax.quiver(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            t[:, 0],
            t[:, 1],
            t[:, 2],
            color="r",
            length=0.2,
            normalize=True,
            label="Tangent (t)",
        )

        # Normal 1 (Green)
        ax.quiver(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            n1[:, 0],
            n1[:, 1],
            n1[:, 2],
            color="g",
            length=0.2,
            normalize=True,
            label="Normal 1 (n1)",
        )

        # Normal 2 (Blue)
        ax.quiver(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            n2[:, 0],
            n2[:, 1],
            n2[:, 2],
            color="b",
            length=0.2,
            normalize=True,
            label="Normal 2 (n2)",
        )

        ax.set_title("Parallel Transport Frame Verification")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    geo = GeometryEngine()
    print(f"Path Length: {geo.total_length:.2f} m")

    # 1. Math Verification
    geo.verify_mathematically()

    # 2. Visual Verification
    print("\nVisualizing Frame...")
    geo.plot_frame()
