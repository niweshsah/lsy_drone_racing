import json
import os
import shutil
from datetime import datetime
from typing import List, Dict
import plotly.graph_objects as go

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R
from numpy.typing import NDArray

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

# Use non-interactive backend for saving plots
matplotlib.use("Agg")

# Parameters:
v_max_ref = 1.5  # m/s
corner_acc = 1.95
mpc_horizons_global = 50
max_lateral_width = 0.3
safety_radius = 0.1

# ==============================================================================
# 1. PARAMETERS & DYNAMICS
# ==============================================================================


def get_drone_params():  # noqa: ANN201
    """Defines physical parameters and System-ID coefficients."""
    # Attempt to load from file, otherwise use defaults
    params = load_params("so_rpy", "cf21B_500")

    if params is not None:
        # Ensure consistency in return type if loaded from file
        return params
    else:
        return {
            "mass": 0.04338,
            "gravity_vec": np.array([0.0, 0.0, -9.81]),
            "g": 9.81,  # Added for convenience
            "J": np.diag([25e-6, 28e-6, 49e-6]),
            "J_inv": np.linalg.inv(np.diag([25e-6, 28e-6, 49e-6])),
            "thrust_min": 0.0,
            "thrust_max": 0.2,  # Normalized or Newtons depending on cmd_f_coef
            # System ID Coefficients (Linear Response Model)
            "acc_coef": 0.0,  # Bias term for thrust curve
            "cmd_f_coef": 0.96836458,  # Slope for thrust curve
            "rpy_coef": [-188.9910, -188.9910, -138.3109],  # Stiffness
            "rpy_rates_coef": [-12.7803, -12.7803, -16.8485],  # Damping
            "cmd_rpy_coef": [138.0834, 138.0834, 198.5161],  # Input Gain
        }


def symbolic_dynamics_spatial(
    mass: float,
    gravity_vec: np.ndarray,
    J: np.ndarray,
    J_inv: np.ndarray,
    acc_coef: float,
    cmd_f_coef: float,
    rpy_coef: list,
    rpy_rates_coef: list,
    cmd_rpy_coef: list,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    # --- 1. State Vector X (12 States) ---
    # [cite: 302] States: s, w1, w2, ds, dw1, dw2, phi, theta, psi, dphi, dtheta, dpsi
    s, w1, w2 = ca.SX.sym("s"), ca.SX.sym("w1"), ca.SX.sym("w2")
    ds, dw1, dw2 = ca.SX.sym("ds"), ca.SX.sym("dw1"), ca.SX.sym("dw2")

    phi, theta, psi = ca.SX.sym("phi"), ca.SX.sym("theta"), ca.SX.sym("psi")
    dphi, dtheta, dpsi = ca.SX.sym("dphi"), ca.SX.sym("dtheta"), ca.SX.sym("dpsi")

    rpy = ca.vertcat(phi, theta, psi)
    drpy = ca.vertcat(dphi, dtheta, dpsi)

    X = ca.vertcat(s, w1, w2, ds, dw1, dw2, rpy, drpy)

    # --- 2. Control Input U (4 Inputs) ---
    phi_c, theta_c, psi_c, T_c = (
        ca.SX.sym("phi_c"),
        ca.SX.sym("theta_c"),
        ca.SX.sym("psi_c"),
        ca.SX.sym("T_c"),
    )
    cmd_rpy = ca.vertcat(phi_c, theta_c, psi_c)
    U = ca.vertcat(cmd_rpy, T_c)

    # --- 3. Parameters P (13 Elements) ---
    # [cite: 299] Dependencies on t, n1, n2, k1, k2 and their derivatives
    t_vec = ca.SX.sym("t_vec", 3)
    n1_vec = ca.SX.sym("n1_vec", 3)
    n2_vec = ca.SX.sym("n2_vec", 3)
    k1, k2 = ca.SX.sym("k1"), ca.SX.sym("k2")
    dk1, dk2 = ca.SX.sym("dk1"), ca.SX.sym("dk2")  # Spatial derivatives of curvature

    # ORDER MATTERS: This must match the order in the Controller loop exactly
    P = ca.vertcat(t_vec, n1_vec, n2_vec, k1, k2, dk1, dk2)

    # --- 4. Physics Engine ---

    # A. Rotational Dynamics (Fitted Linear Model)
    # ddrpy = Stiffness * angle + Damping * rate + Gain * command
    c_rpy = ca.DM(rpy_coef)
    c_drpy = ca.DM(rpy_rates_coef)
    c_cmd = ca.DM(cmd_rpy_coef)

    ddrpy = c_rpy * rpy + c_drpy * drpy + c_cmd * cmd_rpy

    # B. Translational Acceleration (Inertial Frame)
    thrust_mag = acc_coef + cmd_f_coef * T_c
    F_body = ca.vertcat(0, 0, thrust_mag)

    # Rotation Matrix (Body -> Inertial)
    cx, cy, cz = ca.cos(phi), ca.cos(theta), ca.cos(psi)
    sx, sy, sz = ca.sin(phi), ca.sin(theta), ca.sin(psi)

    R_IB = ca.vertcat(
        ca.horzcat(cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy),
        ca.horzcat(cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
        ca.horzcat(-sy, cy * sx, cx * cy),
    )

    # Global Acceleration
    g_vec_sym = ca.DM(gravity_vec)
    acc_world = g_vec_sym + (R_IB @ F_body) / mass

    # C. Spatial Dynamics Reconstruction
    # h is the scaling factor for path curvature
    h = 1 - k1 * w1 - k2 * w2

    # h_dot requires dk1/dk2 (Chain rule: d/dt = d/ds * ds/dt)
    h_dot = -(k1 * dw1 + k2 * dw2 + (dk1 * w1 + dk2 * w2) * ds)

    coriolis = (
        (ds * h_dot) * t_vec
        + (ds**2 * h * k1) * n1_vec
        + (ds**2 * h * k2) * n2_vec
        - (ds * dw1 * k1) * t_vec
        - (ds * dw2 * k2) * t_vec
    )

    # Project World Acceleration onto Path Frame
    proj_t = ca.dot(t_vec, acc_world - coriolis)
    dds = proj_t / h
    ddw1 = ca.dot(n1_vec, acc_world - coriolis)
    ddw2 = ca.dot(n2_vec, acc_world - coriolis)

    # --- 5. Final Time Derivative ---
    X_Dot = ca.vertcat(
        ds,  # s_dot
        dw1,  # w1_dot
        dw2,  # w2_dot
        dds,  # s_ddot
        ddw1,  # w1_ddot
        ddw2,  # w2_ddot
        drpy,  # rpy_dot
        ddrpy,  # rpy_ddot
    )

    return X_Dot, X, U, P


def export_model(params: dict) -> AcadosModel:
    X_dot, X, U, P = symbolic_dynamics_spatial(
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        J=params["J"],
        J_inv=params.get("J_inv"),
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
    )

    model = AcadosModel()
    model.name = "spatial_mpc_drone"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    model.p = P

    return model


# ==============================================================================
# 2. GEOMETRY ENGINE
# ==============================================================================


class GeometryEngine:
    def __init__(
        self,
        gates_pos: List[List[float]],
        gates_normal: List[List[float]],
        gates_y: List[List[float]],
        gates_z: List[List[float]],
        gate_size: float = 0.5,
        obstacles_pos: List[List[float]] = [],
        start_pos: List[float] = [-1.5, 0.75, 0.01],
        start_orient: List[float] = [0, 0, 0],
        obs: dict[str, NDArray[np.floating]] = {},
        info: dict = {},
        sim_config: dict = {},
    ):
        self.gates_pos = np.array(gates_pos)
        self.gates_normal = np.array(gates_normal)
        self.gates_y = np.array(gates_y)
        self.gates_z = np.array(gates_z)
        self.gate_size = gate_size
        # self.obstacles_pos = obstacles_pos
        self.start_pos = np.array(start_pos)
        self.start_orient = R.from_euler(
            "xyz", start_orient
        ).as_matrix()  # Convert to rotation matrix
        self.obs = obs
        self.info = info
        self.sim_config = sim_config
        self.POLE_HEIGHT = 3.0  # Meters
        self.SAFETY_RADIUS = 0.05  # Meters
        self.MAX_LATERAL_WIDTH = 0.35  # Meters
        self.CONTRACTION_LEN = 0.3  # Meters
        self.CLEARANCE_RADIUS = 0.2  # Meters for obstacle avoidance

        self.debug_dicts = []

        # self.obstacles_pos = self.add_virtual_obstacle(obstacles_pos)
        self.obstacles_pos = np.array(obstacles_pos)

        self.gate_vectors = self.gate_to_gate_vectors()

        self.waypoints = self.__initialize_waypoints()
        self.waypoints = self.__insert_obstacle_avoidance_waypoints(
            self.waypoints, clearance_radius=self.CLEARANCE_RADIUS
        )

        self.spline = self.__get_spline(self.waypoints)

        num_frame_points = int(max(10, self.total_length * 100))
        self.pt_frame = self._generate_parallel_transport_frame(num_points=num_frame_points)

        self.corridor_map = self.__generate_static_corridor()

        # self.__print_debug_vectors()

    def gate_to_gate_vectors(self) -> List[NDArray[np.floating]]:
        gate_vectors = []
        for i in range(1, len(self.gates_pos)):
            vec = self.gates_pos[i] - self.gates_pos[i - 1]
            gate_vectors.append(vec / np.linalg.norm(vec))

        return gate_vectors

    def add_virtual_obstacle(self, obstacle_pos):
        new_obstacles: List[List[float]] = obstacle_pos.copy()

        for idx, gate_pos in enumerate(self.gates_pos):
            gate_y = self.gates_y[idx]

            virtual_obs_pos = gate_pos - gate_y * (self.gate_size / 2)
            new_obstacles.append(virtual_obs_pos)

            virtual_obs_pos = gate_pos + gate_y * (self.gate_size / 2)
            new_obstacles.append(virtual_obs_pos)

        return np.array(new_obstacles)

    def __process_single_obstacle_avoidance(
        self,
        sampled_points: NDArray[np.floating],
        clearance_radius: float,
        obstacle_center: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        collision_free_points = []

        is_inside_obstacle_zone = False
        entry_index = None

        obstacle_xy = obstacle_center[:2]

        for i, point in enumerate(sampled_points):
            point_xy = point[:2]
            distance_xy = np.linalg.norm(obstacle_xy - point_xy)

            if distance_xy < clearance_radius:
                if not is_inside_obstacle_zone:
                    # Just entered the collision zone
                    is_inside_obstacle_zone = True
                    print(f"Entering obstacle zone at index {i}, point {point}")
                    entry_index = i

            elif is_inside_obstacle_zone:
                # Just exited the collision zone
                is_inside_obstacle_zone = False
                exit_index = i

                # --- Avoidance Calculation ---
                entry_point = sampled_points[entry_index]
                exit_point = sampled_points[exit_index]

                # Vectors from obstacle center to entry/exit
                entry_vec = entry_point[:2] - obstacle_xy
                exit_vec = exit_point[:2] - obstacle_xy

                # Bisector vector determines the direction to push the path
                avoid_vec = entry_vec + exit_vec
                avoid_vec /= np.linalg.norm(avoid_vec) + 1e-6

                # Calculate new waypoint
                new_pos_xy = obstacle_xy + avoid_vec * clearance_radius
                new_pos_z = (entry_point[2] + exit_point[2]) / 2  # Maintain average altitude
                new_avoid_waypoint = np.concatenate([new_pos_xy, [new_pos_z]])

                # Insert waypoint at average time
                # avg_time = (sampled_times[entry_index] + sampled_times[exit_index]) / 2
                # collision_free_times.append(avg_time)
                collision_free_points.append(new_avoid_waypoint)

            else:
                # Point is safe, keep it
                # collision_free_times.append(sampled_times[i])
                collision_free_points.append(point)

        return np.array(collision_free_points)

    def __insert_obstacle_avoidance_waypoints(
        self, waypoints: NDArray[np.floating], clearance_radius: float, num_points=1000
    ) -> NDArray[np.floating]:
        temp_spline = self.__get_spline(waypoints)
        s_eval = np.linspace(0, self.total_length, num_points)
        sampled_points = temp_spline(s_eval)

        for obs in self.obstacles_pos:
            new_waypoints = self.__process_single_obstacle_avoidance(
                sampled_points, clearance_radius, obs
            )

        return new_waypoints

    def __print_debug_vectors(self):
        for i, debug_dict in enumerate(self.debug_dicts):
            # print(
            #     f"Debug {i}: Frame Pos: {debug_dict['frame_pos']}, Obstacle: {debug_dict['obs']}, w1_obs: {debug_dict['w1_obs']}, reduced_lb: {debug_dict.get('reduced_lb', 'NA')}, reduced_ub: {debug_dict.get('reduced_ub', 'NA')}, s_knots: {debug_dict['s_knots']}"
            # )
            pass

    def angle_between_vectors(self, v1: NDArray[np.floating], v2: NDArray[np.floating]) -> float:
        """Calculate the angle in radians between two vectors."""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = np.arccos(dot_product)
        return angle

    def __initialize_waypoints(self) -> NDArray[np.floating]:
        """Initialize waypoints with special handling for 180-degree reversals."""
        waypoints = [self.start_pos]

        # Tuning parameters for the geometry
        EXTENSION_DIST = 0.2  # How far to fly straight out after a reversal gate
        TURN_RADIUS = 0.25  # How wide the U-turn should be
        APPROACH_DIST = 0.5  # Distance for pre/post gate guidance

        for idx, gate_pos in enumerate(self.gates_pos):
            # 1. Define Standard Approach (Pre-Gate)
            # Aligns the drone with the normal before entering
            before_gate = gate_pos - self.gates_normal[idx] * APPROACH_DIST
            waypoints.append(before_gate)

            # 2. Add Gate Center
            waypoints.append(gate_pos)

            # Check angle for reversal detection
            # We assume angle > 120 degrees implies a sharp turn/reversal
            is_reversal = False
            if idx < len(self.gates_pos) - 1:
                # Calculate vector to next gate to check turn sharpness
                vec_to_next = self.gates_pos[idx + 1] - gate_pos
                angle = self.angle_between_vectors(vec_to_next, self.gates_normal[idx])

                # If angle is large, the next gate is "behind" the current normal
                if np.degrees(angle) > 120:
                    is_reversal = True

            if is_reversal:
                print(f"[Geometry] Generating Reversal Balloon at Gate {idx}")

                # 3. The "Extension" (Fly Out)
                # Force the drone to fly straight OUT of the gate first.
                # This prevents it from snapping 180 immediately inside the gate.
                extension_point = gate_pos + self.gates_normal[idx] * EXTENSION_DIST
                waypoints.append(extension_point)

                # 4. The "Balloon" Turn (Lateral Offset)
                # To come back, we must turn Left or Right. We use the Gate's Y-axis.
                # Heuristic: Check which side the next gate is on relative to the current gate's Y axis
                vec_to_next = self.gates_pos[idx + 1] - gate_pos

                # Dot product determines if next gate is to the Left (+Y) or Right (-Y)
                # If vectors are orthogonal or zero, default to +Y (Left)
                side_sign = np.sign(np.dot(vec_to_next, self.gates_y[idx]))
                if side_sign == 0:
                    side_sign = 1.0

                # Create a waypoint that pulls the spline into a wide U-turn
                # This point is: Extended out + Shifted sideways
                turn_point = extension_point + (self.gates_y[idx] * TURN_RADIUS * side_sign)
                waypoints.append(turn_point)

            else:
                # Standard Exit (Just follow the normal out)
                after_gate = gate_pos + self.gates_normal[idx] * APPROACH_DIST
                waypoints.append(after_gate)

        return np.array(waypoints)

    def __create_sknots(self, points: NDArray[np.floating], num_points=3000) -> CubicHermiteSpline:
        """Create a cubic Hermite spline from given points and tangents."""

        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        dists = np.maximum(dists, 1e-6)  # Prevent division by zero

        s_knots = np.concatenate(([0], np.cumsum(dists)))  # Cumulative distance along the points

        self.total_length = s_knots[-1]

        return s_knots

    def __add_debug_statement(
        self,
        i: int,
        frame_pos: NDArray[np.floating],
        obs: NDArray[np.floating],
        w1_obs: float,
        proposed_lb: float,
        proposed_ub: float,
    ):
        self.debug_dicts.append(
            {
                "frame_pos": f"{frame_pos[0]:.2f}, {frame_pos[1]:.2f}, {frame_pos[2]:.2f}",
                "obs": obs,
                "w1_obs": f"{w1_obs}",
                "reduced_lb": f"{proposed_lb:.2f} " if proposed_lb is not None else "None",
                "reduced_ub": f"{proposed_ub:.2f} " if proposed_ub is not None else "None",
                "s_knots": f"{self.pt_frame['s'][i]:.2f}",
            }
        )

    def __generate_static_corridor(self) -> Dict[str, NDArray]:
        # print(f"[Geometry] Generating bounds (2D). Safety Radius: {self.SAFETY_RADIUS}")
        num_pts = len(self.pt_frame["s"])
        lb_w1 = np.full(
            num_pts, -self.MAX_LATERAL_WIDTH
        )  # left bound which is filled with -max width
        ub_w1 = np.full(num_pts, self.MAX_LATERAL_WIDTH)

        if len(self.obstacles_pos) == 0:
            return {"lb_w1": lb_w1, "ub_w1": ub_w1}

        for i in range(num_pts):
            frame_pos = self.pt_frame["pos"][i]  # the position at frame i
            frame_t = self.pt_frame["t"][i]  # the tangent at frame i
            n1 = self.pt_frame["n1"][i]  # the normal at frame i

            # --- [NEW] Contract bounds near gates ---
            # self.__contract_for_gates(i, frame_pos, frame_t, {'lb_w1': lb_w1, 'ub_w1': ub_w1})

            pos_2d = np.array([frame_pos[0], frame_pos[1], 0.0])  # project to 2D by making z=0
            t_2d = np.array([frame_t[0], frame_t[1], 0.0])  # project to 2D by making z=0

            if np.linalg.norm(t_2d) < 1e-3:
                continue

            t_2d /= np.linalg.norm(t_2d)  # normalize
            n1_2d = np.array([n1[0], n1[1], 0.0])  # normal vector in 2D

            for gate_idx, gate_pos in enumerate(self.gates_pos):
                gate_pos_2d = np.array([gate_pos[0], gate_pos[1], 0.0])
                r_vec_2d = gate_pos_2d - pos_2d
                d_long = np.dot(r_vec_2d, t_2d)  # longitudinal distance along the tangent

                if abs(d_long) > self.CONTRACTION_LEN:
                    # print(f"[Geometry] Gate {gate_idx} too far from frame {i} for contraction: d_long = {d_long:.2f} m")
                    pass

                # Within contraction length, tighten bounds
                new_bound = self.gate_size / 2 - 0.1

                ub_w1[i] = new_bound
                lb_w1[i] = -new_bound

                self.__add_debug_statement(i, frame_pos, gate_pos, None, -new_bound, new_bound)

            for obs in self.obstacles_pos:
                obs_2d = np.array([obs[0], obs[1], 0.0])  # project obstacle to 2D
                # Compute relative vector

                r_vec_2d = obs_2d - pos_2d  # vector from frame pos to obstacle pos
                d = np.linalg.norm(r_vec_2d)
                d_long = np.dot(r_vec_2d, t_2d)  # longitudinal distance along the tangent
                if abs(d_long) > self.CONTRACTION_LEN:  # if obstacle is too far ahead or behind,
                    continue

                w1_obs = np.dot(r_vec_2d, n1_2d)  # lateral distance along the normal
                if d > self.CONTRACTION_LEN:
                    pass

                if abs(w1_obs) > (self.MAX_LATERAL_WIDTH + self.SAFETY_RADIUS):  # too far laterally
                    continue

                # If we are in the "Danger Zone" where geometry is ambiguous
                if abs(w1_obs) < 0.1:
                    # Check which constraint leaves us more room or is closer to the previous point's decision?
                    # Simple heuristic: Which side allows for a wider corridor?

                    proposed_ub = w1_obs - self.SAFETY_RADIUS
                    proposed_lb = w1_obs + self.SAFETY_RADIUS

                    current_width_if_pass_right = (
                        ub_w1[i] - proposed_lb
                    )  # If we treat obs as Right wall
                    current_width_if_pass_left = (
                        proposed_ub - lb_w1[i]
                    )  # If we treat obs as Left wall

                    # Pick the side that leaves the corridor more open
                    if current_width_if_pass_left > current_width_if_pass_right:
                        # Treat as Left Obstacle (Pass Right)
                        if proposed_ub < ub_w1[i]:
                            ub_w1[i] = proposed_ub
                    else:
                        # Treat as Right Obstacle (Pass Left)
                        if proposed_lb > lb_w1[i]:
                            lb_w1[i] = proposed_lb

                    # self.__add_debug_statement(i, frame_pos, obs, w1_obs, proposed_lb, proposed_ub)

                else:
                    if w1_obs >= 0:
                        safe_edge = w1_obs - self.SAFETY_RADIUS
                        if safe_edge < ub_w1[i]:
                            # self.__add_debug_statement(i, frame_pos, obs, w1_obs, None, safe_edge)
                            ub_w1[i] = safe_edge
                    else:
                        safe_edge = w1_obs + self.SAFETY_RADIUS
                        if safe_edge > lb_w1[i]:
                            # self.__add_debug_statement(i, frame_pos, obs, w1_obs, safe_edge, None)
                            lb_w1[i] = safe_edge

        collapsed = lb_w1 >= ub_w1
        if np.any(collapsed):
            mid = (lb_w1[collapsed] + ub_w1[collapsed]) / 2
            lb_w1[collapsed] = mid - 0.05
            ub_w1[collapsed] = mid + 0.05
        return {"lb_w1": lb_w1, "ub_w1": ub_w1}

    def __get_spline(self, points: NDArray[np.floating]) -> CubicHermiteSpline:
        """Generate a cubic Hermite spline from points and tangents."""
        s_knots = self.__create_sknots(points)
        spline = CubicSpline(s_knots, points)
        return spline

    def _generate_parallel_transport_frame(self, num_points=3000):
        # Evaluate along arc length s
        s_eval = np.linspace(0, self.total_length, num_points)
        ds = s_eval[1] - s_eval[0]

        frames = {
            "s": s_eval,
            "pos": [],
            "t": [],
            "n1": [],
            "n2": [],
            "k1": [],
            "k2": [],
            "dk1": [],
            "dk2": [],
        }

        # Initial Frame Setup
        # 1st derivative of Hermite Spline w.r.t s is the tangent
        t0 = self.spline(0, 1)
        t0 /= np.linalg.norm(t0)

        g_vec = np.array([0, 0, -1])  # Gravity reference

        # Handle case where t0 is parallel to gravity
        if np.linalg.norm(np.cross(t0, g_vec)) < 1e-3:
            n2_0 = np.cross(t0, np.array([1, 0, 0]))
        else:
            n2_0 = g_vec - np.dot(g_vec, t0) * t0

        n2_0 /= np.linalg.norm(n2_0)
        n1_0 = np.cross(n2_0, t0)

        curr_t, curr_n1, curr_n2 = t0, n1_0, n2_0

        k1_list, k2_list = [], []

        for i, s in enumerate(s_eval):
            pos = self.spline(s)

            # Curvature vector (2nd derivative)
            k_vec = self.spline(s, 2)

            k1 = np.dot(k_vec, curr_n1)  # Curvature in n1 direction
            k2 = np.dot(k_vec, curr_n2)

            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            k1_list.append(k1)
            k2_list.append(k2)

            # Bishop Frame Propagation (Parallel Transport)
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t_norm = np.linalg.norm(next_t)
                if next_t_norm > 1e-6:
                    next_t /= next_t_norm

                # Rotation from curr_t to next_t
                axis = np.cross(curr_t, next_t)
                dot_prod = np.clip(np.dot(curr_t, next_t), -1.0, 1.0)
                angle = np.arccos(dot_prod)

                if np.linalg.norm(axis) > 1e-6:
                    axis /= np.linalg.norm(axis)
                    r_vec = R.from_rotvec(axis * angle)
                    next_n1 = r_vec.apply(curr_n1)
                    next_n2 = r_vec.apply(curr_n2)
                else:
                    next_n1, next_n2 = curr_n1, curr_n2

                curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        frames["k1"] = np.array(k1_list)
        frames["k2"] = np.array(k2_list)
        frames["dk1"] = np.gradient(frames["k1"], ds)
        frames["dk2"] = np.gradient(frames["k2"], ds)

        # Convert lists to arrays
        for k in frames:
            if isinstance(frames[k], list):
                frames[k] = np.array(frames[k])
        return frames

    def plot(self):
        """Visualizes Path, Gates, Obstacles, and Corridor using Plotly."""
        # print("[Geometry] Generating interactive Plotly visualization...")
        fig = go.Figure()

        # --- 1. Plot Flight Path ---
        path = self.pt_frame["pos"]
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                line=dict(color="black", width=4),
                name="Centerline",
            )
        )

        # --- 2. Plot Waypoints ---
        # color_map = {0: "green", 1: "blue", 2: "red", 3: "purple"}
        # colors = ["green", "blue", "yellow", "green", "green", "green"]  # Start point
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=self.waypoints[:, 0],
        #         y=self.waypoints[:, 1],
        #         z=self.waypoints[:, 2],
        #         mode="markers",
        #         marker=dict(size=6, color=colors),
        #         name="Waypoints",
        #     )
        # )

        # --- 3. [NEW] Plot Gates (Rectangular Frames) ---
        # Using a single trace with "None" separators for performance
        gate_x, gate_y_list, gate_z_list = [], [], []

        # Half dimensions
        hw = self.gate_size / 2.0
        hh = self.gate_size / 2.0

        for i in range(len(self.gates_pos)):
            center = self.gates_pos[i]
            # Get orientation vectors
            y_vec = self.gates_y[i]
            z_vec = self.gates_z[i]

            # Calculate 4 corners
            # c1: Top-Left, c2: Top-Right, c3: Bot-Right, c4: Bot-Left
            c1 = center + (y_vec * hw) + (z_vec * hh)
            c2 = center - (y_vec * hw) + (z_vec * hh)
            c3 = center - (y_vec * hw) - (z_vec * hh)
            c4 = center + (y_vec * hw) - (z_vec * hh)

            # Append loop: 1->2->3->4->1 -> None
            for p in [c1, c2, c3, c4, c1]:
                gate_x.append(p[0])
                gate_y_list.append(p[1])
                gate_z_list.append(p[2])

            # Add separator to break the line between gates
            gate_x.append(None)
            gate_y_list.append(None)
            gate_z_list.append(None)

        fig.add_trace(
            go.Scatter3d(
                x=gate_x,
                y=gate_y_list,
                z=gate_z_list,
                mode="lines",
                line=dict(color="blue", width=5),
                name="Gates",
            )
        )

        # --- 4. Plot Obstacles ---
        if len(self.obstacles_pos) > 0:
            u = np.linspace(0, 2 * np.pi, 25)
            z_pole = np.linspace(0, self.POLE_HEIGHT, 2)
            U, Z_pole = np.meshgrid(u, z_pole)

            first_obs = True
            for obs in self.obstacles_pos:
                X_pole = self.SAFETY_RADIUS * np.cos(U) + obs[0]
                Y_pole = self.SAFETY_RADIUS * np.sin(U) + obs[1]
                fig.add_trace(
                    go.Surface(
                        x=X_pole,
                        y=Y_pole,
                        z=Z_pole,
                        colorscale=[[0, "red"], [1, "red"]],
                        opacity=0.6,
                        showscale=False,
                        name="Obstacle",
                        showlegend=first_obs,
                    )
                )
                first_obs = False

        # --- 5. Plot Corridor Bounds ---
        step = 5
        p_vis = path[::step]
        n1_vis = self.pt_frame["n1"][::step]
        idx = np.arange(0, len(path), step)
        lb = self.corridor_map["lb_w1"][idx]
        ub = self.corridor_map["ub_w1"][idx]

        wall_left = p_vis + (n1_vis * ub[:, np.newaxis])
        wall_right = p_vis + (n1_vis * lb[:, np.newaxis])

        fig.add_trace(
            go.Scatter3d(
                x=wall_left[:, 0],
                y=wall_left[:, 1],
                z=wall_left[:, 2],
                mode="lines",
                line=dict(color="red", width=2),
                name="Bound L",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=wall_right[:, 0],
                y=wall_right[:, 1],
                z=wall_right[:, 2],
                mode="lines",
                line=dict(color="red", width=2),
                name="Bound R",
            )
        )

        fig.update_layout(
            title="Interactive Flight Corridor (With Gates & Avoidance)",
            scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()

    def get_frame(self, s_query):
        idx = np.searchsorted(self.pt_frame["s"], s_query) - 1
        idx = np.clip(idx, 0, len(self.pt_frame["s"]) - 1)
        return {k: self.pt_frame[k][idx] for k in self.pt_frame if k != "s"}

    def get_closest_s(self, pos_query, s_guess=0.0, window=5.0):
        mask = (self.pt_frame["s"] >= s_guess - 1.0) & (self.pt_frame["s"] <= s_guess + window)
        if not np.any(mask):
            candidates_pos, candidates_s = self.pt_frame["pos"], self.pt_frame["s"]
        else:
            candidates_pos, candidates_s = self.pt_frame["pos"][mask], self.pt_frame["s"][mask]
        dists = np.linalg.norm(candidates_pos - pos_query, axis=1)
        return candidates_s[np.argmin(dists)]

    def get_static_bounds(self, s_query):
        """Lookup pre-computed bounds for a given s."""
        idx = np.searchsorted(self.pt_frame["s"], s_query)
        idx = np.clip(idx, 0, len(self.pt_frame["s"]) - 1)
        return self.corridor_map["lb_w1"][idx], self.corridor_map["ub_w1"][idx]


# ==============================================================================
# 3. ACADOS SOLVER SETUP
# ==============================================================================


class SpatialMPC:
    def __init__(self, params, N=50, Tf=1.0):
        self.N = N
        self.Tf = Tf
        params["g"] = params["gravity_vec"][2]  # Ensure g is consistent
        self.params = params

        # Clean compile directory
        if os.path.exists("c_generated_code"):
            try:
                shutil.rmtree("c_generated_code")
            except Exception:
                pass

        self.solver = self._build_solver()

    def _build_solver(self):
        model = export_model(self.params)
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf

        # --- DIMENSIONS ---
        nx = 12
        nu = 4
        ny = nx + nu
        ny_e = nx

        # --- COST CONFIGURATION ---
        q_diag = np.array(
            [
                1.0,
                20.0,
                20.0,  # Pos (s, w1, w2)
                10.0,
                5.0,
                5.0,  # Vel
                1.0,
                1.0,
                1.0,  # Att
                0.1,
                0.1,
                0.1,  # Rate
            ]
        )

        r_diag = np.array([5.0, 5.0, 5.0, 0.1])  # Input weights

        ocp.cost.W = scipy.linalg.block_diag(np.diag(q_diag), np.diag(r_diag))
        ocp.cost.W_e = np.diag(q_diag)

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # --- CONSTRAINTS ---
        # Hard Inputs
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, self.params["thrust_min"] * 4])
        ocp.constraints.ubu = np.array([+0.5, +0.5, +0.5, self.params["thrust_max"] * 4])

        # Soft State Bounds (Corridor) - Path Stage
        # w1, w2 indices in x are 1 and 2
        ocp.constraints.idxbx = np.array([1, 2, 6, 7, 8])  # w1, w2, phi, theta, psi
        ocp.constraints.lbx = np.array([-0.4, -0.4, -0.5, -0.5, -0.5])
        ocp.constraints.ubx = np.array([+0.4, +0.4, +0.5, +0.5, +0.5])

        # --- FIXED: Terminal Constraints ---
        # Ensure terminal node N also respects the flight corridor
        ocp.constraints.idxbx_e = np.array([1, 2, 6, 7, 8])
        ocp.constraints.lbx_e = np.array([-0.4, -0.4, -0.5, -0.5, -0.5])
        ocp.constraints.ubx_e = np.array([+0.4, +0.4, +0.5, +0.5, +0.5])

        # Slack Config
        ns = 2
        ocp.constraints.idxsbx = np.array([0, 1])  # Slack on 0th and 1st element of idxbx

        BIG_COST = 1000.0
        ocp.cost.zl = BIG_COST * np.ones(ns)
        ocp.cost.zu = BIG_COST * np.ones(ns)
        ocp.cost.Zl = BIG_COST * np.ones(ns)
        ocp.cost.Zu = BIG_COST * np.ones(ns)

        ocp.constraints.x0 = np.zeros(nx)

        # --- OPTIONS ---
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp.solver_options.qp_solver_tol_stat = 1e-4

        # --- PARAMETERS ---
        p0 = np.concatenate([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0], [0, 0]])
        ocp.parameter_values = p0

        return AcadosOcpSolver(ocp, json_file="acados_spatial.json")


# ==============================================================================
# 4. CONTROLLER CLASS
# ==============================================================================


class SpatialMPCController(Controller):
    def __init__(self, obs: dict, info: dict, config: dict, env=None):
        # 1. Setup
        self.params = get_drone_params()
        self.v_target = v_max_ref  # Target speed
        self.env = env

        # --- FIXED: Use Upper Case for Constants to match usage ---
        self.OBS_RADIUS = safety_radius
        self.W1_MAX = max_lateral_width
        self.W2_MAX = max_lateral_width

        # --- FIXED: Robust Obstacle Parsing ---
        # Attempt to load obstacles from config, fallback to info
        raw_obstacles = config.get("env", {}).get("track", {}).get("obstacles", [])
        if not raw_obstacles and "obstacles" in info:
            raw_obstacles = info["obstacles"]

        # Parse into list of numpy arrays
        self.obstacles_pos = []
        for o in raw_obstacles:
            if isinstance(o, dict) and "pos" in o:
                self.obstacles_pos.append(np.array(o["pos"]))
            elif isinstance(o, (list, np.ndarray)):
                self.obstacles_pos.append(np.array(o))
            elif isinstance(o, dict):
                # Fallback if 'pos' key isn't present but dict itself is the pos (unlikely but safe)
                self.obstacles_pos.append(np.array(list(o.values())))

        # 2. Geometry
        gates_list = config.get("env", {}).get("track", {}).get("gates", [])
        if not gates_list and "gates" in info:
            gates_list = info["gates"]

        gates_pos = [g["pos"] for g in gates_list]
        start_pos = obs["pos"]

        gates_quaternions = obs["gates_quat"]
        gates_normals, gates_y, gates_z = self._get_gate_normals_y_z(gates_quaternions)
        gate_size = 1.0
        self.geo = GeometryEngine(
            gates_pos=gates_pos,
            gates_normal=gates_normals,
            gates_y=gates_y,
            gates_z=gates_z,
            gate_size=gate_size,
            obstacles_pos=self.obstacles_pos,
        )

        # 3. Solver
        self.N_horizon = mpc_horizons_global
        self.mpc = SpatialMPC(self.params, N=self.N_horizon, Tf=1.0)

        # 4. State
        self.prev_s = 0.0
        self.episode_start_time = datetime.now()
        self.step_count = 0
        self.control_log = {
            k: []
            for k in [
                "timestamps",
                "phi_c",
                "theta_c",
                "psi_c",
                "thrust_c",
                "solver_status",
                "s",
                "w1",
                "w2",
                "ds",
            ]
        }
        self.debug = True

        self.reset_mpc_solver()

    def _compute_corridor_bounds(self, s_pred, frame_pos, frame_t, frame_n1, frame_n2):
        """Computes dynamic [lb, ub] for w1 and w2 at a specific path location s.
        Projects 'thin rod' obstacles onto the transverse plane.
        """
        # 1. Initialize with full corridor width
        lb_w1, ub_w1 = -self.W1_MAX, self.W1_MAX
        lb_w2, ub_w2 = -self.W2_MAX, self.W2_MAX

        # Sensitivity: How far along s (longitudinal) do we care about an obstacle?
        longitudinal_threshold = 0.5

        # --- FIXED: Loop over self.obstacles_pos instead of undefined self.obstacles ---
        for obs_pos in self.obstacles_pos:
            # Vector from Path Center -> Obstacle
            r_vec = obs_pos - frame_pos

            # Project onto Tangent (s-direction)
            s_dist = np.dot(r_vec, frame_t)

            if abs(s_dist) < longitudinal_threshold:
                # Project onto Transverse Plane (n1, n2)
                w1_obs = np.dot(r_vec, frame_n1)
                w2_obs = np.dot(r_vec, frame_n2)

                # --- Dominant Side Logic ---
                # Check if obstacle is actually inside our max corridor
                if (lb_w1 < w1_obs < ub_w1) and (lb_w2 < w2_obs < ub_w2):
                    # DECISION: Pass Left or Pass Right?
                    if w1_obs > 0:
                        # Obstacle on Left -> Pass Right (Trim Upper Bound)
                        dist_to_surface = w1_obs - self.OBS_RADIUS
                        ub_w1 = min(ub_w1, dist_to_surface)
                    else:
                        # Obstacle on Right -> Pass Left (Trim Lower Bound)
                        dist_to_surface = w1_obs + self.OBS_RADIUS
                        lb_w1 = max(lb_w1, dist_to_surface)

        # --- FIXED: Gap Closing Logic ---
        # Prevent infeasibility if bounds cross
        if lb_w1 >= ub_w1:
            mid = (lb_w1 + ub_w1) / 2
            lb_w1 = mid - 0.05
            ub_w1 = mid + 0.05
            if self.debug and self.step_count % 50 == 0:
                print(f"Warning: Corridor gap closed at s={s_pred:.2f}. Forcing narrow passage.")

        return np.array([lb_w1, lb_w2]), np.array([ub_w1, ub_w2])

    def _get_gate_normals_y_z(self, gates_quaternions: np.ndarray) -> np.ndarray:
        rotations = R.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        gates_normals = rotation_matrices[:, :, 0]  # X-axis (normal)
        gates_y = rotation_matrices[:, :, 1]  # Y-axis
        gates_z = rotation_matrices[:, :, 2]  # Z-axis (up)
        return gates_normals, gates_y, gates_z

    def reset_mpc_solver(self):
        """Warm starts the solver with a forward guess."""
        nx = 12
        hover_T = self.params["mass"] * self.params["g"]

        for k in range(self.N_horizon + 1):
            x_guess = np.zeros(nx)
            # Linear ramp from 0 to v_target
            vel_k = self.v_target * (k / self.N_horizon)
            x_guess[3] = vel_k
            x_guess[0] = vel_k * k * (self.mpc.Tf / self.N_horizon) * 0.5

            self.mpc.solver.set(k, "x", x_guess)
            if k < self.N_horizon:
                self.mpc.solver.set(k, "u", np.array([0, 0, 0, hover_T]))

        self.prev_s = 0.0

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        # Derived states
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])

        ANGLE_LB = np.array([-0.5, -0.5, -0.5])
        ANGLE_UB = np.array([0.5, 0.5, 0.5])

        hover_T = self.params["mass"] * -self.params["g"]

        # 1. State Feedback (World -> Spatial)
        x_spatial = self._cartesian_to_spatial(obs["pos"], obs["vel"], obs["rpy"], obs["drpy"])

        # Update current state constraint (x0)
        self.mpc.solver.set(0, "lbx", x_spatial)
        self.mpc.solver.set(0, "ubx", x_spatial)

        # 2. Horizon Updates
        curr_s = x_spatial[0]
        curr_ds = x_spatial[3]
        dt = self.mpc.Tf / self.mpc.N

        # "Carrot" approach: Set target velocity high
        target_vel = self.v_target

        # if self.env is not None and self.step_count % 5 == 0:
        #     try:
        #         path_points = self.geo.pt_frame['pos'][::5]
        #         draw_line(self.env, points=path_points, rgba=np.array([0.0, 1.0, 0.0, 0.5]))
        #     except Exception:
        #         pass

        if self.env is not None:
            try:
                # 1. Draw Center Line (Green)
                path_points = self.geo.pt_frame["pos"][::5]
                draw_line(self.env, points=path_points, rgba=np.array([0.0, 1.0, 0.0, 0.5]))

                # 2. Draw Left & Right Bounds (Red)
                # We need the normal vector n1 at each point to offset the position
                # Offset = Position + (Width * n1)

                # Slicing [::5] to match the path_points downsampling
                positions = self.geo.pt_frame["pos"][::5]
                normals_n1 = self.geo.pt_frame["n1"][::5]

                # Calculate Left and Right Boundary Points
                # Note: w1 is along n1.
                # Left Bound  = pos + (W1_MAX * n1)
                # Right Bound = pos + (-W1_MAX * n1)
                left_bound_points = positions + (self.W1_MAX * normals_n1)
                right_bound_points = positions - (self.W1_MAX * normals_n1)

                positions + (-self.W1_MAX * normals_n1)

                # Draw them
                draw_line(
                    self.env, points=left_bound_points, rgba=np.array([1.0, 0.0, 0.0, 0.3])
                )  # Red, semi-transparent
                draw_line(self.env, points=right_bound_points, rgba=np.array([1.0, 0.0, 0.0, 0.3]))

            except Exception:
                pass

        # Initialize a running reference s starting from current position
        running_s_ref = curr_s
        max_lat_acc = corner_acc
        epsilon = 0.01

        # --- LOOP 0 to N-1 ---
        for k in range(self.mpc.N):
            # A. Predict s for parameter lookup
            s_pred = curr_s + k * max(curr_ds, 1.0) * dt

            # B. Get Frame & Curvature
            f = self.geo.get_frame(s_pred)

            # C. Dynamic Corridor Bounds
            w_lb, w_ub = self._compute_corridor_bounds(s_pred, f["pos"], f["t"], f["n1"], f["n2"])
            vis_curr_l_points = []
            vis_curr_r_points = []
            vis_curr_l_points.append(f["pos"] + w_ub[0] * f["n1"])
            vis_curr_r_points.append(f["pos"] + w_lb[0] * f["n1"])

            # Update Constraints
            if k > 0:
                lbx_k = np.concatenate([w_lb, ANGLE_LB])
                # print("this is lbx_k:", lbx_k)
                ubx_k = np.concatenate([w_ub, ANGLE_UB])
                self.mpc.solver.set(k, "lbx", lbx_k)
                self.mpc.solver.set(k, "ubx", ubx_k)

            # D. Speed Logic
            k_mag = np.sqrt(f["k1"] ** 2 + f["k2"] ** 2)
            v_corner = np.sqrt(max_lat_acc / (k_mag + epsilon))
            v_ref_k = min(v_corner, target_vel)

            # E. Integrate s_ref
            running_s_ref += v_ref_k * dt

            # F. Set Parameters P
            p_k = np.concatenate(
                [f["t"], f["n1"], f["n2"], [f["k1"]], [f["k2"]], [f["dk1"]], [f["dk2"]]]
            )
            self.mpc.solver.set(k, "p", p_k)

            # G. Set Reference yref
            y_ref = np.zeros(16)
            y_ref[0] = running_s_ref
            y_ref[3] = v_ref_k
            y_ref[15] = hover_T
            self.mpc.solver.set(k, "yref", y_ref)

        # --- FIXED: Terminal Node N Update ---
        # 1. Integrate one last step for terminal s
        s_end = running_s_ref + v_ref_k * dt

        # 2. Get Frame for N
        f_end = self.geo.get_frame(s_end)
        p_end = np.concatenate(
            [
                f_end["t"],
                f_end["n1"],
                f_end["n2"],
                [f_end["k1"]],
                [f_end["k2"]],
                [f_end["dk1"]],
                [f_end["dk2"]],
            ]
        )
        self.mpc.solver.set(self.mpc.N, "p", p_end)

        # 3. Compute Terminal Corridor Bounds (Safety check at end of horizon)
        w_lb_e, w_ub_e = self._compute_corridor_bounds(
            s_end, f_end["pos"], f_end["t"], f_end["n1"], f_end["n2"]
        )
        lbx_e = np.concatenate([w_lb_e, ANGLE_LB])
        ubx_e = np.concatenate([w_ub_e, ANGLE_UB])
        self.mpc.solver.set(self.mpc.N, "lbx", lbx_e)
        self.mpc.solver.set(self.mpc.N, "ubx", ubx_e)

        # 4. Set Terminal Reference
        yref_e = np.zeros(12)
        yref_e[0] = s_end
        yref_e[3] = v_ref_k
        yref_e[11] = hover_T
        self.mpc.solver.set(self.mpc.N, "yref", yref_e)

        if self.debug and self.step_count % 50 == 0:
            print(f"Terminal Step, s_end: {s_end:.2f}, k_mag: {k_mag:.3f}, v_ref: {v_ref_k:.3f}")

        # 4. Solve
        status = self.mpc.solver.solve()

        # Visualization (Blue Line)
        if self.env is not None:
            try:
                mpc_points_cartesian = []
                for k in range(self.mpc.N + 1):
                    x_k = self.mpc.solver.get(k, "x")
                    pos_k = self._spatial_to_cartesian(x_k[0], x_k[1], x_k[2])
                    mpc_points_cartesian.append(pos_k)

                draw_line(
                    self.env,
                    points=np.array(mpc_points_cartesian),
                    rgba=np.array([0.0, 0.0, 1.0, 0.8]),
                )
                draw_line(
                    self.env,
                    points=np.array(vis_curr_l_points),
                    rgba=np.array([0.0, 1.0, 0.0, 0.8]),
                )
                draw_line(
                    self.env,
                    points=np.array(vis_curr_r_points),
                    rgba=np.array([0.0, 1.0, 0.0, 0.8]),
                )
            except Exception:
                pass

        if status != 0:
            print(f"MPC Warning: Solver status {status}")
            u_opt = np.array([0.0, 0.0, 0.0, hover_T])
        else:
            u_opt = self.mpc.solver.get(0, "u")

        self._log_control_step(x_spatial, u_opt, status)
        return np.array([u_opt[0], u_opt[1], u_opt[2], u_opt[3]])

    def _spatial_to_cartesian(self, s, w1, w2):
        """Reconstructs global position from spatial coordinates."""
        f = self.geo.get_frame(s)
        pos_world = f["pos"] + w1 * f["n1"] + w2 * f["n2"]
        return pos_world

    def _cartesian_to_spatial(self, pos, vel, rpy, drpy):
        """Projects global state onto the path frame."""
        s = self.geo.get_closest_s(pos, s_guess=self.prev_s)
        self.prev_s = s
        f = self.geo.get_frame(s)

        r_vec = pos - f["pos"]
        w1 = np.dot(r_vec, f["n1"])
        w2 = np.dot(r_vec, f["n2"])

        h = 1 - f["k1"] * w1 - f["k2"] * w2
        h = max(h, 0.01)

        ds = np.dot(vel, f["t"]) / h
        dw1 = np.dot(vel, f["n1"])
        dw2 = np.dot(vel, f["n2"])

        return np.array(
            [s, w1, w2, ds, dw1, dw2, rpy[0], rpy[1], rpy[2], drpy[0], drpy[1], drpy[2]]
        )

    def reset(self):
        self.prev_s = 0.0
        self.reset_mpc_solver()

    def episode_reset(self):
        self.reset()

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        return False

    def episode_callback(self):
        if len(self.control_log["timestamps"]) > 0:
            self.plot_all_diagnostics()
        return

    def _log_control_step(self, x_spatial: np.ndarray, u_opt: np.ndarray, solver_status: int):
        self.step_count += 1
        elapsed_time = (datetime.now() - self.episode_start_time).total_seconds()

        self.control_log["timestamps"].append(elapsed_time)
        self.control_log["phi_c"].append(float(u_opt[0]))
        self.control_log["theta_c"].append(float(u_opt[1]))
        self.control_log["psi_c"].append(float(u_opt[2]))
        self.control_log["thrust_c"].append(float(u_opt[3]))
        self.control_log["solver_status"].append(int(solver_status))
        self.control_log["s"].append(float(x_spatial[0]))
        self.control_log["w1"].append(float(x_spatial[1]))
        self.control_log["w2"].append(float(x_spatial[2]))
        self.control_log["ds"].append(float(x_spatial[3]))

        if self.debug and self.step_count % 10 == 0:
            print(
                f"[Step {self.step_count}] t={elapsed_time:.3f}s | "
                f"s={x_spatial[0]:.2f} w1={x_spatial[1]:+.4f} w2={x_spatial[2]:+.4f} | "
                f"Status={solver_status}"
            )

    def save_control_log(self, filepath: str = None):
        if filepath is None:
            timestamp = self.episode_start_time.strftime("%Y%m%d_%H%M%S")
            filepath = f"control_log_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(self.control_log, f, indent=2)
        return filepath

    def plot_control_values(self, figsize=(16, 10), save_path: str = None):
        if len(self.control_log["timestamps"]) == 0:
            return

        t = np.array(self.control_log["timestamps"])

        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle("MPC Control Values & State Feedback", fontsize=16, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(t, self.control_log["phi_c"], "b-", label="φ_c")
        ax.set_ylabel("Roll (rad)")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(t, self.control_log["theta_c"], "g-", label="θ_c")
        ax.set_ylabel("Pitch (rad)")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(t, self.control_log["thrust_c"], "r-", label="Thrust")
        ax.set_ylabel("Thrust (N)")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(t, self.control_log["psi_c"], "m-", label="ψ_c")
        ax.set_ylabel("Yaw (rad)")
        ax.legend()

        ax = axes[2, 0]
        ax.plot(t, self.control_log["s"], "c-", label="s")
        ax.set_ylabel("Progress (m)")
        ax.legend()

        ax = axes[2, 1]
        ax.plot(t, self.control_log["w1"], "orange", label="w1")
        ax.plot(t, self.control_log["w2"], "purple", label="w2")
        ax.axhline(y=0.5, color="r", linestyle="--")
        ax.axhline(y=-0.5, color="r", linestyle="--")
        ax.set_ylabel("Deviation (m)")
        ax.legend()

        plt.tight_layout()
        if save_path is None:
            timestamp = self.episode_start_time.strftime("%Y%m%d_%H%M%S")
            save_path = f"control_plot_{timestamp}.png"
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_solver_status(self, save_path: str = None):
        if len(self.control_log["timestamps"]) == 0:
            return
        t = np.array(self.control_log["timestamps"])
        status = np.array(self.control_log["solver_status"])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(t, status, c=["g" if s == 0 else "r" for s in status])
        ax.set_title("Solver Status (0=Success)")
        if save_path is None:
            save_path = "solver_status.png"
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_all_diagnostics(self, save_dir: str = None):
        if save_dir is None:
            timestamp = self.episode_start_time.strftime("%Y%m%d_%H%M%S")
            save_dir = f"mpc_debug/mpc_diagnostics_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        self.save_control_log(os.path.join(save_dir, "control_log.json"))
        self.plot_control_values(save_path=os.path.join(save_dir, "control_values.png"))
        self.plot_solver_status(save_path=os.path.join(save_dir, "solver_status.png"))
        return save_dir
