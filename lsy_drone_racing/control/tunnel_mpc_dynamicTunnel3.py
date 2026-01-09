import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

# Acados Imports
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

# Simulation Environment Imports
try:
    from drone_models.core import load_params
    from drone_models.utils.rotation import ang_vel2rpy_rates

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.utils.utils import draw_line
except ImportError:
    print("Warning: Simulation specific modules not found. Using mocks/defaults.")

    def load_params(*args):
        return None

    def ang_vel2rpy_rates(q, w):
        return np.zeros(3)

    def draw_line(*args, **kwargs):
        pass

    class Controller:
        pass


# Use non-interactive backend
matplotlib.use("Agg")

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

CONSTANTS = {
    "v_max_ref": 1.0,  # m/s
    "corner_acc": 1.5,  # m/s^2
    "mpc_horizon": 50,  # Steps
    "max_lateral_width": 0.35,  # m (Static Corridor width)
    "safety_radius": 0.06,  # m (Obstacle buffer - Increased slightly for safety)
    "tf_horizon": 1.0,  # s
    "detour_radius" : 0.5,
    "gate_norm_detour": 0.5,
    "tangent_scale_factor": 1.05

}

# ==============================================================================
# 2. DYNAMICS & MODEL DEFINITION
# ==============================================================================


def get_drone_params() -> Dict[str, Any]:
    params = load_params("so_rpy", "cf21B_500")
    if params is not None:
        return params

    return {
        "mass": 0.04338,
        "gravity_vec": np.array([0.0, 0.0, -9.81]),
        "g": 9.81,
        "J": np.diag([25e-6, 28e-6, 49e-6]),
        "J_inv": np.linalg.inv(np.diag([25e-6, 28e-6, 49e-6])),
        "thrust_min": 0.0,
        "thrust_max": 0.2,
        "acc_coef": 0.0,
        "cmd_f_coef": 0.96836458,
        "rpy_coef": [-188.9910, -188.9910, -138.3109],
        "rpy_rates_coef": [-12.7803, -12.7803, -16.8485],
        "cmd_rpy_coef": [138.0834, 138.0834, 198.5161],
    }


def symbolic_dynamics_spatial(params: Dict[str, Any]) -> Tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    mass = params["mass"]
    gravity_vec = params["gravity_vec"]
    acc_coef = params["acc_coef"]
    cmd_f_coef = params["cmd_f_coef"]
    rpy_coef = params["rpy_coef"]
    rpy_rates_coef = params["rpy_rates_coef"]
    cmd_rpy_coef = params["cmd_rpy_coef"]

    # State: [s, w1, w2, ds, dw1, dw2, phi, theta, psi, dphi, dtheta, dpsi]
    s, w1, w2 = ca.SX.sym("s"), ca.SX.sym("w1"), ca.SX.sym("w2")
    ds, dw1, dw2 = ca.SX.sym("ds"), ca.SX.sym("dw1"), ca.SX.sym("dw2")
    phi, theta, psi = ca.SX.sym("phi"), ca.SX.sym("theta"), ca.SX.sym("psi")
    dphi, dtheta, dpsi = ca.SX.sym("dphi"), ca.SX.sym("dtheta"), ca.SX.sym("dpsi")
    rpy = ca.vertcat(phi, theta, psi)
    drpy = ca.vertcat(dphi, dtheta, dpsi)
    X = ca.vertcat(s, w1, w2, ds, dw1, dw2, rpy, drpy)

    # Input
    phi_c, theta_c, psi_c, T_c = (
        ca.SX.sym("phi_c"),
        ca.SX.sym("theta_c"),
        ca.SX.sym("psi_c"),
        ca.SX.sym("T_c"),
    )
    cmd_rpy = ca.vertcat(phi_c, theta_c, psi_c)
    U = ca.vertcat(cmd_rpy, T_c)

    # Parameters
    t_vec = ca.SX.sym("t_vec", 3)
    n1_vec = ca.SX.sym("n1_vec", 3)
    n2_vec = ca.SX.sym("n2_vec", 3)
    k1, k2 = ca.SX.sym("k1"), ca.SX.sym("k2")
    dk1, dk2 = ca.SX.sym("dk1"), ca.SX.sym("dk2")
    P = ca.vertcat(t_vec, n1_vec, n2_vec, k1, k2, dk1, dk2)

    # Dynamics
    c_rpy = ca.DM(rpy_coef)
    c_drpy = ca.DM(rpy_rates_coef)
    c_cmd = ca.DM(cmd_rpy_coef)

    ddrpy = c_rpy * rpy + c_drpy * drpy + c_cmd * cmd_rpy

    thrust_mag = acc_coef + cmd_f_coef * T_c
    F_body = ca.vertcat(0, 0, thrust_mag)

    cx, cy, cz = ca.cos(phi), ca.cos(theta), ca.cos(psi)
    sx, sy, sz = ca.sin(phi), ca.sin(theta), ca.sin(psi)
    R_IB = ca.vertcat(
        ca.horzcat(cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy),
        ca.horzcat(cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
        ca.horzcat(-sy, cy * sx, cx * cy),
    )

    g_vec_sym = ca.DM(gravity_vec)
    acc_world = g_vec_sym + (R_IB @ F_body) / mass

    h = 1 - k1 * w1 - k2 * w2
    h_dot = -(k1 * dw1 + k2 * dw2 + (dk1 * w1 + dk2 * w2) * ds)

    coriolis = (
        (ds * h_dot) * t_vec
        + (ds**2 * h * k1) * n1_vec
        + (ds**2 * h * k2) * n2_vec
        - (ds * dw1 * k1) * t_vec
        - (ds * dw2 * k2) * t_vec
    )

    proj_t = ca.dot(t_vec, acc_world - coriolis)
    dds = proj_t / h
    ddw1 = ca.dot(n1_vec, acc_world - coriolis)
    ddw2 = ca.dot(n2_vec, acc_world - coriolis)

    X_Dot = ca.vertcat(ds, dw1, dw2, dds, ddw1, ddw2, drpy, ddrpy)
    return X_Dot, X, U, P


def export_model(params: Dict[str, Any]) -> AcadosModel:
    X_dot, X, U, P = symbolic_dynamics_spatial(params)
    model = AcadosModel()
    model.name = "spatial_mpc_drone"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    model.p = P
    return model


# ==============================================================================
# 3. GEOMETRY ENGINE
# ==============================================================================


class GeometryEngine:
    def __init__(self, gates_pos, gates_normals, start_pos):
        self.DETOUR_ANGLE_THRESHOLD = 60.0
        self.DETOUR_RADIUS = CONSTANTS['detour_radius']
        self.TANGENT_SCALE_FACTOR = CONSTANTS['tangent_scale_factor']

        self.gates_pos = np.asarray(gates_pos, dtype=np.float64)
        self.gate_normals = np.asarray(gates_normals, dtype=np.float64)
        self.start_pos = np.asarray(start_pos, dtype=np.float64)

        self.waypoints, self.wp_types, self.wp_normals = self._initialize_waypoints()
        self.waypoints, self.wp_types, self.wp_normals = self._add_detour_logic(
            self.waypoints, self.wp_types, self.wp_normals
        )
        self.tangents = self._compute_hermite_tangents()

        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_knots = np.insert(np.cumsum(dists), 0, 0.0)
        self.total_length = self.s_knots[-1]
        self.spline = CubicHermiteSpline(self.s_knots, self.waypoints, self.tangents)
        self.pt_frame = self._generate_parallel_transport_frame(num_points=1000)

    def _initialize_waypoints(self):
        wps = [self.start_pos]
        types = [0]
        normals = [np.zeros(3)]
        for i in range(len(self.gates_pos)):
            wps.append(self.gates_pos[i])
            types.append(1)
            normals.append(self.gate_normals[i])
        return np.array(wps), np.array(types), np.array(normals)

    def _add_detour_logic(self, wps, types, normals):
        new_wps = [wps[0]]
        new_types = [types[0]]
        new_normals = [normals[0]]

        for i in range(len(wps) - 1):
            curr_p = wps[i]
            next_p = wps[i + 1]
            curr_type = types[i]
            if curr_type == 1:
                gate_idx = i - 1
                gate_norm = self.gate_normals[gate_idx]
                vec_to_next = next_p - curr_p
                
                dist = np.linalg.norm(vec_to_next)
                if dist > 1e-6:
                    vec_to_next /= dist
                    alignment = np.dot(gate_norm, vec_to_next)
                    angle_deg = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
                    if angle_deg > self.DETOUR_ANGLE_THRESHOLD:
                        proj = vec_to_next - (np.dot(vec_to_next, gate_norm) * gate_norm)
                        detour_dir = (
                            proj / np.linalg.norm(proj)
                            if np.linalg.norm(proj) > 1e-3
                            else np.array([0, 0, 1])
                        )
                        detour_pos = curr_p + (detour_dir * self.DETOUR_RADIUS) + (gate_norm * CONSTANTS['gate_norm_detour'])
                        new_wps.append(detour_pos)
                        new_types.append(2)
                        new_normals.append(np.zeros(3))
            new_wps.append(next_p)
            new_types.append(types[i + 1])
            new_normals.append(normals[i + 1])
        return np.array(new_wps), np.array(new_types), np.array(new_normals)

    def _compute_hermite_tangents(self):
        num_pts = len(self.waypoints)
        tangents = np.zeros_like(self.waypoints)
        for i in range(num_pts):
            dist_prev = np.linalg.norm(self.waypoints[i] - self.waypoints[i - 1]) if i > 0 else 0
            dist_next = (
                np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i]) if i < num_pts - 1 else 0
            )
            base_scale = min(
                dist_prev if dist_prev > 0 else dist_next, dist_next if dist_next > 0 else dist_prev
            )
            scale = base_scale * self.TANGENT_SCALE_FACTOR
            if self.wp_types[i] == 1:
                normal = self.wp_normals[i].copy()
                if i > 0 and i < num_pts - 1:
                    flow_vec = self.waypoints[i + 1] - self.waypoints[i - 1]
                    if np.dot(normal, flow_vec) < 0:
                        normal = -normal
                tangents[i] = normal * scale
            else:
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

    def _generate_parallel_transport_frame(self, num_points=3000):
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

        k1_list, k2_list = [], []
        for i, s in enumerate(s_eval):
            pos = self.spline(s)
            k_vec = self.spline(s, 2)
            k1 = np.dot(k_vec, curr_n1)
            k2 = np.dot(k_vec, curr_n2)
            frames["pos"].append(pos)
            frames["t"].append(curr_t)
            frames["n1"].append(curr_n1)
            frames["n2"].append(curr_n2)
            k1_list.append(k1)
            k2_list.append(k2)
            if i < len(s_eval) - 1:
                next_t = self.spline(s_eval[i + 1], 1)
                next_t /= np.linalg.norm(next_t)
                axis = np.cross(curr_t, next_t)
                angle = np.arccos(np.clip(np.dot(curr_t, next_t), -1.0, 1.0))
                if np.linalg.norm(axis) > 1e-6:
                    axis /= np.linalg.norm(axis)
                    r_vec = R.from_rotvec(axis * angle)
                    next_n1, next_n2 = r_vec.apply(curr_n1), r_vec.apply(curr_n2)
                else:
                    next_n1, next_n2 = curr_n1, curr_n2
                curr_t, curr_n1, curr_n2 = next_t, next_n1, next_n2

        frames["k1"] = np.array(k1_list)
        frames["k2"] = np.array(k2_list)
        frames["dk1"] = np.gradient(frames["k1"], ds)
        frames["dk2"] = np.gradient(frames["k2"], ds)
        for k in frames:
            if isinstance(frames[k], list):
                frames[k] = np.array(frames[k])
        return frames

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


# ==============================================================================
# 4. ACADOS SOLVER SETUP
# ==============================================================================


class SpatialMPC:
    def __init__(self, params, N=50, Tf=1.0):
        self.N, self.Tf, self.params = N, Tf, params
        self.params["g"] = params["gravity_vec"][2]
        if os.path.exists("c_generated_code"):
            try:
                shutil.rmtree("c_generated_code")
            except OSError:
                pass
        self.solver = self._build_solver()

    def _build_solver(self):
        model = export_model(self.params)
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf

        nx, nu = 12, 4
        ny, ny_e = nx + nu, nx

        # s, w1, w2, ds, dw1, dw2, rpy[0], rpy[1], rpy[2], drpy[0], drpy[1], drpy[2]

        q_diag = np.array([1.0, 20.0, 20.0, 10.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1])

        # rpy and thrust
        r_diag = np.array([50.0, 50.0, 50.0, 8.0])

        ocp.cost.W = scipy.linalg.block_diag(np.diag(q_diag), np.diag(r_diag))
        ocp.cost.W_e = np.diag(q_diag)
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, self.params["thrust_min"] * 4])
        ocp.constraints.ubu = np.array([+0.5, +0.5, +0.5, self.params["thrust_max"] * 4])

        # Soft State Bounds (w1, w2, phi, theta, psi)
        ocp.constraints.idxbx = np.array([1, 2, 6, 7, 8])
        ocp.constraints.lbx = np.array([-0.4, -0.4, -0.5, -0.5, -0.5])
        ocp.constraints.ubx = np.array([+0.4, +0.4, +0.5, +0.5, +0.5])

        ocp.constraints.idxbx_e = np.array([1, 2, 6, 7, 8])
        ocp.constraints.lbx_e = np.array([-0.4, -0.4, -0.5, -0.5, -0.5])
        ocp.constraints.ubx_e = np.array([+0.4, +0.4, +0.5, +0.5, +0.5])

        ns = 2
        ocp.constraints.idxsbx = np.array([0, 1])
        BIG_COST = 1000.0
        ocp.cost.zl = BIG_COST * np.ones(ns)
        ocp.cost.zu = BIG_COST * np.ones(ns)
        ocp.cost.Zl = BIG_COST * np.ones(ns)
        ocp.cost.Zu = BIG_COST * np.ones(ns)
        ocp.constraints.x0 = np.zeros(nx)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp.parameter_values = np.zeros(13)

        return AcadosOcpSolver(ocp, json_file="acados_spatial.json")


# ==============================================================================
# 5. CONTROLLER CLASS
# ==============================================================================


class SpatialMPCController(Controller):
    def __init__(self, obs: Dict, info: Dict, config: Dict, env=None):
        self.params = get_drone_params()
        self.v_target = CONSTANTS["v_max_ref"]
        self.env = env
        self.OBS_RADIUS = CONSTANTS["safety_radius"]
        self.W1_MAX = CONSTANTS["max_lateral_width"]
        self.W2_MAX = CONSTANTS["max_lateral_width"]

        # --- DEBUG: Obstacle Loading ---
        raw_obstacles = config.get("env", {}).get("track", {}).get("obstacles", [])
        if not raw_obstacles and "obstacles" in info:
            raw_obstacles = info["obstacles"]

        self.obstacles_pos = []
        for o in raw_obstacles:
            if isinstance(o, dict) and "pos" in o:
                self.obstacles_pos.append(np.array(o["pos"]))
            elif isinstance(o, (list, np.ndarray)):
                self.obstacles_pos.append(np.array(o))
            elif isinstance(o, dict):
                self.obstacles_pos.append(np.array(list(o.values())))

        print(f"\n[INIT] DEBUG: Loaded {len(self.obstacles_pos)} obstacles.")
        for i, o in enumerate(self.obstacles_pos):
            print(f"   Obs {i}: {o}")
        print("--------------------------------------------------\n")

        gates_list = config.get("env", {}).get("track", {}).get("gates", [])
        if not gates_list and "gates" in info:
            gates_list = info["gates"]
        gates_pos = [g["pos"] for g in gates_list]
        gates_normals = self._get_gate_normals(obs["gates_quat"])

        self.geo = GeometryEngine(gates_pos, gates_normals, obs["pos"])
        self.N_horizon = CONSTANTS["mpc_horizon"]
        self.mpc = SpatialMPC(self.params, N=self.N_horizon, Tf=CONSTANTS["tf_horizon"])

        self.prev_s = 0.0
        self.episode_start_time = datetime.now()
        self.step_count = 0
        self.debug = True
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

        subsample = 5
        self.global_viz_center = self.geo.pt_frame["pos"][::subsample]
        self.global_viz_left = self.global_viz_center + (
            self.W1_MAX * self.geo.pt_frame["n1"][::subsample]
        )
        self.global_viz_right = self.global_viz_center - (
            self.W1_MAX * self.geo.pt_frame["n1"][::subsample]
        )

        self.reset_mpc_solver()

    def _draw_global_track(self):
        if self.env is None:
            return
        try:
            # pass
            draw_line(self.env, points=self.global_viz_center, rgba=np.array([0.0, 1.0, 0.0, 0.5]))
            # draw_line(self.env, points=self.global_viz_left, rgba=np.array([1.0, 0.0, 0.0, 0.3]))
            # draw_line(self.env, points=self.global_viz_right, rgba=np.array([1.0, 0.0, 0.0, 0.3]))
            # pass
        except Exception:
            pass

    def _compute_corridor_bounds(
        self, s_pred, frame_pos, frame_t, frame_n1, frame_n2, debug_print=False
    ):
        """Computes dynamic [lb, ub] for w1 and w2.
        Includes extensive debugging logic to track why obstacles might be ignored.
        """
        lb_w1, ub_w1 = -self.W1_MAX, self.W1_MAX
        lb_w2, ub_w2 = -self.W2_MAX, self.W2_MAX

        # Increased threshold to catch obstacles earlier
        longitudinal_threshold = 2.0

        for i, obs_pos in enumerate(self.obstacles_pos):
            r_vec = obs_pos - frame_pos
            s_dist = np.dot(r_vec, frame_t)

            # --- DEBUG BLOCK START ---
            if debug_print and abs(s_dist) < 5.0:
                print(f"   [CHECK] Obs {i} | s_dist: {s_dist:.2f} | s_pred: {s_pred:.2f}")
            # --- DEBUG BLOCK END ---

            if abs(s_dist) < longitudinal_threshold:
                w1_obs = np.dot(r_vec, frame_n1)
                w2_obs = np.dot(r_vec, frame_n2)

                # --- FIXED LOGIC: Overlap Check ---
                # Check if obstacle (with radius) overlaps with current bounds
                # We add OBS_RADIUS to the bounds to see if the obstacle's edge is inside
                overlap_w1 = lb_w1 - self.OBS_RADIUS < w1_obs < ub_w1 + self.OBS_RADIUS
                overlap_w2 = lb_w2 - self.OBS_RADIUS < w2_obs < ub_w2 + self.OBS_RADIUS

                if debug_print:
                    print(f"      -> w1_obs: {w1_obs:.3f}, w2_obs: {w2_obs:.3f}")
                    print(f"      -> Overlap Check: w1={overlap_w1}, w2={overlap_w2}")

                if overlap_w1 or overlap_w2:
                    if w1_obs > 0:
                        # Obstacle on Left -> Trim UB
                        dist_to_surface = w1_obs - self.OBS_RADIUS
                        ub_w1 = min(ub_w1, dist_to_surface)
                        if debug_print:
                            print(f"      -> TRIMMED UPPER: New ub_w1 = {ub_w1:.3f}")
                    else:
                        # Obstacle on Right -> Trim LB
                        dist_to_surface = w1_obs + self.OBS_RADIUS
                        lb_w1 = max(lb_w1, dist_to_surface)
                        if debug_print:
                            print(f"      -> TRIMMED LOWER: New lb_w1 = {lb_w1:.3f}")

        if lb_w1 >= ub_w1:
            mid = (lb_w1 + ub_w1) / 2
            lb_w1 = mid - 0.05
            ub_w1 = mid + 0.05
            if debug_print:
                print("      -> WARNING: GAP CLOSED. Resetting to narrow passage.")

        return np.array([lb_w1, lb_w2]), np.array([ub_w1, ub_w2])

    def _get_gate_normals(self, gates_quaternions):
        rotations = R.from_quat(gates_quaternions)
        return rotations.as_matrix()[:, :, 0]

    def reset_mpc_solver(self):
        nx = 12
        hover_T = self.params["mass"] * self.params["g"]
        for k in range(self.N_horizon + 1):
            x_guess = np.zeros(nx)
            vel_k = self.v_target * (k / self.N_horizon)
            x_guess[3] = vel_k
            x_guess[0] = vel_k * k * (self.mpc.Tf / self.N_horizon) * 0.5
            self.mpc.solver.set(k, "x", x_guess)
            if k < self.N_horizon:
                self.mpc.solver.set(k, "u", np.array([0, 0, 0, hover_T]))
        self.prev_s = 0.0

    def compute_control(self, obs: Dict, info: Optional[Dict] = None) -> np.ndarray:
        self._draw_global_track()

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])

        ANGLE_LB = np.array([-0.5, -0.5, -0.5])
        ANGLE_UB = np.array([0.5, 0.5, 0.5])
        hover_T = self.params["mass"] * -self.params["g"]

        x_spatial = self._cartesian_to_spatial(obs["pos"], obs["vel"], obs["rpy"], obs["drpy"])
        self.mpc.solver.set(0, "lbx", x_spatial)
        self.mpc.solver.set(0, "ubx", x_spatial)

        curr_s = x_spatial[0]
        curr_ds = x_spatial[3]
        dt = self.mpc.Tf / self.mpc.N

        running_s_ref = curr_s
        max_lat_acc = CONSTANTS["corner_acc"]
        epsilon = 0.01
        vis_dynamic_left, vis_dynamic_right = [], []

        # --- DEBUG FLAG ---
        # Only print debug info for the FIRST predicted step (k=0)
        # and only every 20 control steps to avoid spamming.
        do_debug = self.step_count % 20 == 0
        if do_debug:
            print(f"\n[STEP {self.step_count}] Computing Control for s={curr_s:.2f}")

        for k in range(self.mpc.N):
            s_pred = curr_s + k * max(curr_ds, 1.0) * dt
            f = self.geo.get_frame(s_pred)

            # Pass debug flag only for k=0
            w_lb, w_ub = self._compute_corridor_bounds(
                s_pred, f["pos"], f["t"], f["n1"], f["n2"], debug_print=(do_debug and k == 0)
            )

            vis_dynamic_left.append(f["pos"] + w_ub[0] * f["n1"])
            vis_dynamic_right.append(f["pos"] + w_lb[0] * f["n1"])

            if k > 0:
                self.mpc.solver.set(k, "lbx", np.concatenate([w_lb, ANGLE_LB]))
                self.mpc.solver.set(k, "ubx", np.concatenate([w_ub, ANGLE_UB]))

            k_mag = np.sqrt(f["k1"] ** 2 + f["k2"] ** 2)
            v_corner = np.sqrt(max_lat_acc / (k_mag + epsilon))
            v_ref_k = min(v_corner, self.v_target)
            running_s_ref += v_ref_k * dt

            p_k = np.concatenate(
                [f["t"], f["n1"], f["n2"], [f["k1"]], [f["k2"]], [f["dk1"]], [f["dk2"]]]
            )
            self.mpc.solver.set(k, "p", p_k)
            y_ref = np.zeros(16)
            y_ref[0] = running_s_ref
            y_ref[3] = v_ref_k
            y_ref[15] = hover_T
            self.mpc.solver.set(k, "yref", y_ref)

        s_end = running_s_ref + v_ref_k * dt
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
        w_lb_e, w_ub_e = self._compute_corridor_bounds(
            s_end, f_end["pos"], f_end["t"], f_end["n1"], f_end["n2"]
        )
        self.mpc.solver.set(self.mpc.N, "lbx", np.concatenate([w_lb_e, ANGLE_LB]))
        self.mpc.solver.set(self.mpc.N, "ubx", np.concatenate([w_ub_e, ANGLE_UB]))
        yref_e = np.zeros(12)
        yref_e[0] = s_end
        yref_e[3] = v_ref_k
        yref_e[11] = hover_T
        self.mpc.solver.set(self.mpc.N, "yref", yref_e)

        status = self.mpc.solver.solve()

        if status != 0:
            u_opt = np.array([0.0, 0.0, 0.0, hover_T])
        else:
            u_opt = self.mpc.solver.get(0, "u")

        if self.env is not None and self.debug:
            try:
                mpc_points = []
                for k in range(self.mpc.N + 1):
                    x_k = self.mpc.solver.get(k, "x")
                    mpc_points.append(self._spatial_to_cartesian(x_k[0], x_k[1], x_k[2]))
                draw_line(
                    self.env, points=np.array(mpc_points), rgba=np.array([0.0, 0.0, 1.0, 0.8])
                )
                draw_line(
                    self.env, points=np.array(vis_dynamic_left), rgba=np.array([1.0, 0.5, 0.0, 0.9])
                )
                draw_line(
                    self.env,
                    points=np.array(vis_dynamic_right),
                    rgba=np.array([1.0, 0.5, 0.0, 0.9]),
                )
            except Exception:
                pass

        self._log_control_step(x_spatial, u_opt, status)
        return np.array([u_opt[0], u_opt[1], u_opt[2], u_opt[3]])

    def _spatial_to_cartesian(self, s, w1, w2):
        f = self.geo.get_frame(s)
        return f["pos"] + w1 * f["n1"] + w2 * f["n2"]

    def _cartesian_to_spatial(self, pos, vel, rpy, drpy):
        s = self.geo.get_closest_s(pos, s_guess=self.prev_s)
        self.prev_s = s
        f = self.geo.get_frame(s)
        r_vec = pos - f["pos"]
        w1 = np.dot(r_vec, f["n1"])
        w2 = np.dot(r_vec, f["n2"])
        h = max(1 - f["k1"] * w1 - f["k2"] * w2, 0.01)
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

    def _log_control_step(self, x_spatial, u_opt, status):
        self.step_count += 1
        elapsed = (datetime.now() - self.episode_start_time).total_seconds()
        self.control_log["timestamps"].append(elapsed)
        self.control_log["phi_c"].append(float(u_opt[0]))
        self.control_log["theta_c"].append(float(u_opt[1]))
        self.control_log["psi_c"].append(float(u_opt[2]))
        self.control_log["thrust_c"].append(float(u_opt[3]))
        self.control_log["solver_status"].append(int(status))
        self.control_log["s"].append(float(x_spatial[0]))
        self.control_log["w1"].append(float(x_spatial[1]))
        self.control_log["w2"].append(float(x_spatial[2]))
        self.control_log["ds"].append(float(x_spatial[3]))

    def save_control_log(self, filepath=None):
        if filepath is None:
            filepath = f"control_log_{self.episode_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, "w") as f:
            json.dump(self.control_log, f, indent=2)
        return filepath

    def plot_control_values(self, figsize=(16, 10), save_path=None):
        if len(self.control_log["timestamps"]) == 0:
            return
        t = np.array(self.control_log["timestamps"])
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle("MPC Control Values", fontsize=16)
        axes[0, 0].plot(t, self.control_log["phi_c"], "b")
        axes[0, 0].set_ylabel("Roll")
        axes[0, 1].plot(t, self.control_log["theta_c"], "g")
        axes[0, 1].set_ylabel("Pitch")
        axes[1, 0].plot(t, self.control_log["thrust_c"], "r")
        axes[1, 0].set_ylabel("Thrust")
        axes[1, 1].plot(t, self.control_log["psi_c"], "m")
        axes[1, 1].set_ylabel("Yaw")
        axes[2, 0].plot(t, self.control_log["s"], "c")
        axes[2, 0].set_ylabel("s")
        axes[2, 1].plot(t, self.control_log["w1"], "orange", label="w1")
        axes[2, 1].plot(t, self.control_log["w2"], "purple", label="w2")
        axes[2, 1].axhline(y=self.W1_MAX, c="r", ls="--")
        axes[2, 1].axhline(y=-self.W1_MAX, c="r", ls="--")
        axes[2, 1].legend()
        plt.tight_layout()
        if save_path is None:
            save_path = f"control_plot_{self.episode_start_time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path)
        plt.close()

    def plot_solver_status(self, save_path=None):
        if len(self.control_log["timestamps"]) == 0:
            return
        t = np.array(self.control_log["timestamps"])
        status = np.array(self.control_log["solver_status"])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(t, status, c=["g" if s == 0 else "r" for s in status])
        ax.set_title("Solver Status")
        if save_path is None:
            save_path = "solver_status.png"
        plt.savefig(save_path)
        plt.close()

    def plot_all_diagnostics(self, save_dir=None):
        if save_dir is None:
            save_dir = (
                f"mpc_debug/mpc_diagnostics_{self.episode_start_time.strftime('%Y%m%d_%H%M%S')}"
            )
        os.makedirs(save_dir, exist_ok=True)
        self.save_control_log(os.path.join(save_dir, "control_log.json"))
        self.plot_control_values(save_path=os.path.join(save_dir, "control_values.png"))
        self.plot_solver_status(save_path=os.path.join(save_dir, "solver_status.png"))
        return save_dir
