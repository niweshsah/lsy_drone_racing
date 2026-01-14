import os
import shutil
from typing import Any, Dict, Tuple

import casadi as ca
import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

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
    ddrpy = c_rpy * rpy + c_drpy * drpy + c_cmd * cmd_rpy  # This is very important!

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
# 3. GEOMETRY ENGINE (UPDATED WITH 2D PROJECTION LOGIC)
# ==============================================================================

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

        # Cost Matrices
        # s (1), w1 (2), w2 (3), ds (4), dw1 (5), dw2 (6), phi (7), theta (8), psi (9), dphi (10), dtheta (11), dpsi (12)
        q_diag = np.array([1.0, 10.0, 10.0, 10.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        #
        r_diag = np.array([5.0, 5.0, 5.0, 0.1])

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
