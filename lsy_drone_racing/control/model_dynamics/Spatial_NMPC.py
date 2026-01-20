import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import casadi as ca
import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Optional/Simulation Specific Imports ---
try:
    from drone_models.core import load_params
    from drone_models.utils.rotation import ang_vel2rpy_rates

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.control.utils.yaml_import import load_yaml
    from lsy_drone_racing.utils.utils import draw_line
except ImportError:
    logger.warning("Simulation specific modules not found. Using mocks/defaults.")

    def load_yaml(path: str) -> Dict[str, Any]:
        """Mock yaml loader."""
        return {}

    def load_params(*args) -> Optional[Dict[str, Any]]:
        """Mock param loader."""
        return None

    def ang_vel2rpy_rates(q, w):
        return np.zeros(3)

    def draw_line(*args, **kwargs):
        pass

    class Controller:
        pass


@dataclass
class MPCConfig:
    """Configuration data class for MPC weights and constraints."""
    
    # Cost Weights
    state_weights: Dict[str, float] = field(default_factory=lambda: {
        "s": 1.0, "w1": 10.0, "w2": 10.0,
        "ds": 10.0, "dw1": 5.0, "dw2": 5.0,
        "phi": 1.0, "theta": 1.0, "psi": 1.0,
        "dphi": 0.1, "dtheta": 0.1, "dpsi": 0.1
    })
    
    control_weights: Dict[str, float] = field(default_factory=lambda: {
        "roll_cmd": 5.0, "pitch_cmd": 5.0, 
        "yaw_cmd": 5.0, "thrust_cmd": 0.1
    })
    
    # Constraints
    bound_cost: float = 1000.0
    w_bound: float = 0.4  # Lateral/Vertical track width limit
    angle_bound: float = 0.5  # Max roll/pitch/yaw (rad)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MPCConfig":
        """Factory method to load config from YAML if available."""
        try:
            data = load_yaml(yaml_path)
            mpc_params = data.get('mpc_wts', {})
            return cls(
                state_weights=mpc_params.get('state_wts', {}),
                control_weights=mpc_params.get('control_wts', {}),
                bound_cost=mpc_params.get('mpc_bound_cost', 1000.0)
            )
        except Exception as e:
            logger.error(f"Failed to load YAML configuration: {e}")
            return cls()


def get_drone_params() -> Dict[str, Any]:
    """
    Retrieves the physical parameters of the drone.
    
    Returns:
        Dict[str, Any]: Dictionary containing mass, gravity, inertia, and aerodynamic coefficients.
    """
    params = load_params("so_rpy", "cf21B_500")
    if params is not None:
        return params

    # Fallback default parameters (Crazyflie 2.1 approximation)
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


def symbolic_dynamics_spatial(
    mass: float,
    gravity_vec: np.ndarray,
    acc_coef: float,
    cmd_f_coef: float,
    rpy_coef: List[float],
    rpy_rates_coef: List[float],
    cmd_rpy_coef: List[float],
    **kwargs
) -> Tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    """
    Constructs the spatial dynamics model (Frenet-Serret frame) using CasADi.

    The state is represented in path coordinates:
    s: Path progress
    w1: Lateral deviation (horizontal)
    w2: Vertical deviation
    ds, dw1, dw2: Derivatives w.r.t time

    Args:
        mass: Drone mass (kg).
        gravity_vec: Gravity vector in world frame.
        acc_coef: Acceleration coefficient.
        cmd_f_coef: Thrust command coefficient.
        rpy_coef: Aerodynamic coefficients for stiffness.
        rpy_rates_coef: Aerodynamic coefficients for damping.
        cmd_rpy_coef: Aerodynamic coefficients for input gain.

    Returns:
        Tuple[X_Dot, X, U, P]: CasADi expressions for dynamics, state, input, and parameters.
    """
    # --- 1. State Vector X (12 States) ---
    s, w1, w2 = ca.SX.sym("s"), ca.SX.sym("w1"), ca.SX.sym("w2")
    ds, dw1, dw2 = ca.SX.sym("ds"), ca.SX.sym("dw1"), ca.SX.sym("dw2")
    
    # Euler angles (Roll, Pitch, Yaw) and rates
    phi, theta, psi = ca.SX.sym("phi"), ca.SX.sym("theta"), ca.SX.sym("psi")
    dphi, dtheta, dpsi = ca.SX.sym("dphi"), ca.SX.sym("dtheta"), ca.SX.sym("dpsi")

    rpy = ca.vertcat(phi, theta, psi)
    drpy = ca.vertcat(dphi, dtheta, dpsi)

    X = ca.vertcat(s, w1, w2, ds, dw1, dw2, rpy, drpy)

    # --- 2. Control Input U (4 Inputs) ---
    # Commands: Roll, Pitch, Yaw, Thrust
    phi_c, theta_c, psi_c, T_c = ca.SX.sym("phi_c"), ca.SX.sym("theta_c"), ca.SX.sym("psi_c"), ca.SX.sym("T_c")
    cmd_rpy = ca.vertcat(phi_c, theta_c, psi_c)
    U = ca.vertcat(cmd_rpy, T_c)

    # --- 3. Parameters P (13 Elements) ---
    # Path reference parameters: tangent (t), normal (n1), binormal (n2), curvature (k)
    t_vec = ca.SX.sym("t_vec", 3)
    n1_vec = ca.SX.sym("n1_vec", 3)
    n2_vec = ca.SX.sym("n2_vec", 3)
    k1, k2 = ca.SX.sym("k1"), ca.SX.sym("k2") # Curvature components
    dk1, dk2 = ca.SX.sym("dk1"), ca.SX.sym("dk2") # Spatial derivatives of curvature

    P = ca.vertcat(t_vec, n1_vec, n2_vec, k1, k2, dk1, dk2)

    # --- 4. Physics Engine ---

    # A. Rotational Dynamics (Fitted First-Order Response Model)
    # Dynamics: alpha = c1 * angle + c2 * rate + c3 * command
    c_rpy = ca.DM(rpy_coef)
    c_drpy = ca.DM(rpy_rates_coef)
    c_cmd = ca.DM(cmd_rpy_coef)

    ddrpy = c_rpy * rpy + c_drpy * drpy + c_cmd * cmd_rpy

    # B. Translational Acceleration (Inertial Frame)
    thrust_mag = acc_coef + cmd_f_coef * T_c
    F_body = ca.vertcat(0, 0, thrust_mag)

    # Rotation Matrix (Body -> Inertial/World) R_IB
    cx, cy, cz = ca.cos(phi), ca.cos(theta), ca.cos(psi)
    sx, sy, sz = ca.sin(phi), ca.sin(theta), ca.sin(psi)

    R_IB = ca.vertcat(
        ca.horzcat(cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy),
        ca.horzcat(cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
        ca.horzcat(-sy, cy * sx, cx * cy),
    )

    # Global Acceleration: a = g + (R * F_thrust) / m
    g_vec_sym = ca.DM(gravity_vec)
    acc_world = g_vec_sym + (R_IB @ F_body) / mass

    # C. Spatial Dynamics Projection
    # Project the world acceleration onto the Frenet-Serret frame (path frame).
    
    # h: Scaling factor for path curvature (Metric of the curvilinear coordinates)
    h = 1 - k1 * w1 - k2 * w2

    # h_dot (Time derivative via Chain rule: d/dt = d/ds * ds/dt)
    h_dot = -(k1 * dw1 + k2 * dw2 + (dk1 * w1 + dk2 * w2) * ds)

    # Coriolis and Centrifugal terms arising from the moving reference frame
    coriolis = (
        (ds * h_dot) * t_vec
        + (ds**2 * h * k1) * n1_vec
        + (ds**2 * h * k2) * n2_vec
        - (ds * dw1 * k1) * t_vec
        - (ds * dw2 * k2) * t_vec
    )

    # Project World Acceleration onto Path Basis Vectors
    # dds (acceleration along track) = (a_world dot t_vec) / h
    proj_t = ca.dot(t_vec, acc_world - coriolis)
    dds = proj_t / h
    
    # Transverse accelerations
    ddw1 = ca.dot(n1_vec, acc_world - coriolis)
    ddw2 = ca.dot(n2_vec, acc_world - coriolis)

    # --- 5. Final State Derivative ---
    X_Dot = ca.vertcat(
        ds, dw1, dw2,      # Velocities
        dds, ddw1, ddw2,   # Accelerations
        drpy,              # Angular Velocities
        ddrpy,             # Angular Accelerations
    )

    return X_Dot, X, U, P


def export_model(params: Dict[str, Any]) -> AcadosModel:
    """Wrapper to export the symbolic CasADi model to Acados."""
    X_dot, X, U, P = symbolic_dynamics_spatial(**params)
    
    model = AcadosModel()
    model.name = "spatial_mpc_drone"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    model.p = P
    return model


class SpatialMPC:
    """
    Nonlinear Model Predictive Control (NMPC) solver using Acados.
    
    This controller operates in the spatial domain (Frenet Frame), optimizing
    progress 's' along a reference path while minimizing deviation 'w1', 'w2'.
    """

    def __init__(self, drone_params: Dict[str, Any], N: int = 50, Tf: float = 1.0, 
                 config: Optional[MPCConfig] = None):
        """
        Initialize the Spatial MPC Solver.

        Args:
            drone_params: Dictionary of drone physical parameters.
            N: Prediction horizon steps.
            Tf: Prediction horizon time (seconds).
            config: MPCConfig object containing weights and constraints.
        """
        self.N = N
        self.Tf = Tf
        self.params = drone_params
        self.config = config or MPCConfig()

        # Ensure gravity Z component is set for consistency
        self.params["g"] = self.params["gravity_vec"][2]
        
        # Clean previous build artifacts
        self._clean_workspace()
        
        # Build solver
        self.solver = self._build_solver()

    def _clean_workspace(self):
        """Removes generated C code to ensure a fresh build."""
        target_dir = "c_generated_code"
        if os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
            except OSError as e:
                logger.warning(f"Could not remove {target_dir}: {e}")

    def _build_solver(self) -> AcadosOcpSolver:
        """Configures and builds the Acados OCP solver."""
        model = export_model(self.params)
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf

        nx, nu = model.x.size()[0], model.u.size()[0]
        
        # --- Cost Function Setup ---
        self._setup_cost(ocp, nx, nu)

        # --- Constraint Setup ---
        self._setup_constraints(ocp, nx, nu)

        # --- Solver Options ---
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver_cond_N = self.N
        
        # Initialize parameters
        ocp.parameter_values = np.zeros(13)

        return AcadosOcpSolver(ocp, json_file="acados_spatial.json")

    def _setup_cost(self, ocp: AcadosOcp, nx: int, nu: int):
        """Defines the Least Squares cost matrices (W, W_e)."""
        # Unpack weights from config
        sw = self.config.state_weights
        cw = self.config.control_weights

        # State cost vector (diagonal)
        # Order: s, w1, w2, ds, dw1, dw2, phi, theta, psi, dphi, dtheta, dpsi
        q_diag = np.array([
            sw.get('s', 1.0), sw.get('w1', 10.0), sw.get('w2', 10.0),
            sw.get('ds', 10.0), sw.get('dw1', 5.0), sw.get('dw2', 5.0),
            sw.get('phi', 1.0), sw.get('theta', 1.0), sw.get('psi', 1.0),
            sw.get('dphi', 0.1), sw.get('dtheta', 0.1), sw.get('dpsi', 0.1)
        ])

        # Control cost vector (diagonal)
        # Order: Roll_cmd, Pitch_cmd, Yaw_cmd, Thrust_cmd
        r_diag = np.array([
            cw.get('roll_cmd', 5.0), cw.get('pitch_cmd', 5.0),
            cw.get('yaw_cmd', 5.0), cw.get('thrust_cmd', 0.1)
        ])

        # Intermediate cost matrix
        ocp.cost.W = scipy.linalg.block_diag(np.diag(q_diag), np.diag(r_diag))
        
        # Terminal cost matrix (States only)
        ocp.cost.W_e = np.diag(q_diag)

        # Map symbolic state/input to cost function
        ny = nx + nu
        ny_e = nx
        
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)
        
        ocp.cost.Vx_e = np.eye(nx)
        
        # References (set to zero, can be updated at runtime)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

    def _setup_constraints(self, ocp: AcadosOcp, nx: int, nu: int):
        """Sets up physical limitations and safety corridors."""
        
        # 1. Hard Control Bounds (Inputs)
        # [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, self.params["thrust_min"] * 4])
        ocp.constraints.ubu = np.array([+0.5, +0.5, +0.5, self.params["thrust_max"] * 4])

        # 2. Soft State Bounds
        # Constrain lateral error (w1, w2) and Euler angles (phi, theta, psi)
        # Indices in X: w1=1, w2=2, phi=6, theta=7, psi=8
        wb = self.config.w_bound
        ab = self.config.angle_bound
        
        bounds_idx = np.array([1, 2, 6, 7, 8])
        lower_bounds = np.array([-wb, -wb, -ab, -ab, -ab])
        upper_bounds = np.array([+wb, +wb, +ab, +ab, +ab])

        # Apply to path and terminal
        ocp.constraints.idxbx = bounds_idx
        ocp.constraints.lbx = lower_bounds
        ocp.constraints.ubx = upper_bounds

        ocp.constraints.idxbx_e = bounds_idx
        ocp.constraints.lbx_e = lower_bounds
        ocp.constraints.ubx_e = upper_bounds

        # 3. Slack Variables for Soft Constraints (z)
        # Only applying slack to w1 and w2 (indices 0 and 1 in the subset idxbx) to prevent infeasibility
        ns = 2 
        ocp.constraints.idxsbx = np.array([0, 1]) 
        
        # Slack costs
        z_cost = self.config.bound_cost * np.ones(ns)
        ocp.cost.zl = z_cost
        ocp.cost.zu = z_cost
        ocp.cost.Zl = z_cost
        ocp.cost.Zu = z_cost
        
        # Initial State
        ocp.constraints.x0 = np.zeros(nx)


# --- Example Usage ---
if __name__ == "__main__":
    # Load configuration
    config_path = "lsy_drone_racing/control/constants.yaml"
    mpc_config = MPCConfig.from_yaml(config_path)
    
    # Load physical parameters
    drone_params = get_drone_params()
    
    # Initialize Solver
    logger.info("Building Acados Solver...")
    mpc = SpatialMPC(drone_params, N=50, Tf=1.0, config=mpc_config)
    logger.info("Solver built successfully.")