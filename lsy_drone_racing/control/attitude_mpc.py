"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

# This creates an acados model from the symbolic drone model.
def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model

# This function creates the acados OCP and solver.
# This is the main function to modify to change the MPC formulation.
def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows() # number of states
    nu = ocp.model.u.rows() # Number of inputs (4: cmd_rpy, thrust)
    ny = nx + nu  # Size of cost vector for intermediate steps
    ny_e = nx  # Size of cost vector for the final step (no input)

    # Set dimensions
    ocp.solver_options.N_horizon = N # number of look-ahead steps

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    # We use a linear least squares cost function
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    # This defines how "bad" it is to deviate from the trajectory.
    
    Q = np.diag([
    50.0, 50.0, 400.0,  # Position (x, y, z)
    1.0, 1.0, 1.0,      # Orientation (roll, pitch, yaw)
    10.0, 10.0, 10.0,   # Velocity (vx, vy, vz)
    5.0, 5.0, 5.0       # Angular Rates (p, q, r)
])
    
    # Input weights (reference is upright orientation and hover thrust)
    # this defines how "bad" it is to use control inputs.
    R = np.diag([
    1.0, 1.0, 1.0,  # Cmd Orientation
    50.0            # Cmd Thrust
])

    Q_e = Q.copy()
    # combines state and input weights into single weight matrices
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    # Stage cost for k = 0 to N-1 follows this structure:

    # The cost is:

    # 1/2 * ( Vx * x_k + Vu * u_k - y_ref_k )^T * W * ( Vx * x_k + Vu * u_k - y_ref_k )

    # Terminal cost at k = N:

    # 1/2 * ( Vx_e * x_N - y_ref_N )^T * W_e * ( Vx_e * x_N - y_ref_N )

    # Meaning of the matrices:

    # Vx selects which elements of the STATE vector appear inside the cost.

    # Vu selects which elements of the CONTROL INPUT vector appear in the cost.

    # Vx_e selects which states are penalized at the terminal step (only x_N, no inputs).

    # In acados, the cost function does not automatically assume that you want to penalize x and u.
    # Instead, you build an “output vector”:

    Vx = np.zeros((ny, nx))
    # Set the state part to identity matrix
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx


    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e


    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    # Initially hover at origin with zero angles and zero velocities
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))


    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])


    # We have to set x0 even though we will overwrite it later on.
    # xo is the initial state at time step 0.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6


    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    # This triggers acados to write C code based on the Python definitions, compile it using gcc, and load it back into Python.
    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Same waypoints as in the trajectory controller. Determined by trial and error.
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
        self._t_total = 15  # s
        
        # create an evenly spaced time array for the waypoints
        t = np.linspace(0, self._t_total, len(waypoints))
        
        # create cubic spline interpolations for position and velocity
        self._des_pos_spline = CubicSpline(t, waypoints)
        
        # derivative of position spline gives velocity spline
        self._des_vel_spline = self._des_pos_spline.derivative()
        
        # This creates a time series of waypoints for the entire trajectory
        self._waypoints_pos = self._des_pos_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        
        # same as above for velocity
        self._waypoints_vel = self._des_vel_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        
        # set desired yaw to zero for all waypoints
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

        # Load drone parameters
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        
        # Create OCP solver
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        
        # set model dimension size for run-time use
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        
        # max tick
        # We subtract N to ensure we have enough waypoints to fill the horizon
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        
        self._config = config
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust [r_des, p_des, y_des, t_des] as a numpy array.
        """
        # initialize index for waypoints
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Setting initial state
        # create rpy and drpy from quat and ang_vel as they are not provided in obs
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        
        # concatenate position, orientation, velocity, and angular velocity into state vector
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        
        # Set the stage-0 state bounds to x0
        self._acados_ocp_solver.set(0, "lbx", x0) # stage, bound type, value
        self._acados_ocp_solver.set(0, "ubx", x0)


        # Setting state reference
        # initialize yref with zeros
        
        # y_ref is a (N x ny) matrix where each row is the reference for that time step
        yref = np.zeros((self._N, self._ny))
        
        # Setting intermediate state references
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]  # position
        
        
        # zero roll, pitch
        # no roll or pitch tracking
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]  # yaw
        
        # velocity tracking
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]  # velocity

        # set thrust reference to hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        
        # Note: roll, pitch and yaw are all 0 always
        
        # Apply references to the solver from stage 0 to N-1
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])


        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]  # position
        # zero roll, pitch
        yref_e[5] = self._waypoints_yaw[i + self._N]  # yaw
        yref_e[6:9] = self._waypoints_vel[i + self._N]  # velocity

        # set thrust reference to hover thrust
        self._acados_ocp_solver.set(self._N, "yref_e", yref_e)

        
        # Solving problem and getting first input
        # Run the solver to minimize the OCP given current x0, yref for all stages, and constraints. The solver will compute the optimal sequence of inputs and states across the horizon.
        
        self._acados_ocp_solver.solve()
        
        # returns the optimal control input at stage 0
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
