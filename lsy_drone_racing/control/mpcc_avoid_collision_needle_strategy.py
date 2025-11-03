"""MPCC based strategy
handle obstacles with integrated costs
but handle gates with transfered contour error calculation
when approaching gates, compute contour error according to a the gate normal vector (like a needle across the gate center)
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING, List, Dict, Callable, Optional, Union, Set
import os
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, DM, cos, sin, vertcat, dot, norm_2, floor, if_else, exp, Function, power
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from casadi import interpolant

from lsy_drone_racing.control.fresssack_controller import FresssackController
from lsy_drone_racing.control.fresssack_controller import MultiArrayTx, MarkerArrayTx, CapsuleMarkerTx, MeshMarkerTx, PathTx, TFTx

from lsy_drone_racing.control import Controller
from lsy_drone_racing.tools.ext_tools import TrajectoryTool, TransformTool
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.publisher import Publisher
    from rclpy.publisher import MsgType
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Path
    from tf2_ros import TransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray

    from transformations import quaternion_from_euler, euler_from_quaternion
    ROS_AVAILABLE = True
except:
    ROS_AVAILABLE = False
# ROS_AVAILABLE = False



class MPCC(FresssackController):
    """Implementation of MPCC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, env=None):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config, ros_tx_freq = 10)
        self.freq = config.env.freq
        self._tick = 0

        self.env = env
        self.init_gates(obs=obs,
                        gate_outer_size=[0.8, 0.8, 0.8, 0.8],   # capsule built in between inner & outer
                        thickness=[0.3, 0.3, 0.3, 0.3],     # thickness of capsule, gaussian cost
                        # thickness=[0.2, 0.2, 0.2, 0.2],     # thickness of capsule, gaussian cost
                        )

        self.init_obstacles(obs=obs, 
                            obs_safe_radius=[0.3, 0.3, 0.3, 0.3])
        
        # region Trajectory
        # # Demo waypoints
        # waypoints = np.array(
        #     [
        #         [1.0, 1.5, 0.05],
        #         [0.8, 1.0, 0.2],
        #         [0.55, -0.3, 0.5],
        #         [0.2, -1.3, 0.65],
        #         [1.1, -0.85, 1.1],
        #         [0.2, 0.5, 0.65],
        #         [0.0, 1.2, 0.525],
        #         [0.0, 1.2, 1.1],
        #         [-0.5, 0.0, 1.1],
        #         [-0.5, -0.5, 1.1],
        #     ]
        # )
        # trajectory = self.trajectory_generate(self.t_total, waypoints)
        # pre-planned trajectory
        # t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets.csv")
        t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/test_run_third_gate_modified_lots_of_handcraft.csv")
        trajectory = CubicSpline(t, pos)
        # # easy controller trajectory
        # gates_rotates = R.from_quat(obs['gates_quat'])
        # rot_matrices = np.array(gates_rotates.as_matrix())
        # self.gates_norm = np.array(rot_matrices[:,:,1])
        # self.gates_pos = obs['gates_pos']
        # # replan trajectory
        # waypoints = self.calc_waypoints(self.init_pos, self.gates_pos, self.gates_norm)
        # # t, waypoints = self.avoid_collision(waypoints, obs['obstacles_pos'], 0.3)
        # trajectory = self.trajectory_generate(self.t_total, waypoints)

        # region Global params
        self.N = 40
        self.T_HORIZON = 0.6
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05 # the segment interval for trajectory to be input to the model
        self.model_traj_length = 12 # maximum trajectory length the param can take
        self.model_traj_N = int(self.model_traj_length/self.model_arc_length)
        self.num_ostacles = len(obs['obstacles_pos'])  # elements: 4 * (pillar: 1);    each capsule: 7
        self.capsule_list = self._gen_pillar_capsule() # for now only for visualization
        self.num_gates = len(obs['gates_pos'])
        # self.gates_norm_list = [gate.norm_vec for gate in self.gates] # 4 * 3 = 12

        # trajectory reparameterization
        self.traj_tool = TrajectoryTool()
        trajectory = self.traj_tool.extend_trajectory(trajectory)
        self.arc_trajectory = self.traj_tool.arclength_reparameterize(trajectory)

        # build model & create solver
        self.acados_ocp_solver, self.ocp = self.create_ocp_solver(self.T_HORIZON, self.N, self.arc_trajectory)

        # initialize ros tx
        self.init_ros_tx()
        
        # initialize debug function for MPC
        cost_dict = self.mpcc_cost_components()
        self.cost_func_debug = Function(
            'mpcc_cost_debug',
            [self.x, self.u, self.p],
            [cost_dict[k] for k in cost_dict],
            ['x', 'u', 'p'],
            list(cost_dict.keys()
            )  # ['total', 'cost_l', ..., 'miu_cost']    
        )


        # initialize
        self.last_theta = 0.0
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False

    def init_ros_tx(self):
        self.mpc_costs_tx_keys = ['total',
                        'cost_l',
                        'C_l',
                        'e_l_cost',
                        'cost_c',
                        'C_c',
                        'e_c_cost',
                        'ang_cost',
                        'ctrl_cost',
                        'cost_obs',
                        'miu_cost',
                        'cost_g_c']
        
        if ROS_AVAILABLE and self.ros_tx:
            self.mpcc_traj_tx = PathTx(
                node_name = 'mpcc_path_tx',
                topic_name = 'mpcc_traj',
                queue_size = 10
            )
            self.path_tx = PathTx(
                node_name = 'global_path_tx',
                topic_name = 'global_path',
                queue_size = 10
            )
            self.gate_tf_txs = [
                TFTx(node_name = 'gate_' + str(i) + '_TF_tx', topic_name = 'gate_' + str(i))
                for i in range(len(self.gates))
            ]
            
            self.obstacle_tf_txs = [
                TFTx(node_name = 'obstacle_' + str(i) + '_TF_tx', topic_name = 'obstacle_' + str(i))
                for i in range(len(self.obstacles))
            ]
            current_dir = os.path.dirname(os.path.abspath(__file__))

            gate_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'gate.dae')
            gate_unobserved_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'gate_not_sure.dae')
            gate_passed_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'gate_passed.dae')
            self.gate_marker_txs = [
                    MeshMarkerTx(
                    node_name = 'gate_' + str(i) + '_marker_tx',
                    topic_name = 'gate_' + str(i) + '_marker',
                    mesh_path = [os.path.abspath(gate_unobserved_mesh_file),
                                  os.path.abspath(gate_mesh_file),
                                  os.path.abspath(gate_passed_mesh_file)],
                    frame_id = 'gate_' + str(i),
                    queue_size = 1
                )
                for i in range(len(self.gates))
            ]

            self.gate_marker_array_tx = MarkerArrayTx(
                node_name = 'gate_marker_array_tx',
                topic_name = 'gate_marker_array',
                queue_size = 5
            )
            
            obstacle_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'obstacle.dae')
            obstacle_unobserved_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'obstacle_not_sure.dae')
            self.obstacle_marker_txs = [
                    MeshMarkerTx(
                    node_name = 'obstacle_' + str(i) + '_marker_tx',
                    topic_name = 'obstacle_' + str(i) + '_marker',
                    mesh_path = [os.path.abspath(obstacle_unobserved_mesh_file),
                                 os.path.abspath(obstacle_mesh_file)],
                    frame_id = 'obstacle_' + str(i),
                    queue_size = 1
                )
                for i in range(len(self.obstacles))
            ]
            self.obstacle_marker_array_tx = MarkerArrayTx(
                node_name = 'obstacle_marker_array_tx',
                topic_name = 'obstacle_marker_array',
                queue_size = 5
            )

            self.collision_capsule_tx =  CapsuleMarkerTx(
                    node_name = 'collision_capsule' + '_tx',
                    topic_name = 'collision_capsule',
                    frame_id = 'map',
                    queue_size = 20
                )            

            self.capsule_marker_array_tx = MarkerArrayTx(
                node_name = 'capsule_marker_array_tx',
                topic_name = 'capsule_marker_array',
                queue_size = 5,
            )
            
            self.mpc_costs_tx = MultiArrayTx(
                node_name = 'cost_debug_tx', 
                queue_size = 10,
                keys = self.mpc_costs_tx_keys,
                topic_prefix = 'mpcc_cost'
            )
            
    # region Model
    def export_quadrotor_ode_model(self) -> AcadosModel:
        """Symbolic Quadrotor Model."""
        # Define name of solver to be used in script
        model_name = "lsy_example_mpc"

        # Define Gravitational Acceleration
        GRAVITY = 9.806

        # Sys ID Params
        params_pitch_rate = [-6.003842038081178, 6.213752925707588]
        params_roll_rate = [-3.960889336015948, 4.078293254657104]
        params_yaw_rate = [-0.005347588299390372, 0.0]
        params_acc = [20.907574256269616, 3.653687545690674]

        """Model setting"""
        # define basic variables in state and input vector
        self.px = MX.sym("px")  # 0
        self.py = MX.sym("py")  # 1
        self.pz = MX.sym("pz")  # 2
        self.vx = MX.sym("vx")  # 3
        self.vy = MX.sym("vy")  # 4
        self.vz = MX.sym("vz")  # 5
        self.roll = MX.sym("r")  # 6
        self.pitch = MX.sym("p")  # 7
        self.yaw = MX.sym("y")  # 8
        self.f_collective = MX.sym("f_collective")

        self.f_collective_cmd = MX.sym("f_collective_cmd")
        self.r_cmd = MX.sym("r_cmd")
        self.p_cmd = MX.sym("p_cmd")
        self.y_cmd = MX.sym("y_cmd")

        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")

        # expanded observation state
        self.theta = MX.sym("theta")
        # self.v_theta = MX.sym("v_theta")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        # define state and input vector
        states = vertcat(
            self.px,
            self.py,
            self.pz,
            self.vx,
            self.vy,
            self.vz,
            self.roll,
            self.pitch,
            self.yaw,
            self.f_collective,
            self.f_collective_cmd,
            self.r_cmd,
            self.p_cmd,
            self.y_cmd,
            self.theta
        )
        inputs = vertcat(
            self.df_cmd, 
            self.dr_cmd, 
            self.dp_cmd, 
            self.dy_cmd, 
            self.v_theta_cmd
        )

        # Define nonlinear system dynamics
        f = vertcat(
            self.vx,
            self.vy,
            self.vz,
            (params_acc[0] * self.f_collective + params_acc[1])
            * (cos(self.roll) * sin(self.pitch) * cos(self.yaw) + sin(self.roll) * sin(self.yaw)),
            (params_acc[0] * self.f_collective + params_acc[1])
            * (cos(self.roll) * sin(self.pitch) * sin(self.yaw) - sin(self.roll) * cos(self.yaw)),
            (params_acc[0] * self.f_collective + params_acc[1]) * cos(self.roll) * cos(self.pitch) - GRAVITY,
            params_roll_rate[0] * self.roll + params_roll_rate[1] * self.r_cmd,
            params_pitch_rate[0] * self.pitch + params_pitch_rate[1] * self.p_cmd,
            params_yaw_rate[0] * self.yaw + params_yaw_rate[1] * self.y_cmd,
            10.0 * (self.f_collective_cmd - self.f_collective),
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        # define dynamic trajectory input & obstacle parameters input
        self.pd_list = MX.sym("pd_list", 3*self.model_traj_N)
        self.tp_list = MX.sym("tp_list", 3*self.model_traj_N)
        self.qc_dyn = MX.sym("qc_dyn", 4*self.model_traj_N)
        self.obst_list = MX.sym("obst_list", self.num_ostacles * 7) # 4 * 7 = 28
        self.gates_param_list = MX.sym("gates_param_list", self.num_gates * (3+3)) # 4 * 6 = 24
        params = vertcat(
            self.pd_list, 
            self.tp_list,
            self.qc_dyn,
            self.obst_list,
            self.gates_param_list
        )

        # For ease of use, contatenate x, u, and p
        self.x = states
        self.u = inputs
        self.p = params

        # Initialize the nonlinear model for NMPC formulation
        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f
        model.f_impl_expr = None
        model.x = states
        model.u = inputs
        model.p = params

        return model
    
    # region Capsule Dist
    def calc_obst_distance(self, pos, a, b):
        """calculate distances of pos to every obstacles with casadi
        Args:
            pos: CasADi 3x1 
            a, b: define capsule center segment
        Returns:
            distance to closest point and vector from closest point to pos
        """
        ab = b - a
        ab_dot = dot(ab, ab)
        ap = pos - a
        t = dot(ap, ab) / ab_dot
        t_clamped = if_else(t < 0.0, 0.0, if_else(t > 1.0, 1.0, t))
        closest = a + t_clamped * ab
        vec = pos - closest
        dist = norm_2(vec)
        return dist, vec
    
    # region Casadi Interp
    def casadi_linear_interp(self, theta, theta_list, p_flat, dim=3):
        """Manually interpolate a 3D path using CasADi symbolic expressions.
        
        Args:
            theta: CasADi symbol, scalar progress variable (0 ~ model_traj_length)
            theta_list: list or array, thetas of path points [0, 0.1, 0.2, ...]
            p_flat: CasADi symbol, 1D flattened path points [x0,y0,z0, x1,y1,z1, ...]
            dim: int, dimension of a single point (default=3)
        Returns:
            p_interp: CasADi 3x1 vector, interpolated path point at given theta
        """
        M = len(theta_list)
        
        # Find index interval
        # Normalize theta to index scale
        idx_float = (theta - theta_list[0]) / (theta_list[-1] - theta_list[0]) * (M - 1)

        idx_lower = floor(idx_float)
        idx_upper = idx_lower + 1
        alpha = idx_float - idx_lower

        # Handle boundary cases (clamping)
        idx_lower = if_else(idx_lower < 0, 0, idx_lower)
        idx_upper = if_else(idx_upper >= M, M-1, idx_upper)

        # Gather points
        p_lower = vertcat(*[p_flat[dim * idx_lower + d] for d in range(dim)])
        p_upper = vertcat(*[p_flat[dim * idx_upper + d] for d in range(dim)])

        # Interpolated point
        p_interp = (1 - alpha) * p_lower + alpha * p_upper

        return p_interp
    
    # region Traj Param
    def get_updated_traj_param(self, trajectory: CubicSpline):
        """get updated trajectory parameters upon replaning"""
        # construct pd/tp lists from current trajectory
        theta_list = np.arange(0, self.model_traj_length, self.model_arc_length)
        pd_list = trajectory(theta_list)
        tp_list = trajectory.derivative(1)(theta_list)
        gate_theta_list, _ = self.traj_tool.find_gate_waypoint(self.arc_trajectory, [gate.pos for gate in self.gates])
        qc_dyn_list = np.zeros((self.num_gates, theta_list.shape[0]))
        for idx, gate in enumerate(self.gates):
            distances = np.linalg.norm(pd_list - gate.pos, axis=-1)
            qc_dyn_gate_front = np.exp(-distances**2 / (0.5*self.q_c_sigma1[idx])**2) # gaussian
            qc_dyn_gate_behind = np.exp(-distances**2 / (0.5*self.q_c_sigma2[idx])**2) # gaussian
            qc_dyn_list[idx] = qc_dyn_gate_front * (theta_list < gate_theta_list[idx]) + qc_dyn_gate_behind * (theta_list >= gate_theta_list[idx])
        p_vals = np.concatenate([pd_list.flatten(), tp_list.flatten(), qc_dyn_list.flatten()])
        return p_vals
    
    # region MPCC Cost
    def mpcc_cost_components(self):
        pos = vertcat(self.px, self.py, self.pz)
        ang = vertcat(self.roll, self.pitch, self.yaw)
        control_input = vertcat(self.f_collective_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)

        # interpolate spline dynamically
        theta_list = np.arange(0, self.model_traj_length, self.model_arc_length)
        pd_theta = self.casadi_linear_interp(self.theta, theta_list, self.pd_list)
        tp_theta = self.casadi_linear_interp(self.theta, theta_list, self.tp_list)
        qc_dyn_theta_list = [self.casadi_linear_interp(self.theta, theta_list, self.qc_dyn[i * self.model_traj_N: (i+1) * self.model_traj_N], dim=1) for i in range(self.num_gates)]
        qc_dyn_theta = 0
        for qc in qc_dyn_theta_list:
            qc_dyn_theta += qc
        tp_theta_norm = tp_theta / norm_2(tp_theta)
        e_theta = pos - pd_theta
        e_l = dot(tp_theta_norm, e_theta) * tp_theta_norm
        e_c = e_theta - e_l

        # cost for gates
        q_g_c = 0.0
        cost_g_c = 0.0
        for i in range(self.num_gates):
            idx = i * 6
            gate_pos = self.gates_param_list[idx:idx+3]
            gate_norm = self.gates_param_list[idx+3:idx+6]
            e_gate_theta = pos - gate_pos
            e_gate_l = dot(gate_norm, e_gate_theta) * gate_norm
            e_gate_c = e_gate_theta - e_gate_l
            e_gate_c_square = dot(e_gate_c, e_gate_c)
            cost_g_c += self.q_c_peak[i] * qc_dyn_theta_list[i] * e_gate_c_square
            q_g_c += self.q_c_peak[i] * qc_dyn_theta_list[i]

        # cost for obstacles
        q_c_supress = 0.0
        obst_cost = 0.0
        for i in range(self.num_ostacles):
            idx = i * 7
            a = self.obst_list[idx     : idx + 3] # extract params from model.p
            b = self.obst_list[idx + 3 : idx + 6]
            r = self.obst_list[idx + 6]
            dis, vec = self.calc_obst_distance(pd_theta, a, b) # EXP: use trajectory collision to supress q_c & miu
            # trick: to supress q_c & miu when running into obstacle extended surfaces
            q_c_supress += exp( -power(dis/(0.5*(self.d_extend+r)), 2) )
            # soft punish when getting into safe range: when , cost = gaussian(distance) if outside surface else 1.0
            dis, vec = self.calc_obst_distance(pos, a, b)
            obst_cost += exp( -power(dis/(0.5*r), 2) )
            # # punish velocity pointing towards center, weighted by dis | doesn't work
            # v_proj = dot(vel, vec)/norm_2(vec + 1e-6) # project to vec
            # v_proj = if_else(v_proj < 0, v_proj, 0.0) # if vel into obst
            # obst_vel_cost += if_else(diff < 0, v_proj**2 * diff/(r+self.d_safe), 0.0)

        # TODO: EXP: trick: make e_c smaller in direction of obstacle: e_c - factor * dot(obs_pos_vec, e_c)/norm(e_c) BUT: do this to closest obstacle or to all of them?
        q_c_factor = 1 - 0.9 * q_c_supress


        # Break down the costs
        C_l = self.q_l + self.q_l_peak * qc_dyn_theta
        e_l_cost = dot(e_l, e_l)
        cost_l = C_l * e_l_cost

        C_c = self.q_c - self.q_c * qc_dyn_theta
        e_c_cost = dot(e_c, e_c)
        cost_c = C_c * e_c_cost

        ang_cost = ang.T @ self.Q_w @ ang
        ctrl_cost = control_input.T @ self.R_df @ control_input

        
        cost_obs = self.obst_w * obst_cost

        
        miu_cost = q_c_factor * (-self.miu) * self.v_theta_cmd

        
        mpcc_cost = cost_l + cost_c + ang_cost + ctrl_cost + cost_obs + miu_cost + cost_g_c

        return {
            'total': mpcc_cost,
            'cost_l': cost_l,
            'C_l': C_l,
            'e_l_cost': e_l_cost,
            'cost_c': cost_c,
            'C_c': C_c + q_g_c,
            'e_c_cost': e_c_cost + cost_g_c,
            'ang_cost': ang_cost,
            'ctrl_cost': ctrl_cost,
            'cost_obs': cost_obs,
            'miu_cost': miu_cost,
            'cost_g_c': cost_g_c,
        }

    def mpcc_cost(self):
        """calculate mpcc cost function"""
        return self.mpcc_cost_components()['total']

    # region Solver create
    def create_ocp_solver(
        self, Tf: float, N: int, trajectory: CubicSpline,  verbose: bool = False
    ) -> tuple[AcadosOcpSolver, AcadosOcp]:
        """Creates an acados Optimal Control Problem and Solver."""
        ocp = AcadosOcp()

        # set model
        model = self.export_quadrotor_ode_model()
        ocp.model = model

        # Get Dimensions
        self.nx = model.x.rows()
        self.nu = model.u.rows()

        # Set dimensions
        ocp.solver_options.N_horizon = N

        ## Set Cost
        # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

        # Cost Type
        ocp.cost.cost_type = "EXTERNAL"

        # region MPCC Weights
        """DEFINITION of GLOBAL MPC WEIGHTS"""

        # MPCC Cost Weights
        self.q_l = 800
        self.q_l_peak = 8000
        self.q_c = 80
        self.q_c_peak = [800, 800, 800, 800]
        self.q_c_sigma1 = [0.5, 0.5, 0.5, 0.3]
        self.q_c_sigma2 = [0.2, 0.2, 0.5, 0.5]
        self.Q_w = 1 * DM(np.eye(3))
        self.R_df = DM(np.diag([1,0.4,0.4,0.4]))
        self.miu = 0.01
        # obstacle relavent
        self.obst_w = 50
        self.d_extend = 0.15 # extend distance to supress q_c
        # velocity bounds
        self.lb_vel = 0.7
        self.ub_vel = 1.9

        
        ocp.model.cost_expr_ext_cost = self.mpcc_cost()

        # Set State Constraints
        ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

        # Set Input Constraints
        ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, self.lb_vel]) # set a speed lower bound to provide it from stopping at obstacles
        ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, self.ub_vel])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        # We have to set x0 even though we will overwrite it later on.
        ocp.constraints.x0 = np.zeros((self.nx))
        # Set initial reference trajectory
        traj_param = self.get_updated_traj_param(self.arc_trajectory)
        # Set initial obstacle parameters
        obst_param = self.get_capsule_param(include_gate=False)
        # Set initial gate parameters
        gate_param = self.get_gate_param()
        # stuff parameters 
        p_vals = np.concatenate([traj_param, obst_param, gate_param])
        ocp.parameter_values = p_vals


        # Solver Options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI
        ocp.solver_options.tol = 1e-5

        ocp.solver_options.qp_solver_cond_N = N
        ocp.solver_options.qp_solver_warm_start = 1

        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50

        # set prediction horizon
        ocp.solver_options.tf = Tf

        acados_ocp_solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted.json", verbose=verbose)

        return acados_ocp_solver, ocp
    
    def ros_transmit(self, obs):
        if self.need_ros_tx(): 
            self.path_tx.publish(
                raw_data = {
                    'traj' : self.arc_trajectory(self.arc_trajectory.x),
                    'frame_id' : 'map'
                }
            )
            for idx, tx in enumerate(self.gate_tf_txs):
                tx.publish(
                    raw_data = 
                    {
                        'pos' : self.gates[idx].pos,
                        'quat' : self.gates[idx]._quat,
                        'frame_id' : 'map'
                    }
                )
            for idx, tx in enumerate(self.obstacle_tf_txs):
                tx.publish(
                    raw_data = 
                    {
                        'pos' : self.obstacles[idx].pos,
                        'quat' : [0,0,0,1],
                        'frame_id' : 'map'
                    }
                )
        if self.need_ros_tx(slow = True): 
            gate_marker_list : List[Marker] = []
            for idx, tx in enumerate(self.gate_marker_txs):
                marker : Marker = None
                if self.next_gate > idx:
                    marker = tx.process_data(
                        {
                            'idx' : 2
                        }
                    )
                elif self.gates_visited[idx]:
                    marker = tx.process_data(
                        {
                            'idx' : 1
                        }
                    )
                else:
                    marker = tx.process_data(
                        {
                            'idx' : 0
                        }
                    )
                gate_marker_list.append(marker)
            self.gate_marker_array_tx.publish(
                {
                    'markers' : gate_marker_list
                }
            )

            obstacle_marker_list = []
            for idx, tx in enumerate(self.obstacle_marker_txs):
                marker : Marker = None
                if self.obstacles_visited[idx]:
                    marker = tx.process_data(
                        {
                            'idx' : 1
                        }
                        )
                else:
                    marker = tx.process_data(
                        {
                            'idx' : 0
                        }
                        )
                obstacle_marker_list.append(marker)
            self.obstacle_marker_array_tx.publish(
                {
                    'markers' : obstacle_marker_list
                }
            )
            capsule_list:List[Marker] = []
            for idx, item in enumerate(self.capsule_list):
                a,b,r = item
                marker_arr : MarkerArray = self.collision_capsule_tx.process_data(
                    {
                        'a' : a,
                        'b' : b,
                        'r' : r,
                        'rgba': [1.0,1.0,0.0,0.3],
                        'base_idx' : 3 * idx
                    }
                )
                capsule_list = capsule_list + [marker for marker in marker_arr.markers]
            self.capsule_marker_array_tx.publish(
                 {
                    'markers' : capsule_list
                }
            )


    # region Compute Control
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
                np.array([self.last_theta])
            )
        )
        ## warm-start - provide initial guess to guarantee stable convergence
        if not hasattr(self, "x_guess"):
            self.x_guess = [xcurrent for _ in range(self.N + 1)]
            self.u_guess = [np.zeros(self.nu) for _ in range(self.N)]
        else:
            self.x_guess = self.x_guess[1:] + [self.x_guess[-1]]
            self.u_guess = self.u_guess[1:] + [self.u_guess[-1]]

        for i in range(self.N):
            self.acados_ocp_solver.set(i, "x", self.x_guess[i])
            self.acados_ocp_solver.set(i, "u", self.u_guess[i])
        self.acados_ocp_solver.set(self.N, "x", self.x_guess[self.N])

        # region Objects update
        ## update obstacles & write parameters
        need_gate_update, _ = self.update_gate_if_needed(obs)
        need_obs_update, _ = self.update_obstacle_if_needed(obs)
        if need_gate_update or need_obs_update:
            p_traj = self.get_updated_traj_param(self.arc_trajectory)
            p_capsule = self.get_capsule_param(include_gate=False)
            p_gate = self.get_gate_param()
            p_full = np.concatenate([p_traj, p_capsule, p_gate])
            for i in range(self.N):
                self.acados_ocp_solver.set(i, "p", p_full)
            self.capsule_list = self._gen_pillar_capsule() # for now only for visualization
        
        # ## replan trajectory:
        # if self.pos_change_detect(obs):
        #     gates_rotates = R.from_quat(obs['gates_quat'])
        #     rot_matrices = np.array(gates_rotates.as_matrix())
        #     self.gates_norm = np.array(rot_matrices[:,:,1])
        #     self.gates_pos = obs['gates_pos']
        #     # replan trajectory
        #     waypoints = self.calc_waypoints(self.init_pos, self.gates_pos, self.gates_norm)
        #     # t, waypoints = self.avoid_collision(waypoints, obs['obstacles_pos'], 0.3)
        #     # t, waypoints = self.add_drone_to_waypoints(waypoints, obs['pos'], 0.3, curr_theta=self.last_theta+1)
        #     trajectory = self.trajectory_generate(self.t_total, waypoints)
        #     trajectory = self.traj_tool.extend_trajectory(trajectory)
        #     self.arc_trajectory = self.traj_tool.arclength_reparameterize(trajectory, epsilon=1e-3)
        #     # write trajectory as parameter to solver
        #     p_traj = self.get_updated_traj_param(self.arc_trajectory)
        #     # xcurrent[-2], _ = self.traj_tool.find_nearest_waypoint(self.arc_trajectory, obs["pos"]) # correct theta
        #     p_capsule = self.get_capsule_param()
        #     p_full = np.concatenate([p_traj, p_capsule])
        #     for i in range(self.N): # write current trajectory to solver
        #         self.acados_ocp_solver.set(i, "p", p_full)

        # TODO: replan: find a set of waypoints, connect with cubicspline, do simple push with obstacle field
            
            # # EXP: I do an extra solve here, with v_theta fixed, to provide a feasible solution
            # for i in range(self.N):
            #     fixed_vel = self.u_guess[i][-1]
            #     self.acados_ocp_solver.set(i, "lbu", np.array([-10.0, -10.0, -10.0, -10.0, fixed_vel-0.00]))
            #     self.acados_ocp_solver.set(i, "ubu", np.array([10.0, 10.0, 10.0, 10.0, fixed_vel+0.00]))
            # # set initial state
            # self.acados_ocp_solver.set(0, "lbx", xcurrent)
            # self.acados_ocp_solver.set(0, "ubx", xcurrent)
            # # solve with v_theta frozen
            # self.acados_ocp_solver.solve()
            # # Restore constraints
            # for i in range(self.N):
            #     self.acados_ocp_solver.set(i, "lbu", np.array([-10.0, -10.0, -10.0, -10.0, self.lb_vel]))
            #     self.acados_ocp_solver.set(i, "ubu", np.array([10.0, 10.0, 10.0, 10.0, self.ub_vel]))
            # # Update warm start with solution just solved
            # self.x_guess = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
            # self.u_guess = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]
            # # Write new warm start
            # for i in range(self.N):
            #     self.acados_ocp_solver.set(i, "x", self.x_guess[i])
            #     self.acados_ocp_solver.set(i, "u", self.u_guess[i])
            # self.acados_ocp_solver.set(self.N, "x", self.x_guess[self.N])
            # self.x_warmup_traj = np.array([x[:3] for x in self.x_guess]) # for visualization


        # set initial state
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # region Dyn vel lim
        ## EXP: dynamic velocity limit | works well! It's different than only increase q_l.
        # dyn_ub_vel = self.lb_vel + (self.ub_vel - self.lb_vel) * (1 - 0.9 * np.exp(-2 * np.min(np.linalg.norm(obs['pos'] - obs['gates_pos'], axis=-1)) ** 4))
        # dyn_vel_hard_code_factor = [3, 4, 1, 10]
        # dyn_ub_vel = self.lb_vel + (self.ub_vel - self.lb_vel) * (1 - 1.0 * np.exp(-dyn_vel_hard_code_factor[obs['target_gate']] * np.linalg.norm(obs['pos'] - obs['gates_pos'][obs['target_gate']], axis=-1) ** 4))
        # for i in range(self.N):
        #     self.acados_ocp_solver.set(i, "lbu", np.array([-10.0, -10.0, -10.0, -10.0, self.lb_vel]))
        #     self.acados_ocp_solver.set(i, "ubu", np.array([10.0, 10.0, 10.0, 10.0, dyn_ub_vel]))

        # safety exit:
        if self.last_theta >= 8.59:
            self.finished = True

        if self.acados_ocp_solver.solve() == 4:
            pass

        x_result = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        u_result = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]
        p_result = [self.acados_ocp_solver.get(i, "p") for i in range(self.N)]

        ## update initial guess
        self.x_guess = x_result
        self.u_guess = u_result

        # x1 = self.acados_ocp_solver.get(1, "x")
        x1 = x_result[1]
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_theta = self.last_theta * (1 - w) + x1[14] * w
        self.last_f_cmd = self.last_f_cmd * (1-w) + x1[10] * w
        self.last_rpy_cmd = self.last_rpy_cmd * (1-w) + x1[11:14] * w


        cmd = np.concatenate(
            (
                np.array([self.last_f_cmd]),
                self.last_rpy_cmd
            )
        )

        # Calculate costs every steps for tuning
        debug_costs : Dict[str, List[np.floating]] = {key : [] for key in self.mpc_costs_tx_keys}
        
        for i in range(self.N):
            result = self.cost_func_debug(x_result[i], u_result[i], p_result[i])
            for idx, value in enumerate(result):
                debug_costs[self.mpc_costs_tx_keys[idx]].append(float(value))

        # region Visualization

        pos_traj = np.array([x_result[i][:3] for i in range(self.N+1)])
        if self.need_ros_tx():
            self.mpcc_traj_tx.publish(
                raw_data = {
                    'traj' : pos_traj,
                    'frame_id' : 'map'
                }
            )
            self.mpc_costs_tx.publish(raw_data = debug_costs)
        if self.need_ros_tx(slow=True):
            pass

        try:
            # print(np.linalg.norm(obs['vel']))
            draw_line(self.env, self.arc_trajectory(self.arc_trajectory.x), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
            draw_line(self.env, np.stack([self.arc_trajectory(self.last_theta), obs["pos"]]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
            draw_line(self.env, pos_traj[0:-1:5],rgba=np.array([1.0, 1.0, 0.0, 0.2]))
            if hasattr(self, "x_warmup_traj"):
                draw_line(self.env, self.x_warmup_traj[0:-1:5],rgba=np.array([0.0, 1.0, 1.0, 0.2]))
            # obstacles: plot a line from pos to the closest point on capsule when dist < self.d_safe
            for a,b,r in self.capsule_list:
                ab = b - a
                ab_norm = np.dot(ab, ab)
                ap = obs["pos"] - a
                t = np.clip(np.dot(ap, ab) / ab_norm, 0.0, 1.0)
                closest = a + t * ab
                dist = np.linalg.norm(obs["pos"] - closest) - r
                if dist < self.d_extend:
                    draw_line(self.env, np.stack([obs["pos"], closest]), rgba=np.array([0.8, 0.0, 1.0, 0.3]))
        except:
            pass

        return cmd

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
        self.step_update(obs = obs)
        self.update_target_gate(obs = obs)
        self.ros_transmit(obs = obs)

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
