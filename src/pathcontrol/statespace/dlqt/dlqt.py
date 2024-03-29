'''
    Path-Tracking with Feedback Linearization Control with a LQR-Control
    Author: Ilja Stasewisch, Date: 2019-06-05
'''
import numpy as np
from numpy.linalg import inv
from threading import Thread, Lock
from math import fabs, sin, cos, tan
import copy

from model.robot_parameter import Robot_Parameter
from pathplanning.libs.transformations import *
from pathplanning.libs.extend_transformations import *
from pathcontrol.libs.dlqr import dlqr

class Discrete_LQ_Tracker:
    
    def __init__(self, robot_parameters, simulation_parameters, path_handler, model_type="") :
        print('Path Tracking Control:  Discrete LQ Tracker')

        self.robot_param = robot_parameters
        self.sim_param = simulation_parameters
        self.path_handler = path_handler

        self.prediction_length = 1.5
        self.prediction_Ts = 1e-1

        self.Ts_ctrl = simulation_parameters.Ts_ctrl
        print("DLQT - Ts_ctrl:", self.Ts_ctrl)
        self.steer = 0.0

    def get_goalReached(self) :
        with self._mutex:
            return self._goal_reached

    def _getLQR(self, A, B, Q, R) :
        # print("A.shape = ", A.shape)
        # print("B.shape = ", B.shape)
        # print("Q.shape = ", Q.shape)
        # print("R.shape = ", R.shape)
        K, _, _ = dlqr(A, B, Q, R)
        return K

    def execute(self, velocity, nn_index, l10_state) :
<<<<<<< HEAD
        return self.type1(velocity = velocity, l10_state=l10_state)
=======
        # return self.type1(velocity = velocity, l10_state=l10_state)

        # return self._ilqr_try(velocity = velocity, l10_state=l10_state)
>>>>>>> b6edc3f7a8fc41da5ae4b8afc13875be57a356af
        # return self._ackermann_frontSteer_RearDrive__n4__input_steerRate(l10_state, velocity)
        # return self.huan_li(velocity = velocity, l10_state=l10_state)
        # return self._ackermann_frontSteer_RearDrive__n3__input_steer(l10_state, velocity)

        # if self.model_type == "_ackermann_frontSteering_RearDrive_n3" :
        #     return self._ackermann_frontSteering_RearDrive_n3(diff_pose, velocity, steer)
        # elif self.model_type == "_ackermann_frontSteering_RearDrive_n2" :
        #     return self._ackermann_frontSteering_RearDrive_n2(diff_pose, velocity)
        # else :
        #     print("FBLC: Control type does not exist!")
        #     return 0.0


    def _execute_dlqt(self, Q, Qf, R, K, Kv, Ku, v, As, Bs, uRef): # DLQT Loop: Backward
        S_N = Qf
        S_k1 = S_N *1
        k = len(As) - 2
        while k >= 0:
            # Get loop relevant variables
            A_k = As[k]
            B_k = Bs[k]
            AT_k = A_k.transpose()
            BT_k = B_k.transpose()
            u_k = np.array( uRef[k] ).reshape(-1,1)

            # DLQT Algorithm: but only the loop relevant part
            inv_bracket = inv(BT_k @ S_k1 @ B_k + R) # for avoiding to calculate 3 times the inverse
            K[k] = inv_bracket @ BT_k @ S_k1 @ A_k
            Kv[k] = inv_bracket @ BT_k
            Ku[k] = inv_bracket @ R
            S_k1 = AT_k @ S_k1 @ (A_k - B_k @ K[k]) + Q           
            v[k] = (A_k - B_k @ K[k]).transpose() @ v[k+1].reshape(-1,1) - K[k].transpose() @ R @ u_k
            # deincrement step
            k -= 1

        return K, Kv, Ku, v

    def _getPrediction_steps_dLength(self, prediction_length, velocity, Ts):
        # Prediction Parameters: steps, delta_length (based on velocity and prediction length, prediction dTime)
        prediction_steps = prediction_length / velocity / Ts
        delta_length = velocity * self.prediction_Ts
        # print("N steps:", prediction_steps, "; delta_length:", delta_length, "; prediction_length:", self.prediction_length, "; velocity:", velocity)
        return int(prediction_steps), delta_length


    def _get_weights__n4__input_steerRate(self):
        Q = np.identity(4)
        Q[0, 0] = 100.0
        Q[1, 1] = 100.0
        Q[2, 2] = 1.0
        Q[3, 3] = 0.01
        R = np.identity(1)
        R[0, 0] = 10
        # print("Q.shape, R.shape:", Q.shape, R.shape)
        return Q, R
    def _get_A4x4(self, velocity, yaw_k, steer_k):
        v = velocity
        lw = self.robot_param.wheelbase
        lo = self.robot_param.length_offset
        Ts = self.prediction_Ts
        omega = v / lw * np.tan(steer_k)
        A = np.identity(4)
        sigma1 = np.tan(steer_k)**2.0 + 1
        A[0, 2] = -Ts * (v * np.sin(yaw_k) + lo * np.cos(yaw_k) * omega )
        A[0, 3] = -Ts * lo / lw * v * np.sin(yaw_k) * sigma1
        A[1, 2] =  Ts * (v * np.cos(yaw_k) - lo * np.sin(yaw_k) * omega )
        A[1, 3] =  Ts * lo / lw * v * np.cos(yaw_k) * sigma1
        A[2, 3] =  Ts * v * sigma1 / lw
        return A
    def _get_B4x1(self):
        Ts = self.prediction_Ts
        B = np.zeros( (4,1) )
        B[3, 0] =  Ts
        return B
    def _execute_forwardRecursion__n4__input_steerRate(self, velocity, K, Kv, v_, Ku, robot_state, ref):
        v, Ts = velocity, self.prediction_Ts
        lw, lo = self.robot_param.wheelbase, self.robot_param.length_offset
        prediction_steps = len(ref["x"])

        x = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        y = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        yaw = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        steer = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        u = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        x[0], y[0], yaw[0], steer[0] = robot_state["lxOffset"], robot_state["lyOffset"], robot_state["lYaw"], robot_state["steer"]
        
        # print("velocity:", velocity)

        for k in range(0, prediction_steps-1):
            xVec_k = np.array( [x[k], y[k], yaw[k], steer[k]] ).reshape(-1, 1)
            refVec_k = np.array( [ref["x"][k], ref["y"][k], ref["yaw"][k], ref["steer"][k] ] ).reshape(-1, 1)
            dx_k = np.array( [xVec_k - refVec_k] ).reshape(-1, 1)
            uRef_k =  np.array( [ ref["steerRate"][k] ] ).reshape(-1, 1)

            du_k = -K[k] @ dx_k - Kv[k] @ v_[k+1] #- Ku[k] @ uRef_k
            u[k] = uRef_k + du_k

            omega_k = v / lw * np.tan(steer[k])
            x[k+1] = x[k] + Ts * (v * np.cos(omega_k) - lo * np.sin(omega_k) * omega_k)
            y[k+1] = y[k] + Ts * (v * np.sin(omega_k) + lo * np.cos(omega_k) * omega_k)
            yaw[k+1] = yaw[k] + Ts * omega_k
            steer[k+1] = steer[k] + Ts * u[k]

        # print("forward recusion steering: ", steer)
        # print("forward recusion steer RATE == system in: ", u)
        return x, y, yaw, steer, u

    def _ackermann_frontSteer_RearDrive__n4__input_steerRate(self, l10_state, velocity):
        no_states, no_inputs = 4, 1

        # Reference Path with steering and steer-rate
        prediction_steps, delta_length = self._getPrediction_steps_dLength(self.prediction_length, velocity, self.prediction_Ts)
        ref = self.path_handler.get_referencePath(self.prediction_length, delta_length, prediction_steps)
 
        # System and Input Matrix
        As = [ np.eye(no_states, no_states) ] * prediction_steps
        Bs = [ np.eye(no_states, no_inputs) ] * prediction_steps
        for k in range(prediction_steps):
            As[k] = self._get_A4x4(velocity, ref["yaw"][k], ref["steer"][k])
            Bs[k] = self._get_B4x1()

        # Execute DLQT: Backward Recusion
        Q, R = self._get_weights__n4__input_steerRate()
        Qf = Q * 10.0
        v = np.zeros( (no_states, 1) ).reshape(-1, 1)
        v = [v] * prediction_steps
        K = np.zeros( (no_inputs, no_states) ) # Dimension: 1x4
        K = [K] * prediction_steps
        Kv = np.zeros( (no_inputs, no_states) ) # Dimension: 1x4
        Kv = [Kv] * prediction_steps
        Ku = np.zeros( (no_inputs, no_inputs) ) # Dimension: 1x1
        Ku = [Ku] * prediction_steps
        K, Kv, Ku, v = self._execute_dlqt(Q, Qf, R, K, Kv, Ku, v, As, Bs, ref["steerRate"])

        # Forward Recusion for Prediction Path and Calculation of desired Steering Angle for Ackermann Simulation ODEs
        prediction_path = self._execute_forward_recursion(velocity, K, Kv, v, Ku, l10_state, ref)
        self.path_handler.set_predictionPath__dlqt_mpc(prediction_path)
        dsteer = prediction_path[-1][0][0]
        self.steer = self.steer + self.Ts_ctrl * dsteer

        return self.steer

    def _get_weights__n3__input_steer(self):
        Q = np.identity(3)
        Q[0, 0] = 100.0
        Q[1, 1] = 100.0
        Q[2, 2] = 10.0
        R = np.identity(1)
        R[0, 0] = 1.0
        # print("Q.shape, R.shape:", Q.shape, R.shape)
        return Q, R
    def _get_A3x3(self, velocity, yaw_k, steer_k):
        v = velocity
        lo, lw = self.robot_param.length_offset, self.robot_param.wheelbase
        Ts = self.prediction_Ts
        u_k = steer_k
        # u_k = 0.0
        A = np.eye(3,3)
        A[0, 2] = -Ts * (v * np.sin(yaw_k) + lo * v / lw * np.cos(yaw_k) * tan(u_k) )
        A[1, 2] =  Ts * (v * np.cos(yaw_k) - lo * v / lw * np.sin(yaw_k) * tan(u_k) )
        # print("A:", A)
        return A
    def _get_B3x1(self, velocity, yaw_k, steer_k):
        v = velocity
        lo, lw = self.robot_param.length_offset, self.robot_param.wheelbase
        Ts = self.prediction_Ts
        u_k = steer_k
        B = np.zeros( (3,1) ) 
        B[0, 0] = -Ts * lo * v / lw * np.sin(yaw_k) * (np.tan(u_k)**2.0 + 1.0)
        B[1, 0] =  Ts * lo * v / lw * np.cos(yaw_k) * (np.tan(u_k)**2.0 + 1.0)
        B[2, 0] =  Ts * v / lw * (np.tan(u_k)**2.0 + 1.0)
        return B
    def _execute_forwardRecursion__n3__input_steer(self, velocity, K, Kv, v_, Ku, robot_state, ref):
        v, Ts = velocity, self.prediction_Ts
        lw, lo = self.robot_param.wheelbase, self.robot_param.length_offset
        prediction_steps = len(ref["x"])

        x = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        y = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        yaw = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        u = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        x[0], y[0], yaw[0]= robot_state["lxOffset"], robot_state["lyOffset"], robot_state["lYaw"]
        
        # print("velocity:", velocity, prediction_steps)


        steer_max = np.deg2rad(35.0)

        for k in range(0, prediction_steps-1):
            xVec_k = np.array( [x[k], y[k], yaw[k]] ).reshape(-1, 1)
            refVec_k = np.array( [ref["x"][k], ref["y"][k], ref["yaw"][k] ] ).reshape(-1, 1)
            dx_k = np.array( [xVec_k - refVec_k] ).reshape(-1, 1)
            uRef_k =  np.array( [ ref["steer"][k] ] ).reshape(-1, 1)
<<<<<<< HEAD
            if k == 0: print("dx_k:", dx_k.transpose(), -K[k] @ dx_k, "; K[k]:", K[k])
            du_k = -K[k] @ dx_k - Kv[k] @ v_[k+1] - Ku[k] @ uRef_k
            # du_k = -Kv[k] @ v_[k+1] -Ku[k] @ uRef_k
            u[k] = uRef_k + du_k
=======
            du_k = -K[k] @ dx_k - Kv[k] @ v_[k+1] - Ku[k] @ uRef_k
            u[k] = du_k
>>>>>>> b6edc3f7a8fc41da5ae4b8afc13875be57a356af

            u_steer = u[k]
            if np.fabs(u_steer) > steer_max: u_steer = steer_max * np.sign(u_steer)
            omega = v / lw * np.tan(u_steer)
            x[k+1] = x[k] + Ts * (v * np.cos(yaw[k]) - lo * np.sin(yaw[k]) * omega )
            y[k+1] = y[k] + Ts * (v * np.sin(yaw[k]) + lo * np.cos(yaw[k]) * omega )
            yaw[k+1] = yaw[k] + Ts * omega

        return x, y, yaw, u       

    def _ackermann_frontSteer_RearDrive__n3__input_steer(self, l10_state, velocity):
        no_states, no_inputs = 3, 1

        # Reference Path with steering and steer-rate
        prediction_steps, delta_length = self._getPrediction_steps_dLength(self.prediction_length, velocity, self.prediction_Ts)
        ref = self.path_handler.get_referencePath(self.prediction_length, delta_length, prediction_steps)

        # System and Input Matrix
        As = [ np.eye(no_states, no_states) ] * prediction_steps
        Bs = [ np.eye(no_states, no_inputs) ] * prediction_steps
        for k in range(prediction_steps):
            As[k] = self._get_A3x3(velocity, ref["yaw"][k], ref["steer"][k])
            Bs[k] = self._get_B3x1(velocity, ref["yaw"][k], ref["steer"][k])

        # Execute DLQT: Backward Recusion
        Q, R = self._get_weights__n3__input_steer()
        Qf = Q * 1.0
        v = np.zeros( (no_states, 1) ).reshape(-1,1)
        v = [v] * prediction_steps
        K = np.zeros( (no_inputs, no_states) ) # Dimension: 1x3
        K = [K] * prediction_steps
        Kv = np.zeros( (no_inputs, no_states) ) # Dimension: 1x3
        Kv = [Kv] * prediction_steps
        Ku = np.zeros( (no_inputs, no_inputs) ) # Dimension: 1x1
        Ku = [Ku] * prediction_steps
        K, Kv, Ku, v = self._execute_dlqt(Q, Qf, R, K, Kv, Ku, v, As, Bs, ref["steer"])

        # Forward Recusion for Prediction Path and Calculation of desired Steering Angle for Ackermann Simulation ODEs
        prediction_path = self._execute_forwardRecursion__n3__input_steer(velocity, K, Kv, v, Ku, l10_state, ref)
        self.path_handler.set_predictionPath__dlqt_mpc(prediction_path)
        
        # Control Steering Angle
        steer = prediction_path[-1][0][0]
        return steer        

    def type1(self, l10_state, velocity):
    
        no_states = 3

        # Reference Path with steering and steer-rate
        prediction_steps, delta_length = self._getPrediction_steps_dLength(self.prediction_length, velocity, self.prediction_Ts)
        ref = self.path_handler.get_referencePath(self.prediction_length, delta_length, prediction_steps)
        u_steer = ref["steer"]

        # Weighting Matrices for Control
        Q, R = self._get_weights__n3__input_steer()

        # Start Condition (see: Li and Todorov 2004 iLQR Paper "Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems")
        S_N = Q * 10
        v_N = np.zeros( (no_states, 1) ).reshape(-1, 1)

        # System and Input Matrix
        As = [ np.eye(no_states, no_states) ] * prediction_steps
        Bs = [ np.eye(no_states, 1) ] * prediction_steps
        # print("length of ref[yaw]:", len(ref["yaw"]) )
        for k in range(prediction_steps):
            As[k] = self._get_A3x3(velocity, ref["yaw"][k], ref["steer"][k])
            Bs[k] = self._get_B3x1(velocity, ref["yaw"][k], ref["steer"][k])

        # Loop Variables and Parameters
        S_k1 = S_N *10
        v = np.zeros( (no_states, 1) ).reshape(-1,1)
        v = [v] * prediction_steps
        v[-1] = v_N
        k = prediction_steps - 2

        # DLQT Loop
        while k >= 0:
            # Get loop relevant variables
            A_k = As[k]
            B_k = Bs[k]
            AT_k = A_k.transpose()
            BT_k = B_k.transpose()
            # DLQT Algorithm: but only the loop relevant part
            K_k = inv(BT_k @ S_k1 @ B_k + R) @ BT_k @ S_k1 @ A_k
            S_k1 = AT_k @ S_k1 @ (A_k - B_k @ K_k) + Q
            u_k = np.array(u_steer[k]).reshape(-1,1)
            v[k] = (A_k - B_k @ K_k).transpose() @ v[k+1].reshape(-1,1) - K_k.transpose() @ R @ u_k
            # deincrement step
            k -= 1

        K_0 = K_k
        Kv_0 = inv(BT_k @ S_k1 @ B_k + R) @ BT_k
        Ku_0 = inv(BT_k @ S_k1 @ B_k + R) @ R

        x_0 = np.array( [l10_state["lx"], l10_state["ly"], l10_state["lYaw"] ] ).reshape(-1, 1)
        r_0 = np.array( [ref["x"][0], ref["y"][0], ref["yaw"][0] ] ).reshape(-1, 1)
        dx_0 = x_0 - r_0
        v_1 =  v[1]
        u_0 = np.array(u_steer[0]).reshape(-1, 1)


        du_0 = -K_0 @ dx_0 - Kv_0 @ v_1 #- Ku_0 @ u_0
        print("du_0:", du_0, "; u_0:", u_0)
        us_0 = u_0 + du_0

        lw = self.robot_param.wheelbase
        steer = us_0[0]
        # print("steer:", steer)
        return steer

    def huan_li(self, l10_state, velocity):
    
        no_states = 3

        # Reference Path with steering and steer-rate
        prediction_steps, delta_length = self._getPrediction_steps_dLength(self.prediction_length, velocity, self.prediction_Ts)
        ref = self.path_handler.get_predictionPath(self.prediction_length, delta_length, prediction_steps)
        u_steer = ref["steer"]

        # Weighting Matrices for Control
        Q, R = self._get_weights()

        # Start Condition (see: Li and Todorov 2004 iLQR Paper "Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems")
        S_N = Q * 100
        v_N = np.zeros( (no_states, 1) ).reshape(-1, 1)

        # System and Input Matrix
        As = [ np.eye(no_states, no_states) ] * prediction_steps
        Bs = [ np.eye(no_states, 1) ] * prediction_steps
        # print("length of ref[yaw]:", len(ref["yaw"]) )
        for k in range(prediction_steps):
            As[k] = self._get_A3x3(velocity, ref["yaw"][k], ref["steer"][k])
            Bs[k] = self._get_B3x1(ref["yaw"][k])

        # Loop Variables and Parameters
        S_k1 = S_N
        v = np.zeros( (no_states, 1) ).reshape(-1,1)
        v = [v] * prediction_steps
        v[-1] = v_N
        k = prediction_steps - 2

        # DLQT Loop
        while k >= 0:
            # Get loop relevant variables
            A_k = As[k]
            B_k = Bs[k]
            AT_k = A_k.transpose()
            BT_k = B_k.transpose()
            # DLQT Algorithm: but only the loop relevant part
            K_k = inv(BT_k @ S_k1 @ B_k + R) @ BT_k @ S_k1 @ A_k
            S_k1 = AT_k @ S_k1 @ (A_k - B_k @ K_k) + Q
            u_k = np.array(u_steer[k]).reshape(-1,1)
            v[k] = (A_k - B_k @ K_k).transpose() @ v[k+1].reshape(-1,1) - K_k.transpose() @ R @ u_k
            # deincrement step
            k -= 1

        K_0 = K_k
        Kv_0 = inv(BT_k @ S_k1 @ B_k + R) @ BT_k
        Ku_0 = inv(BT_k @ S_k1 @ B_k + R) @ R

        x_0 = np.array( [l10_state["lx"], l10_state["ly"], l10_state["lYaw"] ] ).reshape(-1, 1)
        r_0 = np.array( [ref["x"][0], ref["y"][0], ref["yaw"][0] ] ).reshape(-1, 1)
        dx_0 = x_0 - r_0
        v_1 =  v[1]
        u_0 = np.array(u_steer[0]).reshape(-1, 1)

        du_0 = -K_0 @ dx_0 - Kv_0 @ v_1 - Ku_0 @ u_0
        us_0 = u_0 + du_0



        lw = self.robot_param.wheelbase
        steer = np.arctan2(us_0[0] * lw, velocity)
        # print("steer:", steer)
        return steer

 ###################### ILQR TRY #################################



    def _ilqr_loop(self, As, Bs, uVec):
        Q = np.identity(3)
        Q[0, 0] = 100.0
        Q[1, 1] = 100.0
        Q[2, 2] = 1.0
        R = np.identity(1)
        R[0, 0] = 1.0
        S_N = Q * 10.0
        S_k1 = S_N *1
        k = len(As) - 2
        while k >= 0:
            # Get loop relevant variables
            A_k = As[k]
            B_k = Bs[k]
            AT_k = A_k.transpose()
            BT_k = B_k.transpose()
            u_k = np.array( uVec[k] ).reshape(-1,1)

            # DLQT Algorithm: but only the loop relevant part
            inv_bracket = inv(BT_k @ S_k1 @ B_k + R) # for avoiding to calculate 3 times the inverse
            K[k] = inv_bracket @ BT_k @ S_k1 @ A_k
            Kv[k] = inv_bracket @ BT_k
            Ku[k] = inv_bracket @ R
            S_k1 = AT_k @ S_k1 @ (A_k - B_k @ K[k]) + Q           
            v[k] = (A_k - B_k @ K[k]).transpose() @ v[k+1].reshape(-1,1) - K[k].transpose() @ R @ u_k
            # deincrement step
            k -= 1

        return K, Kv, Ku, v

    def _ilqr_try(self, l10_state, velocity):
        
        no_states, no_inputs = 3, 1

        # Reference Path with steering and steer-rate
        prediction_steps, delta_length = self._getPrediction_steps_dLength(self.prediction_length, velocity, self.prediction_Ts)

        uSteerVec = np.zeros( (prediction_steps, 1) ).reshape(-1,1)

        # Reference Path with steering and steer-rate
        lh_pose = self.path_handler.get_lookaheadPose()
        print("lh_pose:", lh_pose)

        # System and Input Matrix
        As = [ np.zeros( (no_states, no_states) ) ] * prediction_steps
        Bs = [ np.zeros( (no_states, no_inputs) ) ] * prediction_steps
        for k in range(prediction_steps):
            As[k] = self._get_A3x3(velocity, ref["yaw"][k], ref["steer"][k])
            Bs[k] = self._get_B3x1(velocity, ref["yaw"][k], ref["steer"][k])

        return 0.0

