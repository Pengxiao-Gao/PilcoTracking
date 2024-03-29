'''
    Path-Tracking with Feedback Linearization Control with a LQR-Control
    Author: Ilja Stasewisch, Date: 2019-06-05
'''
import numpy as np
from numpy.linalg import inv
from threading import Thread, Lock
from math import fabs, sin, cos, tan

from model.robot_parameter import Robot_Parameter
from pathplanning.libs.transformations import *
from pathplanning.libs.extend_transformations import *
from pathcontrol.libs.dlqr import dlqr

class FBLC:
    
    def __init__(self, robot_parameters, simulation_parameters, model_type="") :
        print('Path Tracking Control:  FBLC')
        self._mutex = Lock()
        self._goal_reached = False
        self.robot_param = robot_parameters
        self.sim_param = simulation_parameters
        self.ctrlMatrix_K = None
        self.model_type = model_type

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

    def execute(self, diff_pose, velocity, steer=0.0) :
        if self.model_type == "_ackermann_frontSteering_RearDrive_n3" :
            return self._ackermann_frontSteering_RearDrive_n3(diff_pose, velocity, steer)
        elif self.model_type == "_ackermann_frontSteering_RearDrive_n2" :
            return self._ackermann_frontSteering_RearDrive_n2(diff_pose, velocity)
        else :
            print("FBLC: Control type does not exist!")
            return 0.0

    def _ackermann_frontSteering_RearDrive_n3(self, diff_pose, velocity, steer):
        v = velocity
        lo = self.robot_param.length_offset
        lw = self.robot_param.wheelbase
        Tsteer = self.robot_param.T_PT1_steer * 1.0

        # GET THE REFERENCE-POSE
        x1 = diff_pose[1, 3]
        x2 = matrix_to_yaw(diff_pose)  
          
        x3 = steer

        # STATE TRANSFORMATION
        z1 = x1 - lo*sin(x2)
        z2 = v*sin(x2)
        z3 = (v**2.0 * cos(x2) * tan(x3) ) / lw
        Z = np.array([z1, z2, z3]).reshape(3,1)
        beta = np.array( [ (v**2.0 * cos(x2) * (tan(x3)**2.0 + 1) ) / (Tsteer*lw)]).reshape(1,1)


        # print("Z = ", Z)
        
        # Calculate the Control
        if self.ctrlMatrix_K is None:
            Ts = self.sim_param.Ts_ctrl
            A = np.array( [[1, Ts, 0], [0, 1, Ts], [0, 0, 1] ] )
            B = np.array( [0, 0, Ts] ).reshape(3, 1)
            Q = np.array( [[1000, 0, 0], [0, 10, 0], [0, 0, 1]] )
            R = np.array( [1.0] ).reshape(1, 1)
            self.ctrlMatrix_K = self._getLQR(A, B, Q, R)
        # print("K.shape = ", self.ctrlMatrix_K.shape)
        # print("Z.shape = ", Z.shape)
        ctrl_linear = - self.ctrlMatrix_K @ Z
        alpha_1 = - (v**3.0 * sin(x2) * tan(x3)**2.0) / lw**2.0

        alpha_2 =  - (v**2.0 * x3 * cos(x2) * (tan(x3)**2.0 + 1.0) ) / (Tsteer*lw)
        alpha = np.array( [alpha_1 + alpha_2] ).reshape(1,1)
        
        # print("ctrl_linear = ", ctrl_linear)

        ctrl = -alpha + ctrl_linear @ inv(beta) # See "Jürgen Adamy Nichtlineare Regelungen" p.338
        # print("ctrl.shape = ", ctrl.shape)
        ctrl = np.arctan2(ctrl, 1)
        # print("ctrl[0,0] = ", ctrl[0,0])

        return ctrl[0,0],x2


    def _ackermann_frontSteering_RearDrive_n2(self, diff_pose, velocity):
        v = velocity
        Ts = self.sim_param.Ts_ctrl
        lo = self.robot_param.length_offset
        lw = self.robot_param.wheelbase

        # GET THE REFERENCE-POSE
        x1 = diff_pose[1, 3] #dy
        x2 = matrix_to_yaw(diff_pose)
        
        # print("diff_pose = ", diff_pose)
        # print("x1, x2 = ", x1, x2)

        # DEFINE THE ERROR-STATE-VECTOR
        

        # STATE TRANSFORMATION
        z1 = x1 - lo*sin(x2)
        z2 = v*sin(x2)
        Z = np.array([z1, z2]).reshape(2,1)

        beta = np.array( [ v**2.0 * cos(x2) / lw]).reshape(1,1)


        # print("Z = ", Z)
        
        # Calculate the Control
        if self.ctrlMatrix_K is None:
            Ts = self.sim_param.Ts_ctrl
            A = np.array( [[1, Ts], [0, 1]] )
            B = np.array( [0, Ts] ).reshape(2,1)
            Q = np.array( [[10, 0], [0, 5]] )
            R = np.array( [1.0] ).reshape(1,1)
            self.ctrlMatrix_K = self._getLQR(A, B, Q, R)
        # print("K.shape = ", self.ctrlMatrix_K.shape)
        # print("Z.shape = ", Z.shape)
        ctrl_linear = - self.ctrlMatrix_K @ Z
        alpha = np.array( [0] ).reshape(1,1)
        
        # print("ctrl_linear = ", ctrl_linear)

        ctrl = -alpha + ctrl_linear @ inv(beta) # See "Jürgen Adamy Nichtlineare Regelungen" p.338
        # print("ctrl.shape = ", ctrl.shape)
        ctrl = np.arctan2(ctrl, 1)
        # print("ctrl[0,0] = ", ctrl[0,0])

        return ctrl[0,0]



    def calc_lqr_ctrl(self, path_offset, pathframe_2_offsetframe, idx_nn) :
        pass
     