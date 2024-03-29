import numpy as np
import copy
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time

def minusPi_to_pi(angle) :
    if angle > np.pi or angle < -np.pi :
        return np.arctan2(np.sin(angle), np.cos(angle))
    else :
        return angle

class ILQR_KINEMATIC_VEHICLE():
    def __init__(self):
        
        self.steer_max = np.deg2rad(35.0)

        self.start_pose = [0.0, 0.0, 0.0]
        self.end_pose = [10.0, -3.0, -1.0]

        self.pred_steps = 500
        self.Ts = 0.01

        self.wheelbase = 1.0

        u_start = np.random.rand( self.pred_steps, 2 ) # [velocity, steer]
        u_start[:, 0] = 1.5
        u_start[:, 1] = 0.055

        path = self._forward_init(u_start)
        self._plot(path)
        K, Kv, v, Ku = self._backward(path, u_start)
        u_old = u_start.copy()

        for i in range(290):
            print("iter number:", i)
            path, u_new = self._forward(K, Kv, v, Ku, path, u_old)
            self._plot(path)
            K, Kv, v, Ku = self._backward(path, u_new)
            u_old = u_new.copy()





    def _forward(self, K, Kv, v, Ku, path, u_old):
        Ts = self.Ts
        lw = self.wheelbase
        prediction_steps = self.pred_steps

        x = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        y = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        yaw = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        x[0], y[0], yaw[0]= path[0][0], path[1][0], path[2][0]

        # print("u_old:", u_old.shape)

        u_new = u_old.copy()


        for k in range(0, prediction_steps-1):

            u_k = u_old[k, :].reshape(-1,1)
            # print("test u_k:", u_k.shape)

            xPath = np.array( [path[0][k], path[1][k], path[2][k]] ).reshape(-1,1) 
            x_current = np.array( [ x[k], y[k], yaw[k]] ).reshape(-1,1) 
            xStar = self._get_xGoal().reshape(-1, 1)
            dx_k = x_current - xPath
            
            # print("test dx_k:", dx_k.shape)

            # print("K[k] @ dx_k :", (K[k] @ dx_k).shape,  (Kv[k] @ v[k+1]).shape, (Ku[k] @ u_k).shape )
            
            v_k1 = v[k+1].reshape(-1,1)
            du_k = -K[k] @ dx_k - Kv[k] @ v_k1 - Ku[k] @ u_k

            # print("test:", u_k.shape, du_k.shape)
            us_k = u_k + du_k
            # print("us_k:", us_k, us_k.shape)

            # print("k=", k, us_k.shape, du_k.shape)
            u_new[k, :] = us_k.reshape(1,-1)

            u_v = us_k[0]
            u_steer = us_k[1]
            # print("u_steer:", type(u_steer), u_steer.shape, u_steer[0])
            omega = u_v[0] / lw * np.tan(u_steer[0])
            x[k+1] = x[k] + Ts * u_v[0] * np.cos(yaw[k]) 
            y[k+1] = y[k] + Ts * u_v[0] * np.sin(yaw[k])
            yaw[k+1] = yaw[k] + Ts * omega
            yaw[k+1] = minusPi_to_pi(yaw[k+1])

        return [x, y, yaw], u_new


    def _forward_init(self, system_input):
        Ts = self.Ts
        lw = self.wheelbase
        u = system_input # [velocity, steer]
        prediction_steps = self.pred_steps

        x = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        y = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        yaw = np.zeros( (prediction_steps, 1) ).reshape(-1, 1)
        x[0], y[0], yaw[0]= self.start_pose[0], self.start_pose[1], self.start_pose[2] 

        for k in range(0, prediction_steps-1):
            v = u[k, 0]
            steer = u[k, 1]
            omega = v / lw * np.tan(steer)
            x[k+1] = x[k] + Ts * v * np.cos(yaw[k]) 
            y[k+1] = y[k] + Ts * v * np.sin(yaw[k])
            yaw[k+1] = yaw[k] + Ts * omega
            yaw[k+1] = minusPi_to_pi(yaw[k+1])

        return [x, y, yaw]

    def _get_weights(self):
        Q = np.eye(3, 3)
        Q[0, 0] = 100.0
        Q[1, 1] = Q[0, 0]
        Q[2, 2] = 10.0
        Qf = Q * 10.0
        R = np.eye(2,2) * 1.0
        return Q, Qf, R
    def _get_xGoal(self):
        xGoal = np.zeros( shape=(3,) )
        xGoal[0] = self.end_pose[0]
        xGoal[1] = self.end_pose[1]
        xGoal[2] = self.end_pose[2]
        return xGoal

    def _get_A(self, xVec, uVec):
        theta_k = xVec[2]
        v_k = uVec[0]
        Ts = self.Ts

        A_k = np.eye(3, 3)
        A_k[0,2] = -Ts * v_k * np.sin(theta_k)
        A_k[1,2] =  Ts * v_k * np.cos(theta_k)
        # print("A_k", A_k)
        return A_k
    def _get_B(self, xVec, uVec):
        theta_k = xVec[2]
        v_k = uVec[0]
        steer_k = uVec[1]
        Ts = self.Ts
        lw = self.wheelbase

        B_k = np.zeros( (3, 2) )
        B_k[0,0] = Ts * np.cos(theta_k)
        B_k[1,0] = Ts * np.sin(theta_k)
        B_k[2,0] = Ts * np.tan(steer_k) / lw
        B_k[2,1] = Ts * v_k * (np.tan(steer_k)**2.0 +1.0) / lw
        return B_k


    def _backward(self, path, sys_inputs):
        no_inputs = 2
        no_states = 3

        As, Bs = [], []
        for k in range(len(path[0])):
            xVec = [ path[0][k], path[1][k], path[2][k] ]
            uVec = [ sys_inputs[k, 0], sys_inputs[k, 1] ]
            As.append( self._get_A(xVec, uVec) )
            Bs.append( self._get_B(xVec, uVec) )

        v = np.zeros( (3, 1) ).reshape(-1,1)
        xStar = self._get_xGoal()
        xN = np.array([path[0][-1], path[1][-1], path[2][-1]]).reshape(-1,)
        v = [v] * self.pred_steps
        Q, Qf, R = self._get_weights()
        v[-1] = Qf @ (xN - xStar)
        print("v[-1]:", v[-1])
        S_k1 = Qf
      
        K = np.zeros( (no_inputs, no_states) ) # Dimension: 2x3
        K = [K] * self.pred_steps
        Kv = np.zeros( (no_inputs, no_states) ) # Dimension: 2x3
        Kv = [Kv] * self.pred_steps
        Ku = np.zeros( (no_inputs, no_inputs) ) # Dimension: 2x2
        Ku = [Ku] * self.pred_steps

        k = self.pred_steps-2
        while k >= 0:
            # Get loop relevant variables
            A_k = As[k]
            B_k = Bs[k]
            AT_k = A_k.transpose()
            BT_k = B_k.transpose()
            # DLQT Algorithm: but only the loop relevant part
            inv_bracket = inv(BT_k @ S_k1 @ B_k + R)
            K[k] = inv_bracket @ BT_k @ S_k1 @ A_k
            Kv[k] = inv_bracket @ BT_k
            Ku[k] = inv_bracket @ R
            S_k1 = AT_k @ S_k1 @ (A_k - B_k @ K[k]) + Q
            u_k = np.array( [sys_inputs[k,0], sys_inputs[k,1] ] ).reshape(-1,1)
            # print("R:", R, "u_k:", u_k)
            x_k = np.array([path[0][k], path[1][k], path[2][k]]).reshape(-1,1)
            # print("v[k]",v[k+1] )
            v[k] = (A_k - B_k @ K[k]).transpose() @ v[k+1].reshape(-1,1) - K[k].transpose() @ R @ u_k + Q @ x_k
            k -= 1

        return K, Kv, v, Ku

    def start(self):
        pass

    def _plot(self, path):
        # plt.cla()
        plt.plot(self.start_pose[0], self.start_pose[1], markersize=4, marker="*")
        plt.plot(self.end_pose[0], self.end_pose[1], markersize=4, marker="*")

        plt.plot(path[0], path[1])
        plt.pause(0.1)
        # plt.show()



if __name__ == "__main__":
    ilqr_kinematic = ILQR_KINEMATIC_VEHICLE()

    ilqr_kinematic.start()



