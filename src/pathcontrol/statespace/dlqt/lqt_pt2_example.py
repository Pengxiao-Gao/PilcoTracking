import numpy as np
import copy
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time

class LQT_PT2():

    def __init__(self):
        
        self.traj = None

        self.T1_PT2 = 0.6
        self.T2_PT2 = 0.4

        
        self.prediction_step = 10
        self.prediction_Ts = 1e-1       # [s]
        self.prediction_horizont = 1.0 # [s]
        self.prediction_Ts = self.prediction_horizont / self.prediction_step


        self.sim_Ts = 1e-2
        self.sim_iter = 1
        self.sim_timeEnd = 10.0
        # self.

        self.sim_Steps = int(self.sim_timeEnd / self.sim_Ts)
        self.TimeStates = np.zeros( (self.sim_Steps, 4) )
        self.TimeStates[:] = np.nan
        self.TimeStates[0:1, :] = np.zeros( (1, 4) )
        print("test:", self.TimeStates)

        self.A_sim, self.B_sim = self.get_system(self.sim_Ts)
        self.A_predict, self.B_predict =  self.get_system(self.prediction_Ts)

        self.u_Vec = []

        fig, self.axes = plt.subplots(2, 2)

    def get_system(self, Ts):
        A = np.zeros( (2, 2)  )
        T1 = self.T1_PT2
        T2 = self.T2_PT2
        A[0 ,0] = 1.0
        A[0, 1] = Ts
        A[1, 0] = -Ts/T2
        A[1, 1] = 1 - Ts * T1 / T2
        B = np.zeros( (2, 1) )
        B[0, 0] = 0.0
        B[1, 0] = Ts/T2
        return A, B

    def trajectory_step(self, sim_time_now):
        step_time = 5.0
        Ts = self.prediction_Ts
        ref = np.zeros( (self.prediction_step, 2) )
        ref_Time = sim_time_now
        for i in range(self.prediction_step):
            if ref_Time < step_time:
                ref[i, 0] = 0.0
                ref[i, 1] = 0.0
            else :
                ref[i, 0] = 1.0
                ref[i, 1] = 0.0
            ref_Time += Ts
        return ref

    def start(self):
        
        # self.visualize()

        while self.sim_iter < self.sim_Steps:


            x_k = self.TimeStates[self.sim_iter-1, 1:3].reshape(-1, 1)
            ref_k = self.trajectory_step( (self.sim_iter-1) * self.sim_Ts)
            u = self.get_lqt_ctrl(x_k, ref_k)


            self.sim_step(u)
            # time.sleep(1e-1)
            if self.sim_iter % 100 == 0:
                self.visualize()

        plt.show()


    def get_lqt_ctrl(self, x_k, ref):
        no_states = 2
        no_inputs = 1
        prediction_steps = self.prediction_step

        v = np.zeros( (no_states, 1) ).reshape(-1,1)
        v = [v] * prediction_steps
        K = np.zeros( (no_inputs, no_states) ) # Dimension: 1x4
        K = [K] * prediction_steps
        Kv = np.zeros( (no_inputs, no_states) ) # Dimension: 1x4
        Kv = [Kv] * prediction_steps
        # u = np.zeros( (no_inputs, 1) ) # Dimension: 1x4

        Q = np.eye(2, 2)
        Q[0, 0] = 100.0
        Qf = Q
        R = np.eye(1, 1)
        R[0, 0] = 0.1

        A = self.A_sim
        B = self.B_sim
        AT = A.transpose()
        BT = B.transpose()

        # print("A: ", A, "\nB: ", B, "\nAT: ", AT, "\nBT: ", BT)

 

        S_N = Qf
        S_k = S_N * 10
        k = prediction_steps - 2
        while k >= 0:
            # print("S_k: ", S_k, "R: ", R)
            inv_bracket = inv(BT @ S_k @ B + R)
            K[k] = inv_bracket @ BT @ S_k @ A
            S_k = AT @ S_k @ (A - B @ K[k]) + Q
            v[k] = (A - B @ K[k]).transpose() @ v[k+1].reshape(-1,1) + Q @ ref[k].reshape(-1, 1)
            Kv[k] = inv_bracket @ BT           
            k -= 1

        # print("v[1]:", v[1])
        # print("Kv[0]:", Kv[0].shape)

        # print("K[0]:", K[0])
        # print("K[0]:", Kv[0].shape)


        # print("inv(A + B @ K[0]:", inv(A + B @ K[0]) )
        T1 = self.T1_PT2
        T2 = self.T2_PT2
        A = [ [0.0, 1.0], [-1.0/T2, -T1/T2] ]
        B = [[0.0], [1.0/T2]]
        test = -inv( np.eye(1, 2) @ inv(A + B @ K[0]) @ B )
        print(test)
        x_k = x_k / 4

        u = -K[0] @ x_k + Kv[1] @ v[1]
        # print(u)
        self.TimeStates[self.sim_iter, 3] = u
        return u

    def sim_step(self, u_k):
        i = self.sim_iter
        T1 = self.T1_PT2
        T2 = self.T2_PT2
        Ts = self.sim_Ts
        x_k = self.TimeStates[i-1, 1:3].reshape(-1, 1)
        x_k1 = self.A_sim @ x_k + self.B_sim @ u_k
        self.TimeStates[i, 1:3] = x_k1.transpose()
        self.TimeStates[i, 0] = self.TimeStates[i-1, 0] + Ts
        self.sim_iter += 1




    def visualize(self):
        for idx, axis in np.ndenumerate(self.axes): axis.clear()

        # self.axes[0, 0].plot( np.random.rand(1000, 1) )

        self.axes[1, 0].plot( self.TimeStates[:, 0], self.TimeStates[:, 1] )

        self.axes[1, 1].plot( self.TimeStates[:, 1], self.TimeStates[:, 2] )

        self.axes[0, 1].plot( self.TimeStates[:, 0], self.TimeStates[:, 3] )

        plt.pause(1e-3)




if __name__ == "__main__":
    lqt_pt2 = LQT_PT2()

    lqt_pt2.start()



