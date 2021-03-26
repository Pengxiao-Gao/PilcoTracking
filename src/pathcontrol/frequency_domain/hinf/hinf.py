
# PYTHON
import control
from control import *
import matplotlib.pyplot as plt
import numpy as np
import time 
from math import sin, cos, tan

# LOCAL
from pathcontrol.frequency_domain.hinf.control_ext import combine
from pathplanning.libs.transformations import *
from pathplanning.libs.extend_transformations import *

class Hinf():

    def __init__(self, model_type="", robot_parameter="", sim_parameter="", velocity=1.0, length_offset=-1.0):
        
        self.robot_parameter = robot_parameter
        self.sim_parameter = sim_parameter

        # Weighting Transfer Functions
        # ----------------------------
        self.Wy1 = self._make_weight(dc=1000.0, crossw=0.1, hf=0)
        # Wy2 = self._make_highpass(hf_gain=0.98, m3db_frq=2.1)
        self.Wy2 = self._make_weight(dc=0.001, crossw=0.1, hf=0)
        # self.Wu = self._make_highpass(hf_gain=0.4, m3db_frq=2.1)
        self.Wu = self._make_weight(dc=0.1, crossw=0.8, hf=1.0)

        # Get System
        # ----------
        if model_type=="errorModel_woSteer_woYawDyn":
            print("H-infinity - model:", "errorModel_woSteer_woYawDyn")
            self.Gss = self._get_errorModel_woSteer_woYawDyn(velocity=velocity, length_offset=length_offset)
        elif model_type=="errorModel_steerPT1_woYawDyn":
            print("H-infinity - model:", "errorModel_steerPT1_woYawDyn")
            try:
                self.Gss = self._get_errorModel_steer_woYawDyn(velocity=velocity, length_offset=length_offset, T_steer=robot_parameter.T_steer)
                print(self.Gss)
            except: 
                self.Gss = self._get_errorModel_steerPT1_woYawDyn(velocity=velocity, length_offset=length_offset, T_steer=0.2)
                print(self.Gss)

        G = tf(self.Gss)

        # Generate Controller
        # -------------------
        # for cross_omega in np.arange(10.0, 0.1, -0.1):
        #     print("cross_omega:", cross_omega)
        #     self.Wy1 = self._make_weight(dc=100.0, crossw=cross_omega, hf=0)
        #     Pss = self._get_extendedSystem(Wy_yE=self.Wy1, Wy_yawE=self.Wy2, Wu=self.Wu, G_yE=G[0,0], G_yawE=G[1,0])
        #     Kss, CL, gamma, rcond = hinfsyn(Pss, 2, 1)
        #     print("hinfsyn gamma:", gamma)
        #     if gamma < 1.0: break

        Pss = self._get_extendedSystem(Wy_yE=self.Wy1, Wy_yawE=self.Wy2, Wu=self.Wu, G_yE=G[0,0], G_yawE=G[1,0])
        Kss, CL, gamma, rcond = hinfsyn(Pss, 2, 1)
        print("hinfsyn gamma:", gamma)

        try:
            self.Kdss = c2d(Kss, self.sim_parameter.Ts_ctrl)
        except:
            self.Kdss = c2d(Kss, 1e-3)

        self.K_states = np.zeros( shape=(self.Kdss.A.shape[0], 1) )
        


    def execute(self, diffPose_matrix, velocity )  :
        K, x = self.Kdss, self.K_states
        lw = self.robot_parameter.wheelbase
        yE = diffPose_matrix[1, 3]
        yawE = matrix_to_yaw(diffPose_matrix)
        u = np.array( [ [-yE], [-yawE] ] )
        y = K.C @ x + K.D @ u
        x = K.A @ x + K.B @ u
        omega = y[0,0]
        self.K_states = x
        return np.arctan(omega * lw / velocity)

    def plot_weights(self):
        # Plot Transfer Functions
        mag, phase, omega = bode(self.Wy1)
        # plt.plot(omega, mag)
        mag, phase, omega = bode(self.Wy2)
        # plt.plot(omega, mag)
        mag, phase, omega = bode(self.Wu)
        # plt.plot(omega, mag)
        plt.show()
    def plot_controller(self):
        # Simulate Control Loop
        # ---------------------
        if plot == True:
            # Desired to Controlled Variable
            fig, ax = plt.subplots(3, 1)
            I = ss([], [], [], np.eye(2))
            w_to_y = feedback(self.Gss*self.K, I, -1)
            T, yout, xout = step_response(sys=w_to_y, T=np.linspace(0, 10, 1000), input=0, return_x=True)
            ax[0].plot(T, yout[0,:], 'r--')
            ax[0].plot(T, yout[1,:], 'r:')
            T, yout, xout = step_response(sys=w_to_y, T=np.linspace(0, 10, 1000), input=1, return_x=True)
            ax[0].plot(T, yout[0,:], 'b--')
            ax[0].plot(T, yout[1,:], 'b:')
            ax[0].grid(True)
            ax[0].legend( ('y-Error for desired y-Error=1', 'yaw-Error for desired y-Error=1', 'y-Error for desired yaw-Error=1', 'yaw-Error for desired yaw-Error=1') )
            # Disturbance z1 to Controlled Variable
            ax[1].grid(True)
            z2_to_y = feedback(self.Gss, self.K, -1)
            T, yout, xout = step_response(sys=z2_to_y, T=np.linspace(0, 10, 1000), input=0, return_x=True)
            ax[1].plot(T, yout[0,:], 'r--')
            ax[1].plot(T, yout[1,:], 'r:')
            ax[1].legend( ('y-Error for z2-Error/G-input-Error=1', 'yaw-Error for z2-Error/G-input-Error=1') )
            # Disturbance z2 to Controlled Variable
            ax[2].grid(True)
            z1_to_y = feedback(I, self.Gss*self.K, -1)
            T, yout, xout = step_response(sys=z1_to_y, T=np.linspace(0, 10, 1000), input=0, return_x=True)
            ax[2].plot(T, yout[0,:], 'r--')
            ax[2].plot(T, yout[1,:], 'r:')
            T, yout, xout = step_response(sys=z1_to_y, T=np.linspace(0, 10, 1000), input=1, return_x=True)
            ax[2].plot(T, yout[0,:], 'b--')
            ax[2].plot(T, yout[1,:], 'b:')
            ax[2].legend( ('y-Error for z1-yError=1', 'yaw-Error for z1-yError=1', 'y-Error for z1-yawError=1', 'yaw-Error for z1-yawError=1') )
            plt.show()

    def _get_extendedSystem(self, Wy_yE, Wy_yawE, Wu, G_yE, G_yawE):
        Wy1, Wy2, = Wy_yE, Wy_yawE
        G1, G2 = G_yE, G_yawE

        # P11 = [ [W1, 0, W1*G1], [0, W2, W2*G2], [0, 0, 0] ]
        # P12 = [ [W1*G1], [W2*G2], [Wu] ]
        # P21 = [ [1, 0, G1], [0, 1, G2] ]
        # P22 = [ [G1], [G2] ]
        # P = [ [P11, P21], [P21, P22]]
        P11=np.block( [ [Wy1, 0, Wy1*G1], [0, Wy2, Wy2*G2], [0,0,0] ] ) 
        P12=np.block( [ [Wy1*G1], [Wy2*G2], [Wu] ])
        P21=np.block( [ [-1, 0, -G1], [0, -1, -G2] ] )
        P22=np.block( [ [-G1], [-G2] ] )
        P = np.block( [ [P11, P12], [P21, P22]] )
        Pc = combine(P)
        Pcss = minreal( Pc ) 
        return Pcss
    def _get_errorModel_woSteer_woYawDyn(self, velocity, length_offset):
        v, lo = velocity, length_offset
        A = [ [0, v], [0, 0] ]
        B = [ [lo], [1] ]
        C = [ [1, 0], [0, 1] ] 
        D = [ [0], [0] ]
        Gss = ss(A, B, C, D)
        return Gss
    def _get_errorModel_steerPT1_woYawDyn(self, velocity, length_offset, T_steer):
        v, lo = velocity, length_offset
        A = [ [0, v, lo], [0, 0, 1], [0, 0, -1/T_steer] ]
        B = [ [0], [0], [1/T_steer] ]
        C = [ [1, 0, 0], [0, 1, 0] ] 
        D = [ [0], [0]]
        Gss = ss(A, B, C, D)
        return Gss

    def _make_lowpass(self, dc, crossw):
        return tf([dc], [crossw, 1])
    def _make_highpass(self, hf_gain, m3db_frq):
        return tf([hf_gain, 0], [1, m3db_frq])
    def _make_weight(self, dc, crossw, hf):
        """weighting(wb,m,a) -> wf 
        crossw - design frequency (where |wf| is approximately 1)
        hf - high frequency gain of 1/wf; should be > 1
        dc - low frequency gain of 1/wf; should be < 1
        """
        s = tf([1, 0], [1])
        return (s * hf + crossw) / (s + crossw / dc)



if __name__ == "__main__":

    hinf = Hinf(model_type="errorModel_woSteer_woYawDyn", velocity=1.0, length_offset=-1.0)
    # hinf = Hinf(model_type="errorModel_steerPT1_woYawDyn", velocity=1.0, length_offset=-1.0)

    hinf.plot_weights()
    hinf.plot_controller()

    # v = 1.1
    # lo = 2.2
    # A = [ [0, v], [0, 0] ]
    # B = [ [lo], [1] ]
    # C = [ [1, 0], [0, 1] ] 
    # D = [ [0], [0]]
    # Gss = ss(A, B, C, D)
    # G = tf(Gss)       
    # G1 = G[0,0]
    # G2 = G[1,0]
    # G1ss = ss(G1)
    # G2ss = ss(G2)

    # W1 = hinf._make_weight(dc=100, crossw=2.1, hf=0)
    # W2 = hinf._make_highpass(hf_gain=0.99, m3db_frq=2.1)
    # Wu = hinf._make_highpass(hf_gain=0.3, m3db_frq=2.1)

    # P11 = [ [W1, 0, W1*G1], [0, W2, W2*G2], [0, 0, 0] ]
    # P12 = [ [W1*G1], [W2*G2], [Wu] ]
    # P21 = [ [1, 0, G1], [0, 1, G2] ]
    # P22 = [ [G1], [G2] ]
    # P = [ [P11, P21], [P21, P22]]

    # P11=np.block( [ [W1, 0, W1*G1], [0, W2, W2*G2], [0, 0,0] ] ) 
    # P12=np.block( [ [W1*G1], [W2*G2], [Wu] ])
    # P21=np.block( [ [1, 0, G1], [0, 1, G2] ] )
    # P22=np.block( [ [G1], [G2] ] )

    # print(P11.shape)
    # print(P12.shape)
    # print(P21.shape)
    # print(P22.shape)


    # P_ = np.block( [ [P11, P12], [P21, P22]] )
    # Pc = combine(P_)
    # print(Pc.A.shape)
    # Pcss = minreal( Pc ) 
    # print(Pcss)

    # K, CL, gam, rcond = hinfsyn(Pcss, 2, 1)
    # print("+++++++++++++++++++++\n", K)