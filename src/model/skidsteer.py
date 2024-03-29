# PYTHON
import numpy as np
from math import sin, cos, tan, fabs

# LOCAL
from pathplanning.libs.extend_transformations import *
from model.yaw_rate_dynamic import *

class SkidSteer() :

    def __init__(self, model, robot_state, robot_parameter, sim_parameter, yaw_dynamic=None) :
        self.Ts = sim_parameter.Ts_sim
        self.model_type = model
        self.state = robot_state
        self.param = robot_parameter       

        self.simTime = 0.0
        self.print_model = False

        if yaw_dynamic is not None:
            if yaw_dynamic == "tractor_yaw_dynamic":
                self.yawDyn = YawRate_Dynamic("tractor")
            elif yaw_dynamic == "max_yaw_dynamic":
                self.yawDyn = YawRate_Dynamic("max")
            elif yaw_dynamic == "min_yaw_dynamic":
                self.yawDyn = YawRate_Dynamic("min")
            elif yaw_dynamic == "median_yaw_dynamic":
                self.yawDyn = YawRate_Dynamic("median")
            else:
                print("ERROR IN ACKERMANN: NO SUCH MODEL FOR YAW DYNAMIC")


    def execute(self, vLeft_input=None, vRight_input=None, deltaV_input=None, velocity=None) :
        if self.model_type == "u=diff vel, vel=saturation+no dynamic, yaw=no dynamic" :
            self._eq__input_dVel__noYawDyn_noVelDyn(self.state, vLeft_input, vRight_input)
        elif self.model_type == "u=diff vel, vel=Saturation+PT1, yaw=no dynamic" :
            self._eq__input_dVel_dVelPT1__noYawDyn(self.state, deltaV_input, velocity)
        else :
            print("Model for Skid-Steering does not exist!")
         
        if self.print_model == False:
            print("Ackermann Model", "\n", "\t", self.model_type)
            self.print_model = True

        self._update_simTime()
        

        return self._get_simTime()

    def _eq__input_dVel_dVelPT1__noYawDyn(self, state, deltaV_input, velocity=None):
        # Get robot states
        state = state.get_asDict()

        # Saturation of Velocities: Calc velocity-Right and velocity-Left and Saturate it; And velocity as well
        if velocity is None:
            velocity = state["velocity"]       
        dV = state["deltaVel"]
        v = velocity   
        vR = v + dV
        vR = self._saturation(vR)
        vL = v - dV
        vL = self._saturation(vL)
        vSat = self._get_velocity(vR, vL)
        
        # Solve ODEs
        yawRate = self._get_yawRate(vR, vL)
        Ts = self.Ts
        yaw = state["yaw"]
        x = state["x"] + Ts * ( v * cos(yaw) )
        y = state["y"] + Ts * ( v * sin(yaw) )
        yaw = yaw + Ts * yawRate     
        yaw = getYawRightRange(yaw)
        # ODE for velocity-difference
        dV_u = deltaV_input
        T_dV = self.param.T_pt1_dV
        dV = dV + Ts *(1.0/T_dV *(dV_u-dV)  )    

 
        # Set Robot State
        self._set_robotState(x=x, y=y, yaw=yaw, yawRate=yawRate, vel=vSat, diff_vel=dV)


    def _eq__input_dVel__noYawDyn_noVelDyn(self, state, vLeft_input, vRight_input):
        # Get robot states
        state = state.get_asDict()

        # Saturation of Velocities
        vel_right = vRight_input
        vel_right = self._saturation(vel_right)
        vel_left = vLeft_input
        vel_left = self._saturation(vel_left)
        vel_sat = self._get_velocity(vel_right, vel_left)
        
        # Solve ODEs
        yawRate = self._get_yawRate(vel_right, vel_left)
        # print("yawRate:", yawRate)
        x, y, yaw = self._eulerExplicit_pose(x=state["x"], y=state["y"], v=vel_sat,
                                             yaw=state["yaw"], yawRate=yawRate)

        # Set Robot State
        self._set_robotState(x=x, y=y, yaw=yaw, yawRate=yawRate, vel=vel_sat)



    def _eulerExplicit_pose(self, x, y, yaw, v, yawRate):
        # Parameters
        Ts = self.Ts
        # Position 
        x = x + Ts * ( v * cos(yaw) )
        y = y + Ts * ( v * sin(yaw) )
        yaw = yaw + Ts * yawRate     
        yaw = getYawRightRange(yaw)
        return x, y, yaw

    def _set_robotState(self, x, y, yaw, vel, yawRate, diff_vel=None):
        self.state.set_velocity(vel=vel, diffVel_skid=diff_vel)
        self.state.set_pose(x, y, yaw, yawRate)

        lo = self.param.length_offset
        xo = x + lo * cos(yaw)
        yo = y + lo * sin(yaw)
        self.state.set_offsetPosition(xo, yo)

    def _saturation(self, velocity): # Limit steering angle
        if fabs(velocity) > self.param.max_velocity :
            return self.param.max_velocity * np.sign(velocity)
        else :
            return velocity

    def _get_yawRate(self, right, left):
        return (right - left) / (2.0*self.param.half_track)

    def _get_velocity(self, right, left):
        return (right + left) / 2.0

    def _update_simTime(self):
        self.simTime += self.Ts
    def _get_simTime(self):
        return self.simTime
