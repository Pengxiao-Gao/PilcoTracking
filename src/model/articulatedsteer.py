# PYTHON
import numpy as np
from math import sin, cos, tan, fabs

# LOCAL
from pathplanning.libs.extend_transformations import *
from model.yaw_rate_dynamic import *

class Articulated_Steering() :

    def __init__(self, model, robot_state, robot_parameter, sim_parameter) :
        self.Ts = sim_parameter.Ts_sim
        self.model_type = model
        self.state = robot_state
        self.param = robot_parameter       
        self.simTime = 0.0

    def execute(self, u_input, velocity=None) :
        if self.model_type == "u=steer, steer+yaw=noDyn, sideslip=no, vel=rear" :
            self._eq__u_steer__v_rear__steer_yaw_nodyn__alpha_no(self.state, u_input)
        elif self.model_type == "_eqs__u_steerDesired__v_rear__steer_PT1__yaw_nodyn__alpha_no":
            self._eqs__u_steerDesired__v_rear__steer_PT1__yaw_nodyn__alpha_no(state=self.state, steerDesired=u_input, velocity=velocity)


        else :
            print("Model for articulated steering does not exist!")
         
        self._update_simTime()
        return self._get_simTime()

    def _eqs__u_steerDesired__v_rear__steer_PT1__yaw_nodyn__alpha_no(self, state,  steerDesired, velocity=None) :
        # Get robot states and velocity
        state = state.get_asDict()
        if velocity is None:
            velocity = state["velocity"]
        v = state["velocity"]
        
        steer = state["steer"]
        T_delta = self.param.T_PT1_steer
        steerDesired_sat = self._saturation(steerDesired)
        lv = self.param.length_front
        lh = self.param.length_rear


        # Solve ODEs
        Ts = self.Ts
        yaw = state["yaw"]
        x = state["x"] + Ts * ( v * cos(yaw) )
        y = state["y"] + Ts * ( v * sin(yaw) )
        steerRate = 1.0 / T_delta * (steerDesired_sat - steer) 
        yawRate = (v * sin(steer) - lv * steerRate) / (lh * cos(steer) + lv)
        yaw = yaw + Ts * yawRate     
        yaw = getYawRightRange(yaw)
        steer = steer + Ts * steerRate
        steer_sat = self._saturation(steer)

        # Set Robot State
        self._set_robotState(x=x, y=y, yaw=yaw, yawRate=yawRate, steer=steer_sat, steerRate=steerRate)


    def _eq__u_steer__v_rear__steer_yaw_nodyn__alpha_no(self, state,  steer_ctrl) :
        # Get robot states
        state = state.get_asDict()
        vel = state["velocity"]
        steer_sat = self._saturation(steer_ctrl)

        # Solve ODEs
        steerRate = self._get_steerRate_from_steerDifference(steer_sat, state["steer"])
        yawRate = self._get_yawRate(vel, steer_sat, steerRate, part="rear", vel_loc="rear")
        x, y, yaw = self._eulerExplicit_pose(x=state["x"], y=state["y"], v=vel,
                                             yaw=state["yaw"], yawRate=yawRate)

        # Set Robot State
        self._set_robotState(x=x, y=y, yaw=yaw, yawRate=yawRate, steer=steer_sat, steerRate=steerRate)

    def _get_steerRate_from_steerDifference(self, steer_curr, steer_prev):
        steerRate = (steer_curr - steer_prev) / self.Ts
        return steerRate

    def _get_yawRate(self, v, steer, steerRate, part="rear", vel_loc="rear", alpha_v=0, alpha_h=0):
        l_v = self.param.length_front
        l_h = self.param.length_rear
        a_v = alpha_v
        a_h = alpha_h
        if part == "front" and vel_loc == "front":
            yawRate = (v * sin(steer+a_v-a_h) + steerRate*l_h*cos(a_h)) / (l_v*cos(steer-a_h) + l_h*cos(a_h) )
        elif part == "front" and vel_loc == "rear":
            yawRate = (v * sin(steer+a_v-a_h) + steerRate*l_h*cos(steer+a_v)) / (l_h*cos(steer+a_v) + l_v*cos(a_h) )
        elif part == "rear" and vel_loc == "rear":
            yawRate = (v * sin(steer+a_v-a_h) - steerRate*l_v*cos(a_v)) / (l_h*cos(steer+a_v) + l_v*cos(a_h) )
        elif part == "rear" and vel_loc == "front":
            yawRate = (v * sin(steer+a_v-a_h) - steerRate*l_v*cos(steer-a_h)) / (l_v*cos(steer-a_h) + l_h*cos(a_h) )                                   
        return yawRate

    def _eulerExplicit_pose(self, x, y, v, yaw, yawRate):
        # Position
        dx = v * cos(yaw) #- lo * dyaw * sin(yaw)
        x = x + self.Ts * ( dx )
        dy = v * sin(yaw) #+ lo * dyaw * cos(yaw)
        y = y + self.Ts * ( dy )
        # Orientation
        yaw = yaw + self.Ts * yawRate
        yaw = getYawRightRange(yaw)      
        return x, y, yaw

    def _eulerExplicit_steer(self, steer, steerRate):
        steer = steer + self.Ts * steerRate
        steer_sat = self._saturation(steer)
        return steer_sat

    def _saturation(self, steer): # Limit steering angle
        if fabs(steer) > self.param.max_steerAngle :
            return self.param.max_steerAngle * np.sign(steer)
        else :
            return steer

    def _update_simTime(self):
        self.simTime += self.Ts
    def _get_simTime(self):
        return self.simTime

    ######################################## TIEFER AUS ACKERMANN NOCH


    def _set_robotState(self, x, y, yaw, yawRate=None, steer=None, steerRate=None):
        lo = self.param.length_offset
        self.state.set_pose(x, y, yaw, yawRate)
        xo = x + lo * cos(yaw)
        yo = y + lo * sin(yaw)
        self.state.set_offsetPosition(xo, yo)
        self.state.set_steerAngle(steer, steerRate)


