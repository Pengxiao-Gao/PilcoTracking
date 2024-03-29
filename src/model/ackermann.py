# PYTHON
import numpy as np
from math import sin, cos, tan, fabs

# LOCAL
from pathplanning.libs.extend_transformations import *
from model.yaw_rate_dynamic import *

class TimeDelay():
    def __init__(self, Ts):
        self.Ts = Ts
        self.time_delay = 0.04 # 1/0.04s = 25Hz

        self.value = 0.0
        self.shiftVec = np.array( [] )

    def getTimeDelayed(self, time, steer):
        time = round(time, 6)

        self.shiftVec = np.append(self.shiftVec, [time+self.time_delay, steer] ).reshape(-1,2)
        if time > self.shiftVec[0,0]:
            self.value = self.shiftVec[0,1]
            self.shiftVec = self.shiftVec[1:-1 , :]
            # self.shiftVec = np.delete(self.shiftVec, 1, 0 )

        # print("self.shiftVec:", self.shiftVec, " @time:", time)
        # return steer
        return self.value

class SteerRamp():
    def __init__(self, Ts):
        self.filtered = 0.0
        self.steer_slope_max = np.deg2rad(48.10164104)      # [rad/s]
        self.steer_slope_max = np.deg2rad(70.0)             # [rad/s]
        self.Ts = Ts

        self.save_filtered = []
        self.save_steer = []

    def getFiltered(self, steer):
        self.steer_slope_max # [rad/s]
        dsteer = (steer - self.filtered) /  self.Ts         # [rad/s]

        if fabs(dsteer) < fabs(self.steer_slope_max): 
            # print("NOT IN THE SATURATION")
            self.filtered += dsteer * self.Ts
        else :
            self.filtered += fabs(self.steer_slope_max) * np.sign(dsteer) *  self.Ts
            # print("IN SATURATION: ", np.rad2deg(dsteer), ";steer=", np.rad2deg(steer), ";filtered=", np.rad2deg(self.filtered) )

        self.save_filtered.append(self.filtered)
        self.save_steer.append(steer)

        return self.filtered 
########### ggggggggggggggggggggggggggggggggggggg
class saveSteer():
    def __init__(self):
        self.save_steer0 = []
        self.save_steer1 = []
    
    def save_steering(self,steer_ctrl,steer):
        self.save_steer0.append(np.rad2deg(steer_ctrl))
        self.save_steer1.append(np.rad2deg(steer))
 


class Ackermann() :

    def __init__(self, model, robot_state, robot_parameter, sim_parameter, yaw_dynamic=None) :
        self.Ts = sim_parameter.Ts_sim
        self.model_type = model
        self.state = robot_state
        self.param = robot_parameter       

        self.steerRamp = SteerRamp(self.Ts)
        self.timeDelay = TimeDelay(self.Ts)
        self.saveSteer = saveSteer()
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



    def execute(self, u_input) :
        if self.model_type == "u=steer, steer=no dynamic, yaw=no dynamic" :
            self._eq__yaw_nodyn__steer_nodyn__input_steer(self.state, u_input)

        elif self.model_type == "u=steer, steer=PT1, yaw=no dynamic" :
            self._eq__yaw_nodyn__steer_PT1__input_steer(self.state, u_input)

        elif self.model_type == "u=steer, steer=PT2, yaw=no dynamic" :
            self._eq__yaw_nodyn__steer_PT2__input_steer(self.state, u_input)

        elif self.model_type == "u=steer, steer=PT2+ramp+timedelay, yaw=no dynamic" :
            self._eq__yaw_nodyn__steer_PT2_ramp_timedelay__input_steer(self.state, u_input)

        elif self.model_type == "u=steer, steer=PT2+ramp+timedelay, yaw=PT1" :
            self._eq__yaw_PT1__steer_PT2_ramp_timedelay__input_steer(self.state, u_input)
        else :
            print("Model for Ackermann does not exist!")
         
        if self.print_model == False:
            print("Ackermann Model", "\n", "\t", self.model_type)
            self.print_model = True

        self._update_simTime()
        

        return self._get_simTime()

    def _eq__yaw_PT1__steer_PT2_ramp_timedelay__input_steer(self, state,  steer_ctrl) :
        # Get robot states
        state = self.state.get_asDict()

        # Modeling for a more realistic steering behaviour
        # ------------------------------------------------
        # Time Delay
        steer_delayed = self.timeDelay.getTimeDelayed(self._get_simTime(), steer_ctrl)
        # Ramp filtered
        steer_ramp = self.steerRamp.getFiltered(steer_delayed)
        # Saturation
        # print("self.param.max_steerAngle:", self.param.max_steerAngle)
        if fabs(steer_ramp) > fabs(self.param.max_steerAngle) : steer_sat = fabs(self.param.max_steerAngle) * np.sign(steer_ramp)
        else : steer_sat = steer_ramp

        # print("steer_sat:", steer_sat)

        # Solve ODEs
        # x, y, yaw = self._eulerExplicit_ode_pose(x=state["x"], y=state["y"], yaw=state["yaw"], v=state["velocity"],
                                                #  steer_u=state["steer"], yawRate=state["yawRate"])
        # yawRate = self._eulerExplicit_ode_yawRatePT1(velocity=state["velocity"], steer=state["steer"], yawRate=state["yawRate"])
        # steer_sys, steerRate_sys = self._eulerExplicit_ode_steerPT2(steer_prev=state["steer"], steerRate_prev=state["steerRate"], steer_u=steer_sat)

        #################################################
        x, y, yaw, yawRate = state["x"], state["y"], state["yaw"], state["yawRate"]
        v, steer, steerRate = state["velocity"], state["steer"], state["steerRate"]
        Ts = self.Ts
        lw = self.param.wheelbase
        V_theta = self.yawDyn.get_gain(v, steer_sat)
        V_theta = V_theta[0]
        # print("V_theta = ", V_theta)
        T_theta = self.yawDyn.get_timeConstant(v, steer_sat)
        # V_theta = 1.0
        T_theta = T_theta[0]
        if T_theta < Ts * 5.0:
            print("WARN IN ACKERMANN: T_theta is too small")
        # print("T_theta = ", T_theta)

        T1 = self.param.T1_PT2_steer
        T2 = self.param.T2_PT2_steer


        # Position
        dx = v * cos(yaw) 
        x_new = x + Ts * ( dx )
        dy = v * sin(yaw) 
        y_new = y + Ts * ( dy )
        # Orientation
        yaw_new = yaw + Ts * yawRate
        yaw_new = getYawRightRange(yaw_new)
        yawRate_desired = v / lw * tan(steer)
        dyawRate = (V_theta * yawRate_desired - yawRate) / T_theta # TODO: V_theta outside of the bracket?
        # print("V_theta * yawRate_desired = ", V_theta * yawRate_desired - yawRate)
        yawRate_new = yawRate + Ts * dyawRate       
        # yawRate_new = yawRate_desired
        
        # print("steer_sat: ", np.rad2deg(steer_sat), self.param.max_steerAngle)

        # Solving ODEs
        steer_new = steer + Ts * steerRate
        dsteerRate =  1.0 / T2 * (steer_sat - steer - T1*steerRate)
        steerRate_new = steerRate + Ts * dsteerRate 

        # print("steer_new:", steer_new)

        # if fabs(steer) > fabs(self.param.max_steerAngle) :
        #     print("steer:", np.rad2deg(steer))


        # Set Robot State
        self._set_robotState(x=x_new, y=y_new, yaw=yaw_new, steer=steer_new, steerRate=steerRate_new, yawRate=yawRate_new)

        # Save Steer
        self.saveSteer.save_steering(steer_ctrl,steer_new)

    def _eq__yaw_nodyn__steer_PT2_ramp_timedelay__input_steer(self, state,  steer_ctrl) :
        # print("NO YAW DYN; STEERING = FULL")
        # Get robot states
        state = self.state.get_asDict()

        # Modeling for a more realistic steering behaviour
        # ------------------------------------------------
        # Time Delay
        steer_delayed = self.timeDelay.getTimeDelayed(self._get_simTime(), steer_ctrl)
        # Ramp filtered
        steer_ramp = self.steerRamp.getFiltered(steer_delayed)
        # Saturation
        if fabs(steer_ramp) > fabs(self.param.max_steerAngle) : steer_sat = fabs(self.param.max_steerAngle) * np.sign(steer_ramp)
        else : steer_sat = steer_ramp

        # Solve ODEs
        # x, y, yaw = self._eulerExplicit_ode_pose(x=state["x"], y=state["y"], yaw=state["yaw"], v=state["velocity"],
                                                #  steer_u=state["steer"], yawRate=state["yawRate"])
        # yawRate = self._eulerExplicit_ode_yawRatePT1(velocity=state["velocity"], steer=state["steer"], yawRate=state["yawRate"])
        # steer_sys, steerRate_sys = self._eulerExplicit_ode_steerPT2(steer_prev=state["steer"], steerRate_prev=state["steerRate"], steer_u=steer_sat)

        #################################################
        x, y, yaw = state["x"], state["y"], state["yaw"]
        v, steer, steerRate = state["velocity"], state["steer"], state["steerRate"]
        Ts = self.Ts
        lw = self.param.wheelbase

        T1 = self.param.T1_PT2_steer
        T2 = self.param.T2_PT2_steer


        # Position
        dx = v * cos(yaw) 
        x_new = x + Ts * ( dx )
        dy = v * sin(yaw) 
        y_new = y + Ts * ( dy )
        # Orientation
        yawRate = v / lw * tan(steer)
        yaw_new = yaw + Ts * yawRate
        yaw_new = getYawRightRange(yaw_new)
    
        
        # Solving ODEs
        steer_new = steer + Ts * steerRate
        dsteerRate =  1.0 / T2 * (steer_sat - steer - T1*steerRate)
        steerRate_new = steerRate + Ts * dsteerRate 

        if fabs(steer) > fabs(self.param.max_steerAngle) :
            pass
            # print("steer:", np.rad2deg(steer))


        # Set Robot State
        self._set_robotState(x=x_new, y=y_new, yaw=yaw_new, steer=steer_new, steerRate=steerRate_new, yawRate=yawRate)
        
        # Save Steer
        self.saveSteer.save_steering(steer_ctrl,steer_new)
       

  

    def _eq__yaw_nodyn__steer_PT2__input_steer(self, state,  steer_ctrl) :
        # print("_eq__yaw_nodyn__steer_PT2__input_steer")
        # Get robot states
        stateVec = self.state.get_asVector()

        # Saturation: Steering angle
        steer_sat = self._saturation(steer_ctrl)


        # Solve ODEs
        x, y, yaw, yawRate = self._eulerExplicit_ode_pose(x=stateVec[0], y=stateVec[1], yaw=stateVec[2], v=stateVec[6], steer_u=stateVec[3])
        steer_sys, steerRate_sys = self._eulerExplicit_ode_steerPT2(steer_prev=stateVec[3], steerRate_prev=stateVec[7], steer_u=steer_sat)

        # Set Robot State
        self._set_robotState(x, y, yaw, steer_sys, steerRate_sys, yawRate)

        # Save Steer
        self.saveSteer.save_steering(steer_ctrl,steer_sys)       

    def _eq__yaw_nodyn__steer_PT1__input_steer(self, state,  steer_ctrl) :
        # Get robot states
        stateVec = self.state.get_asVector()

        # Saturation: Steering angle
        steer_sat = self._saturation(steer_ctrl)
        steer_ramp = self.steerRamp.getFiltered(steer_ctrl)
        
        # Solve ODEs
        x, y, yaw, yawRate = self._eulerExplicit_ode_pose(x=stateVec[0], y=stateVec[1], yaw=stateVec[2], v=stateVec[6], steer_u=stateVec[3])
        steer_sys = self._eulerExplicit_ode_steerPT1(steer_prev=stateVec[3], steer_u=steer_sat)

        # Set Robot State
        self._set_robotState(x=x, y=y, yaw=yaw, steer=steer_sys, yawRate=yawRate)
        # Save Steer
        self.saveSteer.save_steering(steer_ctrl,steer_sys) 

    def _eq__yaw_nodyn__steer_nodyn__input_steer(self, state,  steer_ctrl) :
        # Get robot states
        stateVec = self.state.get_asVector()

        # Saturation: Steering angle
        steer_sat = self._saturation(steer_ctrl)
        # Solve ODEs
        x, y, yaw, yawRate = self._eulerExplicit_ode_pose(x=stateVec[0], y=stateVec[1], yaw=stateVec[2], v=stateVec[6], steer_u=steer_sat)

        # Set Robot State
        self._set_robotState(x=x, y=y, yaw=yaw, steer=steer_sat, yawRate=yawRate)
        self.saveSteer.save_steering(steer_ctrl,steer_sat)       

    def _eulerExplicit_ode_steerPT2(self, steer_prev, steerRate_prev, steer_u):
        # States and Parameters
        Ts = self.Ts
        T1 = self.param.T1_PT2_steer
        T2 = self.param.T2_PT2_steer

        # Solving ODEs
        steer_new = steer_prev + Ts * steerRate_prev
        ddsteer =  1.0 / T2 * (steer_u - steer_prev - T1*steerRate_prev)
        steerRate_new = steerRate_prev + Ts * ddsteer 

        return steer_new, steerRate_new

    def _eulerExplicit_ode_yawRatePT1(self, velocity, steer, yawRate):
        # States and Parameters
        Ts = self.Ts
        lw = self.param.wheelbase
        V_theta = self.yawDyn.get_gain(velocity, steer)
        T_theta = self.yawDyn.get_timeConstant(velocity, steer)
        # print("T_theta:", T_theta)
        # print("V_theta:", V_theta)

        # V_theta = 1.0
        yawRate_desired = velocity / lw * tan(steer)

        # Solving ODE
        dyawRate = (V_theta * yawRate_desired - yawRate) / T_theta
        # dyawRate = yawRate_desired
        yawRate = yawRate + Ts * dyawRate 

        if yawRate > 0.2: print("=============>", yawRate)
        print("yawRate_desired:", yawRate_desired, "yawRate:", yawRate)

        return yawRate

    def _eulerExplicit_ode_steerPT1(self, steer_prev, steer_u):
        # States and Parameters
        Ts = self.Ts
        T_steer = self.param.T_PT1_steer

        # print("T_steer:", T_steer)

        # Solving ODE
        dsteer = 1.0 / T_steer * (steer_u - steer_prev)
        steer_new = steer_prev + Ts * dsteer 

        return steer_new

    def _eulerExplicit_ode_pose(self, x, y, yaw, v, steer_u, yawRate=None):
        # Parameters
        Ts = self.Ts
        lw = self.param.wheelbase

        # Position
        dx = v * cos(yaw) #- lo * dyaw * sin(yaw)
        x = x + Ts * ( dx )
        dy = v * sin(yaw) #+ lo * dyaw * cos(yaw)
        y = y + Ts * ( dy )
        # Orientation
        if yawRate is None:
            yawRate = v / lw * tan( steer_u )
        # print("yaw:", yaw, "yawRate:", yawRate, "Ts:", Ts)
        yaw = yaw + Ts * yawRate
        # print("NEXT yaw:", yaw)
        
        yaw = getYawRightRange(yaw)

        return x, y, yaw, yawRate

    def _set_robotState(self, x, y, yaw, steer, steerRate=None, yawRate=None):
        lo = self.param.length_offset
        self.state.set_pose(x, y, yaw, yawRate)
        xo = x + lo * cos(yaw)
        yo = y + lo * sin(yaw)
        self.state.set_offsetPosition(xo, yo)
        self.state.set_steerAngle(steer, steerRate)

    def _saturation(self, steer): # Limit steering angle
        if fabs(steer) > self.param.max_steerAngle :
            return self.param.max_steerAngle * np.sign(steer)
        else :
            return steer

    def _update_simTime(self):
        self.simTime += self.Ts
    def _get_simTime(self):
        return self.simTime
