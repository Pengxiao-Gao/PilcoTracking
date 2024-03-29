# PYTHON
import numpy as np
import math
from math import sin, cos, tan, pi, atan2, fabs
from threading import Lock

# LOCAL
from model.robot_parameter import Robot_Parameter
import pathplanning.libs.extend_transformations as mytf

class Robot_State() :
    def __init__(self, robot_parameter, x=0.0, y=0.0, yaw=0.0, steerAngle=0.0, velocity=0.0, var_xy=0.0, var_yaw=0.0, l10n_update_distance=0.0) :
    
        self._mutex = Lock()
        self.look_ahead = robot_parameter.look_ahead
        self.length_offset = robot_parameter.length_offset
        self.var_xy = var_xy
        self.var_yaw = var_yaw
        self.l10n_update_distance = l10n_update_distance
        self.xyYaw_update = None
        self.dtfMatrix = mytf.xyYaw_to_matrix(x=0.0, y=0.0, yaw=0.0)

        # POSE
        self.x = x
        self.y = y
        self.xo = x + self.length_offset * cos(yaw)
        self.yo = y + self.length_offset * sin(yaw)
        self.yaw = yaw
        self.yawRate = 0.0
        # Localization
        self.lx = x
        self.ly = y
        self.lYaw = yaw
        self.lxo = self.xo
        self.lyo = self.yo
        # STEERING
        self.steerAngle = steerAngle
        self.steerAngleDes = 0.0
        self.steerAngleRate = 0.0
        self.steerAngleRateDes = 0.0

        self.deltaVel = 0.0
        self.deltaVelDes = 0.0

        # DRIVE
        self.velocity = velocity
        self.accel = 0.0

    ########################
    ##### SET METHODS ######
    def set_velocity(self, vel, diffVel_skid=None) :
        # print("ROBOT STATE set_velocity")
        with self._mutex:
            self.velocity = vel
            self.deltaVel = diffVel_skid
    def set_pose(self, x, y, yaw, yawRate = None) :
        with self._mutex:
            self.x = x
            self.y = y
            self.yaw = yaw
            # print("Robot_State::set_pose(...): yaw = ", yaw)
            self.xo = x + self.length_offset * cos(yaw)
            self.yo = y + self.length_offset * sin(yaw)
            if yawRate is not None:
                self.yawRate = yawRate
        self._update_l10n(yawRate)
    def set_offsetPosition(self, xo, yo) :
        with self._mutex:
            self.xo = xo
            self.yo = yo
    def set_steerAngle(self, steerAngle, steerAngleRate=None):
        with self._mutex:
            self.steerAngle = steerAngle
            if steerAngleRate is not None:
                self.steerAngleRate = steerAngleRate

    ########################
    ##### GET METHODS ######
    def get_asVector(self):
        with self._mutex:
            return [self.x, self.y, self.yaw, self.steerAngle, self.xo, self.yo, self.velocity, self.steerAngleRate, self.yawRate] 
            #          0       1        2          3               4        5        6                  7                   8
    def get_xyYaw(self):
        with self._mutex:
            return self.x, self.y, self.yaw
    def get_asVector4lookahead(self):
        with self._mutex:
            lh = self.look_ahead
            x = self.x + lh * cos(self.yaw)
            y = self.y + lh * sin(self.yaw)
            return [x, y, self.yaw] 
    def get_asDict(self):
        with self._mutex:
            return {"x": self.x, "y": self.y, "yaw": self.yaw, "yawRate": self.yawRate,
                       "steer": self.steerAngle, "steerRate": self.steerAngleRate, "velocity": self.velocity,
                       "xOffset": self.xo, "yOffset": self.yo, "deltaVel": self.deltaVel}
    def get_velocity(self):
        with self._mutex:
            return self.velocity
    def get_offsetAsVector(self):
        with self._mutex:
            return [self.xo, self.yo, self.yaw] 
    ########################

    ##########################
    ###### LOCALIZATION ######
    def _update_l10n(self, yawRate):
        if self.l10n_update_distance >=1e-6 and (self.var_xy > 1e-6 or self.var_yaw > 1e-6) :
            self._l10n_discontinous(yawRate)
            # self._set_l10n_pose(self.x, self.y, self.yaw)
        elif self.var_xy > 1e-6 or self.var_yaw > 1e-6:
            self._l10n_continous()
        else:
            self._set_l10n_pose(self.x, self.y, self.yaw)

    def _l10n_continous(self):
        x, y, yaw = self.get_xyYaw()
        dx = np.random.normal(0.0, self.var_xy)
        dy = np.random.normal(0.0, self.var_xy)
        dYaw = np.random.normal(0.0, self.var_yaw)
        lx = x + dx
        ly = y + dy
        lYaw = mytf.getYawRightRange(yaw + dYaw)
        print("x, y, yaw:", x, y, yaw, "; lx, ly, lYaw:", lx, ly, lYaw)
        self._set_l10n_pose(lx, ly, lYaw)
    def _l10n_discontinous(self, yawRate):
        xyYaw_now = self.get_xyYaw()

        tfMatrix_now = mytf.xyYaw_to_matrix(xyYaw_now)

        if self.xyYaw_update is None:
            self.xyYaw_update = xyYaw_now
            

        if mytf.getLength2D(xyYaw_now, self.xyYaw_update) > self.l10n_update_distance:
            self.xyYaw_update = xyYaw_now
            dx = np.random.normal(0.0, self.var_xy)
            dy = np.random.normal(0.0, self.var_xy)
            dYaw = np.random.normal(0.0, self.var_yaw)
            dYaw = mytf.getYawRightRange(dYaw)
            self.dtfMatrix = mytf.xyYaw_to_matrix(dx, dy, dYaw)


        map_2_l10n = tfMatrix_now @ self.dtfMatrix
        lx, ly, lYaw = map_2_l10n[0,3], map_2_l10n[1,3], mytf.matrix_to_yaw(map_2_l10n)
        self._set_l10n_pose(lx, ly, lYaw)


    def _set_l10n_pose(self, lx, ly, lYaw):
        with self._mutex:
            self.lx = lx
            self.ly = ly
            self.lYaw = lYaw
            # print("self.length_offset:", self.length_offset)
            self.lxo = lx + self.length_offset * cos(lYaw)
            # print("lx:", lx, "lxo:", self.lxo)
            self.lyo = ly + self.length_offset * sin(lYaw)
    def get_l10nVec(self):
        with self._mutex:
            return [self.lx, self.ly, self.lYaw, self.steerAngle, self.lxo, self.lyo, self.velocity, self.steerAngleRate, self.yawRate] 
            #          0       1        2          3               4        5        6                  7                   8
    def get_l10nDict(self):
        with self._mutex:
            return {"lx": self.lx, "ly": self.ly, "lYaw": self.lYaw, "yawRate": self.yawRate,
                       "steer": self.steerAngle, "steerRate": self.steerAngleRate, "velocity": self.velocity,
                       "lxOffset": self.lxo, "lyOffset": self.lyo}
    def get_l10nOffset(self):
        with self._mutex:
            return [self.lxo, self.lyo, self.lYaw] 
    ##################################

    ##################
    ##### PRINT ######
    def __str__(self):
        return "x, y, yaw[°], steer[°]: " + str(self.x) + " " + str(self.y) + " " + str( np.rad2deg(self.yaw))  + " " + str( np.rad2deg(self.steerAngle))
    ##################



if __name__ == "__main__":
    print("MAIN for testing the class")

    rp = Robot_Parameter()
    rs = Robot_State(rp)
    # rs.x = 3.0
  
    # print(rsd.get_asVector() )


    rsd_disc.set_pose(x=4.9, y=2.6, yaw=1.2)

    print("BLA:", rsd_disc.get_asVector())

    # Test Varaiance
    max_ = 0.0
    for a in range(1):
        rand_x = np.random.normal(0.0, rsd_disc.var_xy)
        rand_y = np.random.normal(0.0, rsd_disc.var_xy)
        print("dist", np.hypot(rand_x,rand_y), "rand_x:", rand_x, "rand_y:", rand_y)
        dist = np.hypot(rand_x,rand_y)
        if dist > max_:
            max_ = dist

    print("max_", max_)

    print( mytf.xyYaw_to_matrix(x=1.0, y=2.0, yaw=1.4) )