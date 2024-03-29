import numpy as np

class Feedforward_Knick():
    def __init__(self, path, velocity_sign, robot_param, timeconstant_steer, velocity=None):
        self.robot_param = robot_param

        self.path_x = path[0] - robot_param.length_offset * np.cos(path[2])
        self.path_y = path[1] - robot_param.length_offset * np.sin(path[2])

        self.dx, self.dy = self._calc_derivation(self.path_x), self._calc_derivation(self.path_y)
        self.arclength, ds, ds_mean = self._calc_arclength(self.dx, self.dy)
        print("Feedforward ds_mean:", ds_mean)

        # Arclength-dependent derivatives
        self.dx, self.dy     = self.dx / ds, self.dy / ds
        self.ddx, self.ddy   = self._calc_derivation(self.dx), self._calc_derivation(self.dy)
        self.ddx, self.ddy   = self.ddx / ds, self.ddy / ds
        self.dddx, self.dddy = self._calc_derivation(self.ddx), self._calc_derivation(self.ddy)
        self.dddx, self.dddy = self.dddx / ds, self.dddy / ds

        self.velocity_arclength = self._calc_velocity(self.dx, self.dy)
        self.yaw                =  self._calc_yaw(self.dx, self.dy)
        self.dyaw               = self._calc_dyaw(self.dx, self.dy, self.ddx, self.ddy)


        self.steer, self.steerRate = self._calc_knick_steer_steerRate(yawRate=self.dyaw, velArc=self.velocity_arclength,
                                                                      arclength=self.arclength, ds=ds)

        self.steer_wo_steerRate = self._calc_steer(dx=self.dx, ddx=self.ddx, dy=self.dy, ddy=self.ddy,
                                      wheelbase=self.robot_param.wheelbase, trigonometric_func="arcsin")



        print("len:", len(self.steer), len(self.steer_wo_steerRate) )

    def get_steerDesiredPT1(self, idx, vel):
        return self.steer[idx] + self.robot_param.T_PT1_steer * self.steerRate[idx] * vel / self.velocity_arclength[idx]

    def get_steer(self, idx):
        return self.steer[idx]

    def get_steerRate(self, idx, vel):
        return self.steerRate[idx] * vel / self.velocity_arclength[idx]

    def get_steer_arcsin(self, index=None):
        if index is None:
            return self.steer_arcsin
        elif index >= len(self.steer_arcsin):
            index = len(self.steer_arcsin) - 1
        return self.steer_arcsin[index]

    def get_dot_steer_arcsin(self, index=None, velocity=None):
        if index is None:
            return self.dsteerArcsin_by_derivate * velocity
        elif index >= len(self.dsteerArcsin_by_derivate):
            index = len(self.dsteerArcsin_by_derivate) - 1
        return self.dsteerArcsin_by_derivate[index] * velocity

    def get_desiredSteer_in_time_arcsin(self, index=None, velocity=None, T_PT1_steer=None):
        l_f = self.robot_param.length_front
        l_r = self.robot_param.length_rear

        if T_PT1_steer is None:
            T_PT1_steer = self.robot_param.T_PT1_steer

        if index is None:
            return T_PT1_steer * self.dsteerArcsin_by_derivate * velocity + self.steer_arcsin
        elif index >= self.dyaw.shape[0]:
            index = self.dyaw.shape[0]-1

        if velocity is None:
            return self.steer_PT1_in_time_arcsin[index]
        else:
            return T_PT1_steer * self.dsteerArcsin_by_derivate[index] * velocity + self.steer_arcsin[index]


    ################################################
    # KNICK
    def get_knick_steer_in_time(self, index=None, velocity=None): # KNICK
        l_v = self.robot_param.length_front
        l_r = self.robot_param.length_rear

        if index is None:
            num = (l_v+l_r) * self.dyaw*velocity
            den = self.velocity_arclength*velocity
            return np.arcsin( num/den )
        elif index >= self.dyaw.shape[0]:
            index = self.dyaw.shape[0]-1

        num = (l_v+l_r) * self.dyaw[index]*velocity
        den = self.velocity_arclength[index]*velocity
        return np.arcsin( num/den )

    # KNICK
    def get_knick_steerDesired_in_time(self, index=None, velocity=None, T_PT1_steer= None) : # KNICK
        l_v = self.robot_param.length_front
        l_r = self.robot_param.length_rear

        if T_PT1_steer is None:
            T_PT1_steer = self.robot_param.T_PT1_steer

        if index is None:
            steer = np.arcsin(self.dyaw * (l_v+l_r) * velocity / (self.velocity_arclength* velocity) )
            return T_PT1_steer * velocity * self.dsteerArcsin_by_derivate + steer
        elif index >= self.dyaw.shape[0]:
            index = self.dyaw.shape[0]-1

        steer = np.arcsin(self.dyaw[index] * (l_v+l_r) * velocity / (self.velocity_arclength[index]* velocity) )
        steer_desired = T_PT1_steer * velocity * self.dsteerArcsin_by_derivate[index] + steer
        return steer_desired


    ########################################################################
    ############################ INTERN METHODS ############################

    def _calc_knick_steer_steerRate(self, yawRate, velArc, arclength, ds):
        lf = self.robot_param.length_front
        lr = self.robot_param.length_rear

        steer =  yawRate * (lf+lr)  / velArc


        steerRate = self._calc_derivation(steer) / ds
        steer += lf*steerRate / velArc

        return steer, steerRate


    def _calc_derivation(self, to_diff) :
        diff_in = np.diff(to_diff)
        diff_in = np.append(diff_in, [diff_in[-1]], axis=0)
        return diff_in
    def _calc_velocity(self, dx, dy):
        velocity = np.sqrt(dx**2.0 + dy**2.0)
        return velocity
    def _calc_arclength(self, dx, dy) :
        ds = [ np.sqrt(idx**2.0 + idy**2.0) for (idx, idy) in zip(dx, dy)]
        ds_mean = np.mean(ds)
        arclength = np.array([0.0])
        arclength = np.hstack( (arclength, np.cumsum(ds)) )
        arclength = arclength[0:-1] # exclude last element
        return arclength, ds, ds_mean
    def _calc_yaw(self, dx, dy) :
        yaw = np.arctan(dy/dx)
        return yaw
    def _calc_dyaw(self, dx, dy, ddx, ddy) :
        dyaw = (dx * ddy - ddx * dy ) / (dx**2.0 + dy**2.0)
        return dyaw


    def _calc_steer(self, dx, ddx, dy, ddy, wheelbase, trigonometric_func="arcsin") :
        curve = (dx * ddy - ddx * dy ) / ((dx**2.0 + dy**2.0) ** (3.0 / 2.0))
        if trigonometric_func == "arctan":
            steer = np.arctan(curve * wheelbase)
        elif trigonometric_func == "arcsin":
            try:
                # steer = np.arcsin(curve * wheelbase)
                steer = (curve * wheelbase)
            except:
                steer = None
                print("Error for Arcsin in Feedforward Steer!")
        else:
            print("Feedforward '_calc_steer': No trigonometric function implemented for that string!")
            return -1

        return steer

    def _calc_dsteer(self, dx, ddx, dddx, dy, ddy, dddy, wheelbase) :
        v = np.sqrt(dx**2.0 + dy**2.0)
        num = v**3.0 * (dx * dddy - dddx * dy) - 3.0 * v * (dx * ddy - ddx * dy) * (dx * ddx + dy * ddy)
        den = v**6.0 +  wheelbase**2.0 * (dx * ddy - ddx * dy)**2.0
        dsteer = wheelbase  *num /den
        return dsteer
