import numpy as np



class Feedforward_TimeScaling():
    def __init__(self):
    
        self.dx, self.dy    = self._calc_dot()
        self.ddx, self.ddy  = self._calc_ddot()

    def _calc_arclength(self, dx, dy) :
        ds = [ np.sqrt(idx**2.0 + idy**2.0) for (idx, idy) in zip(dx, dy)]
        ds_mean = np.mean(ds)
        arclength = np.array([0.0])
        arclength = np.hstack( (arclength, np.cumsum(ds)) )
        # print("arclength = ", arclength.shape)
        return arclength, ds, ds_mean

    def _calc_offset_path(self, x, y, yaw) :
        x_offset = x + self.offset * np.cos(yaw)
        y_offset = y + self.offset * np.sin(yaw)
        return x_offset, y_offset

    def _calc_dot(self) :
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dx = np.append(dx, [dx[-1]], axis=0)
        dy = np.append(dy, [dy[-1]], axis=0)
        return dx, dy

    def _calc_ddot(self) :
        ddx = np.diff(self.dx)
        ddy = np.diff(self.dy)
        ddx = np.append(ddx, [ddx[-1]], axis=0)
        ddy = np.append(ddy, [ddy[-1]], axis=0)
        return ddx, ddy

    def _calc_yaw(self, dx, dy, velocity) :
        yaw = np.arctan2(self.dy, self.dx)
        if velocity < 0.0:
            yaw = [minusPi_to_pi(iyaw- math.pi) for (iyaw) in yaw]
            yaw = np.asarray(self.yaw)
        return yaw

    def _calc_steer(self, curve, velocity) :
        steer = np.arctan2(curve * self.wheelbase, 1)
        if velocity < 0.0:
            steer = steer * -1.0
            steer = np.asarray(steer)

        # fig = plt.figure(1)
        # plt.plot( np.rad2deg(steer) )
        # plt.show()

        return steer

    def _calc_curve(self,dx, ddx, dy, ddy) :
        curve = (dx * ddy - ddx * dy ) / ((dx**2.0 + dy**2.0) ** (3.0 / 2.0))
        return curve

    def _calc_steerRate(self, steer, ds, v) :
        dsteer = np.diff(steer)
        dsteer = np.append(dsteer, [dsteer[-1]], axis=0)
        steerRate = dsteer / ds * v
        steerRate = np.asarray(steerRate)
        return steerRate
