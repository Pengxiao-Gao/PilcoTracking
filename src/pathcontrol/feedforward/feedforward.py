import numpy as np

class Feedforward():
    def __init__(self, path, velocity_sign, robot_param, timeconstant_steer):
        self.path_x = path[0] - robot_param.length_offset * np.cos(path[2])
        self.path_y = path[1] - robot_param.length_offset * np.sin(path[2])

        self.dx, self.dy    = self._calc_dot(self.path_x), self._calc_dot(self.path_y)
        self.ddx, self.ddy  = self._calc_dot(self.dx), self._calc_dot(self.dy)
        self.dddx, self.dddy= self._calc_dot(self.ddx), self._calc_dot(self.ddy)
        self.velocity       = self._calc_velocity(self.dx, self.dy)
        self.arclength, ds, ds_mean = self._calc_arclength(self.dx, self.dy)
        # print("init - type(self.arclength):", type(self.arclength) )
        self.steer          = self._calc_steer(self.dx, self.ddx, self.dy, self.ddy, velocity_sign, robot_param.wheelbase)
        self.dsteer         = self._calc_dsteer(self.dx, self.ddx, self.dddx, self.dy, self.ddy, self.dddy, velocity_sign, robot_param.wheelbase)
        self.steer_desired  = self._calc_steerDesired(self.steer, self.dsteer, timeconstant_steer)

    def get_steerDesired(self, index, timeconstant_steer=None) : # If time constant is estimated by a 
        if timeconstant_steer is not None:
            return self.steer[index] + timeconstant_steer * self.dsteer[index]
        else:
            if index >= self.steer_desired.shape[0]:
                return self.steer_desired[-1]
            else :
                return self.steer_desired[index]
        

    def get_steer(self, index=None) :
        if index >= self.steer.shape[0]:
            return self.steer[-1]
        else :
            return self.steer[index]

    def _calc_dot(self, to_diff) :
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
        # print(">>> arclength = ", arclength.shape)
        # print(">>> dx = ", dx.shape)
        return arclength, ds, ds_mean
    def _calc_steer(self, dx, ddx, dy, ddy, velocity_sign, wheelbase) :
        curve = (dx * ddy - ddx * dy ) / ((dx**2.0 + dy**2.0) ** (3.0 / 2.0))
        steer = velocity_sign * np.arctan2(curve * wheelbase, 1)
        return steer
    def _calc_dsteer(self, dx, ddx, dddx, dy, ddy, dddy, velocity_sign, wheelbase) :
        v = np.sqrt(dx**2.0 + dy**2.0)
        num = (dx * dddy - dddx * dy) * v**2.0 - 3.0 * (dx * ddx - ddx * dy) * (dx * ddx + dy * ddy)
        den = v**6.0 / wheelbase + (dx * ddy - ddx * dy)**2.0 * wheelbase
        dsteer = v * num /den  # todo?: |v| => according to Müller 2007 "Orbital Tracking Control for Car Parking via Control of the Clock Using a Nonlinear Reduced Order Steering-angle Observer"
        return dsteer
    def _calc_steerDesired(self, steer, dsteer, timeconstant_steer) :
        steer_desired = steer + timeconstant_steer * dsteer
        return steer_desired