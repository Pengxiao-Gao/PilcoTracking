
# PYTHON
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(main_path) 

from pathplanning.libs.extend_transformations import *
from pathplanning.libs.transformations import *

class Path_Sine() :

    def __init__(self, amp, omega, dLength_path, arclength, plotting=False) :
        self.amp = amp
        self.omega = omega
        self.ds = dLength_path
        self.arclength = arclength
        self.cum_arclength = []

    def _calc_dot(self, x, y) :
        dx = np.diff(x)
        dy = np.diff(y)
        dx = np.append(dx, [dx[-1]], axis=0)
        dy = np.append(dy, [dy[-1]], axis=0)
        return dx, dy
    def _calc_arclength(self, dx, dy) :
        ds = [ np.sqrt(idx**2.0 + idy**2.0) for (idx, idy) in zip(dx, dy)]
        ds_mean = np.mean(ds)
        arclength = np.array([0.0])
        arclength = np.hstack( (arclength, np.cumsum(ds)) )
        return arclength
    def _calc_yaw(self, dx, dy, velocity=1.0) :
        yaw = np.arctan2(self.dy, self.dx)
        if velocity < 0.0:
            yaw = [minusPi_to_pi(iyaw- math.pi) for (iyaw) in yaw]
            yaw = np.asarray(self.yaw)
        return yaw

    def get_arclength(self):
        return self.cum_arclength
    def get_yaw(self):
        return self.yaw

    def get_path(self) :

        x = [0.0]
        arclength = [0.0]

        while True:
            dy = self.amp * self.omega * np.cos(self.omega*x[-1])
            dx = np.sqrt(self.ds**2.0 / (1+dy**2.0 ) )
            x.append(x[-1]+dx)
            if x[-1] > self.arclength:
                break

        x = np.array(x)
        y = self.amp * np.sin(self.omega * x)
        dx, dy = _calc_dot(x, y)

        self.cum_arclength = self._calc_arclength(dx, dy)
        idx = np.argwhere(self.cum_arclength > self.arclength)
        idx = idx[0][0]
        x = x[0:idx]
        y = y[0:idx]
        self.cum_arclength = self.cum_arclength[0:idx]

        self.yaw = _calc_yaw(dx, dy)

        return x, y


def _calc_dot(x, y) :
    dx = np.diff(x)
    dy = np.diff(y)
    dx = np.append(dx, [dx[-1]], axis=0)
    dy = np.append(dy, [dy[-1]], axis=0)
    return dx, dy

def _calc_ddot(dx, dy) :
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    ddx = np.append(ddx, [ddx[-1]], axis=0)
    ddy = np.append(ddy, [ddy[-1]], axis=0)
    return ddx, ddy

def _calc_yaw(dx, dy) :
    yaw = np.arctan2(dy, dx)
    return yaw

def _calc_curve(dx, ddx, dy, ddy) :
    curve = (dx * ddy - ddx * dy ) / ((dx**2.0 + dy**2.0) ** (3.0 / 2.0))
    return curve

if __name__ == "__main__":

    # MAIN FOR CHECK IF IT WORKS

    # path_8 = Path_8_shaped()
    ds_val = 0.01
    path_sine = Path_Sine(amp=10.0, omega=0.1, dLength_path=ds_val, arclength=120.0, plotting=True)
    # path_8 = Path_8_shaped(plotting=True)

    path_x, path_y = path_sine.get_path()



    fig, axs = plt.subplots(2,2)
    axs[0, 0].plot(path_x, path_y)
    axs[0, 0].set_title("path")

    dx, dy = _calc_dot(path_x, path_y)
    ds = np.sqrt(dx**2.0 + dy**2.0)
    ds_mean = np.mean(ds)
    print(ds)
    axs[0, 1].plot( ds )
    axs[0, 1].plot( np.ones(shape=ds.shape)*ds_mean )
    axs[0, 1].set_title("ds")
    axs[0, 1].set_ylim([ds_val-ds_val*0.01, ds_val+ds_val*0.01])

    
    ddx, ddy = _calc_dot(dx, dy)
    curve = _calc_curve(dx, ddx, dy, ddy)
    axs[1, 1].plot( curve )
    axs[1, 1].set_title("curvature")

    axs[1, 0].plot( path_sine.get_arclength() )
    axs[1, 0].set_title("cumulated arclength")




    plt.show()