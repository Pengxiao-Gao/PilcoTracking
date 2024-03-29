8.
# PYTHON
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
# main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(main_path) 

from pathplanning.libs.extend_transformations import *
from pathplanning.libs.transformations import *

class Path_8_shaped() :

    def __init__(self, circ1=[0.0, 0.0, 10.0], circ2=[150.0, 0.0, 15.0], dLength_path=1.5, plotting=False, steering="") :


        if steering == "ackr":
            r50 = 9.51
            r90 = 5.64
            self.x1 = -15.0
            self.y1 = 0.0            
            self.r1 = r90
            self.x2 = 15.0
            self.y2 = 0.0          
            self.r2 = r50
        elif steering == "skid":
            r50 = 8.0
            r90 = r50*0.55            
            self.x1 = -15.0
            self.y1 = 0.0            
            self.r1 = r90
            self.x2 = 15.0
            self.y2 = 0.0          
            self.r2 = r50
        elif steering == "arti":
            r50 = 9.82
            r90 = 6.14
            self.x1 = -15.0
            self.y1 = 0.0            
            self.r1 = r90
            self.x2 = 15.0
            self.y2 = 0.0          
            self.r2 = r50     
        else:
            self.x1 = circ1[0]
            self.y1 = circ1[1]
            self.r1 = circ1[2]
            self.r2 = circ2[2]
            self.x2 = circ2[0]
            self.y2 = circ2[1]

        self.ds = dLength_path

        self.plotting = plotting

    def get_path(self) :
        
        xT1, yT1, xT2, yT2 = self.calc_tangent()
        line_above_x, line_above_y = self.get_line(xT1, yT1, xT2, yT2)
        line_below_x, line_below_y = self.get_line(xT2, -yT2, xT1, -yT1)

        circ_left_x, circ_left_y = self.get_circ(self.r1, self.x1, self.y1, xT1, yT1, start=-self.r1, end=xT1-self.x1, do_sleep=True)
        circ_right_x, circ_right_y = self.get_circ(self.r2, self.x2, self.y2, xT2, -yT2, start=self.r2, end=xT2-self.x2, do_sleep=True)

        dx_cl_la = line_above_x[0] - circ_left_x[-1]
        dy_cl_la = line_above_y[0] - circ_left_y[-1]

        if self.plotting == True:
            print("circ left end:", circ_left_x[-1], circ_left_y[-1], xT1, yT1)
            print("AAAAAAAAAA:", np.hypot(dx_cl_la, dy_cl_la) )
            print("ds line_above:", np.sqrt( np.diff(line_above_x)**2.0 + np.diff(line_above_y)**2.0 )      )
            print("ds line_below:", np.sqrt( np.diff(line_below_x)**2.0 + np.diff(line_below_y)**2.0 )      )
            print("ds circ_left:", np.sqrt( np.diff(circ_left_x)**2.0 + np.diff(circ_left_y)**2.0 )      )
            print("ds circ_right:", np.sqrt( np.diff(circ_right_x)**2.0 + np.diff(circ_right_y)**2.0 )      )


        if self.plotting == True:
            self.plot_me(circ_left_x, circ_left_y, ['y', 'r', 'm'])
            self.plot_me(line_above_x, line_above_y, ['b', 'g', 'c'])
            self.plot_me(circ_right_x, circ_right_y, ['y', 'r', 'm'])
            self.plot_me(line_below_x, line_below_y, ['k', 'y', 'b'])

        # path_x = np.concatenate( (circ_left_x, line_above_x, circ_right_x, line_below_x))
        # path_y = np.concatenate( (circ_left_y, line_above_y, circ_right_y, line_below_y))

        # path_x = np.concatenate( (line_above_x, circ_right_x, line_below_x, circ_left_x))
        # path_y = np.concatenate( (line_above_y, circ_right_y, line_below_y, circ_left_y))

        line_above_x_2nd = line_above_x[ int(line_above_x.shape[0]//(4/3) ) : ]
        line_above_y_2nd = line_above_y[ int(line_above_x.shape[0]//(4/3) ) : ]
        line_above_x_1st = line_above_x[0 : int(line_above_x.shape[0]//(4/3) )]
        line_above_y_1st = line_above_y[0 : int(line_above_x.shape[0]//(4/3) )]

        path_x = np.concatenate( (line_above_x_2nd, circ_right_x, line_below_x, circ_left_x, line_above_x_1st))
        path_y = np.concatenate( (line_above_y_2nd, circ_right_y, line_below_y, circ_left_y, line_above_y_1st))

        lineAbove_start_x = path_x[0]
        lineAbove_start_y = path_y[0]
        lineAbove_yaw = np.arctan2( path_y[1]-path_y[0], path_x[1]-path_x[0] )

        return path_x, path_y, lineAbove_start_x, lineAbove_start_y, lineAbove_yaw

    def plot_me(self, x, y, c):
        plt.plot(x, y, color=c[0])
        plt.plot(x[0], y[0], marker='*', markersize=10, color=c[1])
        plt.plot(x[-1], y[-1], marker='*', markersize=10, color=c[1])
        plt.plot(x[1], y[1], marker='s', markersize=10, color=c[2])
        plt.plot(x[-2], y[-2], marker='s', markersize=10, color=c[2])

    def get_circ(self, radius, xC, yC, xT, yT, start=None, end=None, do_sleep=False):
        
        # Define direction of x
        dx = 0.0001
        if start > end:
            dx *= -1 
        
        xp, yp = [start], [0.0]
        i = 0
        while xp[i] * np.sign(dx) < end * np.sign(dx) :

            xn = xp[i] + dx
            # print("xn:", xn)
            while True:
                # if do_sleep==True: time.sleep(0.1)
                yn = np.sqrt(radius**2.0 - xn**2.0)
                dx = xn - xp[i]
                dy = yn - yp[i]
                dL = np.hypot(dx, dy) 
                # print(dx)

                error_ds = np.fabs(dL - self.ds) 
                # print("error_ds", error_ds, i)

                if error_ds < 1e-3:
                    break
                elif dL > self.ds:
                    dx = dx / 2
                elif dL < self.ds:
                    dx = dx * 1.5
                xn = xp[i] + dx
                # print(i, xn, yn, dL, "; dx=", dx)
            # print(i, "dx=", dx, dL)
            xp.append(xn)
            yp.append(yn)
            i = i + 1

        xp[-1], yp[-1] = xT-xC, yT-yC

        xm = np.flip(xp[1:-1],axis=0)
        ym = -np.sqrt(radius**2.0 - xm**2.0)
        y = np.append(ym, yp) + yC
        x = np.append(xm, xp) + xC

        return x, y

    def calc_tangent(self) :
        dist__M1_S = self.r1 + self.r2
        dist__M1_M2 = np.sqrt( (self.x2 - self.x1)**2.0 + (self.y2 - self.y1)**2.0  )
        beta = np.arcsin(dist__M1_S/dist__M1_M2)
        phi = np.pi/2 - beta
        xT1 = self.x1 + self.r1 * np.cos(phi) 
        yT1 = self.y1 + self.r1 * np.sin(phi) 
        xT2 = self.x2 - self.r2 * np.cos(phi) 
        yT2 = self.y2 - self.r2 * np.sin(phi) 
        return xT1, yT1, xT2, yT2


    def get_line(self, xT1, yT1, xT2, yT2):

        # Define and start of line        
        dx = xT2- xT1
        dy = yT2- yT1
        length = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx)
        ds = self.ds
        xl_2 = xT1 + ds * np.cos(angle)
        yl_2 = yT1 + ds * np.sin(angle)
        xl_n1 = xT2 - ds * np.cos(angle)
        yl_n1 = yT2 - ds * np.sin(angle)

        # Calc points between start and end of line
        dx = xl_n1 - xl_2
        dy = yl_n1 - yl_2
        length = np.hypot(dx, dy)
        idxs = int(length/ds) + 1
        ds = length / idxs
        xl = np.zeros(idxs+2)
        yl = np.zeros(idxs+2)
        xl[0] = xl_2
        yl[0] = yl_2
        xl[-1] = xl_n1
        yl[-1] = yl_n1
        for i in range(0, xl.size-1):
            # print(i)
            xl[i+1] = xl[i] + ds * np.cos(angle)
            yl[i+1] = yl[i] + ds * np.sin(angle)
        return xl, yl

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
    path_8 = Path_8_shaped(circ1=[0.0, 0.0, 6.0], circ2=[50.0, 0.0, 10.0], dLength_path=0.01, plotting=True)
    # path_8 = Path_8_shaped(plotting=True)

    path_x, path_y, lineAbove_start_x, lineAbove_start_y, lineAbove_yaw = path_8.get_path()

    path_x = np.append(path_x, path_x[0])
    path_y = np.append(path_y, path_y[0])

    # print("path_x", path_x)
    plt.plot(path_x, path_y, '--', Linewidth=2.1)
    plt.axis('equal')

    # print("ds:", np.sqrt( np.diff(path_x)**2.0 + np.diff(path_y)**2.0 )      )

    dx, dy    = _calc_dot(path_x, path_y)
    ddx, ddy  = _calc_ddot(dx, dy)
    yaw       = _calc_yaw(dx, dy)
    curve     = _calc_curve(dx, ddx, dy, ddy)

    fig = plt.figure(2)
    plt.plot(curve)
    

    # print("path_x", np.diff(path_x))
    # print("path_y", np.diff(path_y) )

    plt.show()