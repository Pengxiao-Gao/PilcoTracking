
# PYTHON
import threading
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
import time
from numpy.linalg import inv
from scipy.special import fresnel

# LOCAL

# import sys
# sys.path.insert(0,'/path/to/mod_directory')

from pathplanning.path_handler import Path2D_Handler
from pathplanning.extern.atsushi_sakai.bspline_path import bspline_planning
from model.robot_parameter import Robot_Parameter
from model.simulation_parameter import Sim_Parameter

from pathplanning.libs.extend_transformations import *
from pathplanning.libs.transformations import *

class Line():
    def __init__(self, delta_length=0.01):
        self._ds = delta_length

    def get_path(self, start, desired_length):

        print(start)

        line_length = 0.0

        xLine = [start["x"]] 
        yLine = [start["y"]]

        while line_length < desired_length:

            xLine_new = xLine[-1] + self._ds * np.cos(start["yaw"])
            xLine.append(xLine_new)
            yLine_new = yLine[-1] + self._ds * np.sin(start["yaw"])
            yLine.append(yLine_new)
            line_length += self._ds

        end = start = {"x": xLine[-1], "y": yLine[-1], "yaw": start["yaw"]}
        return np.array(xLine), np.array(yLine), end





class Circle():
    def __init__(self, delta_length=0.01):
        self._ds = delta_length

    def get_path(self, radius, degree, direction="left"):
        r, xn = radius, -radius
        dx = self._ds / 100
        xCircle, yCircle = [xn], [0.0]

        while xn < radius:
            # print("xn:", xn)
            xn1 = xn 
            was_nan = False

            while True: # adjust dx to get a constant distance between the points; break if degree satisfied or end reached
                xn = xn1 + dx   
                # print("dx:", dx)

                y = np.sqrt( r**2.0 - xn**2.0 )
                ds = np.hypot(dx, y - yCircle[-1] )


                if np.isnan(ds) == True:
                    was_nan = True
                    break
                else:
                    e = ds - self._ds
                    yaw = np.arctan2(y, xn)

                    # print(yaw < np.deg2rad(180.0 - degree), np.rad2deg(yaw), 180.0 - degree)

                    if yaw < np.deg2rad(180.0 - degree): break
                    elif np.fabs(e) < 1e-4:
                        break
                    else :
                        dx_ = dx - 0.001 * e
                        if dx_ < 0.0: dx_ = dx + 0.001 * np.fabs(e)

                        dx = dx_
                        # print(dx)
                        # time.sleep(0.1)
            if was_nan == True and yaw > np.deg2rad(180.0 - degree):
                xCircle = xCircle[0:-1]
                yCircle = yCircle[0:-1]
                yaw_des = np.deg2rad(180 - degree)
                xCircle.append( np.cos(yaw_des) * r )
                yCircle.append( np.sin(yaw_des) * r )
                # print("as))))))))))))))))))))")
                break
            elif was_nan == True:
                xCircle.append(radius)
                yCircle.append(0.0)
                # print("asaaaaa")
                break
            elif yaw < np.deg2rad(180 - degree):
                yaw_des = np.deg2rad(180 - degree)
                xCircle.append( np.cos(yaw_des) * r )
                yCircle.append( np.sin(yaw_des) * r )
                # print("as)dddddddddddd))))))")
                break
            else:
                xCircle.append(xn)
                yCircle.append(y)       

        xCircle = np.array(xCircle) + radius
        yCircle = np.array(yCircle)

        circle_tf = xyYaw_to_matrix(0.0, 0.0, np.deg2rad(-90.0))
        for i, v in enumerate(xCircle) :
            # print(i)
            circle = xyYaw_to_matrix(xCircle[i], yCircle[i], 0.0)
            
            circle = circle_tf @ circle
            xCircle[i], yCircle[i] = circle[0,3], circle[1,3]
            
        if direction == "left":
            yCircle = -yCircle

        print(xCircle)
        print(yCircle)
        print(yCircle.size)

        dx, dy = np.diff(xCircle), np.diff(yCircle)
        dL = np.hypot(dx, dy)
        dL_mean = np.mean(dL)
        print(dL)
        print("mean = ", np.mean(dL), np.fabs(dL_mean - self._ds))

        dy = yCircle[-1] - yCircle[-2]
        dx = xCircle[-1] - xCircle[-2]
        yaw_circle = np.arctan2(dy, dx)
        print("yaw_circle no tf:", np.rad2deg(yaw_circle) )


        return xCircle, yCircle



class Clothoid():
    def __init__(self, delta_length=0.01):
        self._ds = delta_length

    def _get_clothoid(self, radius, length_end, no_points):
        l = np.linspace(0.0, length_end, no_points)
        L = l[-1]
        R = radius
        A = np.sqrt(R*L)
        x_, y_ = [], []
        
        for il in l:
            y, x = np.array(fresnel(il)) * A * np.sqrt(np.pi)
            x_.append(x)
            y_.append(y)

        x_ = np.array(x_)
        y_ = np.array(y_)
        return x_, y_

    def get_path(self, radius, direction="left"):
        ds_desired = self._ds
        R = radius
        l_end = 1.0 / R / np.pi 
        no_points = 50
        e_dL = np.finfo('d').max
 
        # Loop to generate Clothoid with constant distance between the points
        while np.fabs(e_dL) > 1e-5:
            xClothoid, yClothoid = self._get_clothoid(radius, l_end, no_points)

            # Pick elements for the distance between the points
            x_, y_ = [xClothoid[0]], [yClothoid[0]]
            iPrv = 0
            for i, v in enumerate(xClothoid):
                dx, dy = xClothoid[i] - xClothoid[iPrv], yClothoid[i] - yClothoid[iPrv]
                dL = np.hypot(dx, dy)
                if dL > ds_desired:
                    x_.append(xClothoid[i] )
                    y_.append(yClothoid[i] )
                    iPrv = i


            # Check BREAK-Condition
            dx, dy = np.diff(x_), np.diff(y_)
            mean_dL =  np.mean(np.hypot(dx, dy))
            e_dL = mean_dL - ds_desired

            # Calc new number of points for clothoid generation
            delta_nopoints = int(np.fabs(e_dL * 1000))
            if delta_nopoints < 1: delta_nopoints = 1
            no_points += delta_nopoints            

        # Change Direction 
        xClothoid, yClothoid = np.array(x_), np.array(y_)
        if direction == "right": yClothoid *= -1.0

        return xClothoid, yClothoid


class SmoothCircle():
    def __init__(self, delta_length=0.01):
        self._ds = delta_length

        self.circle = Circle(delta_length)
        self.clothoid = Clothoid(delta_length)

    def get_path(self, radius, degree, direction="left"):
        xClo, yClo = self.clothoid.get_path(radius, direction)
        xSmooth, ySmooth = xClo, yClo
        print("asd:", xClo.size)


        dy = yClo[-1] - yClo[-2]
        dx = xClo[-1] - xClo[-2]
        yaw_clothoid = np.arctan2(dy, dx)
        print("yaw_clothoid:", np.rad2deg(yaw_clothoid) )


        plt.plot(xClo, yClo, color='green', marker='o', markersize=18)

        xCir, yCir = self.circle.get_path(radius=radius, degree=degree-1, direction=direction)

        print("xCir", xCir.size)

        xSmooth, ySmooth = self._merge_path_parts([xClo, yClo], [xCir, yCir], color_="blue", markersize_=12)

        xSmooth, ySmooth = self._merge_path_parts([xSmooth, ySmooth], [xClo, yClo], color_="yellow", markersize_=6)


        
        dy = ySmooth[-1] - ySmooth[-2]
        dx = xSmooth[-1] - xSmooth[-2]
        yaw_smooth = np.arctan2(dy, dx)
        print("yaw_smooth:", yaw_smooth, np.pi)


        plt.plot(xSmooth, ySmooth, 'r.')
        plt.show()



        return xSmooth, ySmooth

    def _merge_path_parts(self, parent, child, color_="blue", markersize_=12):
        xMerge, yMerge = parent[0], parent[1]
        
        # Transformation Matrix
        dy = yMerge[-1] - yMerge[-2]
        dx = xMerge[-1] - xMerge[-2]
        yaw = np.arctan2(dy, dx)
        parentEnd = xyYaw_to_matrix(xMerge[-1], yMerge[-1], yaw)

        child_tf_x, child_tf_y = [], []

        # Transform Child "into" Parent and add to parent
        for i, v in enumerate(child[0]) :
            if i==0: continue # Skip the first element so that point is not twice in parent
            xChild = child[0][i]
            yChild = child[1][i]
            child_tfMat = xyYaw_to_matrix(xChild, yChild, 0.0)
            child_tfMat = parentEnd @ child_tfMat
            child_tf_x.append(child_tfMat[0,3])
            child_tf_y.append(child_tfMat[1,3])

        plt.plot(child_tf_x, child_tf_y, color=color_, marker='o', markersize=markersize_)

        xMerge = np.append(xMerge, child_tf_x )
        yMerge = np.append(yMerge, child_tf_y )
        return xMerge, yMerge


def merge_path_parts(in1, in2) :
    
    xM = in1[0]
    yM = in1[1]

    yaw = np.arctan2( yM[-1] - yM[-2], xM[-1] - xM[-2] )

    lineEnd = xyYaw_to_matrix(xM[-1], yM[-1], yaw)

    for i, v in enumerate(in2[0]) :
        circle = xyYaw_to_matrix(in2[0][i], in2[1][i], 0.0)
        
        circle_tf = lineEnd @ circle
        xM = np.append(xM, circle_tf[0,3] )
        yM = np.append(yM, circle_tf[1,3])

    return xM, yM

if __name__ == "__main__":

    # start = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    # line = Line(0.01)
    # xL, yL, start = line.get_path(start, 10.0)
    # xL, yL, end = line.get_path(start, 10.0)

    # circle = Circle(0.01)
    # xC, yC = circle.get_path(radius=6.0, degree=190.0, direction="right")
    # xC, yC = circle.get_path(radius=1.0, degree=170.0, direction="left")

    # xM, yM = merge_path_parts([xL, yL], [[], []])
    # xM, yM = merge_path_parts([xL, yL], [xC, yC])

    # dx, dy =  np.diff(xC), np.diff(yC) 
    # dL = np.hypot(dy, dx)
    # print(dL)

    # clothoid = Clothoid(0.01)
    # xCloth, yCloth = clothoid.get_path(radius = 5.0, direction="right")


    smCircle = SmoothCircle(0.01)
    xSmooth, ySmooth = smCircle.get_path(radius=5.0, degree=180, direction="left")


    dx = np.diff(xSmooth)
    dy = np.diff(ySmooth)
    ddx = np.diff(dx)
    ddy = np.diff(dy)


    dx = dx[0:-1]
    dy = dy[0:-1]
    print(dx.size)
    print(ddx.size)
    kappa = (dx * ddy - ddx * dy) / (dx**2.0 + dy **2.0)**(3.0/2.0)
    print("RADIUS:\n", 1/kappa)

    ax1 = plt.subplot(211)
    ax1.plot(xSmooth, ySmooth)

    ax2 = plt.subplot(212)
    ax2.plot(kappa, color='b', marker='.')

    # ax = plt.gca()
    # ax.set_aspect("equal")

    # ax.set_xlim(-3.0, 3.0)
    # ax.set_ylim(0.0, 3.0)

    plt.show()