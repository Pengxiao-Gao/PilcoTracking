
# PYTHON
import threading
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys

# LOCAL
from pathplanning.path_handler import Path2D_Handler
from pathplanning.extern.atsushi_sakai.bspline_path import bspline_planning
from model.robot_parameter import Robot_Parameter
from model.simulation_parameter import Sim_Parameter

from pathplanning.libs.extend_transformations import *
from pathplanning.libs.transformations import *

class Cubic_Spline() :

    def __init__(self) :

        # Class Variables
        self.path = None
        self.path_ds = None
        self.rest_nearest_pose()
        self.meters_around_last_nnidx = 1.0
        self.goal_tfMat = None
        self.pathPlus = None
        self.path_length = None
        
    def rest_nearest_pose(self) :
        self.index_nn = -1

    def nearest_pose(self, state, increase_index):

        dL = float('Inf')
        index = 0

        # Define Start-Idx for the Loop
        # print("self.path_ds = ", self.path_ds)
        start_index = self.index_nn - int(self.meters_around_last_nnidx /  np.fabs(self.path_ds) )
        if start_index < 0 : start_index = 0

        # Define End-Idx for the Loop
        if self.index_nn < 0: # self.index_lookahead is not inited
            end_index = self.path.shape[0]
        else: 
            end_index = self.index_nn + int( self.meters_around_last_nnidx / np.fabs(self.path_ds)  )
            if end_index > self.path.shape[0] : end_index = self.path.shape[0]


        x_now = state.x
        y_now = state.y

        ''' Get index for neatest euclidean distance '''
        for i in range(start_index, end_index) :
            px = self.path[i, 0]
            py = self.path[i, 1]
            dL_ = np.hypot(px - x_now, py - y_now)
            if dL_ < dL :
                index = i
                dL = dL_

        if index+1 < self.path.shape[0]: index += 1
        self.index_nn = index


        xyYaw = [self.pathPlus.x[index], self.pathPlus.y[index], self.pathPlus.yaw[index] ]

        return index, dL, xyYaw

    def get_path_ds(self) :
        return self.path_ds

    def get_error(self, path_index, xyYaw=None, state=None) :

        # Define the Tf-Matrices
        if xyYaw is not None :
            actual = xyYaw_to_matrix(xyYaw[0], xyYaw[1], xyYaw[2])
        elif state is not None :
            actual = xyYaw_to_matrix(state.x, state.y, state.yaw)
        des_x = self.pathPlus.x[path_index]
        des_y = self.pathPlus.y[path_index]
        des_yaw = self.pathPlus.yaw[path_index]
        desired = xyYaw_to_matrix(des_x, des_y, des_yaw)

        # Calc diff-Tf
        actual = np.matrix(actual)
        desired = np.matrix(desired)
        diff_tf = actual.I * desired
        error_x = diff_tf[0,3]
        error_y = diff_tf[1,3]
        _,_, error_yaw = matrix_to_euler(diff_tf)
        return [error_x, error_y, error_yaw]

    def get_goal_tfMat(self, path_plus) :
        goal_tfMat = xyYaw_to_matrix(path_plus.x[-1], path_plus.y[-1], path_plus.yaw[-1])
        return goal_tfMat


    def path_to_path2d(self, path) :
        pathPlus = Path2D_Handler(x=path[:,0], y=path[:,1], wheelbase=self.rob_params.wheelbase, offset=self.rob_params.length_offset, velocity=self.rob_params.path_velocity)
        return pathPlus

    ######################################################
    ### METHOD TO GENERATE A CUBIC SPLINE ################
    ######################################################
    def calc(self, dLength, x_points=None, y_points=None) :
        # USING dLength=ds=1m/s as nominal difference in arclength as normalization for the curvature and to scale afterwards the feed forward steering command
        
        print("Cubic Spline:")

        if x_points is None :
            x_points = [0.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0]
        if y_points is None :
            y_points = [0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 1.0, -2.0, 2.0, -2.0, 5.0, -25.0, -10.0]

        # Define Spline
        dx_ = np.diff(np.asarray(x_points))
        dy_ = np.diff(np.asarray(y_points))
        ds = [ np.sqrt(idx**2.0 + idy**2.0) for (idx, idy) in zip(dx_, dy_)]
        length_waypoints = np.sum(ds)
        dLength_ = dLength / 100
        print("\t dLength for generating = ", dLength_)
        no_points = int(math.ceil(length_waypoints / dLength_))
        bx, by = bspline_planning(np.asarray(x_points), np.asarray(y_points), no_points)
        bx = np.array(bx)
        by = np.array(by)

        # Reduce Spline points to desired delta length
        dbx_ = np.diff(bx)
        dby_ = np.diff(by)
        ds = [ np.sqrt(idx**2.0 + idy**2.0) for (idx, idy) in zip(dbx_, dby_)]
        bx_constant_ds = np.array( bx[0] )
        by_constant_ds = np.array( by[0] )
        ds_desired = dLength
        print("\t ds_desired = ", ds_desired)
        ds_local = 0.0
        for i, value in enumerate(ds) :
            ds_local += value
            if ds_local > ds_desired : 
                bx_constant_ds = np.append(bx_constant_ds, bx[i])
                by_constant_ds = np.append(by_constant_ds, by[i])
                ds_local = 0.0

        spline_x = bx_constant_ds
        spline_y = by_constant_ds

        # Calculate Path-dLength and Path-Length
        dx_ = np.diff(np.asarray(spline_x))
        dy_ = np.diff(np.asarray(spline_y))
        ds = [ np.sqrt(idx**2.0 + idy**2.0) for (idx, idy) in zip(dx_, dy_)]
        path_length = np.sum(ds)
        mean_path_ds  = np.mean(ds)
        print("\t path length = ", path_length)
        print("\t mean dLength = ", mean_path_ds)

        # Set Path as Np-Array
        path = np.vstack( (spline_x, spline_y) )
        path = np.swapaxes(path, 1, 0)

        return spline_x, spline_y



    def goal_reached(self, state=None, xyYaw=None, velocity=0.0) :

        if xyYaw is not None :
            now = xyYaw_to_matrix(xyYaw[0], xyYaw[1], xyYaw[2])
        elif state is not None :
            now = xyYaw_to_matrix(state.x, state.y, state.yaw)
        else: 
            # print "goal_reached WRONG argument type"
            return False

        _ , dL, _ = self.nearest_pose(state, increase_index = False)

        if dL > 1.0 : return False

        goal = self.goal_tfMat
        now = np.matrix(now)
        goal = np.matrix(goal)
        # print "goal = ", goal
        diff_tf = goal.I * now

        if diff_tf[0, 3] > 0.0 and velocity >= 0.0 : return True
        elif diff_tf[0, 3] < 0.0 and velocity < 0.0 : return True     
        else : return False


   

if __name__ == "__main__":
    robot_parameter = Robot_Parameter(wheelbase=3.0, length_offset=-1.0, max_steerAngle=0.6, T_steer=0.375, min_velocity=1.0, max_velocity=1.0)
    sim_parameter = Sim_Parameter(Ts_ctrl=1e-2, Ts_sim=1e-2)

    path = Cubic_Spline()
    x, y = path.calc(dLength=0.1)
    print("", type(path.path))
    plt.plot(x,y)
    plt.show()