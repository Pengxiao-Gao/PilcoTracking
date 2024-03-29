import numpy as np
from numpy.linalg import inv
import math
from math import sin, cos
import sys
from threading import Thread, Lock
import matplotlib.pyplot as plt
from collections import namedtuple
import copy
from time import sleep

# LOCAL
from model.robot_parameter import Robot_Parameter
from pathplanning.libs.transformations import *
from pathplanning.libs.extend_transformations import *
from pathplanning.libs.extend_transformations import xyYaw_to_matrix



'''##########################################################################
##############    PATH class for handling control-path    ###################
##########################################################################'''


class DrivenPath() :
    def __init__(self):
        self.xo = []
        self.yo = []
        self.yaw = [] 
        self.steer = []

        self.arclength = [0.0]

    def set_robotState(self, robot_state):
        # if len(self.xo) < 6: 
            # print("DrivenPath - set_robotState:", robot_state)
        self.xo.append(robot_state[0])
        self.yo.append(robot_state[1])
        self.yaw.append(robot_state[2])

        if len(self.xo) > 1:
            ds = np.hypot(self.xo[-1] - self.xo[-2], self.yo[-1] - self.yo[-2])
            self.arclength.append( self.arclength[-1] + ds)


    def get_robotState(self, robot_state):
        return self.arclength

class Path2D_Handler_knss :  

    def __init__(self, x, y, robot_state, robot_param, vel = 1.0, length_path_end=5.0) :
        # Class Variables
        self._mutex = Lock()

        # Arguments
        self.x = x
        self.y = y
        self.wheelbase = robot_param.wheelbase
        self.offset = robot_param.length_offset
        self.min_length_goal_reached = length_path_end
        self.robot_state = robot_state

        # Calculate related path element/stuff/foo
        self.dx, self.dy    = self._calc_dot()
        self.ddx, self.ddy  = self._calc_ddot()
        self.yaw            = self._calc_yaw(self.dx, self.dy, vel)
        self.curve          = self._calc_curve(self.dx, self.ddx, self.dy, self.ddy)
        self.steer          = self._calc_steer(self.curve, vel)
        self.x_offset, self.y_offset            = self._calc_offset_path(self.x, self.y, self.yaw)
        self.arclength, self.ds, self.ds_mean   = self._calc_arclength(self.dx, self.dy)
        self.steerRate      = self._calc_steerRate(self.steer, self.ds, vel)

        # Set Goal
        self.lateral_error = np.full(shape=(len(self.arclength),), fill_value=np.NAN) 
        self.goal = xyYaw_to_matrix(self.x_offset[-1], self.y_offset[-1], self.yaw[-1])
        self.goalInv = inv(self.goal)
        self.index_nn = -1
        self.index_nn_lh = -1
        self.meters_around_last_nnidx = 0.3 * vel # Multiplication with the desired/mean velocity for normalization

        self.drivenPath = DrivenPath( )
        self.referencePath = None
        self.predictionPath__dlqt_mpc = None

    def reset__index_nn(self):
        self.index_nn = -1
        self.index_nn_lh = -1

    ######### EXTERN GET FUNCTION ########
    def get_nnPosition(self):
        with self._mutex:
            if self.index_nn < 0 :
                return [self.x_offset[0], self.y_offset[0]]
            else :
                return [self.x_offset[self.index_nn], self.y_offset[self.index_nn]]

    # def get_referencePath(self, path_length, delta_length, prediction_steps):
    def get_referencePath(self, *argv):
        if len(argv) == 0:
            return self.referencePath
        elif len(argv) == 3:
            path_length = argv[0]
            delta_length = argv[1]
            prediction_steps = argv[2]
            refPath = self._determine_referencePath(path_length, delta_length, prediction_steps)
            with self._mutex:
                self.referencePath = refPath
            return refPath
        else:
            print("Error in Path_Handler.get_referencePath(): Too less or many arguments!")

    def get_predictionPath__dlqt_mpc(self):
        with self._mutex:
            return self.predictionPath__dlqt_mpc

    def get_lookaheadPose(self):
        with self._mutex:
            if self.index_nn_lh < 0 :
                return [self.x_offset[0], self.y_offset[0]]
            else :
                return [self.x_offset[self.index_nn_lh], self.y_offset[self.index_nn_lh], self.yaw[self.index_nn_lh] ]

    def get_lookaheadSteer(self):
        with self._mutex:
            if self.index_nn_lh < 0 :
                return [self.steer[0], self.steer[0]]
            else :
                return self.steer[self.index_nn_lh]

    def get_lookaheadPosition(self):
        with self._mutex:
            if self.index_nn_lh < 0 :
                return [self.x_offset[0], self.y_offset[0]]
            else :
                return [self.x_offset[self.index_nn_lh], self.y_offset[self.index_nn_lh]]

    def get_feedforwardSteerRate(self):
        with self._mutex:
            return self.steerRate[self.index_nn]
    def get_feedforwardSteer(self):
        with self._mutex:
            return self.steer[self.index_nn]
    def get_offsetPath(self, index=None) :
        with self._mutex:
            if index is None:
                return self.x_offset, self.y_offset, self.yaw
            else:
                return self.x_offset[index], self.y_offset[index], self.yaw[index]
    def get_path(self) :
        with self._mutex:
            return self.x, self.y, self.yaw
    def get_arclength(self) :
        with self._mutex:
            return self.arclength
    def get_lateralError(self) :
        with self._mutex:
            return self.lateral_error
    def get_drivenPath(self) :
        with self._mutex:
            return self.drivenPath.xo, self.drivenPath.yo, self.drivenPath.yaw         
    def get_drivenArcLength(self) :
        with self._mutex:
            return self.drivenPath.arclength        

    def plot_me(self, cmd="now"):
        with self._mutex:
            fig, ax = plt.subplots(nrows=2, ncols=2)
            fig.canvas.set_window_title("Path Handler - Path Information")
            
            ax[0,0].plot(self.x_offset, self.y_offset, color='b')
            ax[0,0].plot(self.x, self.y, color='k')

            
            ax[0,1].plot( self.arclength[0:-1], self.curve )
            
            if cmd == "now":
                plt.show()


    ####################################################################
    ######################### EXECUTE FUNCTION #########################
    def execute(self, look_ahead=None, purepursuit=False) :
        with self._mutex:
            l10nOffset = self.robot_state.get_l10nOffset()

            # Nearest Index for Visualization
            index_nn, _, nearestPath_matrix = self._get_indexNearestPosition(l10nOffset)
            # print("index_nn:", index_nn)
            self._update_index_nn(index_nn)
            # Error and Driven Path
            diff_error_matrix_nn = self._get_errorMatrix(nearestPath_matrix, l10nOffset)
            self._update_lateralError(index_nn, diff_error_matrix_nn[1,3])
            self._update_drivenPath(l10nOffset)

            # Nearest Index for  Control
            if purepursuit == False:
                index_nn_lh, _, des_lah_pose_matrix = self._get_indexNearestPosition(l10nOffset, look_ahead)
                diff_error_matrix_lh = self._get_errorMatrix(des_lah_pose_matrix, l10nOffset)
                self._update_indexNnLookahead(index_nn_lh)
            else:
                index_nn_lh, _, des_lah_pose_matrix = self._get_indexNearestPosition_byCircle(l10nOffset, look_ahead)
                diff_error_matrix_lh = self._get_errorMatrix(des_lah_pose_matrix, l10nOffset, in_robotframe=True)
                self._update_indexNnLookahead(index_nn_lh)
                

            # Behind Goal
            velocity = self.robot_state.get_velocity()
            behind_goal = self._is_behindGoal(l10nOffset, velocity)

        return diff_error_matrix_lh, behind_goal, index_nn, index_nn_lh
    ####################################################################
    ####################################################################


    def _is_behindGoal(self, robot_state, velocity):

        if self.arclength[-1]*0.95 > self.drivenPath.arclength[-1]:
            return False

        xo, yo, yaw = robot_state[0], robot_state[1], robot_state[2]
        actual = xyYaw_to_matrix(xo, yo, yaw)
        goal2actual = self.goalInv @ actual
        length_to_goal = np.hypot( goal2actual[0,3], goal2actual[1,3] )

        if length_to_goal > self.min_length_goal_reached :
            return False
        else :
            
            dx = goal2actual[0, 3]

            if velocity > 0.0 and dx > 0.0:
                return True
            elif velocity < 0.0 and dx < 0.0:
                return True 
            else :
                False               


    def _get_errorMatrix(self, desired, l10nOffset, in_robotframe=False) :
        xo, yo, yaw = l10nOffset[0], l10nOffset[1], l10nOffset[2]
        actual = xyYaw_to_matrix(xo, yo, yaw)

        if in_robotframe == False:
            diff_tf = inv(desired) @ actual
        else: # in Robot-Frame
            diff_tf = inv(actual) @ desired
            pass

        return diff_tf

    def _get_indexNearestPosition_byCircle(self, l10nOffset, lookahead_distance):

        idx_around_last_nn = int( self.meters_around_last_nnidx / np.fabs(self.ds_mean)+1  )
        # Define Start-Idx for the Loop
        start_index = self.index_nn_lh - idx_around_last_nn
        if start_index < 0 :
            start_index = 0
        # Define End-Idx for the Loop
        if self.index_nn_lh < 0: # self.index_lookahead is not inited
            end_index = len(self.x)
        else: 
            end_index = self.index_nn_lh + idx_around_last_nn
        if end_index > len(self.x) :
            end_index = len(self.x)

        # GET DESIRED-POSE
        dist_to_lookahead = float("inf")
        path_desired_idx = self.index_nn_lh
        actual_tfMatrix = xyYaw_to_matrix(l10nOffset[0], l10nOffset[1], l10nOffset[2])
        for i in range(start_index, end_index) :           
            iPath_tfMat = xyYaw_to_matrix(self.x_offset[i], self.y_offset[i], self.yaw[i])
            dist_offset2path = getLength2DSign(actual_tfMatrix, iPath_tfMat)
            if dist_offset2path < 0.0 :
                continue
            else :
                eDist = dist_offset2path - lookahead_distance
                
                if eDist <= 0.0 :
                    continue
                elif eDist < dist_to_lookahead :
                    dist_to_lookahead = eDist
                    path_desired_idx = i
                    if np.fabs(dist_to_lookahead) <  np.fabs(self.ds_mean):
                        break

       
        desired_pose_matrix = xyYaw_to_matrix(self.x_offset[path_desired_idx], self.y_offset[path_desired_idx], self.yaw[path_desired_idx])
        return path_desired_idx, dist_offset2path, desired_pose_matrix


    def _get_indexNearestPosition(self, robot_state, look_ahead=None):

        robot_state_lh = [robot_state[0], robot_state[1], robot_state[2] ]
        # print("robot_state_lh:", robot_state_lh)
        # sleep(11.1)
        if look_ahead is not None:
            # print("use look ahead")
            # print("before lh:", robot_state[0] )
            robot_state_lh[0] = robot_state_lh[0] + look_ahead * cos(robot_state[2])
            robot_state_lh[1] = robot_state_lh[1] + look_ahead * sin(robot_state[2])
            # print("after lh:", robot_state[0] )

            last_nn_idx = self.index_nn_lh

        else :
            last_nn_idx = self.index_nn


        # Define Start-Index and End-Index for the Loop       
        start_index = last_nn_idx - int(self.meters_around_last_nnidx /  np.fabs(self.ds_mean)+1 )
        if start_index < 0 : start_index = 0

        if last_nn_idx < 0: # at init, search in whole path
            end_index = len(self.x)-10
        else: 
            end_index = last_nn_idx + int( self.meters_around_last_nnidx / np.fabs(self.ds_mean)+1  )
            if end_index > len(self.x) : end_index = len(self.x)

        # Get index for neatest euclidean distance
        dL = float('Inf')
        index = 0
        xo, yo, yaw = robot_state_lh[0], robot_state_lh[1], robot_state_lh[2]
        # print("start_index: ", start_index)
        # print("end_index: ", end_index)
        # print("self.meters_around_last_nnidx: ", self.meters_around_last_nnidx)
        # print("int( self.meters_around_last_nnidx / np.fabs(self.ds_mean) = ", int( self.meters_around_last_nnidx / np.fabs(self.ds_mean) +1) )
        for i in range(start_index, end_index) :
            dL_ = np.hypot(self.x_offset[i] - xo, self.y_offset[i] - yo)
            if dL_ < dL :
                index = i
                dL = dL_

        # xyYaw = [self.x[index], self.y[index], self.yaw[index] ]
        pose_matrix = xyYaw_to_matrix(self.x_offset[index], self.y_offset[index], self.yaw[index])

        

        return index, dL, pose_matrix

    def _determine_referencePath(self, path_length, delta_length, prediction_steps):
        with self._mutex:
            idx = copy.deepcopy(self.index_nn)
            # print("idx:", idx, self.x[idx])
        
            iLength, idL = 0.0, 0.0
            x, y, yaw, steer, steerRate = [], [], [], [], []

            iter = 0
            while iLength <= path_length or iter < prediction_steps:
                # Path Determination
                if idx >= len(self.ds): idx = len(self.ds) - 1

                idL += self.ds[idx]
                if idL > delta_length:
                    x.append( self.x_offset[idx] )
                    y.append( self.y_offset[idx] )
                    yaw.append( self.yaw[idx] )
                    steer.append( self.steer[idx] )
                    steerRate.append( self.steerRate[idx] )
                    idL = 0.0
                    iter += 1

                # Abort Criteria
                iLength += self.ds[idx]
                idx += 1

            ret = {"x": x, "y": y, "yaw": yaw, "steer": steer, "steerRate": steerRate}
            return ret


    ######################################################################################
    ########################## Set Functions #############################################
    def set_predictionPath__dlqt_mpc(self, path):
        with self._mutex:
            self.predictionPath__dlqt_mpc = path


    ######################################################################################
    ########################## UPDATE CLASS VARIABLES ####################################
    def _update_index_nn(self, index_nearest_neighbour):
        self.index_nn = index_nearest_neighbour
    def _update_indexNnLookahead(self, index_nn_lookahead):
        self.index_nn_lh = index_nn_lookahead
    def _update_lateralError(self, index, y_error):
        if self.lateral_error[index] is None    or    y_error < self.lateral_error[index]    or     np.isnan(self.lateral_error[index]) == True:
            self.lateral_error[index] = y_error


        # test = [x for x in self.lateral_error if x is not None]
        # test = np.array(test)
        # print("max lateral error = ", np.fabs(test).max() )
    def _update_drivenPath(self,robot_state):
        self.drivenPath.set_robotState(robot_state)

    ######################################################################################
    ########################## CALCULACTE PATH FUNCTIONS #################################
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
