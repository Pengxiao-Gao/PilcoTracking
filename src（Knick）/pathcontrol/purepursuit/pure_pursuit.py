'''
    Path-Tracking with Pure Pursuit
    Author: Ilja Stasewisch, Date: 2020-03-31
'''

import numpy as np
from numpy.linalg import inv
from threading import Thread, Lock

from model.robot_parameter import Robot_Parameter
from pathplanning.libs.transformations import *
from pathplanning.libs.extend_transformations import *

from pathplanning.libs.extend_transformations import xyYaw_to_matrix



class PurePursuit:
    
    def __init__(self, params, type="Ackermann") :
        print('PurePursuit: Init')
        self.params = params
        self.index_lookahead = -1
        self.type = type

    def execute(self, diff_mat4x4, l_ah=None, vel=1.0) : # l10n = localization pose as 4x4-matrix
        
        type = self.params['type']

        if type == 'skid':
            l_track = self.params['track']
            l_o = self.params['l_o']
            alpha = np.arctan2(diff_mat4x4[1,3], diff_mat4x4[0,3]) # error angle
            if l_ah == None:
                l_ah = self.params['l_ah']
            dvel = (vel * l_track * 2 * np.sin(alpha)) / (l_ah - 2*l_o*np.cos(alpha))
            return dvel

        elif type == 'articulated':
            l_w = self.params['l_w']
            l_f = self.params['l_front']
            l_r = self.params['l_rear']
            l_o = self.params['l_o']
            if l_ah == None:
                l_ah = self.params['l_ah']

            alpha = np.arctan2(diff_mat4x4[1,3], diff_mat4x4[0,3]) # error angle
            if np.abs(alpha) > np.deg2rad(0.1):
                radius_rear = (l_ah - 2*l_o * np.cos(alpha)) / (2*np.sin(alpha))
                angle_a = np.arctan(l_r * np.sin(alpha) / (l_ah/2.0 - l_o * np.cos(alpha)) )
                sign_ = np.sign(diff_mat4x4[1,3])
                radius_front = sign_ * np.sqrt(radius_rear**2 + l_r**2 - l_f**2)
                angle_b = np.arctan(l_f / radius_front)
                # print("angle_b:", angle_b)
                steer = angle_a + angle_b
                return steer
            else:
                return 0.0

        elif type == 'acker':
            l_w = self.params['l_w']
            l_o = self.params['l_o']
            alpha = np.arctan2(diff_mat4x4[1,3], diff_mat4x4[0,3]) # error angle

            if l_ah == None:
                l_ah = self.params['l_ah']

            return np.arctan(l_w * np.sin(alpha) / (l_ah/2.0 - l_o * np.cos(alpha)) )


