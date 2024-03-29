import numpy as np
import math


def render_articulated(robot_params, robot_state, axis):
        
     # Vehicle parameters
    WIDTH = robot_params.wheelbase * 0.75  # [m]
    
    WHEEL_LEN = robot_params.wheelbase * 0.15  # [m]
    WHEEL_WIDTH = robot_params.wheelbase * 0.07  # [m]
    TREAD = WIDTH * 0.25  # [m]
    WB = robot_params.wheelbase  # [m]
    offset = robot_params.length_offset

    LENGTH_REAR = robot_params.length_rear
    LENGTH_FRONT = robot_params.length_front

    cabcolor="-k"
    truckcolor="-k"

    x = robot_state[0]
    y =robot_state[1]
    yaw = robot_state[2]
    steer = robot_state[3]
    xo =  robot_state[4]
    yo =  robot_state[5]

    yaw_r = yaw
    yaw_f = yaw + steer
    yaw_f = mytf.getYawRightRange(yaw_f)

    # Plot Offset and Rear Axle Point
    # axis.arrow( x, y, 1.0, 2.0 )
    # print(xo, yo)
    axis.plot( xo, yo, marker='x', markersize=10, color='r' )
    # axis.plot(np.array(x), np.array(y), marker='x', markersize=3, color='k')

    LENGTH = robot_params.length_rear*1.35 # [m]
    BACKTOWHEEL = robot_params.wheelbase*0.25  # [m]
    outline_rear = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                    [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])
    LENGTH = robot_params.length_front*1.0 # [m]
    BACKTOWHEEL = robot_params.wheelbase*0.2  # [m]
    outline_front = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                    [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])
    
#     print("outline_front:", outline_front)

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                        [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])
    rr_wheel = np.copy(fr_wheel)
    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1
    Rot1 = np.array([[math.cos(yaw_r), math.sin(yaw_r)],
                    [-math.sin(yaw_r), math.cos(yaw_r)]])

    Rot2 = np.array([[math.cos(yaw_f), math.sin(yaw_f)],
                    [-math.sin(yaw_f), math.cos(yaw_f)]])

    steerRot = np.array([[math.cos(steer), math.sin(steer)],
                    [-math.sin(steer), math.cos(steer)]])

    fr_wheel[0, :] += LENGTH_REAR
    fl_wheel[0, :] += LENGTH_REAR
    fr_wheel = (fr_wheel.T.dot(steerRot)).T
    fl_wheel = (fl_wheel.T.dot(steerRot)).T
    fr_wheel[0, :] += LENGTH_FRONT
    fl_wheel[0, :] += LENGTH_FRONT
    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    



#     fr_wheel = (fr_wheel.T.dot(Rot1)).T
#     fl_wheel = (fl_wheel.T.dot(Rot1)).T
    
    outline_rear = (outline_rear.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T



    outline_rear[0, :] += x
    outline_rear[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    # Plot Vehicle
    axis.plot(np.array(outline_rear[0, :]),
            np.array(outline_rear[1, :]), cabcolor)
        
    axis.plot(np.array(fr_wheel[0, :]),
            np.array(fr_wheel[1, :]), truckcolor)
    axis.plot(np.array(rr_wheel[0, :]),
            np.array(rr_wheel[1, :]), truckcolor)
    axis.plot(np.array(fl_wheel[0, :]),
            np.array(fl_wheel[1, :]), 'k')
    axis.plot(np.array(rl_wheel[0, :]),
            np.array(rl_wheel[1, :]), truckcolor)

    rear_long_axis = np.array([[0.0, LENGTH_REAR], [0.0, 0.0]])
    rear_long_axis = (rear_long_axis.T.dot(Rot1)).T
    rear_long_axis[0, :] += x
    rear_long_axis[1, :] += y
    axis.plot( rear_long_axis[0, :], rear_long_axis[1, :], color='k')

#     print("LENGTH_FRONT:", LENGTH_FRONT, LENGTH_REAR)
    front_long_axis = np.array([[0.0, 0.0+LENGTH_FRONT], [0.0, 0.0]])
    front_long_axis = (front_long_axis.T.dot(steerRot)).T
    front_long_axis[0, :] += 0.0+ LENGTH_REAR
    front_long_axis = (front_long_axis.T.dot(Rot1)).T
    front_long_axis[0, :] += x
    front_long_axis[1, :] += y
    axis.plot( front_long_axis[0, :], front_long_axis[1, :], color='k')

    outline_front[0,:] += LENGTH_REAR
    outline_front = (outline_front.T.dot(steerRot)).T
    outline_front[0,:] += LENGTH_FRONT
    outline_front = (outline_front.T.dot(Rot1)).T
#     outline_front[0,:] += LENGTH_REAR
    outline_front[0, :] += x
    outline_front[1, :] += y
    axis.plot(np.array(outline_front[0, :]),
            np.array(outline_front[1, :]), cabcolor)    
 
    coupling_point = np.array([[LENGTH_REAR], [0.0]])
    coupling_point = (coupling_point.T.dot(Rot1)).T
    coupling_point[0, :] += x
    coupling_point[1, :] += y
    axis.plot( coupling_point[0], coupling_point[1], color='k', marker='o', markersize=5)



def render_skid(robot_params, robot_state, axis):
    
    # Vehicle parameters
    LENGTH = robot_params.wheelbase * 1.75 # [m]
    WIDTH = robot_params.track *1.0  # [m]
    WHEEL_LEN = robot_params.track * 0.25  # [m]
    WHEEL_WIDTH = robot_params.track * 0.15  # [m]
    WIDTH -= 2*WHEEL_WIDTH
    TREAD = robot_params.track*0.52 # [m]
    WB = robot_params.wheelbase  # [m]
    offset = robot_params.length_offset

    cabcolor="-k"
    truckcolor="-k"

    x = robot_state[0]
    y =robot_state[1]
    yaw = robot_state[2]
    xo =  robot_state[4]
    yo =  robot_state[5]
    v = robot_state[6]
    omega = robot_state[8]

    vR = omega*robot_params.track/2
    vL = -omega*robot_params.track/2

    # Plot Offset and Rear Axle Point
    # axis.arrow( x, y, 1.0, 2.0 )
    # print(xo, yo)
    axis.plot( xo, yo, marker='x', markersize=6, color='r' )
    # axis.plot(np.array(x), np.array(y), marker='x', markersize=3, color='k')

    outline = np.array([[-LENGTH/2, LENGTH/2, LENGTH/2, -LENGTH/2, -LENGTH/2],
                    [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    triangle = np.array([[LENGTH/4, LENGTH/2, LENGTH/4, LENGTH/4],
                         [WIDTH/2,  0,        -WIDTH/2, WIDTH/2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                        [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    fr_wheel[0, :] += WB/2
    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rr_wheel = np.copy(fr_wheel)
    rr_wheel[0, :] *= -1.0
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1.0

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                    [-math.sin(yaw), math.cos(yaw)]])
    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T
    outline = (outline.T.dot(Rot1)).T
    triangle = (triangle.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T
    outline[0, :] += x
    outline[1, :] += y
    triangle[0, :] += x
    triangle[1, :] += y    
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    # Plot Vehicle
    axis.plot(np.array(outline[0, :]),
            np.array(outline[1, :]), cabcolor)
    axis.plot(np.array(fr_wheel[0, :]),
            np.array(fr_wheel[1, :]), truckcolor)
    axis.plot(np.array(rr_wheel[0, :]),
            np.array(rr_wheel[1, :]), truckcolor)
    axis.plot(np.array(fl_wheel[0, :]),
            np.array(fl_wheel[1, :]), truckcolor)
    axis.plot(np.array(rl_wheel[0, :]),
            np.array(rl_wheel[1, :]), truckcolor)
    axis.plot(np.array(triangle[0, :]),
            np.array(triangle[1, :]), truckcolor)
    axis.plot(x,y, color=[1, 1, 0], marker='*', markersize=10)
 
    vL_norm = vL / robot_params.max_velocity * 100
    vR_norm = vR / robot_params.max_velocity * 100

#     vL_norm = 1

    arrow_params = {'length_includes_head': True, 'shape': "full",
                    'head_starts_at_zero': True}

    arrow_left_start = np.array( [[0], [TREAD]])
    arrow_left_end = np.array( [[ (LENGTH/2)*vL_norm], [TREAD]])
    arrow_left_diff = arrow_left_end - arrow_left_start 
    arrow_left_start = (arrow_left_start.T.dot(Rot1)).T
    arrow_left_start[0, :] += x 
    arrow_left_start[1, :] += y
    arrow_left_diff = (arrow_left_diff.T.dot(Rot1)).T

    arrow_right_start = np.array( [[0], [-TREAD]])
    arrow_right_end = np.array( [[ (LENGTH/2)*vR_norm], [-TREAD]])
    arrow_right_diff = arrow_right_end - arrow_right_start 
    arrow_right_start = (arrow_right_start.T.dot(Rot1)).T
    arrow_right_start[0, :] += x 
    arrow_right_start[1, :] += y
    arrow_right_diff = (arrow_right_diff.T.dot(Rot1)).T


    axis.arrow(0, 0, dx=1.0, dy=1.0,
                  color='r', width = WHEEL_WIDTH/4, head_width= WHEEL_WIDTH, **arrow_params)

    if np.sum(np.abs(arrow_left_diff)) > 0.0:
        axis.arrow(arrow_left_start[0,0], arrow_left_start[1,0], dx=arrow_left_diff[0,0], dy=arrow_left_diff[1,0],
                  color='r', width = WHEEL_WIDTH/4, head_width= WHEEL_WIDTH, **arrow_params)
    if np.sum(np.abs(arrow_right_diff)) > 0.0:
        axis.arrow(arrow_right_start[0,0], arrow_right_start[1,0], dx=arrow_right_diff[0,0], dy=arrow_right_diff[1,0],
                  color='r', width = WHEEL_WIDTH/4, head_width= WHEEL_WIDTH, **arrow_params)

def render_ackermann(robot_params, robot_state, axis):

    # Vehicle parameters
    LENGTH = robot_params.wheelbase*1.5 # [m]
    WIDTH = robot_params.wheelbase * 0.75  # [m]
    BACKTOWHEEL = robot_params.wheelbase*0.25  # [m]
    WHEEL_LEN = robot_params.wheelbase * 0.15  # [m]
    WHEEL_WIDTH = robot_params.wheelbase * 0.07  # [m]
    TRACK = WIDTH * 0.25  # [m]
    WB = robot_params.wheelbase  # [m]
    offset = robot_params.length_offset

    cabcolor="-b"
    truckcolor="-k"

    x = robot_state[0]
    y =robot_state[1]
    yaw = robot_state[2]
    steer = robot_state[3]
    xo =  robot_state[4]
    yo =  robot_state[5]

    # Plot Offset and Rear Axle Point
    # axis.arrow( x, y, 1.0, 2.0 )
    # print(xo, yo)
    axis.plot( xo, yo, marker='x', markersize=6, color='r' )
    # axis.plot(np.array(x), np.array(y), marker='x', markersize=3, color='k')

    outline = np.array([[-LENGTH/2, LENGTH/2, LENGTH/2, -LENGTH/2, -LENGTH/2],
                    [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])
    outline[0, :] += WB/2
   

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                            [-WHEEL_WIDTH - TRACK, -WHEEL_WIDTH - TRACK, WHEEL_WIDTH - TRACK, WHEEL_WIDTH - TRACK, -WHEEL_WIDTH - TRACK]])
    rr_wheel = np.copy(fr_wheel)
    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1
    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                        [-math.sin(steer), math.cos(steer)]])
    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB
    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T
    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T
    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y
    
    # Plot Vehicle
    axis.plot(np.array(outline[0, :]),
            np.array(outline[1, :]), cabcolor)
    axis.plot(np.array(fr_wheel[0, :]),
            np.array(fr_wheel[1, :]), truckcolor)
    axis.plot(np.array(rr_wheel[0, :]),
            np.array(rr_wheel[1, :]), truckcolor)
    axis.plot(np.array(fl_wheel[0, :]),
            np.array(fl_wheel[1, :]), truckcolor)
    axis.plot(np.array(rl_wheel[0, :]),
            np.array(rl_wheel[1, :]), truckcolor)
    axis.plot(x,y, color=[1, 1, 0], marker='*', markersize=10)


import numpy as np
from matplotlib import pyplot as plt
import sys
import os
main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(main_path)
sys.path.append(main_path) 

from model.robot_state import Robot_State
from model.robot_parameter import Articulated_Parameter, Skid_Parameter, Robot_Parameter
import pathplanning.libs.extend_transformations as mytf


if __name__ == "__main__":
    velocity = 1
    Ts = 1e-2 #/ velocity
    robot_params = Articulated_Parameter(length_front=1.5, length_rear=1.5, length_offset=-1.0, max_steerAngle=np.deg2rad(35.0), max_velocity=velocity)
    robot_params = Skid_Parameter(wheelbase=2.0, track=2.0, length_offset=-2.0, max_velocity=velocity)
    robot_params = Robot_Parameter(wheelbase=3.0, length_offset=-1.0, max_steerAngle=np.deg2rad(35.0), min_velocity=velocity, max_velocity=velocity)

    robot_state = Robot_State(var_xy=0.0, var_yaw=0.0, l10n_update_distance=0.0,
                                x=0.0, y=0.0, yaw=0.0, steerAngle=0.0, velocity=robot_params.max_velocity,
                                robot_parameter=robot_params)



    
    fig, axis = plt.subplots(1,1)
    axis.axis('equal')
    # render_ackermann(robot_params, robot_state.get_asVector(), axis)

    robot_state.set_pose(x=1, y=10, yaw=np.pi/4*3 * 1)
    robot_state.set_pose(x=0, y=0, yaw=np.pi/4*3 * 0)
    robot_state.set_steerAngle(steerAngle=0.5)
#     render_articulated(robot_params, robot_state.get_asVector(), axis)
#     render_skid(robot_params, robot_state.get_asVector(), axis)
    render_ackermann(robot_params, robot_state.get_asVector(), axis)
    plt.show()            