'''
    Example for Feedback Linearization Control and an Ackermann-steered robot
    Author: Ilja Stasewisch, Date: 2019-06-05
'''
import sys
import os
main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(main_path) 


from time import sleep
import numpy as np
import numpy
import matplotlib.pyplot as plt

from model.robot_state import Robot_State
from model.robot_parameter import Articulated_Parameter
from model.articulatedsteer import Articulated_Steering
from model.simulation_parameter import Sim_Parameter

from pathcontrol.feedforward.feedforward_knick import Feedforward_Knick
from pathcontrol.pilco.pilcotracking_kn import PILCOTRAC_kn

from pathplanning.path_sineShaped import Path_Sine
from pathplanning.path_handler_kn import Path2D_Handler_knss
from visualization.visualizer import Visualizer
from mutex_locker import Mutex_Locker
from copy import deepcopy
from pathplanning import path_handler
# from gym.envs.classic_control.rendering import LineStyle
from analysis.analysis_error import Analysis_Error

import numpy as np

import gpflow
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward, CombinedRewards
import tensorflow as tf
from tensorflow import logging

np.random.seed(0)
tf.logging.set_verbosity(tf.logging.ERROR)



if __name__ == "__main__":

    # Parameter and State
    velocity = 1.0
    Ts = 1e-3 #/ velocity
    look_ahead = 0.0
    robot_params = Articulated_Parameter(length_front=1.0, length_rear=1.0, length_offset=-0.0, T_steer=0.01, max_steerAngle=np.deg2rad(35.0), min_velocity=velocity, max_velocity=velocity)
    sim_params = Sim_Parameter(Ts_sim=Ts, Ts_ctrl=Ts)
    robot_state = Robot_State(x=0.0, y=0.0, yaw=0.0, steerAngle=0.0, velocity=robot_params.mean_velocity, robot_parameter=robot_params)
   
    # PATH
    ds = 0.01
    path_sine= Path_Sine(amp=12.0, omega=0.1, dLength_path=ds, arclength=220.0, plotting=True)
    path_x, path_y = path_sine.get_path()
    path_yaw = path_sine.get_yaw()
    pathHandler = Path2D_Handler_knss(path_x, path_y, robot_state=robot_state, robot_param=robot_params, vel=1.0)
    robot_state.set_pose(x=path_x[0], y=path_y[0], yaw=path_yaw[0], yawRate = 0.0)


    # Control and Model
    # -----------------
    # Control and Feedforward
    
    feedforward = Feedforward_Knick(path=pathHandler.get_offsetPath(), velocity_sign=np.sign(robot_state.get_velocity()), \
                              robot_param=robot_params, timeconstant_steer=robot_params.T_PT1_steer)



    # Model
    motion_model = Articulated_Steering(model="_eqs__u_steerDesired__v_rear__steer_PT1__yaw_nodyn__alpha_no",
                                            robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    
    pilcotrac = PILCOTRAC_kn(robot_params,sim_params,robot_state,pathHandler,motion_model,feedforward,SUBS=150) 

    # Wait to load visualizer
    print("Example Knicklenkung PILCO")
    print("\trobot_state = ",  robot_state)

    # Visualization
    vis = Visualizer(robot_params, pathHandler, robot_state)
    # sleep(3.0)
    SUBS = 400 



    T_sim = 2500





    with tf.Session() as sess:
        state_dim = 4
        control_dim =1
        max_yerror = 0.5

        data_mean,data_std = pilcotrac.get_randomState()
        W = np.diag([1e-2,1e-2,1e-3,1e-3])
        T_sim = 600

        R, controller = pilcotrac.get_RewardFun_and_Controller(state_dim,control_dim,data_mean,data_std,max_yerror,W)
        
        pilco = pilcotrac.loadpilco(controller=controller,reward=R,sparse=True)
        X_new, Y_new ,j = pilcotrac.rollout(pilco,max_yerror,data_mean,data_std,lookahead= 1.22,timesteps=T_sim,random=False, verbose=False)

    # print("Max steer = ", np.fabs( np.rad2deg(max(steerff_saver)) ) )

    print("____________________")
    print(np.array([robot_state.xo,robot_state.yo]))
    print("Analysis")
    analysis = Analysis_Error()
    opath, dpath = pathHandler.get_offsetPath(),  pathHandler.get_drivenPath()
    lateral_error = analysis.get_lateralError_byKdTree(opath, dpath)
    analysis.print_keyNumbers(lateral_error)

    fig2, ax = plt.subplots(nrows=2, ncols=2)
    fig2.canvas.set_window_title("Analysis")
    ax[0,0].plot(motion_model.steerRamp.save_filtered, color='y', )
    ax[0,0].plot(motion_model.steerRamp.save_steer, color='b', LineStyle='--')
    ax[0,1].plot( np.rad2deg(feedforward.steer) )
    
    ax[1,1].plot(opath[0], opath[1], marker='.')
    ax[1,1].plot(dpath[0], dpath[1])

    des_arclength = pathHandler.get_arclength()
    

    print("AS:", des_arclength.shape, opath[0].shape)
    ax[1,0].plot(des_arclength[0:-1], lateral_error)

    plt.show()