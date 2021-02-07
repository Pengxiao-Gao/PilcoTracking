
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

from pathcontrol.statespace.lqr.lqr import LQR
from pathplanning.path_sineShaped import Path_Sine
from pathplanning.path_handler import Path2D_Handler
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
from pilco.rewards import ExponentialReward,QuadraticReward, LinearReward, CombinedRewards
import tensorflow as tf
from tensorflow import logging
from pathcontrol.pilco.pilcotracking import PILCOTRAC
np.random.seed(0)
tf.logging.set_verbosity(tf.logging.ERROR)

from utils import save_pilco, load_pilco, predict_wrapper


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
    path_sine= Path_Sine(amp=12.0, omega=0.1, dLength_path=ds, arclength=110.0, plotting=True)
    path_x, path_y = path_sine.get_path()
    path_yaw = path_sine.get_yaw()
    pathHandler = Path2D_Handler(path_x, path_y, robot_state=robot_state, robot_param=robot_params, vel=1.0)
    robot_state.set_pose(x=path_x[0], y=path_y[0], yaw=path_yaw[0], yawRate = 0.0)


    # Control and Model
    # -----------------
    # Control and Feedforward
    
    feedforward = Feedforward_Knick(path=pathHandler.get_offsetPath(), velocity_sign=np.sign(robot_state.get_velocity()), \
                              robot_param=robot_params, timeconstant_steer=robot_params.T_PT1_steer)



    # Model
    motion_model = Articulated_Steering(model="_eqs__u_steerDesired__v_rear__steer_PT1__yaw_nodyn__alpha_no",
                                            robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    pilcotrac = PILCOTRAC(robot_params,sim_params,robot_state,pathHandler,motion_model,feedforward,SUBS=150) 

    # Wait to load visualizer
    print("Example Knicklenkung PILCO")
    print("\trobot_state = ",  robot_state)

    # Visualization
    vis = Visualizer(robot_params, pathHandler, robot_state)
    # sleep(3.0)
    SUBS = 400 



    T = 30
    T_sim = 2500
    J = 10





    with tf.Session() as sess:
        max_yerror = 0.3

        X1,Y1 ,k = pilcotrac.rollout(None,max_yerror,data_mean=None,data_std=None,lookahead= 1.22,timesteps=T_sim,random=True, verbose=False)

        for i in range(1,J):

            robot_state.set_pose(x=path_x[0], y=path_y[0], yaw=path_yaw[0], yawRate = 0.0)
            robot_state.set_steerAngle(steerAngle=0.0)
            pathHandler.reset__index_nn()

            #robot_state.set_pose(x=offsetPath[0][0], y=offsetPath[1][0], yaw=offsetPath[2][0], yawRate = 0.0)
            X1_, Y1_,k_ = pilcotrac.rollout(None,max_yerror,data_mean=None,data_std=None,lookahead= 1.22,timesteps=T_sim,random=True, verbose=False)

            X1 = np.vstack((X1, X1_))
            Y1 = np.vstack((Y1, Y1_))
            print(k_)

        robot_state.set_pose(x=path_x[0], y=path_y[0], yaw=path_yaw[0], yawRate = 0.0)
        robot_state.set_steerAngle(steerAngle=0.0)
        pathHandler.reset__index_nn()
        print(X1.shape)
        # Normalization
        data_mean = np.mean(X1[:,:4],0)
        data_std  = np.std(X1[:,:4],0)
        X = np.zeros(X1.shape)
        print(X1.shape,data_std,data_mean)
        X[:,:4] = np.divide(X1[:,:4]-data_mean,data_std)
        X[:,-1] = X1[:,-1] #Control inputs are not normalised
        Y = np.divide(Y1,data_std)
        



        state_dim = Y.shape[1]
        control_dim = X.shape[1] - state_dim
        m_init =  np.transpose(X[0,:-1,None])
        S_init = 0.1 * np.eye(state_dim)
        max_yerror = 0.3
        np.save(main_path+"/SAVED/data_mean_kn", data_mean)
        np.save(main_path+"/SAVED/data_std_kn", data_std)

        W = np.diag([1e-2,1e-2,1e-3,1e-3]) # Weight of states 
        T_sim = 600
        target = np.divide([0.0,0.0,0.0,0.0] -data_mean,data_std)

        R, controller = pilcotrac.get_RewardFun_and_Controller(state_dim,control_dim,data_mean,data_std,max_yerror,W)

        pilco = PILCO(X, Y, num_induced_points=100,controller=controller, horizon=T,reward=R)
        
        for model in pilco.mgpr.models:


            # model.clear()
            # model.kern.lengthscales.prior = gpflow.priors.Gamma(3,20) #priors have to be included before

            # model.kern.variance.prior = gpflow.priors.Gamma(1.5,4)    #before the model gets compiled
            model.likelihood.variance = 0.1
            model.likelihood.variance.trainable = False
            # model.compile()
            # vals = model.read_values()
            # print(vals)
        

        best_r = 0
        all_Rs = np.zeros((X.shape[0], 1))
        for i in range(len(all_Rs)):
            all_Rs[i,0] = reward_wrapper(R, X[i,None,:-1], 0.001 * np.eye(state_dim))[0]

        ep_rewards = np.zeros((len(X)//T,1))

        for i in range(len(ep_rewards)):
            ep_rewards[i] = sum(all_Rs[i * T: i*T + T])

        

        for rollouts in range(30):

            print("**** ITERATION no", rollouts, " ****")
            pilco.optimize_models(maxiter=2000,restarts=3)
            pilco.optimize_policy(maxiter=2000,restarts=2)

            robot_state.set_pose(x=path_x[0], y=path_y[0], yaw=path_yaw[0], yawRate = 0.0)
            robot_state.set_steerAngle(steerAngle=0.0)
            pathHandler.reset__index_nn()

            X_new, Y_new ,j = pilcotrac.rollout(pilco,max_yerror,data_mean=data_mean,data_std=data_std,lookahead= 1.22,timesteps=T_sim,random=False, verbose=False)

            print("No of ops:", len(tf.get_default_graph().get_operations()))
            r_new = np.zeros((T_sim, 1))
            r_tar = np.zeros((T_sim, 1))
            for i in range(len(X_new)):
                r_new[i, 0] = reward_wrapper(R, X_new[i,None,:-1], S_init)[0]
                r_tar[i, 0] = reward_wrapper(R, target[None, :], S_init)[0] 
            total_r = sum(r_new)
            r = sum(r_tar)
            # _, _, r = predict_trajectory_wrapper(pilco, m_init, S_init, T_sim)
            print("Total ", total_r, " Predicted: ", r)

            X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
            all_Rs = np.vstack((all_Rs, r_new)); ep_rewards = np.vstack((ep_rewards, np.reshape(total_r,(1,1))))
            pilco.mgpr.set_XY(X, Y)

    save_pilco("kn_pt1_", X, Y, pilco,sparse=True)
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