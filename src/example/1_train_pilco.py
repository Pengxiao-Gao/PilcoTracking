
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
from pathplanning.path_sineShaped import Path_Sine

from model.robot_parameter import Robot_Parameter
from model.ackermann import Ackermann
from model.simulation_parameter import Sim_Parameter
from pathcontrol.pilco.pilcotracking import PILCOTRAC
from pathcontrol.pilco.pilcotracking import reward_wrapper, predict_trajectory_wrapper
from pathcontrol.feedforward.feedforward import Feedforward
from pathplanning.spline import Cubic_Spline
from pathplanning.path_handler import Path2D_Handler
from visualization.visualizer import Visualizer
from mutex_locker import Mutex_Locker
from copy import deepcopy
from pathplanning import path_handler
from analysis.analysis_error import Analysis_Error

import numpy as np

import gpflow
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward,QuadraticReward, LinearReward, CombinedRewards
import tensorflow as tf
from tensorflow import logging
np.random.seed(0)
tf.logging.set_verbosity(tf.logging.ERROR)

from utils import save_pilco, load_pilco, predict_wrapper


if __name__ == "__main__":

    # Parameter and State
    velocity = 4.0
    Ts = 1e-3 #/ velocity
    robot_params = Robot_Parameter(wheelbase=3.0, length_offset=-1.0, max_steerAngle=np.deg2rad(35.0), min_velocity=velocity, max_velocity=velocity)
    sim_params = Sim_Parameter(Ts_sim=Ts, Ts_ctrl=Ts)
    robot_state = Robot_State(var_xy=0.0, var_yaw=0.0, l10n_update_distance=0.0,
                              x=0.0, y=0.0, yaw=0.0, steerAngle=0.0, velocity=robot_params.mean_velocity,
                              robot_parameter=robot_params)


    # PATH
    '''
    x_sampling_points = [0.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0]
    y_sampling_points = [0.0, 0.0, 0.0, 0.0, 0.25, -0.25, 0.25, -1.0, 1.0, -1.0, 2.0, -25.0, -10.0]



    spline = Cubic_Spline()
    spline_x, spline_y = spline.calc(dLength=0.01, x_points=x_sampling_points, y_points=y_sampling_points) # USING dLength=ds=1m/s as nominal difference in arclength as normalization for the curvature and to scale afterwards the feed forward steering command
    pathHandler = Path2D_Handler(spline_x, spline_y, robot_state=robot_state, robot_param=robot_params, vel=0.5)
    offsetPath = pathHandler.get_path()
    print("set start pose:", offsetPath[0][0], offsetPath[1][0], offsetPath[2][0])
    robot_state.set_pose(x=offsetPath[0][0], y=offsetPath[1][0], yaw=offsetPath[2][0], yawRate = 0.0)
    '''
    '''
    path_ds = 0.01
    path8 = Path_8_shaped(circ1=[0.0, 0.0, 7.0], circ2=[30.0, 0.0, 10.0], dLength_path=path_ds)
    path8_x, path8_y, startLine_x, startLine_y, startLine_yaw = path8.get_path()
    pathHandler = Path2D_Handler(path8_x, path8_y, robot_state=robot_state, robot_param=robot_params, vel=0.5)
    offsetPath = pathHandler.get_path()   
    robot_state.set_pose(x=startLine_x, y=startLine_y, yaw=startLine_yaw, yawRate = 0.0)
    '''
    ds = 0.01
    path_sine= Path_Sine(amp=12.0, omega=0.1, dLength_path=ds, arclength=120.0, plotting=True)
    path_x, path_y = path_sine.get_path()
    path_yaw = path_sine.get_yaw()
    pathHandler = Path2D_Handler(path_x, path_y, robot_state=robot_state, robot_param=robot_params, vel=1.0)
    offsetPath = pathHandler.get_path()
    print("set start pose:", offsetPath[0][0], offsetPath[1][0], offsetPath[2][0])
    robot_state.set_pose(x=offsetPath[0][0], y=offsetPath[1][0], yaw=offsetPath[2][0], yawRate = 0.0)
  

    # Control and Model
    # fblc = FBLC(robot_params, sim_params, model_type="_ackermann_frontSteering_RearDrive_n3")
    #fblc = FBLC(robot_params, sim_params, model_type="_ackermann_frontSteering_RearDrive_n2")
    
    
    
    
    feedforward = Feedforward(path=pathHandler.get_offsetPath(), velocity_sign=np.sign(robot_state.get_velocity()), \
                              robot_param=robot_params, timeconstant_steer=robot_params.T_PT1_steer)
    # ackermann_model = Ackermann(model="u=steer, steer=no dynamic, yaw=no dynamic", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    # ackermann_model = Ackermann(model="u=steer, steer=PT1, yaw=no dynamic", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    # ackermann_model = Ackermann(model="u=steer, steer=PT2, yaw=no dynamic", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    # ackermann_model = Ackermann(model="u=steer, steer=PT2+ramp+timedelay, yaw=no dynamic", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    ackermann_model = Ackermann(model="u=steer, steer=PT2+ramp+timedelay, yaw=PT1", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params, yaw_dynamic="tractor_yaw_dynamic")

    pilcotrac = PILCOTRAC(robot_params,sim_params,robot_state,pathHandler,ackermann_model,feedforward,SUBS=200) 
    #SUBS: Subsampling. Repeat action in SUBS steps for long time horizon. v=1 SUBS = 300-600; v=4 SUBS=100-200

    # Visualization
    vis = Visualizer(robot_params, pathHandler, robot_state)
    sleep(3.0)



    # Wait to load visualizer
    print("Example PILCO")
    print("\trobot_state = ",  robot_state)


    T = 30 # Horizon for iteratively predict for planning and training, somethinglike MPC I think
    T_sim = 250 # Timesteps for testing of each episode
    J = 5 # Random rollouts before optimisation starts, generate a primar GP





    with tf.Session() as sess:

        max_yerror = 0.5

        # Generate a primar GP
        X1,Y1 ,k = pilcotrac.rollout(None,max_yerror,data_mean=None,data_std=None,lookahead= 1.22,timesteps=T_sim,random=True, verbose=False)

        for i in range(1,J):

            robot_state.set_pose(x=offsetPath[0][0], y=offsetPath[1][0], yaw=offsetPath[2][0], yawRate = 0.0)
            robot_state.set_steerAngle(steerAngle=0.0)
            pathHandler.reset__index_nn()

            X1_, Y1_,k_ = pilcotrac.rollout(None,max_yerror,data_mean=None,data_std=None,lookahead= 1.22,timesteps=T_sim,random=True, verbose=False)
            X1 = np.vstack((X1, X1_))
            Y1 = np.vstack((Y1, Y1_))
            print(k_)

        robot_state.set_pose(x=path_x[0], y=path_y[0], yaw=path_yaw[0], yawRate = 0.0)
        robot_state.set_steerAngle(steerAngle=0.0)
        pathHandler.reset__index_nn()
        print(X1.shape)

        # Normalization: make data sets with mean=0 and std=1
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
        S_init = 0.01 * np.eye(state_dim)
        np.save(main_path+"/SAVED/v4_data_mean_5", data_mean)
        np.save(main_path+"/SAVED/v4_data_std_5", data_std)

        W = np.diag([1e-1,1e-1,1e-3,1e-3]) # Weight of states 
        T_sim = 600
        target = np.divide([0.0,0.0,0.0,0.0] -data_mean,data_std)

        R, controller = pilcotrac.get_RewardFun_and_Controller(state_dim,control_dim,data_mean,data_std,max_yerror,W)

        pilco = PILCO(X, Y, num_induced_points=100,controller=controller, horizon=T,reward=R,m_init=m_init,S_init=S_init) # num_induced_points: Number of sparse GP-pseudo-points
        
        for model in pilco.mgpr.models:


            # model.clear()
            # model.kern.lengthscales.prior = gpflow.priors.Gamma(3,20) # priors have to be included before  the model gets compiled

            # model.kern.variance.prior = gpflow.priors.Gamma(1.5,4)    
            model.likelihood.variance = 0.001  # take a large noise to improve numerical stability (Cholesky decomposition failures)
            model.likelihood.variance.trainable = False
            # model.compile()
            # vals = model.read_values()
            # print(vals)
        

        best_r = 0
        all_Rs = np.zeros((X.shape[0], 1))
        for i in range(len(all_Rs)):
            all_Rs[i,0] = reward_wrapper(R, X[i,None,:-1], 0.001 * np.eye(state_dim))[0] # Reward for each step

        total_rewards = np.zeros((len(X)//T,1))
        ep_rewards = np.zeros((len(X)//T,1))

        for i in range(len(total_rewards)):
            total_rewards[i] = sum(all_Rs[i * T: i*T + T])
            ep_rewards[i] = sum(all_Rs[i * T: i*T + T])
        
        
        for rollouts in range(1,31):
            # Optimazation

            print("**** ITERATION no", rollouts, " ****")
            pilco.optimize_models(maxiter=2000,restarts=3) # Restart to avoid local  minima
            pilco.optimize_policy(maxiter=2000,restarts=2)
            
            # Initialize
            robot_state.set_pose(x=offsetPath[0][0], y=offsetPath[1][0], yaw=offsetPath[2][0], yawRate = 0.0)
            robot_state.set_steerAngle(steerAngle=0.0)
            pathHandler.reset__index_nn()
            # Testing
            X_new, Y_new ,j = pilcotrac.rollout(pilco,max_yerror,data_mean=data_mean,data_std=data_std,lookahead= 1.22,timesteps=T_sim,random=False, verbose=False)

            print("No of ops:", len(tf.get_default_graph().get_operations()))
            r_new = np.zeros((len(X_new), 1))
            var_r = np.zeros((len(X_new), 1))
            r_tar = np.zeros((len(X_new), 1))

            for i in range(len(X_new)):
                r_new[i, 0],var_r[i,0] = reward_wrapper(R, X_new[i,None,:-1], S_init)
                r_tar[i, 0] = reward_wrapper(R, target[None, :], S_init)[0] 
            total_r = sum(r_new)
            r = sum(r_tar)
            # _, _, r = predict_trajectory_wrapper(pilco, m_init, S_init, T_sim)
            print("Total ", total_r, " Predicted: ", r)
            '''
            with plt.style.context(['science', 'ieee']):

                plt.plot(range(len(X_new)),r_tar,'r--',r_new,'k')
                plt.fill_between(range(len(r_new)),
                                r_new[:,0] - 2*np.sqrt(var_r[:, 0]), 
                                r_new[:,0] + 2*np.sqrt(var_r[:, 0]), alpha=0.3)
                plt.title('Echtzeit-Belohnungen Versuch %d'%rollouts)
                plt.xlabel('Anzahl der Schritte')
                plt.ylabel('Belohnungen')
                plt.legend(['vorhersagte','echte'])
                plt.xlim(0,250)
                plt.ylim(r_tar[0]*0.2,r_tar[0]*1.5)
                
                R_path = main_path+'/figs/Linear/R'
                folder = os.path.exists(R_path)
                if not folder:
                    os.makedirs(R_path)
                plt.savefig(R_path+'/%d.png'%rollouts)

                
                plt.close()
            
            # Collect data
            X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
            all_Rs = np.vstack((all_Rs, r_new)); total_rewards = np.vstack((total_rewards, np.reshape(total_r,(1,1))))
            ep_rewards = np.vstack((ep_rewards, np.reshape(r,(1,1))))
            pilco.mgpr.set_XY(X, Y)


            
            for i,m in enumerate(pilco.mgpr.models):
                with plt.style.context(['science', 'ieee']):

                    y_pred_test, var_pred_test = m.predict_y(X_new[:T_sim,:])
                    plt.plot(range(len(y_pred_test)), y_pred_test,'r--', Y_new[:,i],'k')

                    plt.fill_between(range(len(y_pred_test)),
                                    y_pred_test[:,0] - 2*np.sqrt(var_pred_test[:, 0]),
                                    y_pred_test[:,0] + 2*np.sqrt(var_pred_test[:, 0]), alpha=0.3)
                    if i == 0:
                        Zustand = r'$\Delta e_y$'
                        bottom = -3.0
                        top = 3.0  
                    elif i == 1:
                        Zustand = r'$\Delta e_\theta$'
                        bottom = -3.0
                        top = 3.0  
                    elif i == 2:
                        Zustand = r'$\Delta \dot{e}_y$'
                        bottom = -3.0
                        top = 3.0  
                    else:
                        Zustand = r'$\Delta \dot{e}_\theta$'
                        bottom = -3.0
                        top = 3.0 
                        
                    plt.xlabel('Anzahl der Schritte')
                    plt.ylabel(Zustand)
                    plt.legend(['vorhersagte','echte'])
                    plt.title('Einschritt-Prognose Versuch %d'%rollouts)
                    plt.xlim(0,120)
                    plt.ylim(bottom,top)


                    E_path = main_path+'/figs/Linear/Einschritt-Prog/%d'%i
                    folder = os.path.exists(E_path)
                    if not folder:
                        os.makedirs(E_path)

                    plt.savefig(E_path+'/%d.png'%rollouts)

                    
                    plt.close()
            
            np.shape(var_pred_test)
            
                    
            from gaocontroller import predict_trajectory_wrapper

            m_p = np.zeros((T, state_dim))
            S_p = np.zeros((T, state_dim, state_dim))
            
            m_p[0,:] = m_init
            S_p[0, :, :] = S_init
            m_h = m_init
            S_h = S_init
            for h in range(T):
                m_h, S_h, _ = predict_trajectory_wrapper(pilco, m_h, S_h, h)
                m_p[h,:], S_p[h,:,:] = m_h[:], S_h[:,:]
            
            for i in range(state_dim):
                with plt.style.context(['science', 'ieee']):

                    plt.plot(range(T-1), m_p[0:T-1, i],'r--', X_new[1:T, i],'k') # can't use Y_new because it stores differences (Dx)
                    plt.fill_between(range(T-1),
                                    m_p[0:T-1, i] - 2*np.sqrt(S_p[0:T-1, i, i]),
                                    m_p[0:T-1, i] + 2*np.sqrt(S_p[0:T-1, i, i]), alpha=0.3)
                    if i == 0:
                        Zustand = r'$e_y$'
                    elif i == 1:
                        Zustand = r'$e_\theta$'
                    elif i == 2:
                        Zustand = r'$\dot{e}_y$'
                    else:
                        Zustand = r'$\dot{e}_\theta$'
                    plt.xlabel('Anzahl der Schritte')
                    plt.ylabel(Zustand)
                    plt.title('Multischritt-Prognose  Versuch %d'%rollouts)
                    plt.ylim(-10,10)
                    plt.legend(['vorhersagte','echte'])

                    M_path = main_path+'/figs/Linear/Multischritt-Prog/%d'%i
                    folder = os.path.exists(M_path)
                    if not folder:
                        os.makedirs(M_path)
                    plt.savefig(M_path+'/%d.png'%rollouts)

                    
                    plt.close()
            '''
            

    save_pilco(main_path+"/SAVED/v4_pt1pt2_", X, Y, pilco,sparse=True)
    np.save(main_path+'/SAVED/total_rewards', total_rewards)
    np.save(main_path+'/SAVED/ep_rewards', ep_rewards)

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
    ax[0,0].plot(ackermann_model.steerRamp.save_filtered, color='y', )
    ax[0,0].plot(ackermann_model.steerRamp.save_steer, color='b', LineStyle='--')
    ax[0,1].plot( np.rad2deg(feedforward.steer) )

    ax[1,1].plot(opath[0], opath[1], marker='.')
    ax[1,1].plot(dpath[0], dpath[1])

    des_arclength = pathHandler.get_arclength()


    print("AS:", des_arclength.shape, opath[0].shape)
    ax[1,0].plot(des_arclength[0:-1], lateral_error)
    plt.show()


