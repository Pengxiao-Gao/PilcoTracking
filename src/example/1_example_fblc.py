
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
from model.robot_parameter import Robot_Parameter
from model.ackermann import Ackermann
from model.simulation_parameter import Sim_Parameter
from pathcontrol.statespace.fblc.fblc import FBLC
from pathcontrol.feedforward.feedforward import Feedforward
from pathplanning.spline import Cubic_Spline
from pathplanning.path_handler import Path2D_Handler
from visualization.visualizer import Visualizer
from mutex_locker import Mutex_Locker
from copy import deepcopy
from pathplanning import path_handler
# from gym.envs.classic_control.rendering import LineStyle
from analysis.analysis_error import Analysis_Error
from pathplanning.libs.extend_transformations import *



if __name__ == "__main__":

    # Parameter and State
    velocity = 4
    Ts = 1e-3 #/ velocity
    robot_params = Robot_Parameter(wheelbase=3.0, length_offset=-1.0, max_steerAngle=np.deg2rad(35.0), min_velocity=velocity, max_velocity=velocity)
    sim_params = Sim_Parameter(Ts_sim=Ts, Ts_ctrl=Ts)
    robot_state = Robot_State(var_xy=0.0, var_yaw=0.0, l10n_update_distance=0.0,
                              x=0.0, y=0.0, yaw=0.0, steerAngle=0.0, velocity=robot_params.mean_velocity,
                              robot_parameter=robot_params)
    
    # PATH
    x_sampling_points = [0.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0]
    y_sampling_points = [0.0, 0.0, 0.0, 0.0, 0.25, -0.25, 0.25, -1.0, 1.0, -1.0, 2.0, -25.0, -10.0]

    # x_sampling_points = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0]
    # y_sampling_points = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0,  4.0, 4.0, 4.0, 4.0, 4.0]

    spline = Cubic_Spline()
    spline_x, spline_y = spline.calc(dLength=0.01, x_points=x_sampling_points, y_points=y_sampling_points) # USING dLength=ds=1m/s as nominal difference in arclength as normalization for the curvature and to scale afterwards the feed forward steering command
    pathHandler = Path2D_Handler(spline_x, spline_y, robot_state=robot_state, robot_param=robot_params, vel=0.5)
    offsetPath = pathHandler.get_path()   
    print("set start pose:", offsetPath[0][0], offsetPath[1][0], offsetPath[2][0])
    robot_state.set_pose(x=offsetPath[0][0], y=offsetPath[1][0], yaw=offsetPath[2][0], yawRate = 0.0)

    # Visualization
    vis = Visualizer(robot_params, pathHandler, robot_state)
    sleep(3.0)

    # Control and Model
    # fblc = FBLC(robot_params, sim_params, model_type="_ackermann_frontSteering_RearDrive_n3")
    fblc = FBLC(robot_params, sim_params, model_type="_ackermann_frontSteering_RearDrive_n2")
    feedforward = Feedforward(path=pathHandler.get_offsetPath(), velocity_sign=np.sign(robot_state.get_velocity()), \
                              robot_param=robot_params, timeconstant_steer=robot_params.T_PT1_steer)
    # ackermann_model = Ackermann(model="u=steer, steer=no dynamic, yaw=no dynamic", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    ackermann_model = Ackermann(model="u=steer, steer=PT1, yaw=no dynamic", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    # ackermann_model = Ackermann(model="u=steer, steer=PT2, yaw=no dynamic", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    # ackermann_model = Ackermann(model="u=steer, steer=PT2+ramp+timedelay, yaw=no dynamic", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params)
    ackermann_model = Ackermann(model="u=steer, steer=PT2+ramp+timedelay, yaw=PT1", robot_state=robot_state, robot_parameter=robot_params, sim_parameter=sim_params, yaw_dynamic="tractor_yaw_dynamic")


    # Wait to load visualizer
    print("Example FBLC")
    print("\trobot_state = ",  robot_state)
    steerff_saver = []
    cnt= 0

    while True : 
        diff_matrix, behind_goal, index_nn = pathHandler.execute(look_ahead=1.22)
        

        if behind_goal == True:
            print("Goal reached!")
            break
        steer_fb, steer_ff = 0.0, 0.0
        steer_fb = fblc.execute( diff_matrix, robot_state.get_velocity() )  # Pengxiao
        steer_ff = feedforward.get_steer(index=index_nn)
        
        # steer_ff = feedforward.get_steerDesired( index=index_nn )
        # steer_ff = feedforward.get_steerDesired_velocity( index=index_nn, velocity=robot_state.get_velcity() )

        steerff_saver.append(steer_fb)
        steer_ctrl = steer_fb+steer_ff# u input


        sim_time = ackermann_model.execute( steer_ctrl )
        
        # if cnt > 1: break

        cnt += 1
        # sleep(0.0001)
        state = ackermann_model.state.get_asVector()
        
        if state[0]>81:
            break
        
    print(cnt) 
    print(index_nn)   
    # print("Max steer = ", np.fabs( np.rad2deg(max(steerff_saver)) ) )

    print("____________________")
    print(np.array([robot_state.xo,robot_state.yo]))
    print("Analysis")
    analysis = Analysis_Error()
    opath, dpath = pathHandler.get_offsetPath(),  pathHandler.get_drivenPath()
    lateral_error = analysis.get_lateralError_byKdTree(opath, dpath)
    analysis.print_keyNumbers(lateral_error)
    des_arclength = pathHandler.get_arclength()



    # with plt.style.context('science'):

    fig2, ax = plt.subplots(nrows=2, ncols=2)
    fig2.canvas.set_window_title("Analysis")
    ax[0,0].plot(ackermann_model.saveSteer.save_steer0, color='y' )
    ax[0,0].plot(ackermann_model.saveSteer.save_steer1, color='b', LineStyle='--')
    ax[0,1].plot( np.rad2deg(feedforward.steer) )
    
    ax[1,1].plot(opath[0], opath[1], marker='.')
    ax[1,1].plot(dpath[0], dpath[1])

    

    print("AS:", des_arclength.shape, opath[0].shape)
    ax[1,0].plot(des_arclength[0:-1], lateral_error)

    plt.show()
    with plt.style.context(['science', 'ieee']):
        plt.plot( np.rad2deg(feedforward.steer) )
        plt.xlabel('Anzahl der Schritte')
        plt.ylabel(r'$\delta_{ff}$ in $^{\circ}$')
        plt.title(r'Lenkradwinkel der Vorsteuerung $\delta_{ff}$')
        plt.show()

        plt.plot(des_arclength[0:-1], lateral_error)
        plt.xlabel('Weg in m')
        plt.ylabel(r'$e_y$ in m')
        plt.title('Querfehler mit Vorsteuerung $e_y$')
        plt.show()
