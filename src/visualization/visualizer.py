import matplotlib.pyplot as plt
from threading import Thread, Lock
import time
import multiprocessing as mp
import numpy as np
import math
from math import sin, cos, tan, sqrt
from copy import deepcopy
from scipy import stats

from model.robot_state import Robot_State
from pathplanning.path_handler import Path2D_Handler


from visualization.rendering_robot import *

# MATPLOTLIB Multiprocess: https://matplotlib.org/3.1.0/gallery/misc/multiprocess_sgskip.html


class ProcessPlotter(object):
    def __init__(self, robot_parameter):
        self.roParam = robot_parameter
        # self.fig, self.axes = plt.subplots(2, 1)

        self.path_x = []
        self.path_y = []
        self.min_x = -1.0
        self.max_x = 1.0
        self.min_y = -1.0
        self.max_y = 1.0

        self.driven_path_x, self.driven_path_y = [], []


    def terminate(self):
        plt.close('all')

    def set_path(self, path_x, path_y) :
        # with self._mutex:
        self.path_x = path_x
        self.path_y = path_y

    def _render_robot(self, robot_state):
        if self.roParam.type == "Ackermann":
            render_ackermann(self.roParam, robot_state, self.axes[0])
        elif self.roParam.type == "Articulated":
            # render_articulated(self.roParam, robot_state, self.axes[0])
            render_articulated(self.roParam, robot_state, self.axes[0])
        elif self.roParam.type == "Skid":
            render_skid(self.roParam, robot_state, self.axes[0])            
        else:
            print("ProcessPlotter._render_robot(...): Model not found")

    def _render_path(self):
        self.axes[0].plot(self.path_x, self.path_y)
        self.axes[0].set_xlim(self.min_x - 10.0, self.max_x + 10.0)
        self.axes[0].set_ylim(self.min_y - 10.0, self.max_y + 10.0)


    def _render_reference(self, ref_position):
        self.axes[0].plot(ref_position[0], ref_position[1], marker='o', markersize=5, color=(1.0, 1.0, 0.0 , 1.0))

    def _render_referencePath(self, path):
        if path is not None:
            self.axes[0].plot(path["x"], path["y"], color=(0.0, 1.0, 1.0 , 1.0),  linewidth=3)

    def _render_predictionPath(self, path):
        if path is not None:
            self.axes[0].plot(path[0], path[1], color=(1.0, 0.0, 0.0 , 0.5),  linewidth=3)

    def _render_errorLateralYaw(self, arclength, lateralError):
        latErr = np.array(lateralError)
        # print("len(arclength): " , len(arclength))
        # print("len(lateralError): " , len(lateralError))
        # print("(lateralError): " , lateralError)
        self.axes[1].clear()
        idxs = ~np.isnan(latErr)
        self.axes[1].plot(arclength[idxs], latErr[idxs], color='k', linestyle="-")
        # print("lateralError:", lateralError)

        # Prepare lateral error data
        latErr = np.array(lateralError)
        latErr = latErr[ idxs ] # Remove None's in Array
        if latErr.size == 0:
            return
        

        latErr_abs = [math.fabs(err) for err in latErr]

        # # latErr = np.array(latErr)
        # print("latErr:", latErr)
        # print("type(latErr):", type(latErr))
        # print("shape(latErr):", latErr.shape)
        # # print("np.fabs(latErr):", np.fabs(latErr) )
        # test = np.array([-0.0, 0.0])
        # # test = np.append(test, latErr)
        # print("test:", test)
        # print("type(test):", type(test))
        # print("shape(test):", test.shape)
        # print("np.fabs(test):", np.fabs(test) )

        # fabs_latErr = np.fabs(latErr) 
        

        try:
            # fabs_latErr = np.fabs(latErr) 
            mean_abs_lat_error = np.mean( latErr_abs )
            mean_abs_error_Vec = np.ones( shape=arclength.shape)  * mean_abs_lat_error
            self.axes[1].plot(arclength, mean_abs_error_Vec, color='g')
        except:
            pass
            # mean_abs_error_Vec = np.ones( shape=arclength.shape)  * 0.0
            # self.axes[1].plot(arclength, mean_abs_error_Vec, color='g', linewidth=3)

        mean_lat_error = np.mean(latErr) 
        var_lat_error = np.var(latErr)
        std_lat_error = sqrt(var_lat_error)
        N = latErr.shape[0]

        # Plot Mean
        mean_error_Vec = np.ones( shape=arclength.shape)  * mean_lat_error
        self.axes[1].plot(arclength, mean_error_Vec, color='b')



        # Plot Var
        latErr_var = np.ones( shape=arclength.shape)  * np.var(latErr)
        self.axes[1].plot(arclength, latErr_var, color='b', linestyle="--")

        # Plot CONFIDENCE INTERVALL
        conf_intervall = stats.norm.interval(0.95, loc=mean_lat_error, scale=std_lat_error)

        conf = np.ones(arclength.shape) * (conf_intervall[0])
        self.axes[1].plot(arclength, conf, color='r', linestyle="--")
        conf = np.ones(arclength.shape) * (conf_intervall[1])
        self.axes[1].plot(arclength, conf, color='r', linestyle="--")



    def _render_drivenPath(self, robot_state):
        self.driven_path_x.append( robot_state[4] )
        self.driven_path_y.append( robot_state[5] )
        self.axes[0].plot(self.driven_path_x, self.driven_path_y, linestyle="-", linewidth=3, color=(0.0, 0.99, 0.0 , 1.0))

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()

            if command is None:
                self.terminate()
                return False
            else:

                if command[0] is not None:
                    # print ("set path in queue class")
                    self.path_x, self.path_y = command[0][0], command[0][1]
                    self.min_x, self.max_x = min(self.path_x), max(self.path_x)
                    self.min_y, self.max_y = min(self.path_y), max(self.path_y)
                robot_state = command[1]
                ref_position = command[2]
                arclength = command[3][0]
                # print("plotter arclength", arclength[-1])
                lateralError = command[3][1]
                ref_path = command[4]
                prediction_path = command[5]

                self.axes[0].clear()
                self._render_path()
                self._render_drivenPath(robot_state)
                self._render_robot(robot_state)
                self._render_reference(ref_position)
                self._render_errorLateralYaw(arclength, lateralError)
                self._render_referencePath(ref_path)
                self._render_predictionPath(prediction_path)

                
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('\t starting plotter...')

        self.pipe = pipe
        self.fig, self.axes = plt.subplots(2, 1)
        self.axes[0].axis('equal')
        dt = 1.0 / 10.0 * 1000.0
        timer = self.fig.canvas.new_timer(interval=dt)
        timer.add_callback(self.call_back)
        timer.start()
        print('\t ...starting plotter done')
        plt.show()

######################################################################################################
############################################# Visualizer #############################################
######################################################################################################

class Visualizer() : 

    def __init__(self, robot_parameter, path_handler, robotState) :
        print('Visualization')
        self.rp = robot_parameter

        self._updated_path = False

        self.robotState = robotState
        self.path_handler = path_handler

        # self._mutex = Lock()

        # MULTIPROCESSING AND THREAD FOR MATPLOTLIB
        try:
            mp.set_start_method("forkserver")
        except:
            print("forkserver exist")

        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(self.rp)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()
        Thread(target=self._execute).start() # Creating a thread, target a function, .start() start

    def set_robotState(self, robot_state):
        self.robotState = robot_state

    def set_pathHandler(self, path_handler):
        self.path_handler = path_handler

    def _execute(self) :
        print('\t Visualization started')
        dt = 1.0 / 5.0
        send = self.plot_pipe.send

        # time.sleep(0.0)

        while True:
            ref_position = self.path_handler.get_lookaheadPosition()
            ref_path = self.path_handler.get_referencePath()
            prediction_path = self.path_handler.get_predictionPath__dlqt_mpc()
            arclength_latError = [self.path_handler.get_arclength(), self.path_handler.get_lateralError()]
            robot_state_ = self.robotState.get_l10nVec()

            if self._updated_path == False :
                opath = self.path_handler.get_offsetPath()
                data =[opath, robot_state_, ref_position, arclength_latError, ref_path, prediction_path]
                self._updated_path = True
            else :
                data = [None, robot_state_, ref_position, arclength_latError, ref_path, prediction_path]

            # Update Data and Wait
            send(data)
            time.sleep(dt)

    

