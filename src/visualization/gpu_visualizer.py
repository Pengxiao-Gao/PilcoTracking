import sys
import numpy as np
from vispy import app, scene
from threading import Thread, Lock
import time
import math
from math import sin, cos, tan

from model.sim_model import Robot_State
from pathplanning.path_handler import Path2D_Handler
from model import simulation_parameter

######################################################################################################
############################################# Visualizer #############################################
######################################################################################################




class GPU_Visualizer(app.Canvas) : 

    def __init__(self, robot_parameter, simulation_parameter, path_handler, robotState) :


        self.sim_time = 0.0
        self.simParam = simulation_parameter
        self.roParam = robot_parameter
        self.robotState = robotState
        self.path_handler = path_handler


        self.init_visual_elements()

    def init_visual_elements(self) : 
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid(spacing=0)
        self.viewbox = self.grid.add_view(row=0, col=1, camera='panzoom')

        # add some axes
        self.x_axis = scene.AxisWidget(orientation='bottom')
        self.x_axis.stretch = (1, 0.1)
        self.grid.add_widget(self.x_axis, row=1, col=1)
        self.x_axis.link_view(self.viewbox)
        self.y_axis = scene.AxisWidget(orientation='left')
        self.y_axis.stretch = (0.1, 1)
        self.grid.add_widget(self.y_axis, row=0, col=0)
        self.y_axis.link_view(self.viewbox)

        # add a line plot inside the viewbox
        self.line_outline = scene.Line(self._get_robot_outline(), parent=self.viewbox.scene)
        pathx, pathy = self.path_handler.get_path()
        path = np.vstack( (pathx, pathy) )
        path = path.T
        self.line_path = scene.Line(path, parent=self.viewbox.scene)

        # auto-scale to see the whole line.
        self.viewbox.camera.set_range()


        self.t = 0
        self.t_max = 10
        self.d_t = 0.1

        app.Canvas.__init__(self, keys='interactive')


    def _get_robot(self):
        pass

    def _get_robot_outline(self, x=0.0, y=0.0, yaw=0.0):
        # Vehicle parameters
        LENGTH = self.roParam.wheelbase*1.5 # [m]
        WIDTH = self.roParam.wheelbase * 0.75  # [m]
        BACKTOWHEEL = self.roParam.wheelbase*0.25  # [m]
        robot_outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                            [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])
        

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                            [-math.sin(yaw), math.cos(yaw)]])
        robot_outline = (robot_outline.T.dot(Rot1)).T
        robot_outline[0, :] += x
        robot_outline[1, :] += y

        # print("ount:\n", robot_outline.T)
        # print("ount:\n", robot_outline.shape)

        return robot_outline.T

    def _get_robot_wheels(self, x=0.0, y=0.0, yaw=0.0, steer=0.0) :
        WIDTH = self.roParam.wheelbase * 0.75  # [m]
        WHEEL_LEN = self.roParam.wheelbase * 0.15  # [m]
        WHEEL_WIDTH = self.roParam.wheelbase * 0.07  # [m]
        TREAD = WIDTH * 0.25  # [m]
        WB = self.roParam.wheelbase  # [m]

        fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                                [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])
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
        
        rr_wheel = (rr_wheel.T.dot(Rot1)).T
        rl_wheel = (rl_wheel.T.dot(Rot1)).T

        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y


    def update(self):
        # print("gpu vis: update")

        state = self.robotState.get_asVector()


        outline = self._get_robot_outline(x=state[0], y=state[1])
        self.line_outline.set_data(pos=outline)

        # self.viewbox.camera.set_range()



        


    






