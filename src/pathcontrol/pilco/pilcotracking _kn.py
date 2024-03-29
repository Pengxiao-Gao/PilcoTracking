import sys
import os
main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
main_path = os.path.dirname(main_path)
sys.path.append(main_path)

import numpy as np
from threading import Thread, Lock
from model.robot_parameter import Robot_Parameter
import gpflow
from gpflow import autoflow
from gpflow import settings
import pilco
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward, CombinedRewards
from pathplanning.libs.extend_transformations import *
from math import sin,cos


np.random.seed(0)
float_type = settings.dtypes.float_type



class PILCOTRAC_kn:

    def __init__(self, robot_parameters, simulation_parameters,robot_state, pathHandler, robot_model,feedforward, SUBS = 1, verbose=True) :
        print('Path Tracking Control: Pilco')
        self._mutex = Lock()
        self._goal_reached = False
        self.robot_param = robot_parameters
        self.sim_param = simulation_parameters
        self.state = robot_state
        self.pathHandler = pathHandler
        self.robot_model = robot_model
        self.feedforward = feedforward
        self.SUBS = SUBS
        self.verbose = verbose
        self.ctrlMatrix_K = None
        self.model_type = robot_model.model_type



    def get_goalReached(self) :
        with self._mutex:
            return self._goal_reached
   
    def get_randomState(self) :
        print(main_path)
        if self.model_type == "u=steer, steer=PT1, yaw=no dynamic" :
            if self.robot_param.mean_velocity == 1:
                return np.load(main_path+"/SAVED/data_mean_2.npy"), np.load(main_path+"/SAVED/data_std_2.npy")
            elif self.robot_param.mean_velocity == 4:
                return np.load(main_path+"/SAVED/v4data_mean_2.npy"), np.load(main_path+"/SAVED/v4data_std_2.npy")
        elif self.model_type == "u=steer, steer=PT2, yaw=no dynamic" :
            return np.load(main_path+"/SAVED/data_mean_3.npy"), np.load(main_path+"/SAVED/data_std_3.npy")
        elif self.model_type == "u=steer, steer=PT2+ramp+timedelay, yaw=no dynamic" :
            return np.load(main_path+"/SAVED/data_mean_4.npy"), np.load(main_path+"/SAVED/data_std_4.npy")
        elif self.model_type == "u=steer, steer=PT2+ramp+timedelay, yaw=PT1" :
            return np.load(main_path+"/SAVED/data_mean_5.npy"), np.load(main_path+"/SAVED/data_std_5.npy")
        
        elif self.model_type == "_eqs__u_steerDesired__v_rear__steer_PT1__yaw_nodyn__alpha_no":
            return np.load(main_path+"/SAVED/data_mean_kn.npy"), np.load(main_path+"/SAVED/data_std_kn.npy")
 
        
        else :
            print("Pilco: Control type does not exist!")
            return 0.0

    def get_RewardFun_and_Controller(self,state_dim, control_dim, data_mean, data_std, max_yerror, W) :
        t1 = np.divide([max_yerror,0.0,0.0,0.0] - data_mean,data_std)
        t2 = np.divide([-max_yerror,0.0,0.0,0.0] - data_mean,data_std)
        R1 = ExponentialReward(state_dim, W, t=t1)
        R2 = ExponentialReward(state_dim, W, t=t2)
        target = np.divide([0.0,0.0,0.0,0.0] -data_mean,data_std)
        Re = ExponentialReward(state_dim=state_dim,t=target,W=W)
        R = CombinedRewards(state_dim, [Re,R1,R2], coefs=[2,-1,-1])

        controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=30)
        # num_basis_functions: number of RBFuntions, a large number of functions is flexber but increases computational demands. 
        # If v==4: num_basis_functions=50
        return R, controller


    def loadpilco(self,controller, reward, sparse=False) :        
        if self.model_type == "u=steer, steer=PT1, yaw=no dynamic" :
            if self.robot_param.mean_velocity == 1:
                return load_pilco(main_path+"/SAVED/pt1_", controller=controller,reward=reward,sparse=True)
            elif self.robot_param.mean_velocity == 4:
                return load_pilco(main_path+"/SAVED/v4pt1_", controller=controller,reward=reward,sparse=True)
        elif self.model_type == "u=steer, steer=PT2, yaw=no dynamic" :
            return load_pilco(main_path+"/SAVED/pt2_", controller=controller,reward=reward,sparse=True)
        elif self.model_type == "u=steer, steer=PT2+ramp+timedelay, yaw=no dynamic" :
            return load_pilco(main_path+"/SAVED/pt2dl_", controller=controller,reward=reward,sparse=True)
        elif self.model_type == "u=steer, steer=PT2+ramp+timedelay, yaw=PT1" :
            return load_pilco(main_path+"/SAVED/pt2pt1_", controller=controller,reward=reward,sparse=True)
        
        elif self.model_type == "_eqs__u_steerDesired__v_rear__steer_PT1__yaw_nodyn__alpha_no":
                return load_pilco(main_path+"/SAVED/kn_pt1_", controller=controller,reward=reward,sparse=True)
         
        
        else :
            print("Pilco: Trained controller does not exist!")
            return 0.0        


        
    def rollout(self,pilco, max_yerror,data_mean,data_std,lookahead, timesteps,random=False, verbose=True) :

        
        j = 0
        X = []; Y = []
        x = np.array([0.0,0.0,0.0,0.0])
        if not random:
            x = np.divide(x-data_mean,data_std)
        
        for timestep in range(timesteps):

            steer_fb = self._policy(pilco,x,random)

            for i in range(self.SUBS):
                diff_matrix, behind_goal, index_nn,_= self.pathHandler.execute(look_ahead=lookahead)
                steer_ff = self.feedforward.get_steer(index_nn)
                u = steer_fb[0] + steer_ff
                u = u.astype(np.float64)  

                self.robot_model.execute( u )
                state = self.robot_model.state.get_asVector()
                yawRate = state[8]

            x_error = diff_matrix[0,3]
            y_error = diff_matrix[1,3]
            theta_error = matrix_to_yaw(diff_matrix)
            dy_error = self.robot_param.mean_velocity*sin(theta_error)-x_error*yawRate
            dtheta_error = yawRate - self.pathHandler.curve[index_nn]

            if behind_goal == True:
                print("Goal reached!")
                break

            if random:
                x_new = np.array([y_error,theta_error,dy_error,dtheta_error])
            else:
                x_new = np.array([y_error,theta_error,dy_error,dtheta_error])
                x_new = np.divide(x_new-data_mean,data_std)


            done = np.fabs(y_error)>max_yerror or np.fabs(theta_error)>0.3
            done = bool(done)
            if done:
                print(y_error,theta_error,index_nn)
                break

            if verbose:
                print("Action: ", steer_fb)
                print("State : ", x_new)

            X.append(np.hstack((x, steer_fb)))
            Y.append(x_new - x)
            x = x_new

            if behind_goal == True:
                print("Goal reached!")
                break

            j+=1

        return np.stack(X), np.stack(Y), j

    def _policy(self,pilco, x, random):

        steerAngel = np.linspace(-np.deg2rad(10.0),np.deg2rad(10.0),1000)
        
        if random:
            return np.random.choice(steerAngel) # feedforward as a priori
        else:
            u = pilco.compute_action(x[None, :])[0, :]
            return u
 



@autoflow((float_type,[None, None]), (float_type,[None, None]))
def predict_one_step_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)


@autoflow((float_type,[None, None]), (float_type,[None, None]), (np.int32, []))
def predict_trajectory_wrapper(pilco, m, s, horizon):
    return pilco.predict(m, s, horizon)


@autoflow((float_type,[None, None]), (float_type,[None, None]))
def compute_action_wrapper(pilco, m, s):
    return pilco.controller.compute_action(m, s)


@autoflow((float_type, [None, None]), (float_type, [None, None]))
def reward_wrapper(reward, m, s):
    return reward.compute_reward(m, s)
@autoflow((float_type,[None, None]), (float_type,[None, None]), (np.int32, []))
def predict_wrapper(pilco, m, s, horizon):
    return pilco.predict(m, s, horizon)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def predict_gpr_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def compute_action_wrapper(pilco, m, s):
    return pilco.controller.compute_action(m, s)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def compute_action_wrapper(controller, m, s, squash=False):
    return controller.compute_action(m, s, squash)

@autoflow((float_type, [None, None]), (float_type, [None, None]))
def reward_wrapper(reward, m, s):
    return reward.compute_reward(m, s)

@autoflow()
def get_induced_points(smgpr):
    return smgpr.Z

def save_pilco(path, X, Y, pilco, sparse=False):
    np.savetxt(path + 'X.csv', X, delimiter=',')
    np.savetxt(path + 'Y.csv', Y, delimiter=',')
    if sparse:
        with open(path+ 'n_ind.txt', 'w') as f:
            f.write('%d' % pilco.mgpr.num_induced_points)
            f.close()
    np.save(path + 'pilco_values.npy', pilco.read_values())
    for i,m in enumerate(pilco.mgpr.models):
        np.save(path + "model_" + str(i) + ".npy", m.read_values())

def load_pilco(path, controller=None, reward=None, sparse=False):
    X = np.loadtxt(path + 'X.csv', delimiter=',')
    Y = np.loadtxt(path + 'Y.csv', delimiter=',')
    if not sparse:
        pilco = PILCO(X, Y, controller=controller, reward=reward)
    else:
        with open(path+ 'n_ind.txt', 'r') as f:
            n_ind = int(f.readline())
            f.close()
        pilco = PILCO(X, Y, num_induced_points=n_ind, controller=controller, reward=reward)

    params = np.load(path + "pilco_values.npy").item()
    pilco.assign(params)
    for i,m in enumerate(pilco.mgpr.models):
        values = np.load(path + "model_" + str(i) + ".npy").item()
        m.assign(values)
    return pilco