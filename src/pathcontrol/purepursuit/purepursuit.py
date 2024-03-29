'''
    Path-Tracking with Pure Pursuit
    Author: Ilja Stasewisch, Date: 2019-06-04
'''



# PYTHON
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import threading
from threading import Thread, Lock
from math import fabs, sin, cos
import sys
import control

class PurePursuit :
    
    def __init__(self, system=System()) :
        print('Path Tracking Control: Pure Pursuit')

        # CLASS-VARIABLES
        self.mutex = Lock()
        self.tf_handler = Tf_handler()
        self.system = system
        self.offset_frame = rospy.get_param("parameter/system/offset_frameId", "offset")
        self.global_frame = rospy.get_param("parameter/system/global_frameId", "map")
        self.base_frame = rospy.get_param("parameter/system/base_frameId", "base_frame")
        self.Ts_ctrl = rospy.get_param("pure_pursuit/Time_sampling", 1e-1)
        self.length_lookahead = rospy.get_param("pure_pursuit/length/look_ahead", 3.0)

        self.path_ds = 0.0
        self.index_lookahead = -1
        self.index_nn_ff = -1

        # ROS-Subscriber
        rospy.Subscriber("/path_rearaxle", Nav.Path, self.pathCb)

        # ROS-Publisher
        self.cmd_pub = rospy.Publisher('/drive/cmd/ackermann', AckermannDriveStamped, queue_size=10)
        self.pathoffset_pub = rospy.Publisher('/path_offset', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('/path_local', Path, queue_size=10)
        self.desired_pose_pub = rospy.Publisher('/purepursuit/desired_pose/lookahead', PoseStamped, queue_size=10)
        self.desiredpose_ff_pub = rospy.Publisher('/purepursuit/desired_pose/feedforward', PoseStamped, queue_size=10)

        # Path-Tracking-Tread
        self.nav_path = Nav.Path()
        pt_thread = threading.Thread(target=self.pathTrackingCb)
        pt_thread.start()    



    def pathCb(self, msg) :
        self.mutex.acquire()
        self.nav_path = msg
        self.mutex.release()


    def wait_for_rosMsg(self) :
        while not rospy.is_shutdown() :
            self.mutex.acquire()
            nav_path = self.nav_path
            self.mutex.release()
            if nav_path.header.seq > 1 :
                print "asdsasssssssssssssss"
                return
            else:
                rospy.loginfo_throttle(1.0, "PATH-TRACKING-LMPC: wait for path, and other ROS-Msgs...")
                rospy.Rate(10.0).sleep()

    def pathHandling(self, path_src, system) :
        positions = navPath_to_positionNpArray(path_src)     
        path2d = Path2D_Handler(x=positions[:,0], y=positions[:, 1], wheelbase=system.wheelbase, velocity=system.velocity, offset=system.length_offset)
        self.path_ds = path2d.ds_mean
        simple_path = SimplePath(x=path2d.x_offset, y=path2d.y_offset, yaw=path2d.yaw, \
                                 steering_angle=path2d.steer, steering_rate=path2d.steerRate, frame_id=path_src.header.frame_id)



        # TO NAV-PATH
        pathPosEul = np.zeros( (simple_path.x.shape[0], 6) )
        pathPosEul[:, 0] = simple_path.x
        pathPosEul[:, 1] = simple_path.y
        pathPosEul[:, 5] = simple_path.yaw       
        path_offset = positionEulerNpArray_to_navPath(pathPosEul)
        path_offset.header.frame_id = self.global_frame
        path_offset.header.stamp = rospy.get_rostime()

        return path_offset, simple_path



    def pathTrackingCb(self) :
        # Wait for Path and others ros-msgs
        self.wait_for_rosMsg()
        self.mutex.acquire()
        path_rearAxle = self.nav_path
        self.mutex.release()

        # Shortening Variables
        print type(System())
        print type(self.system)
        wheelbase = self.system.wheelbase
        length_offset = self.system.length_offset
        velocity = self.system.velocity
        length_lookahead = self.length_lookahead
        Ts = self.Ts_ctrl

        

        # PATH-HANDLING      
        path_offset, simplepath_offset = self.pathHandling(path_rearAxle, self.system)
        self.pathoffset_pub.publish(path_offset)

        while not rospy.is_shutdown() :
            stopwatch = rospy.get_time()

            stopwatch_tf = rospy.get_time()
            # offsetframe_2_pathframe = self.tf_handler.get_last_tfMatrix(self.offset_frame, self.nav_path.header.frame_id)
            pathframe_2_offsetframe = self.tf_handler.get_last_tfMatrix(self.nav_path.header.frame_id, self.offset_frame)  
            pathframe_2_baseframe = self.tf_handler.get_last_tfMatrix(self.nav_path.header.frame_id, self.base_frame)  

            # Pure-Pursuit Equation: See: ILJA AMAZON PHOTOS (DECEMBER 2018)
            stopwatch_angle = rospy.get_time()
            alpha, desPathPoseIdx = self.get_error_angle(path_offset, pathframe_2_offsetframe)
            _ = self.get_index_nearest_neighbour(pathframe_2_offsetframe, path_offset, True)

            pure_pursuit = np.arctan(wheelbase * sin(alpha) / (length_lookahead/2.0 - length_offset * cos(alpha)) )
            # pure_pursuit = 0.0

            stopwatch_nn = rospy.get_time()
            idx_ff = self.get_index_nearest_neighbour(pathframe_2_baseframe, path_rearAxle)
            # print "calc nn = ", rospy.get_time() - stopwatch_nn

            steerangle_ff = simplepath_offset.steering_angle[idx_ff]
            desired_steer_angle = steerangle_ff + pure_pursuit

            velocity_adjust = self.reduce_velocity_near_goal(pathframe_2_offsetframe, path_offset, desired_velocity=velocity)
            # velocity_adjust = 0.0



            # Check Goal reached otherwise send Command!
            if self.behind_goal(actual=pathframe_2_offsetframe, path=path_offset, desired_velocity=velocity) == True :      
                print "BREAK CONTROL - BEHIND GOAL"   
                self.publish_driveCmd(desired_steer_angle, 0.0)
                break
            else :
                self.publish_driveCmd(desired_steer_angle, velocity_adjust)

            # T_sampling-Loop-Waiting
            looptime = rospy.get_time() - stopwatch
            waittime = Ts - looptime
            # print "ctrl looptime = ", looptime
            if looptime > Ts:
                # pass
                rospy.logerr("Looptime is greater than Samling: looptime = " + str(looptime) + " > " + str(Ts) + " = Ts")
            else :
                pass
                # print "ctrl looptime = ", looptime
            if waittime > 1e-5: 
                rospy.Rate(1.0 / waittime).sleep()



    def publish_driveCmd(self, steering_angle, velocity) : 
        cmd_msg = AckermannDriveStamped()
        cmd_msg.header.stamp = rospy.get_rostime()
        cmd_msg.drive.steering_angle = steering_angle
        cmd_msg.drive.steering_angle_velocity = 0.0
        cmd_msg.drive.speed = velocity
        self.cmd_pub.publish(cmd_msg)

    def get_index_nearest_neighbour(self, actual_tfMatrix, path, print_distance = False) : # Get nearest neighbour to path by position
        ''' Set Loop_Variables'''
        dL = float('Inf')
        index = 0

        # Define Start-Idx for the Loop
        meters_around_last_idx = 1.0
        start_index = self.index_nn_ff - int(meters_around_last_idx / self.path_ds)
        if start_index < 0 : start_index = 0

        # Define End-Idx for the Loop
        if self.index_lookahead < 0: # self.index_lookahead is not inited
            end_index = len(path.poses)
        else: 
            end_index = self.index_nn_ff + int( meters_around_last_idx / np.fabs(self.path_ds)  )
        if end_index > len(path.poses) : end_index = len(path.poses)


        ''' Get index for neatest euclidean distance '''
        for i in range(start_index, end_index) :
            ''' OLD, NICER BUT SLOWER '''
            # iPath_tfMat = geoPose_to_tfMatrix(path.poses[i].pose)
            # dL_ = getLength(actual_tfMatrix, iPath_tfMat)
            ''' NEW, LESS NICER BUT FASTER '''
            px = path.poses[i].pose.position.x;
            py = path.poses[i].pose.position.y;
            rx = actual_tfMatrix[0, 3]
            ry = actual_tfMatrix[1, 3]
            dL_ = np.hypot(px-rx, py-ry)

            if dL_ < dL :
                index = i
                dL = dL_
            # elif dL_ * 3.0 > dL :
                # break
     
        if print_distance == True:
            rospy.loginfo_throttle(period=0.1, msg="dist to path = " + str(dL) )

        self.index_nn_ff = index

        desired_pose = path.poses[index]
        desired_pose.header.frame_id = path.header.frame_id
        desired_pose.header.stamp = rospy.get_rostime()
        self.desiredpose_ff_pub.publish(desired_pose);

        return index


    def transform_pathoffset_to_offsetframe(self, simplepath_offset) :
        # Prepare Transformation        
        offset_frame = self.offset_frame
        xG = simplepath_offset.x
        yG = simplepath_offset.y
        yawG = simplepath_offset.yaw
        xL = np.zeros(xG.shape)
        xL = np.reshape(xL, (xL.shape[0],1) )
        yL = np.zeros(yG.shape)
        yL = np.reshape(yL, (yL.shape[0],1) )
        yawL = np.zeros(yawG.shape)
        yawL = np.reshape(yawL, (yawL.shape[0],1) )
        offsetframe_2_pathframe = self.tf_handler.get_last_tfMatrix(offset_frame, simplepath_offset.frame_id)
        
        # Execute Transformation
        for i, (ixG, iyG, iyawG) in enumerate(zip(xG, yG, yawG)) :
            pathframe_2_pathpose = xyYaw_to_tfMatrix(ixG, iyG, iyawG)
            offsetframe_2_pathpose = np.matmul(offsetframe_2_pathframe, pathframe_2_pathpose)
            _, _, yaw = tfMatrix_to_eulerAngles(offsetframe_2_pathpose)
            yawL[i] = yaw
            position = tfMatrix_to_position(offsetframe_2_pathpose)
            xL[i] = position[0]
            yL[i] = position[1]

        # Publish LOCAL-PATH
        pos2D_yaw_array = np.hstack( (xL, yL, yawL))
        local_navpath = position2DyawNpArray_to_navPath(pos2D_yaw_array)
        local_navpath.header.stamp = rospy.get_rostime()
        local_navpath.header.frame_id = offset_frame
        self.local_path_pub.publish(local_navpath)

        # Define and Return-Variale
        local_path = SimplePath()
        local_path.x = xL
        local_path.y = yL
        local_path.yaw = yawL
        return local_navpath, local_path

    def get_error_angle(self, path_offset, actual_tfMatrix) :
        # INIT VARIABLES TO CHOOSE THE DESIRED POSE
        dist_to_lookahead = float("inf")
        path_desired_idx = 0
        

        # Define Start-Idx for the Loop
        meters_around_last_idx = 1.0
        # print "self.path_ds = ", self.path_ds
        # print "t = ", int(meters_around_last_idx / self.path_ds)
        start_index = self.index_lookahead - int(meters_around_last_idx / self.path_ds)
        if start_index < 0 : start_index = 0
        # print "start_index = ", start_index

        # Define End-Idx for the Loop
        if self.index_lookahead < 0: # self.index_lookahead is not inited
            end_index = len(path_offset.poses)
        else: 
            end_index = self.index_lookahead + int( meters_around_last_idx / np.fabs(self.path_ds)  )
        if end_index > len(path_offset.poses) : end_index = len(path_offset.poses)
        # print "end_index = ", end_index

        # GET DESIRED-POSE
        for i in range(start_index, end_index) :           
            iPath_tfMat = geoPose_to_tfMatrix(path_offset.poses[i].pose)
            dist_offset2path = getLength2DSign(actual_tfMatrix, iPath_tfMat)
            if dist_offset2path < 0.0 :
                continue
            else :
                eDist = dist_offset2path - self.length_lookahead 

                if eDist <= 0.0 :
                    continue
                elif eDist < dist_to_lookahead :
                    dist_to_lookahead = eDist
                    path_desired_idx = i
                    if dist_to_lookahead < self.path_ds:
                        break


        self.index_lookahead = path_desired_idx
        desired_pose = path_offset.poses[path_desired_idx]
        desired_pose.header.frame_id = path_offset.header.frame_id
        desired_pose.header.stamp = rospy.get_rostime()
        self.desired_pose_pub.publish(desired_pose);

        global_2_path = geoPose_to_tfMatrix(desired_pose.pose)
        actual_2_global =  np.linalg.inv(actual_tfMatrix)
        dTf = np.matmul(actual_2_global, global_2_path)
        
        error_angle = np.arctan2(dTf[1,3], dTf[0,3])
        return error_angle, path_desired_idx


    def reduce_velocity_near_goal(self, actual, path, desired_velocity) :
        now = actual
        goal = geoPose_to_tfMatrix(path.poses[-1].pose)
        dL = getLength(goal, now)
        length_reduce_velocity = rospy.get_param("pure_pursuit/length/reduce_velocity", 0.5)
        if dL < length_reduce_velocity :
            velocity = desired_velocity * dL * 2.0
            if math.fabs(velocity) < 0.1 :
                velocity = 0.1 * np.sign(desired_velocity)
            elif velocity > desired_velocity :
                velocity = desired_velocity
        else :
            velocity = desired_velocity


        return velocity

    def behind_goal(self, actual, path, desired_velocity) :
        now = actual
        goal = geoPose_to_tfMatrix(path.poses[-1].pose)
        dL = getLength2DSign(goal, now)
        if fabs(dL) > 1.0:
            return False
        elif dL * np.sign(desired_velocity) > 0.0:
            print "GOAL REACHED!"
            return True
        else :
            return False


