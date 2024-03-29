
'''
    DESCRIPTION: Class for robot parameter
    Author: Ilja Stasewisch, Date: 2019-03-26
'''

class Robot_Parameter() :
    def __init__(self,
                wheelbase=3.0, length_offset=-1.0, look_ahead=0.0, # Length
                max_steerAngle=0.6, T_steer=0.375, min_velocity=1.0, max_velocity=1.0) :  # Velocity
        self.type = "Ackermann"

        # Geometry
        self.wheelbase = wheelbase
        self.length_offset = length_offset
        self.look_ahead = look_ahead
                
        # Velocity
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.mean_velocity = (min_velocity + max_velocity) / 2.0

        # Steering
        self.max_steerAngle = max_steerAngle
        self.T_PT1_steer = T_steer
        self.T2_PT2_steer = 0.0186
        self.T1_PT2_steer = 0.1893
        self.tDead_steer = 0.02
        self.ramp_delta = 0.8395
    
class Skid_Parameter() :
    def __init__(self,
                 wheelbase=3.0, track=1.0, length_offset=-1.0,
                 max_velocity=0.6, look_ahead = 2.0, T_pt1_vel=None, min_velocity=None) :   
        
        self.type = "Skid"
        
        # Geometry
        self.wheelbase = wheelbase
        self.track = track
        self.half_track = track/2.0
        self.length_offset = length_offset
        self.look_ahead = look_ahead
                
        # Velocity
        if min_velocity is None:
            self.min_velocity = max_velocity
        else:
            self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        # Velocity Dynamic
        self.T_pt1_vel = T_pt1_vel
        self.T_pt1_dV = T_pt1_vel


class Articulated_Parameter() :
    def __init__(self,
                 length_front=1.5, length_rear=1.5, length_offset=-1.0,
                 max_steerAngle=0.6, T_steer=0.02, min_velocity=None, max_velocity=1.0,look_ahead=0.0) :   
        
        self.type = "Articulated"
        
        # Geometry
        self.wheelbase = length_front + length_rear
        self.length_front = length_front
        self.length_rear = length_rear
        self.length_offset = length_offset
        self.look_ahead = look_ahead
                
        # Velocity
        if min_velocity is None:
            self.min_velocity = max_velocity
        else:
            self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.mean_velocity = (self.min_velocity + max_velocity) / 2.0

        # Steering
        self.max_steerAngle = max_steerAngle
        self.T_PT1_steer = T_steer
        self.T2_PT2_steer = 0.0186
        self.T1_PT2_steer = 0.1893
        self.tDead_steer = 0.02
        self.ramp_delta = 0.8395

