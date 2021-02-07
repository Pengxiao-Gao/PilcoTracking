
'''
    DESCRIPTION: Class for robot parameter
    Author: Ilja Stasewisch, Date: 2019-03-26
'''

class Sim_Parameter() :
    def __init__(self, Ts_sim=1e-2, Ts_ctrl=1e-2, timeout=1000.0) :   
        self.Ts_sim = Ts_sim
        self.Ts_ctrl = Ts_ctrl
        self.timeout = timeout
