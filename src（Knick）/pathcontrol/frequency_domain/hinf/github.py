
import control
from control import *
import matplotlib.pyplot as plt
import numpy as np
from control.statesp import _convertToStateSpace

def combine(systems):
    """ systems: 2D array of systems to combine """
    
    rrows=[]
    for srow in systems:
        s1 = srow[0]
        if not isinstance(s1,StateSpace):
            s1=_convertToStateSpace(s1)            
            
        for s2 in srow[1:]:
            if not isinstance(s2, StateSpace):
                s2 = _convertToStateSpace(s2)
            if s1.dt != s2.dt:
                raise ValueError("Systems must have the same time step")            
            n = s1.states + s2.states
            m = s1.inputs + s2.inputs
            p = s1.outputs
            if s2.outputs != p:
                raise ValueError('inconsistent systems')
            A = np.zeros((n, n))
            B = np.zeros((n, m))
            C = np.zeros((p, n))
            D = np.zeros((p, m))
            A[:s1.states, :s1.states] = s1.A
            A[s1.states:, s1.states:] = s2.A
            B[:s1.states, :s1.inputs] = s1.B
            B[s1.states:, s1.inputs:] = s2.B
            C[:, :s1.states] = s1.C
            C[:, s1.states:] = s2.C
            D[:, :s1.inputs] = s1.D
            D[:, s1.inputs:] = s2.D
            s1=StateSpace(A,B,C,D,s1.dt)
        rrows.append(s1)
    r1=rrows[0]
    for r2 in rrows[1:]:
        if r1.dt != r2.dt:
            raise ValueError("Systems must have the same time step")            
        n = r1.states + r2.states
        m = r1.inputs
        if r2.inputs != m:
            raise ValueError('inconsistent systems')
        p = r1.outputs + r2.outputs
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        C = np.zeros((p, n))
        D = np.zeros((p, m))
        A[:r1.states, :r1.states] = r1.A
        A[r1.states:, r1.states:] = r2.A
        B[:r1.states, :] = r1.B
        B[r1.states:, :] = r2.B
        C[:r1.outputs, :r1.states] = r1.C
        C[r1.outputs:, r1.states:] = r2.C
        D[:r1.outputs, :] = r1.D
        D[r1.outputs:, :] = r2.D
        r1=StateSpace(A,B,C,D,r1.dt)
    return r1
    
def make_lowpass(dc, crossw):
    return tf([dc], [crossw, 1])
def make_highpass(hf_gain, m3db_frq):
    return tf([hf_gain, 0], [1, m3db_frq])
def make_weight(dc, crossw, hf):
    s = tf([1, 0], [1])
    return (s * hf + crossw) / (s + crossw / dc)

v = 1.1    
lo = 2.2
A = [ [0, v], [0, 0] ]
B = [ [lo], [1] ]
C = [ [1, 0], [0, 1] ] 
D = [ [0], [0]]
Gss = ss(A, B, C, D)
G = tf(Gss)       
G1 = G[0,0]
G2 = G[1,0]


Wy1 = make_weight(dc=1, crossw=2.1, hf=0)
Wy2 = make_highpass(hf_gain=0.99, m3db_frq=2.1)
Wu = make_highpass(hf_gain=0.3, m3db_frq=2.1)

# P11 = [ [W1, 0, W1*G1], [0, W2, W2*G2], [0, 0, 0] ]
# P12 = [ [W1*G1], [W2*G2], [Wu] ]
# P21 = [ [1, 0, G1], [0, 1, G2] ]
# P22 = [ [G1], [G2] ]
# P = [ [P11, P21], [P21, P22]]
P11=np.block( [ [Wy1, 0, Wy1*G1], [0, Wy2, Wy2*G2], [0, 0,0] ] ) 
P12=np.block( [ [Wy1*G1], [Wy2*G2], [Wu] ])
P21=np.block( [ [1, 0, G1], [0, 1, G2] ] )
P22=np.block( [ [G1], [G2] ] )
P_ = np.block( [ [P11, P12], [P21, P22]] )
Pc = combine(P_)
print(Pc.A.shape)
Pcss = minreal( Pc ) 
print(Pcss)

K, CL, gam, rcond = hinfsyn(Pcss, 2, 1)
Wy = np.block([ [Wy1, 0], [0, v] ])
Wy = combine(Wy)
print(K)

