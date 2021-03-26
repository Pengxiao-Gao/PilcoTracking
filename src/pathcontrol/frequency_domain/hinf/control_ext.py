import numpy as np
from control import StateSpace,tf,minreal
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