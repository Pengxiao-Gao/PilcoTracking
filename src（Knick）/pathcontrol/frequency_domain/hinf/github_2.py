from control import *

def make_weight(dc, crossw, hf):
    s = tf([1, 0], [1])
    return (s * hf + crossw) / (s + crossw / dc)

def merge_tfs(tf1, tf2):
    num = [ [tf1.num[0][0], [0] ], [ [0], tf2.num[0][0]] ]   
    den = [ [tf1.den[0][0], [1] ], [ [1], tf2.den[0][0]] ]   
    return tf( num, den ) 

v, lo = 1.1, 1.4
A = [ [0, v], [0, 0] ]
B = [ [lo], [1] ]
C = [ [1, 0], [0, 1] ] 
D = [ [0], [0] ]
Gss = ss(A, B, C, D)


w1y = make_weight(dc=100.0, crossw=0.1, hf=0)
w1Y = make_weight(dc=0.9, crossw=0.5, hf=0)
w1 =  merge_tfs(w1y, w1Y)
w2 = make_weight(dc=0.1, crossw=0.8, hf=1.0)

print("self.Gss:\n", Gss)
Paug = augw(g=Gss, w1=ss(w1) )
print(Paug)
Kss, CL, gamma, rcond = hinfsyn(Paug, 1, 1)
print("hinfsyn gamma:", gamma)