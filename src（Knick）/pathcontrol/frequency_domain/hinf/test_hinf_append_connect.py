#%%
import control
from control import *
import matplotlib.pyplot as plt

Gu = tf([1], [1])
Gz = tf([1], [1])
Gy = tf([1], [1])
W1 = tf([2], [1])
W2 = tf([3], [1])
G = tf([1], [1, 1])

sysvec = append(ss(G), ss(W1), ss(W2), ss(Gu), ss(Gz), ss(Gy))


print(sysvec)
P = connect(sysvec, [[1, 4], [3, 4], [6, 1], [6, 5], [2,6] ], [5, 4], [2, 3, 6])

print("\n-------\nP:", P)






#%%
