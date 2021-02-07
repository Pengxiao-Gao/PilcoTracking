import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel


x_, y_ = [], []
R = 5.0
l_end = 1.0 / R / np.pi 
l = np.linspace(0.0, l_end, 50)
print(l, l_end)
A = np.sqrt(R*l_end)

for il in l:
    y, x = np.array(fresnel(il)) * A * np.sqrt(np.pi)
    x_.append(x)
    y_.append(y)


dx = np.diff(x_)
dy = np.diff(y_)
ddx = np.diff(dx)
ddy = np.diff(dy)


dx = dx[0:-1]
dy = dy[0:-1]
print(dx.size)
print(ddx.size)
kappa = (dx * ddy - ddx * dy) / (dx**2.0 + dy **2.0)**(3.0/2.0)
print("RADIUS:\n", 1/kappa)

ax1 = plt.subplot(211)
ax1.plot(x_, y_)

ax2 = plt.subplot(212)
ax2.plot(kappa)

plt.show()