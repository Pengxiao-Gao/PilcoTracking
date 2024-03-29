from scipy.special import fresnel
import numpy as np
import matplotlib.pyplot as plt
from pathplanning.libs.extend_transformations import *
from pathplanning.libs.transformations import *
import time



###################################################
###################################################
R = 5.0
l_end = 1.0 / R / np.pi 
ds_desired = 1e-2
no_points = 2
while True:
    print("###################################################")
    xClothoid, yClothoid = get_clothoid(l_end, no_points)
    


    # Pick elements for the distance between the points
    x_, y_ = [xClothoid[0]], [yClothoid[0]]
    iPrv = 0
    for i, v in enumerate(xClothoid):
        dx = xClothoid[i] - xClothoid[iPrv]
        dy = yClothoid[i] - yClothoid[iPrv]
        dL = np.hypot(dx, dy)

        if dL > ds_desired:
            x_.append(xClothoid[i] )
            y_.append(yClothoid[i] )
            iPrv = i

    # Check BREAK-Condition
    dx, dy = np.diff(x_), np.diff(y_)
    dL = np.hypot(dx, dy)
    mean_dL = np.mean(dL) 
    e_dL = mean_dL - ds_desired
    if np.fabs(e_dL) < 1e-5:
        break

    # Calc new number of points for clothoid generation
    delta_nopoints = int(np.fabs(e_dL * 1000))
    if delta_nopoints < 1: delta_nopoints = 1
    no_points += delta_nopoints
 


    


xClothoid = np.array(x_)
yClothoid = np.array(y_)




# yClothoid *= -1.0

dx = np.diff(x_)
dy = np.diff(y_)
ddx = np.diff(dx)
ddy = np.diff(dy)


dx = dx[0:-1]
dy = dy[0:-1]
print(dx.size)
print(ddx.size)
kappa = (dx * ddy - ddx * dy) / (dx**2.0 + dy **2.0)**(3.0/2.0)


ax1 = plt.subplot(211)
ax1.plot(xClothoid, yClothoid)

ax2 = plt.subplot(212)
ax2.plot(1/kappa)

print("kappa[0], kappa[-1], 1/R:", kappa[0], kappa[-1], 1/R)
print("l_end:", l_end)
# ax1.set_aspect("equal")

plt.show()