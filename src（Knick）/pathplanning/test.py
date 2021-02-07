
import numpy as np


idx = np.array([1,1])

idx = (1,1)

t = np.array( [[1,2], [3,4]] )
print("t:\n", t)
print("via idx:\n", t[idx] )

import matplotlib.pyplot as plt


fig, axs = plt.subplots(2,2)

axs[0,0].set_visible(False)
plt.show()


