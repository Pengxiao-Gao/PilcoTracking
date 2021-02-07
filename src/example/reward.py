import numpy as np
import matplotlib.pyplot as plt
import sys
import os
main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(main_path)


Total = np.array([0,89.623249,89.73282327,89.74366625])
Predict = np.array([0,96.70415903,96.70415903,96.70415903])

Total = np.load(main_path+"/SAVED/total_rewards.npy")
Predict = np.load(main_path+"/SAVED/ep_rewards.npy")

with plt.style.context(['science', 'ieee']):

    Zustand = r'$\Delta \dot{e}_\theta$'
    plt.plot(Total,'k')
    plt.plot(Predict,'r')
    # plt.title(Zustand)
    plt.xlabel('Versuchen')
    plt.ylabel('Kulmulative Belohneng')
    # plt.xticks(x)
    plt.ylim(0,30)
    plt.legend(['echte','Ziel'])
    plt.title('Lineares Modell')
    plt.show()

