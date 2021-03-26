import numpy as np
from gpflow import config
from pilco.models.pilco import PILCO
float_type = config.default_float()



def save_pilco(path, X, Y, pilco, sparse=False):
    np.savetxt(path + 'X.csv', X, delimiter=',')
    np.savetxt(path + 'Y.csv', Y, delimiter=',')
    if sparse:
        with open(path+ 'n_ind.txt', 'w') as f:
            f.write('%d' % pilco.mgpr.num_induced_points)
            f.close()
    np.save(path + 'pilco_values.npy', pilco.read_values())
    for i,m in enumerate(pilco.mgpr.models):
        np.save(path + "model_" + str(i) + ".npy", m.read_values())

def load_pilco(path, controller=None, reward=None, sparse=False):
    X = np.loadtxt(path + 'X.csv', delimiter=',', allow_pickle=True)
    Y = np.loadtxt(path + 'Y.csv', delimiter=',', allow_pickle=True)
    if not sparse:
        pilco = PILCO((X, Y), controller=controller, reward=reward)
    else:
        with open(path+ 'n_ind.txt', 'r') as f:
            n_ind = int(f.readline())
            f.close()
        pilco = PILCO((X, Y), num_induced_points=n_ind, controller=controller, reward=reward)
    params = np.load(path + "pilco_values.npy").item()
    pilco.assign(params)
    for i,m in enumerate(pilco.mgpr.models):
        values = np.load(path + "model_" + str(i) + ".npy").item()
        m.assign(values)
    return pilco