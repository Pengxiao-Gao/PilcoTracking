import pickle

def save_data(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_data(name ):
    try:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        with open(name, 'rb') as f:
            return pickle.load(f)