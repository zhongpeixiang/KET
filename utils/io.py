import pickle

def to_pickle(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
