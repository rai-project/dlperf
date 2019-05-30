import numpy as np


def get(model):
    np.random.seed(0)
    return np.asarray(np.random.uniform(model.shape), dtype=np.float32)

