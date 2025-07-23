import numpy as np

def upsample(x, M = 2, remove_leading_zeros = True):
    y = np.zeros(shape = (x.shape[0] * M, ))
    y[0:y.shape[0]:M] = x
    if remove_leading_zeros:
        y = y[: len(y) - (M - 1)]
    return y
