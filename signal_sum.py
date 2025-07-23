import numpy as np

def signal_sum(x, y):
    if len(y) <= len(x):
        ya = np.zeros(shape = x.shape)
        ya[: len(y)] = y
        return x + ya
    else:
        return signal_sum(y, x)
