import numpy as np

def downsample(x, n = [], M = 2):
    if len(n) == 0:
        n = np.arange(0, len(x))
    k = [n[i] == 0 for i in range(0, len(n))]
    k = np.where(k)[0][0]
    x_right = x[k :]
    y_right = x_right[: len(x_right) : M]
    x_left = x[: k + 1]
    x_left = x_left[::-1]
    y_left = x_left[: len(x_left) : M]
    y_left = y_left[::-1]
    y_left = y_left[: len(y_left) - 1]
    n_ = np.arange(-len(y_left), len(y_right))
    y = np.concatenate((y_left, y_right))
    return y, n_
