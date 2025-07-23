# multivel_multirate_decomposition.py
# map <leader><leader> :wall<cr>:!python %<cr>

from copy import deepcopy
from extract_filters import extract_filters
from filter_iterator import filter_iterator
from multivel_multirate_decomposition import multivel_multirate_decomposition
from matplotlib import pyplot as plt
import numpy as np
from plot_signal import plot_signal
from qmf_filters_validator import qmf_filters_validator
from scipy.io import loadmat
from signal_sum import signal_sum
from upsample import upsample

def multivel_multirate_reconstruction(x_decomp, g0, g1, A, d):
    g, multirate_factors, rescale_factors, advance_values = filter_iterator(g0, g1, A, d, levels = len(x_decomp) - 1)
    xd = deepcopy(x_decomp)
    xr = np.zeros(shape = [1, ])
    for k in range(0, len(xd)):
        xd[k] = upsample(xd[k], M = multirate_factors[k])
        xd[k] = np.convolve(g[k], xd[k])
        xd[k] *= rescale_factors[k]
        x_ = xd[k]
        xd[k] = x_[advance_values[k]:]
        xr = signal_sum(xr, xd[k])
    return xr

def font_configuration():
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "font.size": 26,
        })
    except:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "font.size": 16,
        })

if __name__ == '__main__':
    M = loadmat('ECG_1.mat')
    x = M['x']
    x = x[:, 0]
    x = x
    fs = M['fs']
    plot_signal(x, fs, xlabel = 'Tempo t (segundos)', ylabel = 'x_c(t)')
    h0, h1, g0, g1 = extract_filters(4)
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)
    x_hat, x_decomp = multivel_multirate_decomposition(x, h0, h1, A, d, levels = 5)
    plot_signal(x_hat, 1, xlabel = 'Índice da transformada k', ylabel = 'x_hat[k]')
    xr = multivel_multirate_reconstruction(x_decomp, g0, g1, A, d)
    plot_signal(xr, fs)
    err = xr[:len(x)] - x
    plot_signal(err, fs)
    print(len(x))
    print(len(xr))
    print(len(h0))
    print(len(h1))
    print(len(g0))
    print(len(g1))
