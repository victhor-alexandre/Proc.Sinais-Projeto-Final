# filter_iterator.py
# map <leader><leader> :wall<cr>:!python %<cr>

from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from qmf_filters_validator import qmf_filters_validator
from scipy.io import loadmat
from upsample import upsample

def font_configuration():
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "font.size": 25,
        })
    except:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "font.size": 16,
        })

def filter_iterator(x0, x1, A, d, levels = 4): # valid both for analysis and synthesis
    x = [0] * (levels + 1)
    x0_ = deepcopy(x0)
    x[levels] = x1
    for k in range(levels - 1, 0, -1):
        x[k] = np.convolve(upsample(x[k + 1], 2), x0)
        x0_ = np.convolve(upsample(x0_, 2), x0)
    x[0] = x0_
    T = np.arange(levels + 1, 0, -1)
    T[0] = levels
    multirate_factors = deepcopy(T)
    multirate_factors = 2 ** multirate_factors
    rescale_factors = deepcopy(T)
    rescale_factors = 1.0 / (A ** T)
    advance_values = deepcopy(T)
    advance_values = d * ((2 ** T) - 1)
    return x, multirate_factors, rescale_factors, advance_values

if __name__ == '__main__':
    M = loadmat('wfilters.mat')
    chosen_wfilter = 44
    h0 = M['h0']
    h1 = M['h1']
    g0 = M['g0']
    g1 = M['g1']
    h0 = h0[0, chosen_wfilter][0]
    h1 = h1[0, chosen_wfilter][0]
    g0 = g0[0, chosen_wfilter][0]
    g1 = g1[0, chosen_wfilter][0]
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)
    h, multirate_factors, rescale_factors, advance_values = filter_iterator(h0, h1, A = A, d = d, levels = 10)
    font_configuration()
    for k in range(len(h) - 1, -1, -1):
        plt.plot(h[k])
        plt.show()
    indices = "1"
    final_indice = "0"
    for k in range(len(h) - 1, -1, -1):
        H = np.fft.fft(h[k], 1000000)
        f = np.linspace(-0.5, 0.5, len(H))
        label = '${|H_{' + indices + '}(f)|}$'
        indices = "0" + indices
        if k == 1:
            indices = final_indice
        else:
            final_indice = "0" + final_indice
        plt.plot(f, np.abs(np.fft.fftshift(H)), label = label)
    plt.xlim((0, 0.5))
    plt.legend(fontsize = 25.0)
    plt.show()
    print(len(h))
