# multivel_multirate_decomposition.py
# map <leader><leader> :wall<cr>:!python %<cr>

from downsample import downsample
from filter_iterator import filter_iterator
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat

def unite(x):
    N = 0
    for k in range(0, len(x)):
        N += len(x[k])
    y = np.zeros(shape = (N,))
    n = 0
    for k in range(0, len(x)):
        y[n : n + len(x[k])] = x[k]
        n += len(x[k])
    return y

def multivel_multirate_decomposition(x, h0, h1, A, d, levels = 4):
    h, multirate_factors, _, _ = filter_iterator(h0, h1, A, d, levels = levels)
    x_decomp = [0] * len(h)
    for k in range(0, len(h)):
        x_ = np.convolve(h[k], x)
        x_, _ = downsample(x_, M = multirate_factors[k])
        x_decomp[k] = x_
    x_hat = unite(x_decomp)
    return x_hat, x_decomp

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
    font_configuration()
    M = loadmat('ECG_1.mat')
    x = M['x']
    x = x[:, 0]
    x = x - np.mean(x)
    fs = M['fs']
    t = np.zeros(shape = x.shape)
    t[0 : len(x)] = np.arange(0, len(x)) * 1.0 / fs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xrange = np.max(t) - np.min(t)
    yrange = np.max(x) - np.min(x)
    # ax.set_aspect(5.0 * xrange / yrange / 8.0)
    plt.tight_layout()
    plt.plot(t, x)
    plt.xlabel('Time ${t}$ (seconds)')
    plt.ylabel('${x_c(t)}$', rotation = 0.0, labelpad = 20)
    plt.grid()
    plt.show()
    M = loadmat('wfilters.mat')
    chosen_wfilter = 4
    h0 = M['h0']
    h1 = M['h1']
    g0 = M['g0']
    g1 = M['g1']
    h0 = h0[0, chosen_wfilter][0]
    h1 = h1[0, chosen_wfilter][0]
    g0 = g0[0, chosen_wfilter][0]
    g1 = g1[0, chosen_wfilter][0]
    # h0 = np.array([1, 1])
    # h1 = np.array([1, -1])
    # g0 = np.array([1, 1])
    # g1 = np.array([-1, 1])
    x_hat, x_decomp = multivel_multirate_decomposition(x, h0, h1, levels = 4)
    print(x_hat)
    n = np.arange(0, len(x_hat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xrange = np.max(n) - np.min(n)
    yrange = np.max(x_hat) - np.min(x_hat)
    # ax.set_aspect(5.0 * xrange / yrange / 8.0)
    plt.tight_layout()
    plt.plot(x_hat)
    plt.xlabel('Índice da transformada k')
    plt.ylabel('${\hat{x}[k]}$', rotation = 0.0, labelpad = 20)
    plt.grid()
    plt.show()
    print(len(x))
    print(len(x_hat))
