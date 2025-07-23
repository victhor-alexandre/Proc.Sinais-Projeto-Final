# map <leader><leader> :wall<cr>:!python %<cr>

from extract_filters import extract_filters
from multivel_multirate_decomposition import multivel_multirate_decomposition
from multivel_multirate_reconstruction import multivel_multirate_reconstruction
import numpy as np
from plot_signal import plot_signal
from qmf_filters_validator import qmf_filters_validator
from scipy.io import loadmat

def elimanated_percentage(x, th):
    k = np.where(np.abs(x) < th)[0]
    k = len(k)
    return k / len(x) * 100

def apply_threshold(x, percent_to_keep = 80, tol = 1):
    finished = False
    th_max = np.max(np.abs(x))
    th_min = 0
    th = 0
    while not(finished):
        ep = elimanated_percentage(x, th)
        if np.abs(percent_to_keep - (100 - ep)) < tol:
            finished = True
        else:
            if ep > 100 - percent_to_keep
                # decrease th
                th_max = th
                th = (th_min + th) / 2.0
            else:
                # decrease th
                th_min = th
                th = (th + th_max) / 2.0
    # Apply the determined threshold 
    # return the thresholded signal.


if __name__ == '__main__':
    M = loadmat('ECG_1.mat')
    x = M['x']
    x = x[:, 0]
    x = x
    fs = M['fs']
    plot_signal(x, fs, xlabel = 'Tempo t (segundos)', ylabel = 'x_c(t)')
    h0, h1, g0, g1 = extract_filters(4)
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)
    x_hat, x_decomp = multivel_multirate_decomposition(x, h0, h1, A, d, levels = 4)
    plot_signal(x_hat, 1, xlabel = 'Índice da transformada k', ylabel = 'x_hat[k]')
    apply_threshold(x_hat, percent_to_keep = 100)
