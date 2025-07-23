from font_configuration import font_configuration
from matplotlib import pyplot as plt
import numpy as np

def plot_signal(x, fs, xlabel = '', ylabel = ''):
    font_configuration()
    t = np.zeros(shape = x.shape)
    t[0 : len(x)] = np.arange(0, len(x)) * 1.0 / fs
    print(t[-1])
    print(len(x)/fs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xrange = np.max(t) - np.min(t)
    yrange = np.max(x) - np.min(x)
    # ax.set_aspect(5.0 * xrange / yrange / 8.0)
    plt.tight_layout()
    plt.plot(t, x)
    if len(xlabel) > 0:
        plt.xlabel(xlabel)
    if len(ylabel) > 0:
        plt.ylabel(ylabel, rotation = 0.0, labelpad = 20)
    plt.grid()
    plt.show()
