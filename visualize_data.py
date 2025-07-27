#python libraries
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat

from plot_signal import plot_signal
from multivel_multirate_decomposition import multivel_multirate_decomposition
from extract_filters import extract_filters
from qmf_filters_validator import qmf_filters_validator
from font_configuration import font_configuration

def plot_signal_single_electrode(M, electrode, person_id, epoch_id, fs):
    print(f"Chosen person: {chosen_person}")
    print(f"Chosen epoch: {chosen_epoch}")
    print(f"Chosen electrode: {chosen_electrode}")
    x = M[person_id == chosen_person]
    x = np.concatenate([seg[:,chosen_electrode] for seg in x])
    print(len(x))
    print(x)

    
    x_title = f'EEG for {chosen_person} person and {chosen_electrode} electrode'
    x_hat_title = f"Wavelet (db4) decomposition for {chosen_person} person and {chosen_electrode} electrode"

    
    h0, h1, g0, g1 = extract_filters(5) #db4
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)

    x_hat, _ = multivel_multirate_decomposition(x, h0, h1, A, d, levels=4)
    

    font_configuration()
    t = np.zeros(shape = x.shape)
    t[0 : len(x)] = np.arange(0, len(x)) * 1.0 / fs

    n = np.zeros(shape = x_hat.shape)
    n[0 : len(x_hat)] = np.arange(0, len(x_hat)) *1.0 / 1



    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    axes[0].plot(t, x)
    axes[0].set_title(x_title)
    axes[0].grid()

    axes[1].plot(n, x_hat)
    axes[1].set_title(x_hat_title)
    axes[1].grid()


    plt.tight_layout()

    plt.grid(True)
    plt.show()
    return

def plot_id_all_electrodes(M, chosen_person, chosen_epoch, chosen_target):
    font_configuration()
    fs = 128

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 10))
    x_title = f'EEG for ID {chosen_person} and epoch nº {chosen_epoch}. Class = {chosen_target}'

    x_ = np.transpose(M[person_id == chosen_person])
    a, b, c = x_.shape
    x = x_.reshape(a, b * c)
    print(x.shape)
    t = np.zeros(shape = x.shape)
    print(t.shape)
    print(len(range(5)))
    for electrode in range(5):
        t[electrode][0 : len(x[electrode])] = np.arange(0, len(x[electrode])) * 1.0 / fs
        axes[electrode].plot(t[electrode], x[electrode])
        axes[electrode].set_title(f"electrode number {electrode+1}")
        axes[electrode].grid()

    plt.xlabel(x_title)
    plt.ylabel("Electrodes (1 to 5)")
    plt.tight_layout()

    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 10))
    for n in range(5):        
        electrode = n+5
        t[electrode][0 : len(x[electrode])] = np.arange(0, len(x[electrode])) * 1.0 / fs
        axes[n].plot(t[electrode], x[electrode])
        axes[n].set_title(f"electrode number {electrode+1}")
        axes[n].grid()

    plt.xlabel(x_title)
    plt.ylabel("Electrodes (6 to 10)")
    plt.tight_layout()
    plt.show()


    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 10))
    for n in range(5):
        electrode = n+10
        t[electrode][0 : len(x[electrode])] = np.arange(0, len(x[electrode])) * 1.0 / fs
        axes[n].plot(t[electrode], x[electrode])
        axes[n].set_title(f"electrode number {electrode+1}")
        axes[n].grid()

    plt.xlabel(x_title)
    plt.ylabel("Electrodes (11 to 15)")
    plt.tight_layout()

    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
    for n in range(4):
        electrode = n+15
        t[electrode][0 : len(x[electrode])] = np.arange(0, len(x[electrode])) * 1.0 / fs
        axes[n].plot(t[electrode], x[electrode])
        axes[n].set_title(f"electrode number {electrode+1}")
        axes[n].grid()

    plt.xlabel(x_title)
    plt.ylabel("Electrodes (16 to 19)")
    plt.tight_layout()

    plt.grid(True)
    plt.show()



    return


def plot_epoch_all_electrodes(M, chosen_person, chosen_epoch, chosen_target):
    font_configuration()
    fs = 128

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 10))
    x_title = f'EEG for ID {chosen_person} and epoch nº {chosen_epoch}. Class = {chosen_target}'
    
    x = np.transpose(M[chosen_epoch])
    print(x.shape)
    t = np.zeros(shape = x.shape)
    print(t.shape)
    print(len(range(5)))
    for electrode in range(5):
        t[electrode][0 : len(x[electrode])] = np.arange(0, len(x[electrode])) * 1.0 / fs
        axes[electrode].plot(t[electrode], x[electrode])
        axes[electrode].set_title(f"electrode number {electrode+1}")
        axes[electrode].grid()

    plt.xlabel(x_title)
    plt.ylabel("Electrodes (1 to 5)")
    plt.tight_layout()

    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 10))
    for n in range(5):        
        electrode = n+5
        t[electrode][0 : len(x[electrode])] = np.arange(0, len(x[electrode])) * 1.0 / fs
        axes[n].plot(t[electrode], x[electrode])
        axes[n].set_title(f"electrode number {electrode+1}")
        axes[n].grid()

    plt.xlabel(x_title)
    plt.ylabel("Electrodes (6 to 10)")
    plt.tight_layout()
    plt.show()


    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 10))
    for n in range(5):
        electrode = n+10
        t[electrode][0 : len(x[electrode])] = np.arange(0, len(x[electrode])) * 1.0 / fs
        axes[n].plot(t[electrode], x[electrode])
        axes[n].set_title(f"electrode number {electrode+1}")
        axes[n].grid()

    plt.xlabel(x_title)
    plt.ylabel("Electrodes (11 to 15)")
    plt.tight_layout()

    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
    for n in range(4):
        electrode = n+15
        t[electrode][0 : len(x[electrode])] = np.arange(0, len(x[electrode])) * 1.0 / fs
        axes[n].plot(t[electrode], x[electrode])
        axes[n].set_title(f"electrode number {electrode+1}")
        axes[n].grid()

    plt.xlabel(x_title)
    plt.ylabel("Electrodes (16 to 19)")
    plt.tight_layout()

    plt.grid(True)
    plt.show()


    return

if __name__ == "__main__":

    D = loadmat("test_data.mat")
    M = D['M']
    target = D['t'].flatten()
    person_id = D['i'].flatten()
    epoch_id = D['e']
    fs = 128
    unique_id = np.unique(person_id)
    np.random.shuffle(unique_id)
    chosen_person = unique_id[0]

    chosen_electrode = np.random.randint(0,19)

    chosen_epoch = np.random.randint(0,len(M[person_id == chosen_person]))
    chosen_target = target[person_id == chosen_person][0]


    plot_id_all_electrodes(M, chosen_person, chosen_epoch, chosen_target)

    