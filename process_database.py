import numpy as np
import csv
from matplotlib import pyplot as plt
from os.path import exists
from scipy.io import loadmat, savemat
from sklearn import svm
from extract_csv import load_data
from plot_signal import plot_signal
from multivel_multirate_decomposition import multivel_multirate_decomposition
from extract_filters import extract_filters
from qmf_filters_validator import qmf_filters_validator

if __name__ == '__main__':

    #the load function will bring data from the train or test dataset, or the original dataset

    file_type = "train"

    M, target, person_id = load_data(file_type)
    # print(person_id)
    IDs = np.unique(person_id)
    # for ID in IDs:  #for each ID
    #     for i in range(len(M[0,:])):    #for each electrode
            # The idea is to separate the signal of the electrode in multiple 10seconds small signals
            # then apply the QMF, and calculate the energy in each band
            # And finally, create a secondary matrix, that is 3D, where for each electrode at each 10s chunk it tells us the value of the energy of each bandwidth
            # Or maybe create 5 matrixes, each corresponding to one of the bandwidth's energy
            # Or maybe even, create 19 matrixes (one for each elecrode) with 5 columns (the bandwidths), and each 10s signal is a different "patient"

    # print(unique[0])
    m = M[person_id == IDs[0]]
    # print(m)
    x = m[:,0]
    # print(len(x))
    # plot_signal(x, 128, "t", f"eeg for the id {unique[0]}")
    fs = 128
    segment_samples = int(fs * 10)
    num_segments = int(np.ceil(len(x)/segment_samples))

    segments = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = x[start:end]
        segments.append(segment)

    # maybe use both daubechies 4 and 6, and compare results
    h0, h1, g0, g1 = extract_filters(5) #db6
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)
    x_hat, x_decomp = multivel_multirate_decomposition(segments[0], h0, h1, A, d, levels = 5)
    print(f"x_hat: \n{x_hat}")
    print(f"len of x_decomp = {len(x_decomp)}")
    # print(f"x_decomp: \n{x_decomp}")
    plot_signal(x_hat, 1, "f", "EEG")

    #TODO
    #next steps:
    #decompose the signal in the five frequencies (alpha, beta...)
    #data augmentation, by taking chunks of 10 seconds from each signal, from each electrode
    #features will be, in this approach, the energy in each frequency band, in each time chunk, in each electrode
    #essentially, for a single patient, we have multiple 10 seconds chunks of data, that will have 5 features each (the energy in each frequency band)
