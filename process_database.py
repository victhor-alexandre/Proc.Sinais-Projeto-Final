#python libraries
import numpy as np
import csv
from matplotlib import pyplot as plt
from os.path import exists
from os import makedirs
from scipy.io import loadmat, savemat
from sklearn import svm
from tqdm import tqdm 

#libraries developed in class
from plot_signal import plot_signal
from multivel_multirate_decomposition import multivel_multirate_decomposition
from extract_filters import extract_filters
from qmf_filters_validator import qmf_filters_validator

#my libraries
from extract_csv import load_data


def extract_qmf_features(band_list, Fs, signal):
    # Standard EEG frequency bands will coincide with the qmf bands
    standard_bands = {
        'delta': (0, 4),
        'theta': (4, 8),
        'alpha': (8, 16),
        'beta': (16, 32),
        'gamma': (32, Fs/2)
    }
    
    features = []
    # Raw signal temporal features
    features.extend([
        np.std(signal),     #temporal standard deviation
    ])

    for i, (name, (f_low, f_high)) in enumerate(standard_bands.items()):
        if i >= len(band_list):
            break
            
        band = band_list[i]
        freqs = np.linspace(f_low, f_high, len(band))
        psd = np.abs(band)**2  # Power spectral density
        dominant_ratio = np.max(psd) / (np.sum(psd) + 1e-10)    #Dominant Frequency Ratio
        features.extend([
            np.sum(band**2),       # energy
            freqs[np.argmax(band)], # peak freq
            np.sum(freqs * psd) / (np.sum(psd) + 1e-10),  # Spectral centroid
            dominant_ratio
        ])
    
    return np.array(features)

def create_feature_whole_data(M, target, person_id, fs, QMF_levels, db, data_type):
    unique_ids = np.unique(person_id)
    n_electrodes = 19
    
    h0, h1, g0, g1 = extract_filters(db+1)
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)

    # Initialize list to store 3D features (n_persons × n_electrodes × n_features)
    all_features = []
    person_labels = []

    for uid in tqdm(unique_ids, desc="Processing IDs"):
        mask = (person_id == uid)
        segments = M[mask]
        label = target[mask][0]
        
        # Process each electrode
        electrode_features = []
        for elec in range(n_electrodes):
            # Concatenate all segments for this electrode
            signal = np.concatenate([seg[:, elec] for seg in segments])
            
            # Extract features (returns 1D array of fixed length)
            _, qmf_bands = multivel_multirate_decomposition(signal, h0, h1, A, d, levels=QMF_levels)
            features = extract_qmf_features(qmf_bands, fs, signal)
            electrode_features.append(features)
        
        # Stack electrode features for this person
        all_features.append(np.stack(electrode_features))
        person_labels.append(label)
    
    # Convert to 3D numpy array
    features_3d = np.stack(all_features)
    
    mat_data = {
        'features': features_3d,  # Now a proper 3D array
        'target': np.array(person_labels),
        'person_id': unique_ids,
        'fs': fs
    }
    
    makedirs('processed_data', exist_ok=True)
    savemat(f'processed_data/{data_type}_whole_qmf_features_db{db}.mat', mat_data)


def create_feature_epoched_data(M, target, person_id, epoch_id, fs, QMF_levels, db, data_type):
    n_epochs = len(M)
    n_electrodes = 19 
    
    h0, h1, g0, g1 = extract_filters(db+1)
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)

    all_features = []
    
    for epoch_idx in tqdm(range(n_epochs), desc="Processing epochs"):
        epoch = M[epoch_idx] 
        epoch_features = []
        
        for electrode in range(n_electrodes):
            signal = epoch[:, electrode]
            _, qmf_bands = multivel_multirate_decomposition(signal, h0, h1, A, d, levels=QMF_levels)
            
            # Get features as numpy array (ensure this returns 1D array)
            electrode_features = extract_qmf_features(qmf_bands, fs, signal)
            epoch_features.append(electrode_features)
        
        # Stack electrode features for this epoch
        all_features.append(np.stack(epoch_features))
    
    features_3d = np.stack(all_features)
    
    mat_data = {
        'features': features_3d, 
        'target': target,
        'person_id': person_id,
        'epoch_id': epoch_id,
        'fs': fs
    }

    makedirs('processed_data', exist_ok=True)
    savemat(f'processed_data/{data_type}_epoched_qmf_features_db{db}.mat', mat_data)

    return

def process_database(data_type='epoched', db=4, QMF_levels=4, fs=128):
    #will create a feature data for both train and test datasets.
    #if these datasets doesn't exist, load_data() will create them, as long as the .csv is in the same folder
    #defaults epoched, daubechies4 wavelets, with 4 levels of decomposition, at a sample rate of 128 Hz

    if data_type == 'epoched':
        file_type = "test"
        M, target, person_id, epoch_id = load_data(file_type)
        create_feature_epoched_data(M, target, person_id, epoch_id, fs, QMF_levels, db, data_type=file_type)

        file_type = "train"
        M, target, person_id, epoch_id = load_data(file_type)
        create_feature_epoched_data(M, target, person_id, epoch_id, fs, QMF_levels, db, data_type=file_type)

    elif data_type == 'whole':
        file_type = "test"
        M, target, person_id, epoch_id = load_data(file_type)
        create_feature_whole_data(M, target, person_id, fs, QMF_levels, db, data_type=file_type)

        file_type = "train"
        M, target, person_id, epoch_id = load_data(file_type)
        create_feature_whole_data(M, target, person_id, fs, QMF_levels, db, data_type=file_type)
        

    else:
        raise ValueError(f"Invalid type of data segmentation. Choose 'epoched' to extract features for each epoch of the data, or 'whole' to extract features for the whole duration")

    return

if __name__ == '__main__':
    #by default, i'm generating epoched and whole versions for two filters: db4 and db6
    #each call of the process_database creates 2 .mat files with the features and targets, one for training and one for testing
    process_database(data_type='epoched', db=4, QMF_levels=4, fs=128)
    process_database(data_type='epoched', db=6, QMF_levels=4, fs=128)
    process_database(data_type='whole', db=4, QMF_levels=4, fs=128)
    process_database(data_type='whole', db=6, QMF_levels=4, fs=128)
