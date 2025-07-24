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
from tqdm import tqdm 



def extract_qmf_features(band_list, Fs):
    # Standard EEG frequency bands
    standard_bands = {
        'delta': (0, 4),
        'theta': (4, 8),
        'alpha': (8, 16),
        'beta': (16, 32),
        'gamma': (32, Fs/2)
    }
    
    features = []
    
    for i, (name, (f_low, f_high)) in enumerate(standard_bands.items()):
        if i >= len(band_list):
            break
            
        band = band_list[i]
        freqs = np.linspace(f_low, f_high, len(band))
        psd = np.abs(band)**2  # Power spectral density
        
        features.extend([
            np.sum(band**2),       # energy
            freqs[np.argmax(band)], # peak freq
            np.sum(freqs * psd) / (np.sum(psd) + 1e-10)  # Spectral centroid
        ])
    
    return np.array(features)

def create_feature_whole_data(M, target, person_id, fs, QMF_levels, db, data_type):
    unique_ids = np.unique(person_id)
    n_electrodes = 19
    
    h0, h1, g0, g1 = extract_filters(db)
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)


    features = np.empty((len(unique_ids), n_electrodes), dtype=object)
    person_labels = []

    # for each id, for each electrode, perform the qmf decomposition and extract the features
    for idx, uid in enumerate(tqdm(unique_ids, desc="Processing IDs")):
        mask = (person_id == uid)
        segments = M[mask] 
        label = target[mask][0] 
        
        electrode_features = []
        for elec in range(n_electrodes):
            signal = np.concatenate([seg[:, elec] for seg in segments])
            _, qmf_bands = multivel_multirate_decomposition(signal, h0, h1, A, d, levels=QMF_levels)
            electrode_features.append(extract_qmf_features(qmf_bands, fs))
        
        features[idx] = electrode_features
        person_labels.append(label)
    
    mat_data = {
        'features': features,
        'target': np.array(person_labels),
        'person_id': unique_ids,
        'fs': fs
    }
    
    savemat(f'prepared_data/{data_type}_whole_qmf_features_db{db-1}.mat', mat_data)
    print(f"Saved features for {len(unique_ids)} examples × {n_electrodes} electrodes")

def create_feature_epoched_data(M, target, person_id, epoch_id, fs, QMF_levels, db, data_type):
    n_epochs = len(M)
    n_electrodes = 19 
    
    h0, h1, g0, g1 = extract_filters(db)
    _, _, _, A, d = qmf_filters_validator(h0, h1, g0, g1)

    features = np.empty((n_epochs, n_electrodes), dtype=object)
    # for each epoch, for each electrode, perform the qmf decomposition and extract the features
    for epoch_idx in tqdm(range(n_epochs), desc="Processing epochs"):
        epoch = M[epoch_idx] 
        
        for electrode in range(n_electrodes):
            signal = epoch[:, electrode]
            
            _, qmf_bands = multivel_multirate_decomposition(signal, h0, h1, A, d, levels = QMF_levels) 
            
            features[epoch_idx, electrode] = extract_qmf_features(qmf_bands, fs)

    mat_data = {
        'features': features,
        'target': target,
        'person_id': person_id,
        'epoch_id': epoch_id,
        'fs': fs
    }

    savemat(f'prepared_data/{data_type}_epoched_qmf_features_db{db-1}.mat', mat_data)
    print(f"Saved features for {n_epochs} epochs × {n_electrodes} electrodes")

    return

def process_database():
    fs = 128
    QMF_levels = 4
    
    file_type = "test"
    M, target, person_id, epoch_id = load_data(file_type)

    create_feature_epoched_data(M, target, person_id, epoch_id, fs, QMF_levels, db=5, data_type=file_type)
    create_feature_epoched_data(M, target, person_id, epoch_id, fs, QMF_levels, db=7, data_type=file_type)

    create_feature_whole_data(M, target, person_id, fs, QMF_levels, db=5, data_type=file_type)
    create_feature_whole_data(M, target, person_id, fs, QMF_levels, db=7, data_type=file_type)

    file_type = "train"
    M, target, person_id, epoch_id = load_data(file_type)

    create_feature_epoched_data(M, target, person_id, epoch_id, fs, QMF_levels, db=5, data_type=file_type)
    create_feature_epoched_data(M, target, person_id, epoch_id, fs, QMF_levels, db=7, data_type=file_type)

    create_feature_whole_data(M, target, person_id, fs, QMF_levels, db=5, data_type=file_type)
    create_feature_whole_data(M, target, person_id, fs, QMF_levels, db=7, data_type=file_type)

    return

if __name__ == '__main__':

    process_database()
