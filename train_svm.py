from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat, savemat
from os.path import exists
from os import makedirs

import joblib
from joblib import Parallel, delayed

from process_database import process_database

def train_electrode_svm_parallel(electrode, features, targets, data_type, db, kernel='linear', C=1.0):
    X = features[:, electrode, :]

    X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = targets

    svm = SVC(kernel=kernel, C=C)
    svm.fit(X_scaled, y)

    # checks if the folders exist, create them if not
    makedirs('trained_data', exist_ok=True)
    makedirs(f'trained_data/{kernel}_kernel', exist_ok=True)
    makedirs(f'trained_data/{kernel}_kernel/db{db}_{data_type}', exist_ok=True)

    #saving the svm models for the current electrode
    joblib.dump(svm, f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_trained_{electrode}_svm.savedsvm')
    joblib.dump(scaler, f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_scaler_{electrode}.savedscaler')
    return (electrode, svm, scaler)

def train_electrode_svms_parallel(features, targets, data_type, db, kernel, n_electrodes=19):
    results = Parallel(n_jobs=-1)(
        delayed(train_electrode_svm_parallel)(e, features, targets, data_type, db, kernel)
        for e in range(n_electrodes)
    )
    
    # Reorganize results
    models = [None] * n_electrodes
    scalers = [None] * n_electrodes
    
    for e, svm, scaler in results:
        models[e] = svm
        scalers[e] = scaler
    
    return

def train_generalized_svm(features, target, data_type, db, kernel='linear', C=1.0):
    # Reshape features to combine all electrodes
    # Reshape features to (n_samples, n_electrodes * n_features)

    n_samples, n_electrodes, n_features = features.shape
    X = features.reshape(-1, n_features)  # Shape: (n_samples * n_electrodes, n_features)
    
    # Add electrode index as a feature (normalized)
    electrode_indices = np.repeat(np.arange(n_electrodes), n_samples).reshape(-1, 1)
    X = np.hstack([X, electrode_indices / n_electrodes])  # Normalize to [0,1]
    
    # Repeat targets for each electrode
    y = np.repeat(target, n_electrodes)
    
    # 2. Scale and train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(kernel=kernel, C=C)
    svm.fit(X_scaled, y)

    # checks if the folders exist, create them if not
    makedirs('trained_data', exist_ok=True)
    makedirs(f'trained_data/{kernel}_kernel', exist_ok=True)
    makedirs(f'trained_data/{kernel}_kernel/db{db}_{data_type}', exist_ok=True)
    joblib.dump(svm, f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_trained_generalized_svm.savedsvm')
    joblib.dump(scaler, f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_scaler_generalized.savedscaler')
    return

def train_svm(data_type='epoched', db=4, QMF_levels=4, fs=128, kernel="linear"):
    match data_type:
        case "epoched":
            matfile = f"processed_data/train_epoched_qmf_features_{db}.mat"
        case "whole":
            matfile = f"processed_data/train_whole_qmf_features_{db}.mat"
        case _:  # Default case
            raise ValueError(f"Invalid type of data segmentation. Choose 'epoched' to train the svm with epoched data, or 'whole' to to train with whole data")


    if (not exists(f"processed_data/train_{data_type}_qmf_features_db{db}.mat")):
        process_database(data_type, db, QMF_levels, fs)
        
    D = loadmat(f"processed_data/train_{data_type}_qmf_features_db{db}.mat")
    print("Training SVMs for each electrode...")

    features = D['features']
    target = np.array(D['target']).flatten()
    fs = D['fs'][0][0]

    train_electrode_svms_parallel(features, target, data_type, db, kernel=kernel)
    
    print("Training a generalized SVM...")
    train_generalized_svm(features, target, data_type, db, kernel=kernel)

    print(f"The trained_data was successfully created at: trained_data/{kernel}_kernel/db{db}_{data_type}")
    return

if __name__ == "__main__":
    QMF_levels=4
    fs=128

    train_svm(data_type='epoched', db=4, QMF_levels=4, fs=128, kernel="linear")
    train_svm(data_type='whole', db=4, QMF_levels=4, fs=128, kernel="linear")
    train_svm(data_type='epoched', db=6, QMF_levels=4, fs=128, kernel="linear")
    train_svm(data_type='whole', db=6, QMF_levels=4, fs=128, kernel="linear")
    
    train_svm(data_type='epoched', db=4, QMF_levels=4, fs=128, kernel="rbf")
    train_svm(data_type='whole', db=4, QMF_levels=4, fs=128, kernel="rbf")
    train_svm(data_type='epoched', db=6, QMF_levels=4, fs=128, kernel="rbf")
    train_svm(data_type='whole', db=6, QMF_levels=4, fs=128, kernel="rbf")

    





