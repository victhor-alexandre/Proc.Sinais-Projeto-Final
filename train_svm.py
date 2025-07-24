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
    X = np.concatenate(features[:, electrode])
    y = targets
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVC(kernel=kernel, C=C)
    svm.fit(X_scaled, y)

    # checks if the folders exist, create them if not
    makedirs('trained_data', exist_ok=True)
    makedirs(f'trained_data/db{db}_{data_type}', exist_ok=True)

    #saving the svm models for the current electrode
    joblib.dump(svm, f'trained_data/db{db}_{data_type}/db{db}_{data_type}_trained_{electrode}_svm.savedsvm')
    return (electrode, svm, scaler)

def train_electrode_svms_parallel(features, targets, data_type, db, n_electrodes=19):
    """Parallel version using joblib"""
    results = Parallel(n_jobs=-1)(
        delayed(train_electrode_svm_parallel)(e, features, targets, data_type, db)
        for e in range(n_electrodes)
    )
    
    # Reorganize results
    models = [None] * n_electrodes
    scalers = [None] * n_electrodes
    
    for e, svm, scaler in results:
        models[e] = svm
        scalers[e] = scaler
    
    return

def train_svm(data_type='epoched', db=4, QMF_levels=4, fs=128):
    match data_type:
        case "epoched":
            matfile = f"prepared_data/test_epoched_qmf_features_{db}.mat"
        case "whole":
            matfile = f"prepared_data/test_whole_qmf_features_{db}.mat"
        case _:  # Default case
            raise ValueError(f"Invalid type of data segmentation. Choose 'epoched' to train the svm with epoched data, or 'whole' to to train with whole data")


    if (not exists(f"prepared_data/test_{data_type}_qmf_features_db{db}.mat")):
        process_database(data_type, db, QMF_levels, fs)
        
    D = loadmat(f"prepared_data/test_{data_type}_qmf_features_db{db}.mat")
    print("Training SVMs for each electrode...")

    features = D['features']
    target = np.array(D['target']).flatten()
    fs = D['fs'][0][0]

    train_electrode_svms_parallel(features, target, data_type, db)
    print(f"The trained_data was successfully created at:")
    print(f'trained_data/db{db}_{data_type}')

    return

if __name__ == "__main__":
    QMF_levels=4
    fs=128

    data_type = 'whole'
    db = 4

    train_svm(data_type='epoched', db=4, QMF_levels=4, fs=128)
    train_svm(data_type='whole', db=4, QMF_levels=4, fs=128)
    train_svm(data_type='epoched', db=6, QMF_levels=4, fs=128)
    train_svm(data_type='whole', db=6, QMF_levels=4, fs=128)






