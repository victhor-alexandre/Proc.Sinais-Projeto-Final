from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat, savemat
from os.path import exists
import joblib
import os
import csv

from avaliacao_desempenho_classificadores import metrics

from train_svm import train_svm

def save_to_csv(data_type, wavelet, kernel, model_type, metrics_tuple, filename='results.csv'):
    M, accuracy, recall, specificity, precision, F1 = metrics_tuple
    
    # Convert confusion matrix to string format
    if isinstance(M, np.ndarray):
        confusion_str = np.array2string(M, separator=', ')
    else:
        confusion_str = str(M)
    
    # Prepare row data
    row = {
        'data_type': data_type,
        'wavelet': f'db{wavelet}',
        'kernel': kernel,
        'model': 'generalized' if model_type else 'per-electrode',
        'accuracy': f"{accuracy:.4f}",
        'recall': f"{recall:.4f}",
        'specificity': f"{specificity:.4f}",
        'precision': f"{precision:.4f}",
        'F1': f"{F1:.4f}",
        'confusion_matrix': confusion_str
    }
    
    # Write to file
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def run_and_log_test(data_type, db, kernel, generalized):
    targets, results = test_svm(
        data_type=data_type, 
        db=db, 
        kernel=kernel, 
        generalized=generalized
    )
    
    # Get all metrics
    metrics_result = metrics(targets, results)  # Returns (M, accuracy, recall...)
    
    # Save to CSV
    save_to_csv(
        data_type=data_type,
        wavelet=db,
        kernel=kernel,
        model_type=generalized,
        metrics_tuple=metrics_result
    )

def test_svm(data_type, db, kernel, generalized=True):
    D = loadmat(f"processed_data/test_{data_type}_qmf_features_db{db}.mat")
    features = D['features']
    target = np.array(D['target']).flatten()

    n_epochs = features.shape[0]
    n_electrodes = features.shape[1]

    prediction = np.empty((n_epochs, n_electrodes), dtype='object')
    targets = np.empty((n_epochs, n_electrodes), dtype='object')
    acc = np.empty((n_epochs, n_electrodes), dtype='object')

    # Pre-load all models/scalers
    if(generalized):
        model_type = "generalized"
        scalers = joblib.load(f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_scaler_generalized.savedscaler')
        models = joblib.load(f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_trained_generalized_svm.savedsvm') 
    else:
        model_type = "per-electrode"
        scalers = [joblib.load(f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_scaler_{e}.savedscaler') for e in range(n_electrodes)]
        models = [joblib.load(f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_trained_{e}_svm.savedsvm') for e in range(n_electrodes)]


    for epoch in range(n_epochs):
        for elec in range(n_electrodes):

            if(generalized):
                X = np.hstack([features[epoch, elec], [elec/n_electrodes]]).reshape(1, -1)
                X_scaled = scalers.transform(X)
                prediction[epoch, elec] = str(models.predict(X_scaled)[0])
            else:
                X = features[epoch, elec].reshape(1, -1) 
                X_scaled = scalers[elec].transform(X) 
                prediction[epoch, elec] = str(models[elec].predict(X_scaled)[0])

            targets[epoch,elec] = str(target[epoch])
            acc[epoch,elec] = (prediction[epoch,elec] == target[epoch])
    
    result = np.array(prediction).flatten()
    t = np.array(targets).flatten()         
    
    print(f"Testing with {kernel} kernels, and {model_type}_trained models")
    print(f"test with db{db} and {data_type} data")

    return t, result

if __name__ == '__main__':
    #this will test for all electrodes at the same time, with different parameters
    
    test_scenarios = [
        {'data_type': 'epoched', 'db': 4, 'kernel': 'linear', 'generalized': True},
        {'data_type': 'epoched', 'db': 4, 'kernel': 'linear', 'generalized': False},
        {'data_type': 'epoched', 'db': 4, 'kernel': 'rbf',    'generalized': True},
        {'data_type': 'epoched', 'db': 4, 'kernel': 'rbf',    'generalized': False},

        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'generalized': False},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'rbf',    'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'rbf',    'generalized': False},

        {'data_type': 'whole',   'db': 4, 'kernel': 'linear', 'generalized': True},
        {'data_type': 'whole',   'db': 4, 'kernel': 'linear', 'generalized': False},
        {'data_type': 'whole',   'db': 4, 'kernel': 'rbf',    'generalized': True},
        {'data_type': 'whole',   'db': 4, 'kernel': 'rbf',    'generalized': False},

        {'data_type': 'whole',   'db': 6, 'kernel': 'linear', 'generalized': True},
        {'data_type': 'whole',   'db': 6, 'kernel': 'linear', 'generalized': False},
        {'data_type': 'whole',   'db': 6, 'kernel': 'rbf',    'generalized': True},
        {'data_type': 'whole',   'db': 6, 'kernel': 'rbf',    'generalized': False},
    ]

    # Run all tests
    i=1
    for config in test_scenarios:
        print(f"running test on scenario {i}/16")
        run_and_log_test(**config)
        i=i+1


    