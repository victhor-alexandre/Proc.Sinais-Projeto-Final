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

def save_to_csv(electrode, data_type, wavelet, kernel, model_type, metrics_tuple, filename='results_per_electrode_wavelet2.csv'):
    M, accuracy, recall, specificity, precision, F1 = metrics_tuple
    
    # Convert confusion matrix to string format
    if isinstance(M, np.ndarray):
        confusion_str = np.array2string(M, separator=', ')
    else:
        confusion_str = str(M)
    
    # Prepare row data
    row = {
        'electrode' : electrode,
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
    # print(row)
    # Write to file
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def run_and_log_test(data_type, db, kernel, electrode, generalized):
    targets, results = test_svm(
        data_type=data_type, 
        db=db, 
        kernel=kernel, 
        electrode=electrode,
        generalized=generalized
    )
    
    # Get all metrics
    metrics_result = metrics(targets, results)  # Returns (M, accuracy, recall...)
    
    # Save to CSV
    save_to_csv(
        electrode=electrode, 
        data_type=data_type,
        wavelet=db,
        kernel=kernel,
        model_type=generalized,
        metrics_tuple=metrics_result
    )

def test_svm(data_type, db, kernel, electrode, generalized=True):
    #this version will test a single electrode, instead of all of them. This way, we have results per electrode
    #so each test_case will be a single electrode
    #to limit the amount of tests, i will be differentiating only epoched vs whole data, and generalized vs per-electrode svm
    #the test will run on db4, with 'rbf' kernel
    D = loadmat(f"processed_data/test_{data_type}_qmf_features_db{db}.mat")
    features = D['features']

    n_epochs = features.shape[0]      # n of signals or epochs
    n_electrodes = features.shape[1]


    target = np.array(D['target']).flatten()
    prediction = np.empty((n_epochs), dtype='object')
    targets = np.empty((n_epochs), dtype='object')
    acc = np.empty((n_epochs), dtype='object')


    if(generalized):
        model_type = "generalized"
        scalers = joblib.load(f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_scaler_generalized.savedscaler')
        models = joblib.load(f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_trained_generalized_svm.savedsvm') 
    else:
        # Pre-load all models/scalers
        model_type = "per-electrode"
        scalers = joblib.load(f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_scaler_{electrode}.savedscaler')
        models = joblib.load(f'trained_data/{kernel}_kernel/db{db}_{data_type}/db{db}_{data_type}_trained_{electrode}_svm.savedsvm')


    elec = electrode
    for epoch in range(n_epochs):
        if(generalized):
            X = np.hstack([features[epoch, elec], [elec/n_electrodes]]).reshape(1, -1)
            X_scaled = scalers.transform(X)
            prediction[epoch] = str(models.predict(X_scaled)[0])
        else:
        # 1. Extract and reshape features
            X = features[epoch, elec].reshape(1, -1) 
            # 2.  Scale and predict
            X_scaled = scalers.transform(X) 
            prediction[epoch] = str(models.predict(X_scaled)[0])

        targets[epoch] = str(target[epoch])
        acc[epoch] = (prediction[epoch] == target[epoch])
    
    result = np.array(prediction).flatten()
    t = np.array(targets).flatten()
    # print(f"Results: \n{result[1:10]}")
    # print(f"targets: \n{t[1:10]}")
    
    # print(result.shape)
    # print(t.shape)

    # n_correct = np.sum(acc)          # Count all True values in the accuracy matrix
    # n_total = acc.size               # Total predictions (x_size * y_size)
    # accuracy = n_correct / n_total   # Overall accuracy
    print(f"Testing with {kernel} kernels, and {model_type}_trained models")
    print(f"test with db{db} and {data_type} data")
    # print(f"Correct predictions: {n_correct}/{n_total}")
    # print(f"Accuracy: {accuracy:.2%}")


    return t, result

if __name__ == '__main__':
    #this version will test two sets of parameters for each electrode. 
    
    test_scenarios = [
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 0, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 1, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 2, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 3, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 4, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 5, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 6, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 7, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 8, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 9, 'generalized':  True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 10, 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 11, 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 12, 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 13, 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 14, 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 15, 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 16, 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 17, 'generalized': True},
        {'data_type': 'epoched', 'db': 6, 'kernel': 'linear', 'electrode' : 18, 'generalized': True},

    ]

    # Run all tests
    i=1
    for config in test_scenarios:
        print(f"running test on scenario {i}/19")
        run_and_log_test(**config)
        i=i+1


    