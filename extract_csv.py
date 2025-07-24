import numpy as np
import csv
from matplotlib import pyplot as plt
from os.path import exists
from scipy.io import loadmat, savemat
from sklearn import svm

def read_csv_database(filename):
    M = []
    target = []
    person_id = []

    with open(filename, 'r') as fid:
        csv_reader = csv.reader(fid)
        next(csv_reader)

        for row in csv_reader:
            if not row:  # Skip empty rows (if any)
                continue

            # Extract features (all except last 2 columns), convert to float
            features = [float(x) for x in row[:-2]]
            M.append(features)

            # Extract target (second last column) and ID (last column)
            target.append(row[-2])
            person_id.append(row[-1])

    return M, target, person_id

def epoch_data(ids, M, target, person_id, epoch_sec=8, Fs=128):
    """
    Args:
        M: EEG data array (n_samples × 19 electrodes)
        target: Class labels (n_samples,)
        person_id: Person IDs (n_samples,)
        epoch_sec: Epoch duration in seconds
        Fs: Sampling rate (Hz)
    """
    samples_per_epoch = int(Fs * epoch_sec)
    n_electrodes = 19  # Assuming 19 columns = electrodes
    
    # Storage for results
    epoched_data = []
    epoched_labels = []
    epoched_person_ids = []
    epoched_epoch_ids = []
    
    for current_id in ids:
        # Get all rows for this person
        person_mask = (person_id == current_id)
        person_eeg = M[person_mask]  # (n_person_samples × 19)
        person_label = target[person_mask][0]
        
        n_samples = person_eeg.shape[0]
        n_epochs = n_samples // samples_per_epoch
        
        if n_epochs == 0:
            print(f"Skipping {current_id}: Not enough samples ({n_samples/Fs:.1f}s < {epoch_sec}s)")
            continue
        
        # Reshape to (n_epochs, samples_per_epoch, n_electrodes)
        epochs = person_eeg[:n_epochs*samples_per_epoch].reshape(
            n_epochs, samples_per_epoch, n_electrodes
        )
        
        # Store results
        epoched_data.append(epochs)
        epoched_labels.extend([person_label] * n_epochs)
        epoched_person_ids.extend([current_id] * n_epochs)
        epoched_epoch_ids.extend(range(n_epochs))
    
    if not epoched_data:
        raise ValueError("No valid epochs created. Check your epoch_sec and data lengths.")
    
    return {
        'M': np.concatenate(epoched_data, axis=0),  # (total_epochs, samples_per_epoch, 19)
        't': np.array(epoched_labels),
        'i': np.array(epoched_person_ids),
        'e': np.array(epoched_epoch_ids)
    }

def split_train_test(D):
    #shuffle ids, because they are in order
    M = np.array(D['M'])
    target = np.array(D['t']).flatten()
    person_id = np.array(D['i']).flatten()
    # epoch_id = np.array(D['e']).flatten()

    unique_ids = np.unique(person_id)
    np.random.shuffle(unique_ids)
    
    # split for training and testing
    train_ids = unique_ids[:85]
    test_ids = unique_ids[85:]

    # train_mask = np.isin(person_id, train_ids)
    # test_mask = np.isin(person_id, test_ids)

     # Epoch both train and test data
    train_data = epoch_data(train_ids, M, target, person_id)
    test_data = epoch_data(test_ids, M, target, person_id)

    # train_data = {
    #     'M': M[train_mask],
    #     't': target[train_mask],
    #     'i': person_id[train_mask]
    # }

    # test_data = {
    #     'M': M[test_mask],
    #     't': target[test_mask],
    #     'i': person_id[test_mask]
    # }
    #save to different files
    savemat('train_data.mat', train_data)
    savemat('test_data.mat', test_data)
    print(f"Training set: {len(train_ids)} IDs, {len(train_data['M'])} rows")
    print(f"Test set: {len(test_ids)} IDs, {len(test_data['M'])} rows")

    return


def load_data(file_type):
    match file_type:
        case "train":
            matfile = "train_data.mat"
        case "test":
            matfile = "test_data.mat"
        case "original":
            matfile = "adhdata.mat"
        case _:  # Default case
            raise ValueError(f"Invalid file_type: {file_type}. Choose 'train', 'test', or 'original'")
    if exists(matfile):
        D = loadmat(matfile)
        M = D['M']
        target = D['t'].flatten()
        person_id = D['i'].flatten()
        epoch_id = D['e'].flatten()
        return M, target, person_id, epoch_id
    else:
        M, target, person_id = read_csv_database('adhdata.csv')
        D = {}
        D['M'] = M
        D['t'] = target
        D['i'] = person_id
        D['e'] = np.zeros_like(person_id, dtype=int)

        savemat('adhdata.mat', D)
        split_train_test(D)
        return load_data(file_type)

    return M, target, person_id, epoch_id


if __name__ == '__main__':

    M, t, i, e = load_data("test")
    print(M)
    print(t)
    print(i)
    print(e)
