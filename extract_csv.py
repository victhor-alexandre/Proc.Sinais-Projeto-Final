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

def split_train_test(D):
    #shuffle ids, because they are in order
    M = np.array(D['M'])
    target = np.array(D['t']).flatten()
    person_id = np.array(D['i']).flatten()


    unique_ids = np.unique(person_id)
    np.random.shuffle(unique_ids)
    
    # split for training and testing
    train_ids = unique_ids[:85]
    test_ids = unique_ids[85:]

    train_mask = np.isin(person_id, train_ids)
    test_mask = np.isin(person_id, test_ids)

    train_data = {
        'M': M[train_mask],
        't': target[train_mask],
        'i': person_id[train_mask]
    }

    test_data = {
        'M': M[test_mask],
        't': target[test_mask],
        'i': person_id[test_mask]
    }
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
        return M, target, person_id
    else:
        M, target, person_id = read_csv_database('adhdata.csv')
        D = {}
        D['M'] = M
        D['t'] = target
        D['i'] = person_id
        savemat('adhdata.mat', D)
        split_train_test(D)
        return load_data(file_type)

    return M, target, person_id



def mostre_imagem_aleatoria(M, t):
    k = np.random.permutation(len(t))
    k = k[0]
    img = np.reshape(M[k], (28, 28))
    plt.imshow(img, cmap = 'gray')
    plt.title('Rótulo de referência: ' + str(t[k]))
    plt.show()

def classifique_imagem_aleatoria(s, M, t):
    k = np.random.permutation(len(t))
    k = k[0]
    result = s.predict([M[k]])
    img = np.reshape(M[k], (28, 28))
    plt.imshow(img, cmap = 'gray')
    plt.title('Rótulo de referência: ' + str(t[k]) + \
    '; resultado da classificação: ' + str(result))
    plt.show()

if __name__ == '__main__':

    M, t, i = load_data("test")
    print(M)
    print(t)
    print(i)
