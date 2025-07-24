from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from scipy.io import loadmat, savemat
from os.path import exists


from joblib import Parallel, delayed

def train_electrode_svm(elec_idx):
    X = np.array([extract_qmf_features(epoch[elec_idx]) for epoch in all_epochs])
    return SVC(kernel='linear').fit(X, y)




if __name__ == "__main__":
    D = loadmat("prepared_data/test_epoched_qmf_features_db4.mat")
    print(D.keys())

    features = D['features']
    target = D['target']
    person_id = D['person_id']
    epoch_id = D['epoch_id']
    fs = D['fs'][0][0]
    print(f"features: {features[0][0]}")
    print(f"len features: {len(person_id)}")
    # print(f"target: {target}")
    # print(f"person_id: {person_id}")
    # print(f"epoch_id: {epoch_id[0]}")
    # print(f"fs: {fs}")


    # svms = Parallel(n_jobs=-1)(delayed(train_electrode_svm)(i) for i in range(19))
