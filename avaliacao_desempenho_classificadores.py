#!/usr/bin/env python
# map <leader><leader> :wall<cr>:!python avaliacao_desempenho_classificadores.py<cr>

import numpy as np

def get_label_position(labels, r):
    k = np.where(labels == r)[0][0]
    return k

def confusion_matrix(t, results):
    labels = np.unique(t)
    n_classes = len(labels)
    M = np.zeros(shape = (n_classes, n_classes))
    for k in range(0, len(results)):
        i = get_label_position(labels, t[k])
        j = get_label_position(labels, results[k])
        M[i, j] += 1
    return M, labels

def reduced_confusion_matrix(M, k):
    Mr = np.zeros(shape = (2, 2))
    Mr[0, 0] = M[k, k] 
    Mr[0, 1] = np.sum(M[k, :]) - M[k, k]
    Mr[1, 0] = np.sum(M[:, k]) - M[k, k]
    Mr[1, 1] = np.sum(M) - np.sum(Mr)
    TP = Mr[0, 0]
    TN = Mr[1, 1]
    FP = Mr[1, 0]
    FN = Mr[0, 1]
    P = TP + FN
    N = TN + FP
    accuracy = (TP + TN) / (P + N)
    recall = TP / P
    specificity = TN / N
    precision = TP / (TP + FP)
    return Mr, accuracy, recall, specificity, precision

def metrics(t, results):
    M, _ = confusion_matrix(t, results)
    accuracy = 0
    recall = 0
    specificity = 0
    precision = 0
    for k in range(0, M.shape[0]):
        _, accuracy_, recall_, specificity_, precision_ = \
        reduced_confusion_matrix(M, k)
        accuracy += accuracy_ / M.shape[0]
        recall += recall_ / M.shape[0]
        specificity += specificity_ / M.shape[0]
        precision += precision_ / M.shape[0]
    F1 = 2.0 / (1.0/precision + 1.0/recall)
    return M, accuracy, recall, specificity, precision, F1

if __name__ == '__main__':
    t = ['a', 'b', 'c', 'd', 'a', 'd', 'b', 'c']
    r = ['b', 'b', 'c', 'c', 'a', 'd', 'b', 'b']
    M, accuracy, recall, specificity, precision, F1 = metrics(t, r)
    print(M)
    print(accuracy)
    print(recall)
    print(specificity)
    print(precision)
    print(F1)
