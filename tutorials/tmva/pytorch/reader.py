"""
Module for parsing the TTree and generating the train and test data
"""
import numpy as np
from tqdm import tqdm
import pickle
import os
from root_numpy import root2array

def get_dataset(data_file, test_split=0.2):
    """
    @data_file: path of the root file
    @test_split: fraction of data to be used as test set
    """
    assert 0 <= test_split <= 0.3

    # Load data
    signal = root2array(data_file, 'sig_tree')
    background = root2array(data_file, 'bkg_tree')

    tree_data = [
        np.array([img[0] for img in signal]),
        np.array([img[0] for img in background])
    ]

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    # Deterministic Random
    np.random.seed(0)

    for label, data in enumerate(tree_data):
        np.random.shuffle(data)

        test_size = int(len(data) * test_split)
        X_train.append(data[:-test_size])
        X_test.append(data[-test_size:])
        Y_train.append([label] * (len(data) - test_size))
        Y_test.append([label] * test_size)

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)

    assert len(Y_train) == len(X_train)
    assert len(Y_test) == len(X_test)

    # Shuffling the data
    train_perm = np.random.permutation(len(X_train))
    X_train = X_train[train_perm]
    Y_train = Y_train[train_perm]
    
    test_perm = np.random.permutation(len(X_test))
    X_test = X_test[test_perm]
    Y_test = Y_test[test_perm]
    
    return (X_train, Y_train), (X_test, Y_test)