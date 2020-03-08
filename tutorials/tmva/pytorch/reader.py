"""
Module for parsing the TTree and generating the train and test data
"""
import numpy as np
from tqdm import tqdm
import pickle
import os

def get_dataset(tree_data, test_split=0.2):
    """
    @tree_data: list of all the trees. label is same as the index in the data array
    @test_split: fraction of data to be used as test set
    """
    assert 0 <= test_split <= 0.3

    data_cache = 'dataset_cache.pkl'
    if os.path.exists(data_cache):
        return pickle.load(open(data_cache, 'rb'))

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    # Deterministic Random
    np.random.seed(0)

    for label, tree in enumerate(tree_data):
        print ('Loading tree #%d...' % label)
        data = np.array([np.array(entry.vars) for entry in tqdm(tree)])
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
    
    pickle.dump(((X_train, Y_train), (X_test, Y_test)),
                open(data_cache, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    return (X_train, Y_train), (X_test, Y_test)