#!/usr/bin/env python
# coding=utf-8

import os
import sys
import conf
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

sys.setrecursionlimit(10000) 
np.random.seed(42)

class DataSet(object):

    def __init__(self):
        self._images = None
        self._labels = None
        self._index_in_epoch = 0
        self._num_examples = 0
        self._epochs_completed = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
    
    def load(self, test=False, cols=None):
        fname = conf.SRC_TESTFILE if test else conf.SRC_TRAINFILE
        df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

        if cols:  # get a subset of columns
            df = df[list(cols) + ['Image']]

        print(df.count())  # prints the number of values for each column
        df = df.dropna()  # drop all rows that have missing values in them

        X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
        X = X.astype(np.float32)

        if not test:  # only FTRAIN has any target columns
            y = df[df.columns[:-1]].values
            y = (y - 48) / 48  # scale target coordinates to [-1, 1]
            X, y = shuffle(X, y, random_state=42)  # shuffle train data
            y = y.astype(np.float32)
        else:
            y = None

        self._images = X
        self._labels = y
        self._num_examples = len(X)

if __name__ == "__main__":
    obj = DataSet()
    obj.load()
    imgs, labels = obj.next_batch(10)
    print imgs
    print labels
