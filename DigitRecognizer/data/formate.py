#!/usr/bin/env python
# coding=utf-8

import csv
import numpy as np
import os, pickle

def convertData(filename, flag):
    if os.path.exists(filename) == False:
        raise ValueError
        return None

    with open(filename,'rb+') as handle:
        reader = csv.reader(handle)
        index = 0
        data = []
        labels = []
        for line in reader:
            index = index + 1
            if index == 1:
                continue
            if flag == "train":
                res = map(float,line[:28*28+1])
                labels.append(res[0])
                data.append(np.array(res[1:]).reshape([28,28,1]))
            if flag == "test":
                data.append( np.array( map( float, line[:28*28] ) ).reshape([28,28,1]) )
        if flag == "train":
            labels_handle = open("labels.pkl","wb")
            pickle.dump(np.array(labels),labels_handle)
            labels_handle.close()

        saveHandle = open(filename.split(".")[0] + ".pkl",'wb')
        pickle.dump(np.array(data),saveHandle)
        saveHandle.close()

if __name__ == "__main__":
    convertData("train.csv","train")
    convertData("test.csv","test")
