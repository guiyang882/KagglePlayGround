#!/usr/bin/env python
# coding=utf-8

import csv
import numpy as np
import os, pickle
import cv2 as cv

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
                res = map(np.uint8,line[:28*28+1])
                labels.append(res[0])
                info_arr = np.array(res[1:]).reshape([28,28,1])
                info_img = cv.adaptiveThreshold(info_arr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
                data.append(255 - info_img.astype(np.float).reshape([28,28,1]))
                #cv.imwrite("train/S%05d_%d.jpg" % (index-1,res[0]),info_arr)
            if flag == "test":
                info_arr = np.array( map( np.uint8, line[:28*28] ) ).reshape([28,28,1])
                info_img = cv.adaptiveThreshold(info_arr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
                data.append(255 - info_img.astype(np.float).reshape([28,28,1]))
                #cv.imwrite("test/S%05d.jpg" % (index-1),info_arr)
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
