#!/usr/bin/env python
# coding=utf-8

import csv 
import cv2
import numpy as np
from conf import *

def CSV2IMG_TEST():
    with open(SRC_TESTFILE,'rb+') as handle:
        reader = csv.reader(handle)
        index = 0
        for line in reader:
            index = index + 1
            if index == 1:
                continue
            img_id, img_data = line[0], line[1]
            img_data = map(np.uint8, img_data.split(" "))
            img_data = np.array(img_data)
            img_data.shape = IMG_WIDTH, IMG_HEIGH
            #print img_data
            cv2.imwrite( DIR_TESTIMG + "%04d" % int(img_id) + ".jpg", img_data)


def CSV2IMG_TRAIN():
    with open(SRC_TRAINFILE, 'rb+') as handle:
        reader = csv.reader(handle)
        index = 0
        for line in reader:
            index = index + 1
            if index == 1:
                continue
            img_data = np.array(map(np.uint8, line[-1].split(" ")))
            img_data.shape = IMG_WIDTH, IMG_HEIGH
            #print img_data
            cv2.imwrite(DIR_TRAINIMG + "%04d" % (index-1) + ".jpg", img_data) 

if __name__ == "__main__":
    #CSV2IMG_TEST()
    #CSV2IMG_TRAIN()

