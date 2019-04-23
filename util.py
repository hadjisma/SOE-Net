###################################################
# Miscellaneous Utlitary functions
# Coded by Isma Hadji (hadjisma@cse.yorku.ca)
###################################################
""" Implementation of various utilitary functions
e.g: visualzation functions
"""
import cv2
import numpy as np

def VideoVisualization (data):
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            img = cv2.normalize(data[i,j,:,:,:], None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imshow('frame',img)
            cv2.waitKey(100)

def EnergyVisualization (data):
    M = np.tile(np.amax(data[:,:,:,:,0:10], axis=4, keepdims=True),(1,1,1,1,data.shape[4])) + np.spacing(1)
    normdata = np.divide(data,M)
    for e in range(np.shape(data)[4]):
        raw_input('LOOPING THROUGH CHANNELS == > Press enter to continue ... ')
        for b in range(np.shape(data)[0]):
            for t in range(np.shape(data)[1]):
                #img = cv2.normalize(normdata[b,t,:,:,e], None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                #cv2.imshow('frame',img)
                #cv2.waitKey(100)
                cv2.imshow('frame',normdata[b,t,:,:,e])
                cv2.waitKey(50)

