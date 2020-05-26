#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:43:53 2018

@author: huawei
"""

import os
import cv2

root_dir = '/Users/huawei/Desktop/SemanticSeg_TestVideos/CrossWalk_2/frames/'
img_size = (256,340)

for subdir in os.listdir(root_dir):
    print subdir
    img = cv2.imread(os.path.join(root_dir, subdir))
    if subdir.startswith('.'):
        raw_input('press')
        continue
    else:
        print img.shape
        img = cv2.resize(
                        img,
                        img_size[::-1],
                        interpolation=cv2.INTER_LINEAR)
        print img.shape
        cv2.imwrite(os.path.join(root_dir, subdir), img)

