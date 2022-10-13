# -*- coding: utf-8 -*-
"""
 BRIEF Algorithm for feature extraction
 @author: Kmedrano101
 Created on Tue Oct 10 14:23:13 2022
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../img1.jpg')
# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(img,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)
img=cv.drawKeypoints(gray,kp,img)
cv.imshow('dst',img)
cv.waitKey(0)
cv.destroyAllWindows()