# -*- coding: utf-8 -*-
"""
 SIFT Algorithm for feature extraction
 @author: Kmedrano101
 Created on Tue Oct 10 12:55:34 2022
"""

import numpy as np
import cv2 as cv

img = cv.imread('../img1.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
#cv.imwrite('sift_keypoints.jpg',img)