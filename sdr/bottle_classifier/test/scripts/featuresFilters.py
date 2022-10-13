# -*- coding: utf-8 -*-
"""
Testin features extrantion
Created on Thu Oct 04 13:10:59 2022

@author: Kmedrano101
"""
import cv2 as cv
import imutils
import numpy as np

img = cv.imread('../../data/train/bottle1/1.jpeg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Kernel for filters
kernel1 = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

img_sharped = cv.filter2D(img_gray,-1,kernel1)
# Resize the image
img_resized = imutils.resize(img_sharped, width=300)

# threshold the image
img_mBlur = cv.medianBlur(img_resized, 3)
_,img_thresh = cv.threshold(img_mBlur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
img_morph = cv.morphologyEx(img_mBlur, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)))

img_edges = cv.Canny(img_morph,20,60)

# Get key points and descriptors
orb = cv.ORB_create()
img_kp = np.ones(img_resized.shape[:2], dtype="uint8") * 255
kp, des = orb.detectAndCompute(img_resized,None)
img_kp = cv.cvtColor(cv.drawKeypoints(img_kp,kp,None),cv.COLOR_BGR2GRAY)

cv.imshow("Feature 1", img_resized)
cv.imshow("Feature 2", img_morph)
cv.imshow("Feature 3", img_thresh)
cv.imshow("Feature 4", img_edges)
cv.imshow("Feature 5", img_kp)

cv.waitKey(0)
cv.destroyAllWindows()
