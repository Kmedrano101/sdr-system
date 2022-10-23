# -*- coding: utf-8 -*-
"""
Features Extraction from an image
@author: Kmedrano101
Created on Thu Oct 13 10:59:07 2022
"""

# import the necessary packages
import cv2 as cv
import numpy as np
import os
import pickle
from utils.constants import PATH_TRAIN,PATH_DATA, CATEGORIES

class featureExtractor(object):
    """
    feature extractor class for get image features
    Args:
        object (_object_): inheritance from base object class
    """    
    def __init__(self):
        self.__features = [None]*6      # [0: feature 1, 1: feature 2... so on]
        self.__data = []                # [0: numpy data]

    # Properties
    @property
    def features(self):
        """The features property."""
        return self.__features

    @features.setter
    def features(self, value):
        self.__features = value

    @property
    def data(self):
        """The data property."""
        return self.__data

    @data.setter
    def data(self, value):
        self.__data = value

    # Functions
    def get_features(self, image,size: tuple=(50,50), n_feature:int=1)-> list:
        """
            Method to extract features from a given image
        Args:
            image (_Image_): Image to extract features from.
            n_feature (int, optional): Number of features to extract, from 1 to 6. Defaults to 1.
        """
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Kernel for filters
        kernel1 = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        # Sharpening the image
        img_sharped = cv.filter2D(img_gray,-1,kernel1)
        # Resize the image
        img_resized =  cv.resize(img_sharped, size, interpolation=cv.INTER_AREA)
        # Save firts feature 
        self.features[0] = img_resized
        # threshold the image
        img_mBlur = cv.medianBlur(img_resized, 3)
        self.features[1] = img_mBlur
        _,img_thresh = cv.threshold(img_mBlur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        self.features[2] = img_thresh 
        img_morph = cv.morphologyEx(img_mBlur, cv.MORPH_OPEN, 
                                    cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)))
        self.features[3] = img_morph
        img_edges = cv.Canny(img_morph,20,60)
        self.features[4] = img_edges
        # Get key points and descriptors
        orb = cv.ORB_create()
        img_kp = np.ones(img_resized.shape[:2], dtype="uint8") * 255
        kp, des = orb.detectAndCompute(img_resized,None)
        img_kp = cv.cvtColor(cv.drawKeypoints(img_kp,kp,None),cv.COLOR_BGR2GRAY)
        self.features[5] = img_kp
        return self.features[n_feature-1]

    def get_dataset(self) -> list:
        """
            Method to get dataset from the features array
        Args:
            None
        """
        # data
        dataset = []
        # Collecting dataset
        for c in CATEGORIES:
            path = PATH_TRAIN+c
            label = CATEGORIES.index(c)
            for img in os.listdir(path):
                # Get image path
                imgpath = os.path.join(path, img)
                # Read the image
                image = cv.imread(imgpath)
                self.get_features(image)
                data_f1 = np.array(self.features[0]).flatten()
                data_f2 = np.array(self.features[1]).flatten()
                data_f3 = np.array(self.features[2]).flatten()
                data_f4 = np.array(self.features[3]).flatten()
                data_f5 = np.array(self.features[4]).flatten()
                data_f6 = np.array(self.features[5]).flatten()
                data = np.concatenate((data_f1,data_f2,data_f3,data_f4,data_f5,data_f6),axis=0)
                dataset.append([data,label])
        self.data = dataset
        return self.data

    def save_data(self,name: str = 'data') -> None:
        """
        Save the dataset as a file into a directory
        Args:
            name (str, optional): name of the file. Defaults to 'data'.
        Raises:
            Exception: When variable self.data does not have data yet.
        """        
        if self.data:
            file = open(PATH_DATA+name+'.pkl','wb')
            pickle.dump(self.data,file)
            file.close()
        else:
            raise Exception("No data found!, make sure to run get dataset function first") 

    def show_graphic_features(self, path_save: str) -> None:
        """
        Show on a single windows the whole features
        Raises:
            Exception: When variable self.features does not have data yet.
        """        
        if self.features:
            top_img = np.concatenate((self.features[0],self.features[1],
                                      self.features[2]),axis=1)
            bottom_img = np.concatenate((self.features[3],self.features[4],
                                         self.features[5]),axis=1)
            image = np.concatenate((top_img,bottom_img),axis=0)
            if path_save:
                cv.imwrite(path_save,image)
            cv.imshow('Features Image',image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            raise Exception("No features found!, make sure to run get features function first")

    def get_sideBottle(self, matrix) -> int:
        """
        Args:
            img (_Image_): Image object to detect side bottle

        Returns:
            int: 1 if the bottle is the right side otherwise 0
        """        
        side = None
        row,_ = matrix.shape
        up_black = np.sum(matrix[:int(row/2),:]==0)
        down_black = np.sum(matrix[int(row/2):,:]==0)
        if up_black < down_black:
            side = 1
            return side
        else:
            side = 0
        return side
        
        

# Testing v0.2 working
obj = featureExtractor()
img = cv.imread(PATH_TRAIN+'n_lvirgen/IMG20221020155231.jpg')
img_feature = obj.get_features(img,n_feature=3)
if obj.get_sideBottle(img_feature) == 1:
    print("Side of the bottle is right!")
else:
    print("Side of the bottle is wrong!")
obj.show_graphic_features()
data = obj.get_dataset()
obj.save_data('data_glass_plastic')
obj.show_graphic_features()
print("Finish")

