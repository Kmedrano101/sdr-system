# -*- coding: utf-8 -*-
"""
Machine Learning Predictor
Created on Fri Oct 14 12:07:30 2022

@author: Kmedrano101
"""

# import modules
import joblib
from featuresExtractor import featureExtractor
from utils.constants import PATH_MODEL
import cv2 as cv
import numpy as np

PATH_MODEL = 'model/glass_beers/' # this just for test

class predictor(featureExtractor):
    def __init__(self):
        self.result = None
        self.model = None
        super().__init__()

    def load_model(self, name:str='model_lgb.sav')-> None:
        self.model = joblib.load(PATH_MODEL+name)

    def make_prediction(self, img)-> int:
        self.get_features(img)
        d_f1 = np.array(self.features[0]).flatten()
        d_f2 = np.array(self.features[1]).flatten()
        d_f3 = np.array(self.features[2]).flatten()
        d_f4 = np.array(self.features[3]).flatten()
        d_f5 = np.array(self.features[4]).flatten()
        d_f6 = np.array(self.features[5]).flatten()
        data = np.concatenate((d_f1,d_f2,d_f3,d_f4,d_f5,d_f6),axis=0)
        if self.model != None:
            self.result = self.model.predict(data.reshape(1,-1))
        else:
            raise Exception("model not loaded!")
        return self.result
    
obj = predictor()
image = cv.imread('data/train/n_lvirgen/IMG20221020153326.jpg')
obj.load_model()
print(obj.make_prediction(image))
print("finish")
        
