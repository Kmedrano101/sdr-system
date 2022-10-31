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

class predictorModel(featureExtractor):
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
    
"""obj = predictorModel()
image = cv.imread('data/test/test_Y_2.jpeg')
img = cv.resize(image, (560,580), interpolation=cv.INTER_AREA)
img = cv.rectangle(img,(140,0),(420,580),color=(0, 255, 0), thickness=3)
obj.load_model()
if obj.make_prediction(image) == 1:
    img = cv.putText(img, 'Bottle: YES', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0),2, cv.LINE_AA)
else:
    img = cv.putText(img, 'Bottle: NO', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0),2, cv.LINE_AA)
cv.imshow('Prediction',img)
cv.waitKey(0)
cv.destroyAllWindows()
print("finish")"""
        
