# -*- coding: utf-8 -*-
"""
Machine learning model selector
@author: Kmedrano101
Created on Thu Oct 13 10:59:07 2022
"""

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score, f1_score
from utils.constants import PATH_DATA, PATH_MODEL, MODEL_NAMES
import pickle, random
from datetime import datetime
from time import time
from statistics import mean
# Algorithms for classifications
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb



class modelSelector(object):
    """
    Class for model selection using machine learning algorithms.
    Args:
        object (_object_): inheritance from base object class
    """
    def __init__(self):
        self.__dataset = None
        self.__models = [None]*4
        self._cmatrix_plots = [None]*4      # 0:SVM, 1:GB, 2:XGB, 3:LGB
        self._accuracy_values = {0: {'Accuracy': 0,'crossVal': 0,'Speed': 0},
                                 1:  {'Accuracy': 0,'crossVal': 0,'Speed': 0},
                                 2: {'Accuracy': 0,'crossVal': 0,'Speed': 0},
                                 3: {'Accuracy': 0,'crossVal': 0,'Speed': 0}}
    # Properties
    @property
    def dataset(self):
        """The dataset property."""
        return self.__dataset

    @dataset.setter
    def dataset(self, value):
        self.__dataset = value

    @property
    def cmatrix_plots(self):
        """The accuracy_plots property."""
        return self._cmatrix_plots

    @cmatrix_plots.setter
    def cmatrix_plots(self, value):
        self._cmatrix_plots = value    

    @property
    def accuracy_values(self):
        """The accuracy_values property."""
        return self._accuracy_values

    @accuracy_values.setter
    def accuracy_values(self, value):
        self._accuracy_values = value   

    @property
    def models(self):
        """The models property."""
        return self.__models

    @models.setter
    def models(self, value):
        self.__models = value

    # Methods
    def load_data(self, path: str='data/dataset/data.pkl') -> None:
        """
        Function to save the data as a pickle file
        Args:
            path (str, optional): path of the file. Defaults to 'data/dataset/data.pkl'.
        """        
        file = open(path, 'rb')
        self.dataset = pickle.load(file)
        file.close()

    def train_models(self, iterations:int=1) -> None:
        """
        Train differents machine learning algorithms in order to get the models
        Args:
            iterations (int, optional): Number of iterations to run. Defaults to 1.

        Raises:
            Exception: When variable self.dataset is None
        """
        if self.dataset is None:
            raise Exception("Dataset not loaded!")
        else:
            X,y = [],[]
            # Get X and y values
            for feature,label in self.dataset:
                X.append(feature)
                y.append(label)
            # SVM Classifier
            svm_model = SVC(C=1,kernel='poly',gamma='auto') 
            self.models[0] = svm_model
            # Gradient Boosting Classifier
            gb_model = GradientBoostingClassifier()
            self.models[1] = gb_model
            # Extreme Gradient Boosting Classifier
            xgb_model = XGBClassifier()
            self.models[2] = xgb_model
            # lighGBM Classifier
            lgb_model = lgb.LGBMClassifier()
            self.models[3] = lgb_model
            y_pred = [None]*len(self.models)
            acc = [None]*len(self.models)
            
            # Training
            print("Training start time",datetime.now().strftime("%H:%M:%S"))
            for n_iter in range(iterations):
                # Split dataset into X train-test and y train-test 
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
                for i in range(len(self.models)):
                    self.models[i] = self.models[i].fit(X_train, y_train)
                    
                # Predictors
                for i in range(len(self.models)):
                    y_pred[i] = self.models[i].predict(X_test)
                    cm = confusion_matrix(y_test, y_pred[i])
                    self.cmatrix_plots[i] = ConfusionMatrixDisplay(confusion_matrix=cm)
                
                # Save accuracy values
                for i in range(len(self.models)):
                    start = time()
                    acc = accuracy_score(y_test, y_pred[i])
                    cval = cross_val_score(self.models[i], X_train, y_train, 
                                           cv=5, scoring='f1_macro').mean()
                    tm = np.round(time()-start,3)
                    if n_iter > 0:
                        acc_mean = mean([self.accuracy_values[i]['Accuracy'], acc])
                        cval_mean = mean([self.accuracy_values[i]['crossVal'], cval])
                        tm_mean = sum([self.accuracy_values[i]['Speed'], tm])
                        self.accuracy_values[i]['Accuracy'] = acc_mean
                        self.accuracy_values[i]['crossVal'] = cval_mean
                        self.accuracy_values[i]['Speed'] = tm_mean
                    else:
                        self.accuracy_values[i]['Accuracy'] = acc
                        self.accuracy_values[i]['crossVal'] = cval
                        self.accuracy_values[i]['Speed'] = tm

                progress = round((((n_iter+1) * 100)/ iterations),2)
                print(f"Progress percentage... {progress} %")
            print("Training finish time",datetime.now().strftime("%H:%M:%S"))

    def save_models(self)-> None:
        """
        Save all models in the directory
        """        
        for i in range(len(self.models)):
            joblib.dump(self.models[i], PATH_MODEL+MODEL_NAMES[i])

# Testing v0.2 working
obj = modelSelector()
obj.load_data(path='data/dataset/data_glass_plastic.pkl')
obj.train_models(iterations=10)
obj.save_models()
print(obj.accuracy_values)
obj.cmatrix_plots[0].plot()
obj.cmatrix_plots[1].plot()
obj.cmatrix_plots[2].plot()
obj.cmatrix_plots[3].plot()
plt.show()
