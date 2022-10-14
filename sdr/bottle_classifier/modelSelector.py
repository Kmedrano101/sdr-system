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

# Algorithms for classifications
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier


class modelSelector(object):
    """
    Class for model selection using machine learning algorithms.
    Args:
        object (_object_): inheritance from base object class
    """
    def __init__(self):
        self.__dataset = None
        self.__models = [None]*7
        self._cmatrix_plots = [None]*7      # 0:RF, 1:GB, 2:GP, 3:SGD, 4:PC, 5:KN, 6:SVM 
        self._accuracy_values = [None]*7    # 0:RF, 1:GB, 2:GP, 3:SGD, 4:PC, 5:KN, 6:SVM 

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
            X = []
            y = []
            # Random Forest Classifier
            rf_model = RandomForestClassifier()
            self.models[0] = rf_model
            # Gradient Boosting Classifier
            gb_model = GradientBoostingClassifier()
            self.models[1] = gb_model
            # Gaussian Process Classifier
            kernel = 1.0 * RBF(1.0)
            gp_model = GaussianProcessClassifier(kernel=kernel)
            self.models[2] = gp_model
            # SGD Classifier
            sgd_model = SGDClassifier()
            self.models[3] = sgd_model
            # Perceptron Classifier
            p_model = Perceptron()
            self.models[4] = p_model
            # KNeighbors Classifier
            kn_model = KNeighborsClassifier()
            self.models[5] = kn_model
            # SVM Classifier
            svm_model = SVC(C=1,kernel='poly',gamma='auto') 
            self.models[6] = svm_model
            y_pred = [None]*7
            acc = [None]*2
            
            # Training
            print("Training start time",datetime.now().strftime("%H:%M:%S"))
            for n_iter in range(iterations):
                # Get X and y values
                for feature,label in self.dataset:
                    X.append(feature)
                    y.append(label)
                # Split dataset into X train-test and y train-test 
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
                for i in range(7):
                    self.models[i] = self.models[i].fit(X_train, y_train)
                # Predictors
                for i in range(7):
                    y_pred[i] = self.models[i].predict(X_test)
                    cm = confusion_matrix(y_test, y_pred[i])
                    self.cmatrix_plots[i] = ConfusionMatrixDisplay(confusion_matrix=cm)
                # Save accuracy values
                for i in range(7):
                    acc[0] = accuracy_score(y_test, y_pred[i])
                    acc[1] = cross_val_score(self.models[i], X_train, y_train, cv=5, scoring='f1_macro').mean()
                    if n_iter > 0:
                        acc1 = (self.accuracy_values[i][0] + acc[0]) / 2
                        acc2 = (self.accuracy_values[i][1] + acc[1]) / 2
                        self.accuracy_values[i][0] = acc1
                        self.accuracy_values[i][1] = acc2
                    else:
                        self.accuracy_values[i] = acc.copy()
            print("Training finish time",datetime.now().strftime("%H:%M:%S"))

    def save_models(self)-> None:
        """
        Save all models in the directory
        """        
        for i in range(7):
            joblib.dump(self.models[i], PATH_MODEL+MODEL_NAMES[i])

# Testing v0.1 working
obj = modelSelector()
obj.load_data(path='data/dataset/data.pkl')
obj.train_models()
obj.save_models()
print(obj.accuracy_values)
obj.cmatrix_plots[0].plot()
plt.show()
