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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score

# Algorithms for classifications
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier


"""class modelSelector(object):
    #Class for model selection using machine learning algorithms.
    def __init__(self):
        self.name = name
        self.age = age
"""