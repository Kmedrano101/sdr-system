# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:29:31 2022

@author: Kmedrano101
"""

import numpy as np
import pandas as pd
import pickle

dataN1 = np.ones((10,1))
dataN2 = np.zeros((10,1))
dataN3 = np.ones((10,1))

dataFinal = np.concatenate((dataN1,dataN2,dataN3),axis=1)


test = []

if not test:
    print("thres is data")

#data['Feature 2'] = dataN2
#print(data)



"""pick_in = open('data/dataset/data.pkl','rb')
data = pickle.load(pick_in)
pick_in.close()
"""
