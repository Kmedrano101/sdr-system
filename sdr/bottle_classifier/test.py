# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:29:31 2022

@author: Kmedrano101
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

p = Path('.')
#[print(x) for x in p.iterdir() if x.is_dir()]

dicc = {'SVM': {'accuracy':0.0,'cross val': 3.0,'speed':0.0}}

print(dicc['SVM']['cross val'])

listaA = [None]*4

print(len(listaA))

print(sum([2,3]))


