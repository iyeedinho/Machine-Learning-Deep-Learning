# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:46:22 2019

@author: Admin
"""

from sklearn.datasets.samples_generator import make_blobs
datav,y=make_blobs(n_samples=400,n_features=2,centers=4)

import matplotlib.pyplot as plt
plt.scatter(datav[:,0],datav[:,1])
