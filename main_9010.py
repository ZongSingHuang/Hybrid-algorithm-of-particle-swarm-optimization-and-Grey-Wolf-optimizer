# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB
"""

from bGWO import bGWO
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

np.random.seed(42)

# 讀資料
Breastcancer = pd.read_csv('Breastcancer.csv', header=None).values

X = Breastcancer[:, :-1]
y = Breastcancer[:, -1]

def Breastcancer_test(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, x[i, :]], y, cv=skf)
            loss[i] = 0.99*(1-score.mean()) + 0.01*(np.sum(x[i, :])/X.shape[1])
        else:
            loss[i] = np.inf
            print(666)
    return loss

skf = StratifiedKFold(n_splits=10, shuffle=True)
optimizer = bGWO(fit_func=Breastcancer_test, 
                  num_dim=X.shape[1], num_particle=5, max_iter=70, x_max=1, x_min=0)
optimizer.opt()

score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, optimizer.gBest_X], y, cv=skf)
print(np.sum(optimizer.gBest_X))
print(score.mean())

score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=skf)
print(X.shape[1])
print(score.mean())