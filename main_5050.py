# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB
"""

from HPSOGWO import HPSOGWO
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# 讀資料
Breastcancer = pd.read_csv('M-of-n.csv', header=None).values

X_train, X_test, y_train, y_test = train_test_split(Breastcancer[:, :-1], Breastcancer[:, -1], stratify=Breastcancer[:, -1], test_size=0.5)

def Breastcancer_test(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            knn = KNeighborsClassifier(n_neighbors=5).fit(X_train[:, x[i, :].astype(bool)], y_train)
            score = accuracy_score(knn.predict(X_test[:, x[i, :].astype(bool)]), y_test)
            loss[i] = 0.01*(1-score) + 0.99*(np.sum(x[i, :])/X_train.shape[1])
        else:
            loss[i] = np.inf
            print(666)
    print(loss.max())
    return loss

optimizer = HPSOGWO(fit_func=Breastcancer_test, 
                    num_dim=X_train.shape[1], num_particle=10, max_iter=100, x_max=1, x_min=0)
optimizer.opt()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[:, optimizer.gBest_X.astype(bool)], y_train)
print(np.sum(optimizer.gBest_X))
print(accuracy_score(knn.predict(X_test[:, optimizer.gBest_X.astype(bool)]), y_test))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(X_train.shape[1])
print(accuracy_score(knn.predict(X_test), y_test))