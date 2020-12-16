# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB

Main reference:
https://doi.org/10.1016/j.advengsoft.2013.12.007
https://seyedalimirjalili.com/gwo
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class HPSOGWO():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500, 
                 x_max=1, x_min=0, a_max=2, a_min=0):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter
        self.x_max = x_max
        self.x_min = x_min
        self.a_max = a_max
        self.a_min = a_min
        self.C1 = 0.5
        self.C2 = 0.5
        self.C3 = 0.5
        self.w = 0.5 + np.random.uniform()/2
        
        self._iter = 0
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        self.score_alpha = np.inf
        self.score_beta = np.inf
        self.score_delta = np.inf
        self.X_alpha = None
        self.X_beta = None
        self.X_delta = None

        self.X = np.random.choice(2, size=[self.num_particle, self.num_dim]).astype(float)
        self.V = 0.3 * np.random.normal(size=[self.num_particle, self.num_dim])
        
        self.update_score()
        
        self._itter = self._iter + 1

        
    def opt(self):
        while(self._iter<self.max_iter):
            a = self.a_max - (self.a_max-self.a_min)*(self._iter/self.max_iter) # (8)
            
            for i in range(self.num_particle):
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a # (3)
                D = np.abs(self.C1*self.X_alpha - self.w*self.X[i, :]) #(19)
                cstep_alpha = 1/(1+np.exp(-10*A*D-0.5)) # (18)
                bstep_alpha = (cstep_alpha >= np.random.uniform(size=self.num_dim))*1 # (17)
                X1 = (self.X_alpha + bstep_alpha)>=1 # (16)
                
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a # (3)
                D = np.abs(self.C2*self.X_beta - self.w*self.X[i, :]) #(19)
                cstep_beta = 1/(1+np.exp(-10*A*D-0.5)) # (18)
                bstep_beta = (cstep_beta >= np.random.uniform(size=self.num_dim))*1 # (17)
                X2 = (self.X_beta + bstep_beta)>=1 # (16)
                
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a # (3)
                D = np.abs(self.C3*self.X_delta - self.w*self.X[i, :]) #(19)
                cstep_delta = 1/(1+np.exp(-10*A*D-0.5)) # (18)
                bstep_delta = (cstep_delta >= np.random.uniform(size=self.num_dim))*1 # (17)
                X3 = (self.X_delta + bstep_delta)>=1 # (16)

                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                r3 = np.random.uniform(size=self.num_dim)
                self.V[i, :] = self.w*(self.V[i, :] \
                                       + self.C1*r1*(X1-self.X[i, :])
                                       + self.C2*r2*(X2-self.X[i, :])
                                       + self.C3*r3*(X3-self.X[i, :])) # (20)
                self.X[i, :] = self.sigmoid((X1+X2+X3)/3) + self.V[i, :] # (21)
                
                self.X[i, :] = self.X[i, :] >= np.random.uniform(size=self.num_dim) # (14)
                
            # self.X = np.clip(self.X, self.x_min, self.x_max)
            
            self.update_score()
            
            self._iter = self._iter + 1
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
    
    def update_score(self):
        score_all = self.fit_func(self.X)
        for idx, score in  enumerate(score_all):
            if score<self.score_alpha:
                self.score_alpha = score.copy()
                self.X_alpha = self.X[idx, :].copy()
                
            if score>self.score_alpha and score<self.score_beta:
                self.score_beta = score.copy()
                self.X_beta = self.X[idx, :].copy()
            
            if score>self.score_alpha and score>self.score_beta and score<self.score_delta:
                self.score_delta = score.copy()
                self.X_delta = self.X[idx, :].copy()
        
        self.gBest_X = self.X_alpha.copy()
        self.gBest_score = self.score_alpha.copy()
        self.gBest_curve[self._iter] = self.score_alpha.copy()
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-10*(x-0.5))) # (15)
            