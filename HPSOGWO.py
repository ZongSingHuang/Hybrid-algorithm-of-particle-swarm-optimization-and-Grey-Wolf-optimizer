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
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0, a_max=2, a_min=0,
                 init_w=0.5, c1=0.5, c2=0.5, c3=0.5, k=0.2):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub
        self.lb = lb
        self.a_max = a_max
        self.a_min = a_min
        self.init_w = init_w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.k = k
        self.v_max = self.k*(self.ub-self.lb)

        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        self.F_alpha = np.inf
        self.F_beta = np.inf
        self.F_delta = np.inf
        self.X_alpha = np.zeros([self.D])
        self.X_beta = np.zeros([self.D])
        self.X_delta = np.zeros([self.D])

    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        self.V = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        
        # 迭代
        for g in range(self.G):
            # 適應值計算
            F = self.fitness(self.X)
            
            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 更新
            for i in range(self.P):
                if F[i]<self.F_alpha:
                    self.F_alpha = F[i].copy()
                    self.X_alpha = self.X[i].copy()
                elif F[i]<self.F_beta:
                    self.F_beta = F[i].copy()
                    self.X_beta = self.X[i].copy()
                elif F[i]<self.F_delta:
                    self.F_delta = F[i].copy()
                    self.X_delta = self.X[i].copy()
                    
            a = self.a_max - (self.a_max-self.a_min)*(g/self.G)
            self.w = self.init_w + np.random.uniform()/2
            
            r1 = np.random.uniform(size=[self.P, self.D])
            A = 2*a*r1 - a
            D = np.abs(self.c1*self.X_alpha - self.w*self.X)
            X1 = self.X_alpha - A*D

            r1 = np.random.uniform(size=[self.P, self.D])
            A = 2*a*r1 - a
            D = np.abs(self.c2*self.X_beta - self.w*self.X)
            X2 = self.X_beta - A*D

            r1 = np.random.uniform(size=[self.P, self.D])
            A = 2*a*r1 - a
            D = np.abs(self.c3*self.X_delta - self.w*self.X)
            X3 = self.X_delta - A*D

            r2 = np.random.uniform(size=[self.P, self.D])
            r3 = np.random.uniform(size=[self.P, self.D])
            r4 = np.random.uniform(size=[self.P, self.D])
            
            self.V = self.w*(self.V + self.c1*r2*(X1-self.X)
                                    + self.c2*r3*(X2-self.X)
                                    + self.c3*r4*(X3-self.X))
            self.V = np.clip(self.V, -self.v_max, self.v_max) # 邊界處理

            self.X = self.X + self.V
            self.X = np.clip(self.X, self.lb, self.ub) # 邊界處理

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.loss_curve[-1], 3))+']')
        plt.plot(self.loss_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()