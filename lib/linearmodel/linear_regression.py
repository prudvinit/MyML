# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:27:26 2018

@author: pmaddala
"""

import numpy as np
from enum import Enum
import math
import matplotlib.pyplot as plt

class Cost(Enum):
    MSE = 1
    MAE = 2
    
class LinearRegression:
    def __init__(self,x,y,alpha=0.01,cost=Cost.MSE,L=0):
        self.input = x
        t = []
        for i in range(len(x)):
            t.append(np.insert(x[i],0,1))
        self.x = np.array(t)  
        self.y = y
        self.alpha = alpha
        self.coef = np.random.random(1+len(x[0]))
        self.coef = np.zeros(1+len(x[0]))
        self.cost = cost
        self.errors = []
        self.L = L
        
    def train(self,iterations=100):        
        for i in range(iterations):
            error = self.getError()
            self.errors.append(error)
            #print('Coef is : ',self.coef)
            self.coef = self.coef - self.alpha*self.diff()
            
    def train_tolerannce(self,e=0.05):
        error = self.getError()
        self.errors.append(error)
        prev = error
        while True:
            self.coef = self.coef - self.alpha*self.diff()
            terr = self.getError()
            self.errors.append(terr)
            if abs(terr-prev)<=e:
                break
            prev = terr
            
        
    def getError(self):
        if self.cost == Cost.MSE:
            error = 0
            for i in range(len(x)):
                error = error + (self.y[i]-np.dot(self.coef,self.x[i]))**2
            return math.sqrt(error/len(self.x))
            #return math.sqrt(np.sum((self.y-np.dot(self.coef,self.x))**2)/len(self.x))
        if self.cost == Cost.MAE:
            error = 0
            for i in range(len(x)):
                error = error + abs(self.y[i]-np.dot(self.coef,self.x[i]))
            return error
                
    def diff(self):
        if self.cost == Cost.MSE:
            tcoef = []
            for c in range(len(self.coef)):
                error = 0
                for i in range(len(self.x)):
                    error = error - (self.y[i]-np.dot(self.coef,self.x[i]))*self.x[i][c]
                error = error + self.L*self.coef[c]
                error = 2*error
                
                tcoef.append(error)
            tcoef = np.array(tcoef)
            
            return tcoef
        
        if self.cost == Cost.MAE:
            tcoef = []
            for c in range(len(self.coef)):
                error = 0
                for i in range(len(self.x)):
                    if self.y[i]-np.dot(self.coef,self.x[i])>0:
                        error = error -self.x[i][c]
                    else:
                        error = error + self.x[i][c]
                    #error = error - (self.y[i]-np.dot(self.coef,self.x[i]))*self.x[i][c]
                #error = 2*error
                
                tcoef.append(error)
            tcoef = np.array(tcoef)            
            return tcoef
            
    def predict(self,x_new):        
        return self.coef*np.insert(x_new,0,1)
        
    def plotInput(self):
        plt.scatter(self.input,self.y)
        plt.plot(np.linspace(0,10,1000),self.coef[1]*np.linspace(0,10,1000) + self.coef[0])
        
    def plot_errors(self):
        plt.plot(range(len(self.errors)),self.errors)
        
        
if __name__ == '__main__':
    x = np.array([[1],[5],[3]])
    yl = [5,13,14]
    y = np.array(yl)
    l = LinearRegression(x,y,cost=Cost.MSE,L=0)
    l.train()
    l.plotInput()