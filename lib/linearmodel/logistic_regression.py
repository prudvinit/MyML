#!/usr/bin/env python
# coding: utf-8

# In[285]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action="ignore",category=DeprecationWarning)

data = pd.read_csv('LoR2b_train.csv', header=None)
dataTest = pd.read_csv('LoR2a_test.csv', header=None)
# data = data.apply(LabelEncoder().fit_transform)
mTraining = data.shape[0]

#to encode both training and testing data together
data = data.append(dataTest, ignore_index = True)

mask = data.dtypes==object         #filtering categorical columns
categoricalColumns = data.columns[mask].tolist()
encoder = LabelEncoder()
data[categoricalColumns] = data[categoricalColumns].apply(lambda col: encoder.fit_transform(col))

data = data.to_numpy()

def normalizer(data):
    return data / data.max(axis=0)

data = normalizer(data)

dataTest = data[mTraining:,:]
data     = data[:mTraining,:]

# In[286]:


# cor = data.corr()
# cor_target = abs(cor[14])           #Selecting highly correlated features
# relevant_features = cor_target[cor_target>0]

# In[287]:


data     = np.delete(data,    2, axis=1)
dataTest = np.delete(dataTest,2, axis=1)

numFeatures = np.size(data,1)-1     #n = number of features
X     = data[:, :-1]                 
Y     = data[:, numFeatures]
Xtest = dataTest[:,:-1]
Ytest = dataTest[:,numFeatures]


# In[288]:


def sigmoidFunc(x):
    return 1 / (1 + np.exp(-x))
sigmoidV = np.vectorize(sigmoidFunc)


# In[289]:


L1 = Lasso(max_iter = 10000)
L2 = Ridge(max_iter = 10000)

parameters     = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1 ,1, 5, 10, 20 ]}
ridgeRegressor = GridSearchCV(L2, parameters, scoring = 'neg_mean_squared_error', cv=5)
ridgeRegressor.fit(X,Y)

lambdaL2 =(ridgeRegressor.best_params_.get('alpha'))      #optimal regularization parameter for L2

lassoRegressor = GridSearchCV(L1, parameters, scoring = 'neg_mean_squared_error', cv=5)
lassoRegressor.fit(X,Y)
lambdaL1 = (lassoRegressor.best_params_.get('alpha'))     #optimal regularization parameter for L1

print("Lambda1, Lambda2 ", lambdaL1, lambdaL2)


# In[290]:


def GradientDescent(data,testData,noOfIterations,regType,regParam): #regType = L1/L2 or 0 for none,regParam=lambda        
    numFeatures     = np.size(data,1)-1           #n
    numTrainingRows = np.size(data,0)             #m
    numTestRows     = np.size(testData,0)         #m for test data
    thetavector     = np.ones(numFeatures+1)      #thetavector initialized with ones
    alpha           = 0.1                         #learning rate
    temp            = np.zeros(numFeatures+1)     #for saving temp values of theta(s) and simultaneous updation
    ErrortrainItr    = np.zeros(noOfIterations)    #for saving error value after each iteration for training set
    ErrortestItr     = np.zeros(noOfIterations)    #for saving error value after each iteration for test set

    X  = np.insert(data, 0, 1, axis = 1)
    X  = X[:,:-1]
    Y  = data[:, numFeatures]
    
    Xtest = np.insert(testData, 0, 1, axis = 1)
    Xtest = Xtest[:,:-1]
    Ytest = testData[:,numFeatures]
    
    
    for x in range(noOfIterations):               #number of iterations for gradient descent
        hypVector  = np.dot(X,thetavector)      #hypvector is hypothesis vector from i=1 to m
        hypVector  = sigmoidV(hypVector)      
        difference = hypVector - Y
        gradient   = np.dot(X.T,difference)     #d/d(theta)(J(theta))
        gradient   = gradient/numTrainingRows

        if(regType == 0):        #No Regression
            thetavector = (thetavector - (alpha*gradient))
        elif(regType == 1):      #L1 regression
            thetavector = (thetavector - (alpha* (gradient + (regParam*np.sign(thetavector))/numTrainingRows)))
        elif(regType == 2):    #L2 regression  #(1-alpha*(lambda/m))
            thetavector = thetavector*(1-alpha*(regParam/numTrainingRows)) - alpha*gradient

        #calculate cost function for training set
        costsum = 0.0
        yTrans = Y.transpose()
        hypVlog = np.log(hypVector)
        hypVlog1 = np.log(1-hypVector)
        costsum = ((np.dot(yTrans, hypVlog)) + np.dot(((1-Y).transpose()),hypVlog1))*(-1.0)
        costsum = costsum/numTrainingRows
        
        ErrortrainItr[x] = costsum
        
        #calculate cost function for test set      
        hypVectorTest = np.dot(Xtest, thetavector)
        hypVectorTest = sigmoidV(hypVectorTest)
        differenceTest = hypVectorTest - Ytest
        costsum = 0.0
        yTransTest = Ytest.transpose()
        hypVlogTest = np.log(hypVectorTest)
        hypVlog1Test = np.log(1-hypVectorTest)
        costsum = ((np.dot(yTransTest, hypVlogTest)) + np.dot(((1-Ytest).transpose()),hypVlog1Test))*(-1.0)
        costsum = costsum/numTestRows
        
        ErrortestItr[x] =costsum
        
    return ErrortrainItr, ErrortestItr


# In[291]:


def plot(Errortrain, Errortest, title,numberOfIterations):
    xaxis = np.arange(1,numberOfIterations+1)    
    plt.plot(xaxis,Errortrain)
    plt.plot(xaxis,Errortest)
    plt.title(title)
    plt.xlabel('#iterations')
    plt.ylabel('Error')
    plt.legend(['Training Set Error', 'Testing Set Error'], loc='upper right')
    plt.show()


# In[293]:


ErrortrainItrCatch,   ErrortestItrCatch   =  GradientDescent(data,dataTest,100,0,1)
ErrortrainItrCatchL1, ErrortestItrCatchL1 =  GradientDescent(data,dataTest,100,1,lambdaL1)
ErrortrainItrCatchL2, ErrortestItrCatchL2 =  GradientDescent(data,dataTest,100,2,lambdaL2)
# print(ErrortrainItrCatch[99])
plot(ErrortrainItrCatch, ErrortestItrCatch, "Without Regularization",100)
plot(ErrortrainItrCatchL1, ErrortestItrCatchL1, "With L1 Regularization",100)
plot(ErrortrainItrCatchL2, ErrortestItrCatchL2, "With L2 Regularization",100)


# In[ ]:




