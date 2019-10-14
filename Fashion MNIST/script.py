# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:37:41 2019

@author: Prudvi
"""
import os
import pandas as pd
import time
import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Dropout,Flatten
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


os.chdir('D:\Coding\ML\Fashion MNIST')
start = time.time()
train = pd.read_csv('data/fashion-mnist_train.csv')
images = []
for i in range(train.shape[0]):
    img = train.iloc[i][1:].to_numpy().reshape(28,28,1)
    images.append(img/255)
    
    
X = np.array(images)
y = to_categorical(train.label.values)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

start= time.time()
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))
    
print('Took ',time.time()-start,' Seconds')
from matplotlib import pyplot as plt
plt.imshow(images[678].to_numpy().reshape(28,28), interpolation='cubic')
plt.show()