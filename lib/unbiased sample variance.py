# -*- coding: utf-8 -*-
"""
Created on Wed May  1 01:10:02 2019

@author: pmaddala
"""

import random 
import matplotlib.pyplot as plt

#This generates a random distribution
def gen(POP_SIZE=10000,MIN=0,MAX=1000):
    data = []
    for _ in range(POP_SIZE):
        data.append(random.randint(MIN,MAX))
    return data
   
#This plots frequence curve
def draw(data):
    c = {}
    for e in data:
        if e not in c:
            c[e] = 0
        c[e]+=1
    x = list(c.keys())
    y = [c[t] for t in x]
    plt.scatter(x,y)
    #return x,y
 
#This function calculates variance of data of population   
def variance(data,bias=0):
    mean = sum(data)/len(data)
    sq = [(x-mean)**2 for x in data]
    return sum(sq)/(len(data)-bias)

#Generate data
data = gen()
#Find population variance
popvar = variance(data)

#Perform random sampling
def getsample(data,n=100):
    ans = []
    for i in range(n):
        ans.append(data[random.randint(0,len(data)-1)])
    return ans
    
tsvar = 0
avg = []
#Randomly generate sample lot of times and plot of average of unbiased sample variance. 
#In some time, it will match the original population variance. 
for i in range(1000):
    sdata = getsample(data)
    #Sample unbiased variance, We must use n-1, not n while calculating variance
    svar = variance(sdata,bias=1)
    tsvar += svar
    asvar = tsvar/(i+1)
    avg.append(asvar)
plt.plot(avg)
plt.plot([popvar]*len(avg))