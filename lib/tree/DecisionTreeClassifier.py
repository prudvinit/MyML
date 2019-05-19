# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:01:42 2019

@author: pmaddala
"""


import pandas as pd
import numpy as np

class Node:
    def __init__(self):
        self.split = None
        self.feature_values = None
        self.leaf = False
        self.left = None
        self.right = None
        self.result = None
        self.gini = 0
        self.data = None
        self.level=0
        
    def __repr__(self):
        return '['+self.split+'  '+str(self.feature)+']'

class DecisionTree:
    #takes data and target column
    def __init__(self,data,target='Eat',depth=None,v=0):
        self.data = data
        self.target = target
        self.root = None
        self.depth = depth
        self.v = v
        
    #This helper function calculates gini index of the dataset
    def giniIndex(dset,target):
        return 1-(len(dset[dset[target]==True])/len(dset))**2-(len(dset[dset[target]==False])/len(dset))**2
     
    #This method builds the Decision tree
    def build(self,data,level=0):
        if self.v==1:
            print('======================================')
            print('Building Tree for the data')
            print('======================================')
            print(data)
        left_dataset = None
        right_dataset = None
        min_gini = np.inf
        best_feature = None
        feature_values = None
        
        gini = len(data)*DecisionTree.giniIndex(data,self.target)
        if self.v==1:
            print('GiniIndex for total data = ',gini)
        node = Node()
        node.data = data
        node.level = level
        node.gini = gini
        
        if self.depth is not None and level==self.depth:
            node.leaf = True
            node.result = len(data[data[self.target]==True])>=len(data[data[self.target]==False])
            return node
        
        #If there is no impurity in this data, make it a leaf node
        if gini==0:
            if self.v==1:
                print('The data is pure, no split is needed ')
            node.gini=0
            node.leaf = True
            node.result = len(data[data[self.target]==True])>=len(data[data[self.target]==False])
            return node
            
        
        for feature in data.columns:
            #We need not split on target column
            if feature == self.target:
                continue
            
            #Find all unique values for the feauture
            unique = data[feature].unique()
            if self.v==1:
                print('________________________________________________________')
                print('Evaluating all possible splits for the feature "',feature,'"')
                print('________________________________________________________')
                print('All the values for this feature are ',unique)
            
            #Initialize gini, left and right datasets and best feature values
            tmngini = np.inf
            tldset = None
            trdset = None
            tbftr = None

            #We can't split based on a single value,There must be atleast 2 unique values to be able to split
            if len(unique)==1:
                print('Ignoring this feature as it has only a single unique value')
                continue
            
            #To find the best values for split on the given feature
            for st in range(1,2**len(unique)-1):
                
                lvals = [unique[x] for x in [t[0] for t in enumerate(list(bin(st)[2:])[::-1]) if t[1]=='1']]
                #Find left data set
                lset = data[data[feature].isin(lvals)]
                rvals = list(set(unique)-set(lvals))
                #Find right data set
                rset = data[data[feature].isin(rvals)]
                #Avoid dealing with duplicate sets
                if len(lvals)>len(rvals):
                    continue
                #Find gini index for left split
                lgini = DecisionTree.giniIndex(lset,self.target)
                #Find gini impurity for right split
                rgini = DecisionTree.giniIndex(rset,self.target)
                #Find the total weighted gini. 
                tgini = len(lset)*lgini+len(rset)*rgini
                if self.v==1:                    
                    print(' ******************************************')
                    print(' ***** split based on values ',lvals,'*****')
                    print(' ******************************************')
                    print('-----------------------')
                    print('Left dataset')
                    print('-----------------------')
                    print(lset)
                    print('-----------------------')
                    print('Right dataset')
                    print('-----------------------')
                    print(rset)
                    print('Weighted Gini for this split ',tgini)
                #Update minimum gini
                if tgini<tmngini:
                    tmngini=tgini
                    tldset = lset
                    trdset = rset
                    tbftr = lvals
            if self.v==1:
                print('Best gini for ',feature,' is ',tmngini)
                
            #Update minimum gini
            if tmngini<min_gini:
                min_gini = tmngini
                left_dataset = tldset
                right_dataset = trdset
                feature_values = tbftr
                best_feature = feature
                
        #No improvement in gini value after split, Make it as leaf node
        if min_gini>tmngini:
            node.leaf = True
            node.result = len(data[data[self.target]])>len(data[data[self.target]])
            return node
            
        node.min_gini= min_gini
        node.feature_values = feature_values
        node.split = best_feature
        if self.v==1:
            print('Best split is "',best_feature,'" values are ',feature_values,' and GiniIndex is ',min_gini)
        
        #Build tree for left dataset
        if left_dataset is not None:
            node.left = self.build(left_dataset,level+1)
            
        #Build tree for right dataset
        if right_dataset is not None:
            node.right = self.build(right_dataset,level+1)
            
        #If both the trees are not built, it has to be leaf
        if node.left==None and node.right==None:
            node.leaf = True
        
        return node        
        
    
    def fit(self):
        self.root = self.build(data)
        return self.root
        
    def __predict__(self,s,root):
        if root is None:
            return False
        if root.leaf:
            return root.result
        if s[root.split] in root.feature_values:
            return self.__predict__(s,root.left)
        return self.__predict__(s,root.right)
        
        
    def predict(self,s):
        return self.__predict__(s,self.root)
   
def t(x):
    if x is None:
        return ""
    return str(x)
def build_tree(root):
    if root==None:
        return ""
    return {"split" : t(root.split)+' '+str(root.level), "left":build_tree(root.left), "right":build_tree(root.right) }


if __name__ == '__main__':
    
    dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],
       'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],
       'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],
       'Eat':[False, False, True, False, True, True, False, True, True, True]}

    data = pd.DataFrame(dataset)[:]
       
    tree = DecisionTree(data,'Eat',depth=None,v=1)
    root = tree.fit()
    
    
    g = build_tree(root)
    g = str(g).replace('\'','"')
    g = g.replace('["','')
    g = g.replace('"]','')
    #print(g)
    
    score = 0
    for i in range(len(data)):
        pred = tree.predict(data.loc[i])
        if pred and data.loc[i].Eat:
            score+=1
        elif not pred and not data.loc[i].Eat:
            score+=1
    print('Accuracy on training data is ',(score*100/len(data)))
            