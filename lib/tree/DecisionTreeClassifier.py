# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:01:42 2019

@author: pmaddala
"""


import pandas as pd
import numpy as np

dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],
       'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],
       'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],
       'Eat':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}

data = pd.DataFrame(dataset)


class Node:
    def __init__(self):
        self.split = None
        self.feature = None
        self.leaf = False
        self.left = None
        self.right = None
        self.result = None
        self.data = None
        
    def __repr__(self):
        return '['+self.split+' * '+str(self.feature)+' * '+str(self.data)+']'

class DecisionTree:
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.root = None
        
    def giniImpurity(self,dset):
        return 1-(len(dset[dset['Eat']=='Yes'])/len(dset))**2-(len(dset[dset['Eat']=='No'])/len(dset))**2
     
    def build(self,data):
        print('Building ')
        print('Data is ',data)
        overalllset = None
        overallrset = None
        overallgini = np.inf
        overallftr = None
        split = None
        
        gini = len(data)*giniImpurity(data)
        print('Gini impurity for total data is ',gini)
        node = Node()
        node.data = data
        #If there is no impurity in this data, make it a leaf node
        if gini==0:
            node.leaf = True
            node.result = len(data[data[self.target]=='Yes'])>=len(data[data[self.target]=='No'])
            return node
            
        
        for feature in data.columns:
            if feature == self.target:
                continue
            print('For feature ',feature)
            unique = data[feature].unique()
            tmngini = np.inf
            tmxlset = None
            tmxrset = None
            tmxftr = None

            #We can't split based on a single value, right??
            if len(unique)==1:
                continue
            
            for st in range(1,2**len(unique)-1):
                lvals = [unique[x] for x in [t[0] for t in enumerate(list(bin(st)[2:])[::-1]) if t[1]=='1']]
                lset = data[data[feature].isin(lvals)]
                rvals = list(set(unique)-set(lvals))
                rset = data[data[feature].isin(rvals)]
                if len(lvals)>len(rvals):
                    continue
                lgini = self.giniImpurity(lset)
                rgini = self.giniImpurity(rset)
                tgini = len(lset)*lgini+len(rset)*rgini
                print('lvals ',lvals)
                print(lset)
                print('rvals ',rvals)
                print('right set ',rset)
                print('tgini is ',tgini)
                if tgini<tmngini:
                    tmngini=tgini
                    tmxlset = lset
                    tmxrset = rset
                    tmxftr = lvals
            print('Best gini for ',feature,' is ',tmngini)
            if tmngini<overallgini:
                overallgini = tmngini
                overalllset = tmxlset
                overallrset = tmxrset
                overallftr = tmxftr
                split = feature
                
        #No improvement in gini value after split, so, let's make it as leaf node
        if overallgini>gini:
            node.leaf = True
            node.result = len(data[data[self.target]=='Yes'])>=len(data[data[self.target]=='No'])
            return node
            
        node.feature = overallftr
        node.split = split
        print('Best split is ',split)
            
        #Perfect split
        #if overallgini==0:
        #    print('Perfect*****************************************')
        #    return node
        if overalllset is not None:
            node.left = self.build(overalllset)
        if overallrset is not None:
            node.right = self.build(overallrset)
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
        if s[root.split] in root.feature:
            return self.__predict__(s,root.left)
        return self.__predict__(s,root.right)
        
        
    def predict(self,s):
        return self.__predict__(s,self.root)
        
        
tree = DecisionTree(data,'Eat')
root = tree.fit()

def t(x):
    if x is None:
        return ""
    return str(x)
def build_tree(root):
    if root==None:
        return ""
    return {"split" : t(root.split)+' '+t(root.feature), "left":build_tree(root.left), "right":build_tree(root.right) }

g = build_tree(root)
g = str(g).replace('\'','"')
g = g.replace('["','')
g = g.replace('"]','')
print(g)

score = 0
for i in range(len(data)):
    if tree.predict(data.loc[i]) and data.loc[i].Eat=='Yes':
        score+=1
    elif not tree.predict(data.loc[i]) and data.loc[i].Eat=='No':
        score+=1
print(score)
        