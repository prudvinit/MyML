#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import pickle
import numpy as np
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


# In[3]:


data = unpickle('cifar-10-batches-py/data_batch_2')    #data_batch_2 from 9000-10000, std=min=4.77
X_train = data['data']
Y_train = data['labels']
Y_train = np.array(Y_train)
X_train = X_train[9000:10000, :]
Y_train = Y_train[9000:10000]

test_data = unpickle('cifar-10-batches-py/test_batch')
X_test = test_data['data']
Y_test = test_data['labels']
Y_test = np.array(Y_test)
X_test = X_test[0:1000,:]
Y_test = Y_test[0:1000]
# arr = np.zeros(10)
# for i in range(1000):
#     arr[Y[i]]+=1
# print(arr)
# print(np.std(arr))


# In[4]:


def classifier(X_train,Y_train, X_test, Y_test, kernel_type, decFuncShape):
    print(kernel_type," kernel", " & ", decFuncShape , " classification method" )
    kf = KFold(n_splits = 5, shuffle = False)
    fold_count = 1
    for train_index, validation_index in kf.split(X_train):
        print("Fold : ", fold_count)
        X_train_fold, X_validation_fold = X_train[train_index], X_train[validation_index]
        Y_train_fold, Y_validation_fold = Y_train[train_index], Y_train[validation_index]
        
        if(kernel_type == 'rbf'):
            clf = svm.SVC(kernel = kernel_type, gamma = 'scale', C=1,decision_function_shape=decFuncShape,probability = True).fit(X_train_fold, Y_train_fold)
        elif(kernel_type == 'poly'):
            clf = svm.SVC(kernel = kernel_type, degree = 2, C=1,decision_function_shape=decFuncShape,probability = True).fit(X_train_fold, Y_train_fold)
        elif(kernel_type == 'linear'):
            clf = svm.SVC(kernel = kernel_type, C=1,decision_function_shape=decFuncShape, probability = True).fit(X_train_fold, Y_train_fold)
        else:
            print("Invalid kernel")
            break
          
        #confusion matrix: 
        ticklabels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        y_pred1 = clf.predict(X_test)
        acc = accuracy_score(Y_test, y_pred1) * 100
        print("Accuracy : ", acc , "%")
        cm = confusion_matrix(Y_test, y_pred1)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt='d',xticklabels= ticklabels, yticklabels= ticklabels, cmap="YlGnBu")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        #roc-auc curve:
        y_pred = clf.predict_proba(X_test)
        Y_test_f = label_binarize(Y_test, classes=[0,1,2,3,4,5,6,7,8,9])
#         y_pred = label_binarize(y_pred, classes=[0,1,2,3,4,5,6,7,8,9])
        numClasses = 10
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(numClasses):
            fpr[i], tpr[i], _ = roc_curve(Y_test_f[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = ['aqua', 'darkorange', 'cornflowerblue','darkolivegreen','navy','fuchsia','dimgray','red','yellow','lawngreen']
        for i, color in zip(range(numClasses), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kernel: ' + kernel_type + " ,Method: " + decFuncShape + " ,Fold: " + str(fold_count))
        plt.legend(loc=(1.04,0))
        plt.show()
        
        fold_count+=1


# In[ ]:


if __name__ == '__main__':
    #SVM WITH NO KERNEL
    classifier(X_train,Y_train,X_test,Y_test,'linear','ovr')  #ONE-VS-ALL CLASSIFICATION METHOD
    classifier(X_train,Y_train,X_test,Y_test,'linear','ovo')  #ONE-VS-ONE CLASSIFICATION METHOD

    #SVM WITH RBF KERNEL
    classifier(X_train,Y_train,X_test,Y_test,'rbf','ovr')     #ONE-VS-ALL CLASSIFICATION METHOD
    classifier(X_train,Y_train,X_test,Y_test,'rbf','ovo')     #ONE-VS-ONE CLASSIFICATION METHOD

    #SVM WITH QUADRATIC POLYNOMIAL KERNEL
    classifier(X_train,Y_train,X_test,Y_test,'poly','ovr')    #ONE-VS-ALL CLASSIFICATION METHOD
    classifier(X_train,Y_train,X_test,Y_test,'poly','ovo')    #ONE-VS-ONE CLASSIFICATION METHOD


# In[ ]:





# In[ ]:




