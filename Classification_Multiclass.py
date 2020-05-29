# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 09:22:59 2019

@author: Admin
"""
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import confusion_matrix 
import time 
#1)
from sklearn import datasets 
dataset=datasets.load_iris()
#2)
data=dataset.data
#3)
label=dataset.target
#5)
xtrain,xtest,ytrain,ytest=train_test_split(data,label,test_size=0.33,random_state=0)
#6)
#svm multiclass
#7)
help(svm.SVC)

model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
start_time= time.time()
model.fit(xtrain,ytrain)
temps=time.time()-start_time
ypredict=model.predict(xtest)
acc=accuracy_score(ytest,ypredict)
fpr,tpr,t=roc_curve(ytest,ypredict,pos_label=1)
aucc=auc(fpr,tpr)*100

