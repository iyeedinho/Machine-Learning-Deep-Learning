# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:30:18 2019

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import confusion_matrix 
import time 

dataset=pd.read_csv('C:/diabetes.csv')

Label=dataset['Outcome']

Data=dataset.drop(['Outcome'],axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Data,Label,test_size=0.33,random_state=0)

help(svm.SVC)


model=svm.SVC(kernel='linear',C=0.5)
start_time= time.time()
model.fit(xtrain,ytrain)
temps=time.time()-start_time
ypredict=model.predict(xtest)
acc=accuracy_score(ytest,ypredict)
fpr,tpr,t=roc_curve(ytest,ypredict,pos_label=1)
aux1=auc(fpr,tpr)*100
p=tpr/(fpr+tpr)
cm=confusion_matrix(ytest,ypredict)
help(confusion_matrix)
plt.figure()
plt.plot(fpr,tpr)
plt.xlabel('FP')
plt.ylabel('TP')
plt.title('Roc curve')

Label.value_counts()
#partie2)
dataset_select1=dataset.loc[dataset['Outcome'].isin([1])]
dataset_select0=dataset.loc[dataset['Outcome'].isin([0])]

label1=dataset_select1['Outcome']
label0=(dataset_select0['Outcome']+1)*(-1)
data1=dataset_select1.drop(['Outcome'],axis=1)
data0=dataset_select0.drop(['Outcome'],axis=1)
#()==> data1 label1 car on a besoin des gens maladies

model_one_class=svm.OneClassSVM(nu=0.2,kernel='rbf',gamma=0.9)
xtrain1,xtest1,ytrain1,ytest1=train_test_split(data1,label1,test_size=0.33,random_state=0)
model_one_class.fit(xtrain1)
testData=np.concatenate((xtest1,data0),axis=0)
testLabel=np.concatenate((ytest1,label0),axis=0)
Yprod=model_one_class.predict(testData)
Acc=accuracy_score(testLabel,Yprod)





