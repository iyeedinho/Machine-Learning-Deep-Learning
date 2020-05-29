# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:31:44 2019

@author: Admin
"""
from PIL import Image 
import glob 
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
Img=Image.open("C:/yalefaces/subject01.gif")
Img2=Image.open("C:/yalefaces1/subject01.centerlight")
path=glob.glob("C:/yalefaces1/subject*")
Imaage=[]
for i in path:
    Img3=Image.open(i)
    IM=np.array(Img3)
    VECT=IM.reshape(IM.shape[0]*IM.shape[1])
    Imaage.append(VECT)
    


from sklearn.decomposition import PCA
pca = PCA(n_components=165)
principalPCA = pca.fit_transform(Imaage)

y=[]
for i in range(1, 16):
    path2=glob.glob('C:/yalefaces1/subject'+str(i).zfill(2)+"*")
    for fname in path2:
        y.append(i)

X_train, X_test, y_train, y_test = train_test_split(principalPCA, y, test_size=0.33,random_state=0)
model=svm.SVC(kernel = 'rbf' , gamma = 0.1 , C = 4 , decision_function_shape='ovo') 
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
acc=accuracy_score(y_test,y_predict)

model2=svm.SVC(kernel = 'linear' , C = 2 , decision_function_shape='ovo') 
model2.fit(X_train,y_train)
y_predict2=model2.predict(X_test)
acc2=accuracy_score(y_test,y_predict2)

model3=svm.SVC(kernel = 'sigmoid' , gamma= 0.5,coef0 = 10 , decision_function_shape='ovo') 
model3.fit(X_train,y_train)
y_predict3=model3.predict(X_test)
acc3=accuracy_score(y_test,y_predict3)

model4=svm.SVC(kernel = 'poly' , gamma= 0.5,coef0 = 10,degree=9 , decision_function_shape='ovo') 
model4.fit(X_train,y_train)
y_predict4=model4.predict(X_test)
acc4=accuracy_score(y_test,y_predict4)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
modelLDA=LDA()
newData=modelLDA.fit_transform(Imaage,y)
X_trainLDA, X_testLDA, y_trainLDA, y_testLDA = train_test_split(newData, y, test_size=0.33,random_state=0)
modelLDA.fit(X_trainLDA,y_trainLDA)
y_predictLDA=modelLDA.predict(X_testLDA)
accLDA=accuracy_score(y_testLDA,y_predictLDA)
print(accLDA)


















