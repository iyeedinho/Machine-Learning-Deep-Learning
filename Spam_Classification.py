# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:47:29 2019

@author: Iyeed
"""
"""Partie I : Data Cleaning """
"""1er methode """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
import pandas as pd
df = pd.read_csv('C:/spam.csv',delimiter=',',encoding='latin-1') 

df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)

y=df.v1
x=df.v2

from sklearn.preprocessing import LabelEncoder 
Model = LabelEncoder() 
Y = Model.fit_transform(y)  
Y = Y.reshape(-1,1) 
"""==> elle transforme les données textuelles en données numérique """

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing import sequence 
max_words = 1000 
max_len = 150 
tok = Tokenizer(num_words=max_words) 
tok.fit_on_texts(x) 
sequences = tok.texts_to_sequences(x) 
"""==> elle fait la coupure de text en des mots et elle les donne des numeros index de 0 à 1
et chaqye feature contient le nombre des observations """

import tensorflow as tf
seq=sequence.pad_sequences(sequences,padding='post')
"""==> on a obtenu une dataset qui est n'est pas equilibré c pour cela on a utilisé
cette commande pour faire l'équilibre de notre dataset on completent les valeurs
manquantes par 0 à droit au à gauche ça ne différe pas"""

"""2éme methode """
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer() 
X1 = vectorizer.fit_transform(x) 
print(X1)
"""==> on a transofrmer notre données sour forme d'une matrice """

"""Partie 2 : Classification """ 
"""Méthode 1: SVM """
"""On va utiliser la model One class car on a une dataset unbalanced """

y1=pd.DataFrame(data=Y[0:,0],columns=list('Y'))
"""==>Transofrmer Y en dataframe y1 """

seq1=pd.DataFrame(data=seq[0:,0:])
"""==>Transofrmer seq en dataframe seq1 """
Dataconc=np.concatenate((seq1,y1), axis=1)
datafm=pd.DataFrame(data=Dataconc[0:,0:])
"""==>concatination entre seq1 et y1 et le transofrmer dans une dataFrame """

Dataset_selected1=datafm.loc[datafm[172].isin([1])]
Dataset_selected0=datafm.loc[datafm[172].isin([0])]

label1=Dataset_selected1[172]
label0=Dataset_selected0[172]
data1=Dataset_selected1.drop(172,axis=1)
data0=Dataset_selected0.drop(172,axis=1)

model_one_class=svm.OneClassSVM(nu=0.2,kernel='rbf',gamma=0.9)
xtrain1,xtest1,ytrain1,ytest1=train_test_split(data1,label1,test_size=0.33,random_state=0)
start_time1=time.time()
model_one_class.fit(xtrain1)
calcul_time1=time.time()-start_time1
testData=np.concatenate((xtest1,data0),axis=0)
testLabel=np.concatenate((ytest1,label0),axis=0)
Yprod=model_one_class.predict(testData)


Acc=accuracy_score(testLabel,Yprod)


fpr,tpr,t=roc_curve(testLabel,Yprod,pos_label=1)
aucc=auc(fpr,tpr)*100

precision=(tpr[1]/(tpr[1]+fpr[1]))*100
cm=confusion_matrix(testLabel,Yprod)



""" Méthode 2  Gaussian"""
""" GaussianNB"""
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 
xxtrain, xxtest, yytrain, yytest = train_test_split(seq,Y,test_size=0.333,random_state=0)

model_gauss = GaussianNB(priors=None, var_smoothing=1e-08)
start_timeg=time.time()
model_gauss.fit(xxtrain,yytrain)
calcul_timeg=time.time()-start_timeg
ypred_gauss = model_gauss.predict(xxtest)

acc_gauss=accuracy_score(yytest,ypred_gauss)

fpr_gauss,tpr_gauss,T_gauss=roc_curve(yytest,ypred_gauss,pos_label=1)
auc_gauss=auc(fpr_gauss,tpr_gauss)*100

precision_gauss=(tpr_gauss[1]/(tpr_gauss[1]+fpr_gauss[1]))*100

cm_gauss=confusion_matrix(yytest,ypred_gauss)

""" MultinomialNB"""

model_mg = MultinomialNB(alpha=0.9, class_prior=None, fit_prior=True)

start_timemg=time.time()
model_mg.fit(xxtrain,yytrain)
calcul_time_mg=time.time()-start_timemg
y_pred_mg = model_mg.predict(xxtrain)

acc_mg=accuracy_score(yytest,y_pred_mg)

fpr_mg,tpr_mg,T_mgl=roc_curve(yytest,y_pred_mg,pos_label=1)
auc_mg=auc(fpr_mg,tpr_mg)*100

precision_mg=(tpr_mg[1]/(tpr_mg[1]+fpr_mg[1]))*100

cm_mg=confusion_matrix(yytest,y_pred_mg)