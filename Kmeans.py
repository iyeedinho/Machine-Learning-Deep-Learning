# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:46:38 2019

@author: Iyeed
"""
from sklearn.model_selection import train_test_split
"""question 1:"""
from sklearn import datasets 
dataset=datasets.load_iris()
"""question 2:"""
data=dataset.data
#3)
label=dataset.target

from sklearn.cluster import KMeans
help(KMeans)
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)

centre=kmeans.cluster_centers_
kameanslabels=kmeans.labels_

"""partie PCA """
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalPCA = pca.fit_transform(data)
plt.scatter(principalPCA[:,0],principalPCA[:,1])


kmpredict=kmeans.predict(data)


"""Partie II :"""
"""question1"""
"""datasets est une base de données artificielle"""
from sklearn.datasets.samples_generator import make_blobs
data2,y=make_blobs(n_samples=200,n_features=2,centers=3)
"""question 2"""
"""Vissualitation II :"""
import matplotlib.pyplot as plt
plt.scatter(data2[:,0],data2[:,1])
"""question3"""
kmeans2 = KMeans(n_clusters=3, random_state=0).fit(data2)
"""question4"""
centre2=kmeans2.cluster_centers_
kmeanslabels2=kmeans2.labels_
"""question5"""
kmpred=kmeans2.predict(data2)
"""question6"""
plt.scatter(data2[:,0],data2[:,1],c=kmpred)
plt.scatter(centre2[:,0],centre2[:,1],c='red')



