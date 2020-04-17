# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:05:21 2020

@author: Jie.Hu
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


print("Current Working Directory " , os.getcwd())
os.chdir(r'C:\Users\Jie.Hu\Desktop\Data Science\Practice\ml\Dimension Reduction')


df = pd.read_csv('seg_ex.csv')
#df = pd.read_csv('seg_ex2.csv')
df['Cluster9'] = df['Cluster9'].astype(str)
#di = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I'}
#df.replace({'Cluster9': di}, inplace=True)


X = df.iloc[:,1:35].values
y = df.loc[:,'Cluster9'].values


''' transformation '''
from sklearn.preprocessing import MinMaxScaler, StandardScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)



''' affinity propagation '''
from sklearn.cluster import AffinityPropagation
# define the model
ap = AffinityPropagation(damping=0.9)
# fit model and predict clusters
df['Cluster_AP'] = ap.fit_predict(X).astype(str)



''' Kmeans '''
from sklearn.cluster import KMeans
# define the model
km = KMeans(n_clusters = 9, init = 'k-means++', random_state = 1337)
# fit model and predict clusters
df['Cluster_KM'] = km.fit_predict(X).astype(str)



''' MiniBatch Kmeans '''
from sklearn.cluster import KMeans, MiniBatchKMeans
# define the model
mbkm = MiniBatchKMeans(n_clusters = 9, init = 'k-means++', random_state = 1337)
# fit model and predict clusters
df['Cluster_MBKM'] = mbkm.fit_predict(X).astype(str)



''' mean shift '''
from sklearn.cluster import MeanShift
# define the model
ms = MeanShift()
# fit model and predict clusters
df['Cluster_MS'] = ms.fit_predict(X)



''' agglomerative '''
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
# define the model
agg = AgglomerativeClustering(n_clusters=9, affinity = 'cosine', linkage='complete')
# fit model and predict clusters
df['Cluster_AGG'] = agg.fit_predict(X).astype(str)

# dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, metric = 'euclidean', method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()



''' birch '''
from sklearn.cluster import Birch
bh = Birch(threshold=0.01, branching_factor = 100, n_clusters=9)
# fit model and predict clusters
df['Cluster_BH'] = bh.fit_predict(X).astype(str)



''' dbscan clustering '''
from sklearn.cluster import DBSCAN
# define the model
db = DBSCAN(eps=0.3, min_samples=100)
# fit model and predict clusters
df['Cluster_DB'] = db.fit_predict(X)



''' optics '''
from sklearn.cluster import OPTICS
# define the model
op = OPTICS(eps=0.8, min_samples=10)
# fit model and predict clusters
df['Cluster_OP'] = op.fit_predict(X)



''' spectral '''
from sklearn.cluster import SpectralClustering
# define the model
sp = SpectralClustering(n_clusters=9)
# fit model and predict clusters
df['Cluster_SP'] = sp.fit_predict(X)


df['Cluster_SP'].value_counts(normalize=True)


''' TSNE '''
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, 
            perplexity = 100, 
            #early_exaggeration = 20, 
            #learning_rate = 200, 
            random_state = 1337)
result = tsne.fit_transform(X) 

df['D1'] = result[:, 0]
df['D2'] = result[:, 1]

#plt.figure(figsize=(12,9), dpi = 300)
sns.scatterplot(x='D1', y='D2', hue='Cluster_SP', palette=sns.color_palette(n_colors = df['Cluster_SP'].nunique()), data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
#plt.savefig('tsne.png')


