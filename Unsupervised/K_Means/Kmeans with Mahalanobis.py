# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:10:21 2020

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


# correlations
corrmat = df.iloc[:,1:35].corr()
corrmat[corrmat < 0.5] = np.nan
sns.heatmap(corrmat, annot=False, cmap="RdYlGn")



''' PCA '''
from sklearn.decomposition import PCA
pca = PCA(whiten=True)
whitened = pca.fit_transform(X)
#a = np.cov(whitened.T)



''' kmeans '''
from sklearn.cluster import KMeans
# define the model
km = KMeans(n_clusters = 9, init = 'k-means++', random_state = 1337)
# fit model and predict clusters
df['Cluster_KM'] = km.fit_predict(whitened).astype(str)



''' TSNE '''
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, 
            #perplexity = 50, 
            #early_exaggeration = 20, 
            #learning_rate = 200, 
            random_state = 1337)

result = tsne.fit_transform(whitened) 

df['D1'] = result[:, 0]
df['D2'] = result[:, 1]

#plt.figure(figsize=(12,9), dpi = 300)
sns.scatterplot(x='D1', y='D2', hue='Cluster_KM', palette=sns.color_palette(n_colors = df['Cluster_KM'].nunique()), data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
#plt.savefig('tsne.png')

 


