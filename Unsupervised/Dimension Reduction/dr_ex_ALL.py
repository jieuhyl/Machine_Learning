# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:49:34 2020

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



''' PCA '''
from sklearn.decomposition import PCA
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca = PCA(n_components=2, random_state=1337)
pca.fit(X)  

print(pca.explained_variance_ratio_)  

print(pca.explained_variance_ratio_.sum())  

result = pca.transform(X)

df['D1'] = result[:, 0]
df['D2'] = result[:, 1]

sns.scatterplot(x='D1', y='D2', hue='Cluster9', data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)



''' SVD '''
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=1337)
svd.fit(X) 

print(svd.explained_variance_ratio_)  

print(svd.explained_variance_ratio_.sum())  

print(svd.singular_values_) 

result = svd.transform(X)

df['D1'] = result[:, 0]
df['D2'] = result[:, 1]

sns.scatterplot(x='D1', y='D2', hue='Cluster9', data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)



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

plt.figure(figsize=(12,9), dpi = 300)
sns.scatterplot(x='D1', y='D2', hue='Cluster9', palette=sns.color_palette(n_colors = df['Cluster9'].nunique()), data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('tsne.png')



''' MDS '''
from sklearn.manifold import MDS
mds = MDS(n_components=2)
result = mds.fit_transform(X)

print(mds.embedding_)  

print(mds.stress_)  

df['D1'] = result[:, 0]
df['D2'] = result[:, 1]

plt.figure(figsize=(12,9), dpi = 300)
sns.scatterplot(x='D1', y='D2', hue='Cluster9', palette=sns.color_palette(n_colors = df['Cluster9'].nunique()), data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('mds.png')



''' SE '''
from sklearn.manifold import SpectralEmbedding
se = SpectralEmbedding(affinity = 'rbf', n_components=2)
result = se.fit_transform(X)

df['D1'] = result[:, 0]
df['D2'] = result[:, 1]

sns.scatterplot(x='D1', y='D2', hue='Cluster9', data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)



''' LTSA '''
from sklearn.manifold import LocallyLinearEmbedding
ltsa = LocallyLinearEmbedding(n_neighbors=300, n_components=2, method = 'ltsa')
result = ltsa.fit_transform(X)
  
df['D1'] = result[:, 0]
df['D2'] = result[:, 1]

#plt.figure(figsize=(12,9), dpi = 300)
sns.scatterplot(x='D1', y='D2', hue='Cluster9', data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)


