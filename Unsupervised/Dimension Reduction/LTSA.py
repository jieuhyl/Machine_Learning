# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:48:21 2020

@author: Jie.Hu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.manifold import LocallyLinearEmbedding

# import some data to play with
iris = datasets.load_iris()

iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']
dct = {0:'setosa', 1:'versicolor', 2:'virginica'}
iris_df.replace({"species": dct}, inplace = True)


X = iris_df.iloc[:,0:4].values
y = iris_df['species'].values


ltsa = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method = 'ltsa')
result = ltsa.fit_transform(X)
  

iris_df['D1'] = result[:, 0]
iris_df['D2'] = result[:, 1]

sns.scatterplot(x="D1", y="D2", hue="species", data=iris_df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)