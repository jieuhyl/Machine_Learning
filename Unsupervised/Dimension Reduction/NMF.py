# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:27:41 2020

@author: Jie.Hu
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import NMF

# import some data to play with
iris = datasets.load_iris()

iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']
dct = {0:'setosa', 1:'versicolor', 2:'virginica'}
iris_df.replace({"species": dct}, inplace = True)


X = iris_df.iloc[:,0:4].values
y = iris_df['species'].values


mod_nmf = NMF(n_components=2)
W = mod_nmf.fit_transform(X)
H = mod_nmf.components_

iris_df['D1'] = W[:, 0]
iris_df['D2'] = W[:, 1]

sns.scatterplot(x="D1", y="D2", hue="species", data=iris_df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)