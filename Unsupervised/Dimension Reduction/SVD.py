# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:51:23 2020

@author: Jie.Hu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import TruncatedSVD


# import some data to play with
iris = datasets.load_iris()

iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']
dct = {0:'setosa', 1:'versicolor', 2:'virginica'}
iris_df.replace({"species": dct}, inplace = True)


X = iris_df.iloc[:,0:4].values
y = iris_df.iloc[:,0:5].values


svd = TruncatedSVD(n_components=2, random_state=1337)
svd.fit(X) 

print(svd.explained_variance_ratio_)  

print(svd.explained_variance_ratio_.sum())  

print(svd.singular_values_) 

result = svd.transform(X)
#df_trans = pd.DataFrame(result)
iris_df['D1'] = result[:, 0]
iris_df['D2'] = result[:, 1]
sns.scatterplot(x="D1", y="D2", hue="species", data=iris_df)

