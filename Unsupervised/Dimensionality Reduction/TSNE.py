# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 02:01:59 2020

@author: Jie.Hu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.manifold import TSNE


# import some data to play with
iris = datasets.load_iris()

iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']
dct = {0:'setosa', 1:'versicolor', 2:'virginica'}
iris_df.replace({"species": dct}, inplace = True)


X = iris_df.iloc[:,0:4].values
y = iris_df.iloc[:,0:5].values


tsne = TSNE(n_components=2, 
            perplexity = 10, 
            #early_exaggeration = 20, 
            #learning_rate = 200, 
            random_state = 1337)
result = tsne.fit_transform(X) 

#df_trans = pd.DataFrame(result)
iris_df['D1'] = result[:, 0]
iris_df['D2'] = result[:, 1]
sns.scatterplot(x="D1", y="D2", hue="species", data=iris_df)