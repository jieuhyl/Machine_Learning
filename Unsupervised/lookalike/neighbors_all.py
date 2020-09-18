# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:12:19 2020

@author: Jie.Hu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold
import time

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=2, random_state=1337)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=1337)



from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


# demension reduction and classify
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state=42)
nca.fit(X_train, y_train)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))


knn.fit(nca.transform(X_train), y_train)
print(knn.score(nca.transform(X_test), y_test))


result = nca.transform(X_test)
  

a = result[:, 0]
b = result[:, 1]

sns.scatterplot(x=a, y=b, hue=y_test)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)



# similarity 
# First let's create a dataset called X, with 6 records and 2 features each.
X = np.array([[-1, 2], [4, -4], [-2, 1], [-1, 3], [-3, 2], [-1, 4], [3,4], [5,6],[9,10],[5,7],[8,6],[5,6],[9,1],[1,13]])
x,y = X.T
plt.scatter(x,y)
plt.show()

# Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = nbrs.kneighbors(X)

# Let's print out the indices of neighbors for each record in object X.
indices
distances

print(nbrs.kneighbors([[8,5]], return_distance=False))