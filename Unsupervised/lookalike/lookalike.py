# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 23:38:07 2020

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
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, weights=[0.9], n_redundant=5, n_classes=2, random_state=1337)
df = pd.DataFrame(data=X, columns=['V'+str(i) for i in range(1,20+1)])
df.insert(0, 'ID', np.array(range(2000)))
df['Target'] = y


# check data
df.info()
df.isnull().values.sum()


# neighbors
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import NearestNeighbors


# demension reduction and classify
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state=42)
nca.fit(X, y)
trans_nca = nca.transform(X)


# mtx
X = df.iloc[:,1:21].values
y = df['Target'].values

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X = mms.fit_transform(X)



# fit neighbors
# metrics minkowski p 2, 'cosine'
n_size = 50
nbrs = NearestNeighbors(n_neighbors=n_size, metric='minkowski', p = 2).fit(X)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = nbrs.kneighbors(X)

# Let's print out the indices of neighbors for each record in object X.
indices
distances


# get all the sub
sub = df.loc[df.Target == 1, 'ID'].tolist()
sub_indices = indices[sub]

# all condidates
all_cand = []
for i in sub_indices.tolist():
    for j in range(1, n_size):
        all_cand.append(i[j])
len(all_cand)
len(set(all_cand))

# find intersection
sub_inter = list(set(sub) & set(all_cand))
print('the capture capacity is {}'.format(round(len(sub_inter)/len(sub), 4)))

# recommend
sub_recommend = set(all_cand) - set(sub)
print('the recommend capacity is {}'.format(round(len(sub_recommend)/(len(df)-len(sub)), 4)))




# add weights
all_cand = {}
for i in sub_indices.tolist():
    for j in range(1, n_size):
        all_cand[i[j]] = all_cand.get(i[j], 0) + 1
len(all_cand)

# check sub distribution
dist = []
for i,j in all_cand.items():
    if i in sub:
        dist.append(j)
plt.hist(dist)
np.mean(dist)


thres = 10
sub_cand = []
for i,j in all_cand.items():
    if j >= thres:
        sub_cand.append(i)
len(sub_cand)

# find intersection
sub_inter = list(set(sub_cand) & set(sub))
print('the capture size is {} \nthe capture capacity is {:.2%}'.format(len(sub_inter), round(len(sub_inter)/len(sub), 4)))

# recommend
sub_recommend = set(sub_cand) - set(sub)
print('the recommend size is %1d \nthe recommend capacity is %.2f%%' % (len(sub_recommend), round(100*len(sub_recommend)/(len(df)-len(sub)), 4)))




