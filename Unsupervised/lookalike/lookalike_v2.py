# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 23:29:54 2020

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

'''
# demension reduction and classify
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state=42)
nca.fit(X, y)
trans_nca = nca.transform(X)
'''


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
sub_id = df.loc[df.Target == 1, 'ID'].tolist()
sub = indices[sub_id]


# all condidates
all_cand = {}
for i in sub.tolist():
    for j in range(1, n_size):
        all_cand[i[j]] = all_cand.get(i[j], 0) + 1
len(all_cand)

# check sub distribution
dist = []
for i,j in all_cand.items():
    if i in sub_id:
        dist.append(j)
plt.hist(dist)
np.mean(dist), np.median(dist), np.min(dist), np.max(dist)


thres = 13
first_cand = []
for i,j in all_cand.items():
    if j >= thres:
        first_cand.append(i)
len(first_cand)


# find intersection
first_cand_sub = list(set(first_cand) & set(sub_id))
print('the capture size is {} \nthe capture capacity is {:.2%}'.format(len(first_cand_sub), round(len(first_cand_sub)/len(sub_id), 4)))

# recommend
first_recommend = list(set(first_cand) - set(sub_id))
print('the recommend size is %1d \nthe recommend capacity is %.2f%%' % (len(first_recommend), round(100*len(first_recommend)/(len(df)-len(sub_id)), 4)))



thres = 13
cand_sub = []
first_recommend = []
for i,j in all_cand.items():
    if i in sub_id and j >= thres:
        cand_sub.append(i)
    if i not in sub_id and j >= thres:
        first_recommend.append(i)
len(cand_sub)
len(first_recommend)



# add connection
second_connection = indices[first_recommend]
thres2 = 40
second_cand = set()
for i in second_connection.tolist():
    for j in range(1, thres2):
        if i[j] not in first_recommend and i[j] not in sub_id:
            second_cand.add(i[j])
second_cand = list(second_cand)
len(second_cand)


second_cand_sub = set()
for i in second_connection.tolist():
    for j in range(1, thres2):
        if i[j] in sub_id and i[j] not in first_cand_sub:
            second_cand_sub.add(i[j])
second_cand_sub = list(second_cand_sub)
len(second_cand_sub)

# find intersection
second_cand_sub = first_cand_sub + second_cand_sub
print('the capture size is {} \nthe capture capacity is {:.2%}'.format(len(second_cand_sub), round(len(second_cand_sub)/len(sub_id), 4)))

# recommend
second_recommend = first_recommend + second_cand
print('the recommend size is %1d \nthe recommend capacity is %.2f%%' % (len(second_recommend), round(100*len(second_recommend)/(len(df)-len(sub_id)), 4)))




second_connection = indices[first_recommend]
thres2 = 40
cand_sub2 = set()
second_recommend = set()
for i in second_connection.tolist():
    for j in range(1, thres2):
        if i[j] in sub_id and i[j] not in first_cand_sub:
            cand_sub2.add(i[j])
        if i[j] not in first_recommend and i[j] not in sub_id:
            second_recommend.add(i[j])
cand_sub2 = list(cand_sub2)
second_recommend = list(second_recommend)

cand_sub2 = cand_sub2 + cand_sub
len(cand_sub2)

cand_sub2 = list(set(cand_sub2) & set(cand_sub))
len(cand_sub2)


second_recommend = first_recommend + second_cand
len(second_recommend)


