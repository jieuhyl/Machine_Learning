# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:20:38 2020

@author: Jie.Hu
"""


import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression


# generate dataset
X, y = make_classification(n_samples=10000, n_features=5, n_redundant=2,
	n_clusters_per_class=1, weights=[0.5], flip_y=0.1, random_state=1234)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, shuffle=True, random_state=1337)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, stratify=y_train, shuffle=True, random_state=1337)


# define model
model = LogisticRegression(solver='lbfgs')
#
model.fit(X_train, y_train)

# Predicting the train set results
y_pred = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)

plt.hist(y_pred, bins = 100)

#==============================================================================
''' Psuedo Labeling '''


pos_threshold = 0.85
neg_threshold = 0.15
idx_pseudo = np.argwhere(np.logical_or(y_pred > pos_threshold, y_pred < neg_threshold ))[:,0]

X_test_pseudo = X_test[idx_pseudo]
y_test_pseudo = y_pred[idx_pseudo]
y_test_pseudo = y_test_pseudo.round().astype(int)

#y_test_pseudo[y_test_pseudo > 0.5] = 1
#y_test_pseudo[y_test_pseudo <= 0.5] = 0


auc_pred = []
test_pred = []
pos_threshold = 0.85
neg_threshold = 0.15
n_fold = 0

X_train_valid = np.concatenate([X_train, X_valid], axis = 0)
y_train_valid = np.concatenate([y_train, y_valid], axis = 0)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
for idx_train, idx_valid in rskf.split(X_train_valid, y_train_valid):
    train_x, train_y = X_train_valid[idx_train], y_train_valid[idx_train]
    valid_x, valid_y = X_train_valid[idx_valid], y_train_valid[idx_valid]
    
    train_x = np.concatenate([train_x, X_test_pseudo], axis = 0)
    train_y = np.concatenate([train_y, y_test_pseudo], axis = 0)
    print('train size: {}, pseudo size: {}'.format(train_x.shape, len(idx_pseudo)))
    
    model = LogisticRegression(solver='lbfgs')
    model.fit(train_x, train_y)
    oof_preds = model.predict_proba(valid_x)[:, 1]
    
    print('Fold %1d AUC: %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_pred.append(roc_auc_score(valid_y, oof_preds))
    
    # pseudo labeling
    test_prob = model.predict_proba(X_test)[:, 1]
    idx_pseudo = np.argwhere(np.logical_or(test_prob > pos_threshold, test_prob < neg_threshold ))[:,0]
    X_test_pseudo = X_test[idx_pseudo]
    y_test_pseudo = y_pred[idx_pseudo]
    y_test_pseudo = y_test_pseudo.round().astype(int)

    test_pred.append([model.predict_proba(X_test)[:, 1]])
    n_fold = n_fold + 1
print('Full AUC score %.6f' % np.mean(auc_pred)) 


res = sorted(zip(auc_pred, test_pred), reverse=True)[:2]
lst = [j for (i,j) in res]
lst_res = np.mean(lst, axis = 0)[0]
roc_auc_score(y_test, lst_res)









