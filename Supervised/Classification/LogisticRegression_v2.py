# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:41:01 2020

@author: Jie.Hu
"""


''' 1: Logistic Regression_v2'''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression


clf_lr = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, random_state = 1337)
clf_lr.fit(X, y)

# Predicting the train set results
y_pred = clf_lr.predict(X)
roc_auc_score(y, y_pred)

skf = StratifiedKFold(n_splits=5)
acc = cross_val_score(estimator = clf_lr, X = X, y = y, cv = skf, scoring='roc_auc')
acc.mean(), acc.std()


# KF n GS
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 
              'class_weight':[{0: w} for w in [1, 2, 4, 6, 10]]}

grid_search = GridSearchCV(estimator = clf_lr,
                           param_grid = parameters,
                           scoring='f1',
                           cv = 5,
                           n_jobs = -1)

grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_

# last step
clf_lr = LogisticRegression(penalty = 'elasticnet', 
                            solver = 'saga',
                            C = 10,
                            l1_ratio = 0.1, 
                            class_weight = {0:1},
                            random_state = 1337)
clf_lr.fit(X, y)
y_pred = clf_lr.predict(X)
roc_auc_score(y, y_pred)

print(classification_report(y, y_pred))
