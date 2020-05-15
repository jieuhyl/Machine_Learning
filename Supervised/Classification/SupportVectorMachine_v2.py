# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:44:06 2020

@author: Jie.Hu
"""


''' 4: Support Vector Machine'''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
clf_svc = SVC(random_state = 1337)
clf_svc.fit(X, y)

# Predicting the train set results
y_pred = clf_svc.predict(X)
roc_auc_score(y, y_pred)

skf = StratifiedKFold(n_splits=5)
acc = cross_val_score(estimator = clf_svc, X = X, y = y, cv = skf, scoring='roc_auc')
acc.mean(), acc.std()


# KF n GS
parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'gamma': ['auto', 'scale'], 'degree': [1,2,3]},
              {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': ['auto', 'scale']}]

grid_search = GridSearchCV(estimator = clf_svc,
                           param_grid = parameters,
                           scoring='f1',
                           cv = 5,
                           n_jobs = -1)

grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_

# last step
clf_svc = SVC(kernel = 'poly',
              C = 1000,
              gamma = 'scale', 
              random_state = 1337)
clf_svc.fit(X, y)
y_pred = clf_svc.predict(X)
roc_auc_score(y, y_pred)

print(classification_report(y, y_pred))
