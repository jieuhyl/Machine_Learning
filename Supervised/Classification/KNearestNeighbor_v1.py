# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:43:17 2020

@author: Jie.Hu
"""



''' KNN '''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


clf_knn = KNeighborsClassifier()
clf_knn.fit(X, y)

# Predicting the train set results
y_pred = clf_knn.predict(X)
roc_auc_score(y, y_pred)

skf = StratifiedKFold(n_splits=5)
acc = cross_val_score(estimator = clf_knn, X = X, y = y, cv = skf, scoring='roc_auc')
acc.mean(), acc.std()


# KF n GS
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'weights': ['uniform', 'distance'], 
              'p': [1, 2, 3, 4]}


grid_search = GridSearchCV(estimator = clf_knn,
                           param_grid = parameters,
                           scoring='f1',
                           cv = 5,
                           n_jobs = -1)

grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_

# last step
clf_knn = KNeighborsClassifier(n_neighbors = 1,
                               weights = 'uniform')

clf_knn.fit(X, y)
y_pred = clf_knn.predict(X)
roc_auc_score(y, y_pred)

print(classification_report(y, y_pred))
