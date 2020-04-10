# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:42:42 2020

@author: Jie.Hu
"""




''' Naive Bayes '''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB


clf_nb = BernoulliNB()
clf_nb.fit(X, y)

# Predicting the train set results
y_pred = clf_nb.predict(X)
roc_auc_score(y, y_pred)

skf = StratifiedKFold(n_splits=5)
acc = cross_val_score(estimator = clf_nb, X = X, y = y, cv = skf, scoring='roc_auc')
acc.mean(), acc.std()


# KF n GS
parameters = {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]}

grid_search = GridSearchCV(estimator = clf_nb,
                           param_grid = parameters,
                           scoring='f1',
                           cv = 5,
                           n_jobs = -1)

grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_

# last step
clf_nb = BernoulliNB(alpha = 0.3)

clf_nb.fit(X, y)
y_pred = clf_nb.predict(X)
roc_auc_score(y, y_pred)

print(classification_report(y, y_pred))
