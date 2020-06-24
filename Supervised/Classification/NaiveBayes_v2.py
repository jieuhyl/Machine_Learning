# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:44:15 2020

@author: Jie.Hu
"""



import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold

# define dataset
rng = np.random.RandomState(1)
X = rng.randint(5, size=(100, 10))
y = np.random.choice([0, 1], size=(100,), p=[2./3, 1./3])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)



''' Naive Bayes '''
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_nb.predict(X_test)


# empirical log probability
clf_nb.feature_log_prob_[1]
np.exp(clf_nb.coef_)


# KF n GS
parameters = {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]}

grid_search = GridSearchCV(estimator = clf_nb,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = 3,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_

# last step
clf_nb = BernoulliNB(alpha = 0.3)

clf_nb.fit(X, y)
y_pred = clf_nb.predict(X)
roc_auc_score(y, y_pred)

print(classification_report(y, y_pred))








