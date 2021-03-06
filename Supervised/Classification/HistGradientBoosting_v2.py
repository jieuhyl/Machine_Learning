#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:35:46 2020

@author: qianqianwang
"""


# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold
import time

# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform


# Skopt functions
from skopt import BayesSearchCV
from skopt import gp_minimize # Bayesian optimization using Gaussian Processes
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args # decorator to convert a list of parameters to named arguments
from skopt.callbacks import DeltaXStopper # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta
from skopt.callbacks import DeadlineStopper # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback # Callback to control the verbosity


# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, weights=[0.5], random_state=1337)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1337)


''' 8: HistGradient Boosting'''
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
clf_hgb = HistGradientBoostingClassifier(validation_fraction=0.2,
                                         n_iter_no_change=10, 
                                         scoring = 'roc_auc',
                                         random_state= 1337)


scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_hgb, X = X_train, y = y_train, cv = rskf, scoring='roc_auc')
acc.mean(), acc.std()


# GridSearchCV needs a predefined plan of the experiments
param_grid = {'learning_rate':[0.3], 
              'max_depth':[7,9],
              'max_leaf_nodes':[13],
              'max_bins': [100, 255, 500],
              'max_iter':[100],
              'min_samples_leaf':[9,11],
              'l2_regularization':[0.005,0.001]}


                                       
grid_search = GridSearchCV(estimator = clf_hgb,
                           param_grid = param_grid,
                           scoring='roc_auc',
                           cv = 3,
                           verbose = True,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_

# last step
clf_hgb_grid = grid_search.best_estimator_

y_pred = clf_hgb_grid.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_hgb_grid.predict_proba(X_test)[:, 1]
print('HGB_GRID AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



# RandomizedSearchCV needs the distribution of the experiments to be tested
# If you can provide the right distribution, the sampling will lead to faster and better results.
param_distribution = {'learning_rate': uniform(0.01, 1.0),
                      'max_depth':randint(2, 10),
                      'max_leaf_nodes': randint(2, 21),
                      'max_bins': randint(10, 255),
                      'max_iter': randint(50, 500),
                      'min_samples_leaf': randint(1, 21),
                      'l2_regularization': uniform(0.0001, 1.0)}

random_search = RandomizedSearchCV(estimator = clf_hgb, 
                                   param_distributions=param_distribution,
                                   n_iter=100,
                                   cv=3,
                                   scoring='roc_auc',
                                   verbose = True,
                                   n_jobs=-1,
                                   random_state=1337)

start_time = time.time()
random_search = random_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
random_search.best_params_, random_search.best_score_

# last step
clf_hgb_random = random_search.best_estimator_

y_pred = clf_hgb_random.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_hgb_random.predict_proba(X_test)[:, 1]
print('HGB_RANDOM AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))




# also BayesSearchCV needs to work on the distributions of the experiments but it is less sensible to them
bayes_space = {'learning_rate': Real(0.01, 1.0),
                'max_depth':Integer(2, 10),
                'max_leaf_nodes':Integer(2, 21),
                'max_bins': Integer(10, 255),
                'max_iter': Integer(50, 500),
                'min_samples_leaf': Integer(1, 10),
                'l2_regularization': Real(0.0001, 1.0)}

for baseEstimator in ['GP', 'RF', 'ET', 'GBRT']:
    bayes_search = BayesSearchCV(clf_hgb,
                                 bayes_space,
                                 n_iter=100,
                                 cv=3,
                                 scoring='roc_auc', 
                                 optimizer_kwargs={'base_estimator': baseEstimator},
                                 verbose= False,
                                 n_jobs=-1,                             
                                 random_state=1337)
        
    start_time = time.time()
    bayes_search = bayes_search.fit(X_train, y_train)
    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    print(bayes_search.best_params_, bayes_search.best_score_)
    
    
    
bayes_search = BayesSearchCV(estimator = clf_hgb,
                             search_spaces = bayes_space,
                             n_iter=100,
                             cv=3,
                             scoring='roc_auc', 
                             optimizer_kwargs={'base_estimator': 'GP'},
                             verbose= False,
                             n_jobs=-1,                             
                             random_state=1337)
        
start_time = time.time()
bayes_search = bayes_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
bayes_search.best_params_, bayes_search.best_score_ 

# last step
clf_hgb_bayes = bayes_search.best_estimator_

y_pred = clf_hgb_bayes.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_hgb_bayes.predict_proba(X_test)[:, 1]
print('HGB_BAYES AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))
 
