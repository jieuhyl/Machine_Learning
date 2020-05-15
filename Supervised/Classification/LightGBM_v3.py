#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 02:49:46 2020

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
import datetime

# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import loguniform


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


# lgb =========================================================================
import lightgbm as lgb
clf_lgb = lgb.LGBMClassifier(objective = 'binary',
                             boosting_type = 'dart',
                             verbose = -1,
                             random_state=1337)


scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)

acc = cross_val_score(estimator = clf_lgb, X = X_train, y = y_train, cv = rskf, scoring='roc_auc')
acc.mean(), acc.std()


# GridSearchCV needs a predefined plan of the experiments
param_grid = {
              'learning_rate':[0.1],
              'max_depth':[5,7,9,-1],
              'min_data_in_leaf':[5,10,15],
              'num_leaves':[10,20,30], 
              'bagging_freq' : [7],
              'bagging_fraction':[0.5],
              'feature_fraction':[0.5],
              'lambda_l1':[0,0.1,1,10],
              'lambda_l2':[0,0.1,1,10],
              'min_split_gain':[0,0.1,1,10],
              'num_iterations':[400],
              'drop_rate':[0,0.1],
              'max_drop':[50],
              'skip_drop':[0.5],
              'uniform_drop': [True, False]
              }


grid_search = GridSearchCV(estimator = clf_lgb,
                           param_grid = param_grid,
                           scoring='roc_auc',
                           cv = 3,
                           verbose = -1,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
grid_search.best_params_, grid_search.best_score_

# last step
clf_lgb_grid = grid_search.best_estimator_

y_pred = clf_lgb_grid.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_lgb_grid.predict_proba(X_test)[:, 1]
print('LGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



# RandomizedSearchCV needs the distribution of the experiments to be tested
# If you can provide the right distribution, the sampling will lead to faster and better results.
param_distribution = {'learning_rate': loguniform(0.0001, 1.0),
                      'num_leaves': randint(2, 100),
                      'max_depth':randint(0, 20),
                      'min_child_samples':randint(2, 100),
                      'min_child_weight':randint(0, 10),
                      'max_bin': randint(50, 500),
                      'subsample': uniform(0.01, 1),
                      'subsample_freq': randint(0,10),
                      'colsample_bytree': uniform(0.01, 1),
                      'colsample_bynode': uniform(0.01,1),
                      'reg_alpha': loguniform(0.0001, 100),
                      'reg_lambda': loguniform(0.001, 100),
                      'min_split_gain': loguniform(0.001, 100),
                      'scale_pos_weight': loguniform(1e-6, 100),
                      'n_estimators': randint(50, 500),
                      'drop_rate':uniform(0.01, 1),
                      'max_drop':randint(10, 100),
                      'skip_drop':uniform(0.01, 1),
                      'uniform_drop': [True, False]}


random_search = RandomizedSearchCV(estimator = clf_lgb, 
                                   param_distributions=param_distribution,
                                   n_iter=100,
                                   cv=rskf,
                                   scoring='roc_auc',
                                   verbose = -1,
                                   n_jobs=-1,
                                   random_state=1337)

start_time = time.time()
random_search = random_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
random_search.best_params_, random_search.best_score_

# last step
clf_lgb_random = random_search.best_estimator_

y_pred = clf_lgb_random.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_lgb_random.predict_proba(X_test)[:, 1]
print('LGB_RANDOM AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))





# also BayesSearchCV needs to work on the distributions of the experiments but it is less sensible to them

bayes_space  = {'learning_rate': Real(0.0001, 1, 'log-uniform'),
                'num_leaves': Integer(2, 500),
                'max_depth': Integer(1, 50),
                'min_child_samples':Integer(2, 30),
                'min_child_weight': Integer(0, 20),
                'max_bin': Integer(20, 500),
                'subsample': Real(0.01, 1, 'uniform'),
                'subsample_freq': Integer(0, 10),
                'colsample_bytree': Real(0.01, 1, 'uniform'),
                'colsample_bynode': Real(0.01, 1, 'uniform'),
                'reg_lambda': Real(1e-6, 1000, 'log-uniform'),
                'reg_alpha': Real(1e-6, 1000, 'log-uniform'),
                'min_split_gain': Real(1e-6, 1000, 'log-uniform'),
                'scale_pos_weight': Real(1e-2, 100, 'log-uniform'),
                'n_estimators': Integer(50, 500),
                'drop_rate': Real(0.01, 1, 'uniform'),
                'max_drop': Integer(10, 50),
                'skip_drop': Real(0.01, 1, 'uniform'),
                'uniform_drop': Categorical(categories=[True, False])}

    
    
bayes_search = BayesSearchCV(estimator = clf_lgb,
                             search_spaces = bayes_space,
                             n_iter=100,
                             cv=rskf,
                             scoring='roc_auc', 
                             optimizer_kwargs={'base_estimator': 'GP'},
                             verbose= -1,
                             n_jobs= -1,                             
                             random_state=1337)

        
start_time = time.time()
bayes_search = bayes_search.fit(X_train, y_train, callback=[DeltaXStopper(0.0001), DeadlineStopper(60*60)])
print('Training time: {} minutes'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
bayes_search.best_params_, bayes_search.best_score_ 


# last step
clf_lgb_bayes = bayes_search.best_estimator_

y_pred = clf_lgb_bayes.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_lgb_bayes.predict_proba(X_test)[:, 1]
print('LGB_BAYES AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))





# not the best the params======================================================
#[grid_search, random_search, bayes_search]
cv_hist = pd.DataFrame(bayes_search.cv_results_)
cv_hist.sort_values('rank_test_score', inplace = True)
cv_hist.reset_index(inplace = True)

# n_best
params_best_n = cv_hist.loc[0, 'params']
#print(params_best_n)

clf_best_n = clf_lgb
clf_best_n.set_params(**params_best_n)

clf_best_n.fit(X_train, y_train)

y_pred = clf_best_n.predict_proba(X_test)[:, 1]
print('LGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))


res = []
for i in range(20):
    params_best_n = cv_hist.loc[i, 'params']

    clf_best_n = clf_lgb
    clf_best_n.set_params(**params_best_n)

    clf_best_n.fit(X_train, y_train)

    y_pred = clf_best_n.predict_proba(X_test)[:, 1]
    res.append(roc_auc_score(y_test, y_pred))
