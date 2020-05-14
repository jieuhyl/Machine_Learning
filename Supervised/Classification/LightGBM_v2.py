#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:30:42 2020

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
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1337)


# lgb =========================================================================
import lightgbm as lgb
clf_lgb = lgb.LGBMClassifier(objective = 'binary',
                             boosting_type = 'gbdt',
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
              'num_iterations':[400]}

fit_params = {'early_stopping_rounds': 100,
              'eval_metric': ['auc'],
              'eval_set': [(X_valid, y_valid)]}

grid_search = GridSearchCV(estimator = clf_lgb,
                           param_grid = param_grid,
                           scoring='roc_auc',
                           cv = 3,
                           verbose = 100,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
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
                      'scale_pos_weight': loguniform(1e-6, 100),
                      'n_estimators': randint(50, 500)}

fit_params = {'early_stopping_rounds': 20,
              'eval_metric': ['auc'],
              'eval_set': [(X_valid, y_valid)]}

random_search = RandomizedSearchCV(estimator = clf_lgb, 
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
clf_lgb_random = random_search.best_estimator_

y_pred = clf_lgb_random.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_lgb_random.predict_proba(X_test)[:, 1]
print('LGB_RANDOM AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



# last step
clf_lgb_random = random_search.best_estimator_
clf_lgb_random.fit(X_train, y_train, **fit_params)
y_pred = clf_lgb_random.predict(X_test, num_iteration = clf_lgb_random.best_iteration_)
print(classification_report(y_test, y_pred))

y_pred = clf_lgb_random.predict_proba(X_test, num_iteration = clf_lgb_random.best_iteration_)[:, 1]
print('XGB_BAYES AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))




# also BayesSearchCV needs to work on the distributions of the experiments but it is less sensible to them


bayes_space  = {'learning_rate': Real(0.0001, 1, 'log-uniform'),
                'num_leaves': Integer(2, 100),
                'max_depth': Integer(1, 100),
                'min_child_samples':Integer(2, 100),
                'min_child_weight': Integer(0, 10),
                'max_bin': Integer(20, 500),
                'subsample': Real(0.01, 1, 'uniform'),
                'subsample_freq': Integer(0, 10),
                'colsample_bytree': Real(0.01, 1, 'uniform'),
                'colsample_bynode': Real(0.01, 1, 'uniform'),
                'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
                'reg_alpha': Real(1e-9, 1000, 'log-uniform'),
                'scale_pos_weight': Real(1e-6, 100, 'log-uniform'),
                'n_estimators': Integer(50, 500)}

    
    
bayes_search = BayesSearchCV(estimator = clf_lgb,
                             search_spaces = bayes_space,
                             n_iter=100,
                             cv=rskf,
                             scoring='roc_auc', 
                             optimizer_kwargs={'base_estimator': 'GP'},
                             verbose= -1,
                             n_jobs=-1,                             
                             random_state=1337)

'''
for baseEstimator in ['GP', 'RF', 'ET', 'GBRT']:
    bayes_search = BayesSearchCV(clf_lgb,
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
'''
        
start_time = time.time()
bayes_search = bayes_search.fit(X_train, y_train, callbacks=[DeltaXStopper(0.0001), DeadlineStopper(60*10)])
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
bayes_search.best_params_, bayes_search.best_score_ 

# last step
clf_lgb_bayes = bayes_search.best_estimator_

y_pred = clf_lgb_bayes.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_lgb_bayes.predict_proba(X_test)[:, 1]
print('LGB_BAYES AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



# last step
def learning_rate_decay_power(current_iter):
    base_learning_rate = 0.31
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


params = {'objective': 'binary',
          'boosting_type': 'gbdt',
          'metric': 'auc',
         'colsample_bytree': 1.0,
         'learning_rate': 0.31402983358009906,
         'max_depth': 66,
         'min_child_samples': 2,
         'min_child_weight': 0,
         'min_split_gain': 0.0,
         'n_estimators': 131,
         'num_leaves': 41,
         'random_state': 1337,
         'reg_alpha': 2.5650838626319216e-06,
         'reg_lambda': 0.00024724195223517915,
         'subsample': 1.0,
         'subsample_for_bin': 200000,
         'subsample_freq': 3,       
         'colsample_bynode': 0.3423093458222358,
         'max_bin': 500,
         'scale_pos_weight': 84.34040301808572}

dtrain = lgb.Dataset(X_train, y_train)

dvalid = lgb.Dataset(X_valid, y_valid)

evals_results = {}

model = lgb.train(params, 
                  dtrain, 
                  valid_sets=[dvalid], 
                  valid_names=['valid'], 
                  evals_result=evals_results, 
                  callbacks = [lgb.reset_parameter(learning_rate=learning_rate_decay_power)],
                  num_boost_round=500,
                  early_stopping_rounds=50,
                  verbose_eval=10)

y_pred = model.predict(X_test, num_iteration = model.best_iteration)

print('LGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))

y_pred = (y_pred >= 0.5).astype(bool)  
print(classification_report(y_test, y_pred))





#==============================================================================
# 1 tts
auc_pred = []
y_pred = np.zeros(X_test.shape[0])
TTS = 0
for i in range(0, 10):
    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    
    params = {'objective': 'binary',
          'boosting_type': 'gbdt',
          'metric': 'auc',
         'colsample_bytree': 1.0,
         'learning_rate': 0.31402983358009906,
         'max_depth': 66,
         'min_child_samples': 2,
         'min_child_weight': 0,
         'min_split_gain': 0.0,
         'n_estimators': 131,
         'num_leaves': 41,
         'random_state': 1337,
         'reg_alpha': 2.5650838626319216e-06,
         'reg_lambda': 0.00024724195223517915,
         'subsample': 1.0,
         'subsample_for_bin': 200000,
         'subsample_freq': 3,       
         'colsample_bynode': 0.3423093458222358,
         'max_bin': 500,
         'scale_pos_weight': 84.34040301808572,
         'verbose': -1}

    dtrain = lgb.Dataset(train_x, train_y)
    dvalid = lgb.Dataset(valid_x, valid_y)
    dtest = lgb.Dataset(X_test, y_test)
    
    evals_results = {}
    
    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dvalid], 
                      valid_names=['valid'], 
                      evals_result=evals_results, 
                      num_boost_round=500,
                      early_stopping_rounds=50,
                      verbose_eval=100)


    y_pred += model.predict(X_test, num_iteration = model.best_iteration)/10
    
    oof_preds = model.predict(valid_x, num_iteration = model.best_iteration)
    print('TTS %2d AUC : %.6f' % (TTS + 1, roc_auc_score(valid_y, oof_preds)))
    auc_pred.append(roc_auc_score(valid_y, oof_preds))
    TTS += 1
    
print('Full AUC score %.3f' % np.mean(auc_pred))   
print('TTS_LGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))
y_pred = (y_pred >= 0.5).astype(bool)  
print('TTS_LGB ACC: %.3f' % accuracy_score(y_test, y_pred))
print('TTS_LGB F1: %.3f' % f1_score(y_test, y_pred))
print('TTS_LGB GMEAN: %.3f' % geometric_mean_score(y_test, y_pred))    



# 2 skf
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1337)
auc_pred = []
y_pred = np.zeros(X_test.shape[0])
n_fold = 0
for idx_train, idx_valid in skf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    params = {'objective': 'binary',
          'boosting_type': 'gbdt',
          'metric': 'auc',
         'colsample_bytree': 1.0,
         'learning_rate': 0.31402983358009906,
         'max_depth': 66,
         'min_child_samples': 2,
         'min_child_weight': 0,
         'min_split_gain': 0.0,
         'n_estimators': 131,
         'num_leaves': 41,
         'random_state': 1337,
         'reg_alpha': 2.5650838626319216e-06,
         'reg_lambda': 0.00024724195223517915,
         'subsample': 1.0,
         'subsample_for_bin': 200000,
         'subsample_freq': 3,       
         'colsample_bynode': 0.3423093458222358,
         'max_bin': 500,
         'scale_pos_weight': 84.34040301808572,
         'verbose':-1}

    dtrain = lgb.Dataset(train_x, train_y)
    dvalid = lgb.Dataset(valid_x, valid_y)
    dtest = lgb.Dataset(X_test, y_test)
    
    evals_results = {}
    
    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dvalid], 
                      valid_names=['valid'], 
                      evals_result=evals_results, 
                      num_boost_round=500,
                      early_stopping_rounds=50,
                      verbose_eval=100)


    y_pred += model.predict(X_test, num_iteration = model.best_iteration)/5
    
    oof_preds = model.predict(valid_x, num_iteration = model.best_iteration)
    print('Fold %2d AUC : %.3f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_pred.append(roc_auc_score(valid_y, oof_preds))
    n_fold = n_fold + 1

print('Full AUC score %.3f' % np.mean(auc_pred))   
print('SKF_XLB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))
y_pred = (y_pred >= 0.5).astype(bool)  
print('SKF_LGB ACC: %.3f' % accuracy_score(y_test, y_pred))
print('SKF_LGB F1: %.3f' % f1_score(y_test, y_pred))
print('SKF_LGB GMEAN: %.3f' % geometric_mean_score(y_test, y_pred))    




# 3 rskf
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
auc_pred  = []
y_pred = np.zeros(X_test.shape[0])
n_fold = 0
for idx_train, idx_valid in rskf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    params = {'objective': 'binary',
          'boosting_type': 'gbdt',
          'metric': 'auc',
         'colsample_bytree': 1.0,
         'learning_rate': 0.31402983358009906,
         'max_depth': 66,
         'min_child_samples': 2,
         'min_child_weight': 0,
         'min_split_gain': 0.0,
         'n_estimators': 131,
         'num_leaves': 41,
         'random_state': 1337,
         'reg_alpha': 2.5650838626319216e-06,
         'reg_lambda': 0.00024724195223517915,
         'subsample': 1.0,
         'subsample_for_bin': 200000,
         'subsample_freq': 3,       
         'colsample_bynode': 0.3423093458222358,
         'max_bin': 500,
         'scale_pos_weight': 84.34040301808572,
         'verbose': -1}

    dtrain = lgb.Dataset(train_x, train_y)
    dvalid = lgb.Dataset(valid_x, valid_y)
    dtest = lgb.Dataset(X_test, y_test)
    
    evals_results = {}
    
    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dvalid], 
                      valid_names=['valid'], 
                      evals_result=evals_results, 
                      num_boost_round=500,
                      early_stopping_rounds=50,
                      verbose_eval=100)


    y_pred += model.predict(X_test, num_iteration = model.best_iteration)/15
    
    oof_preds = model.predict(valid_x, num_iteration = model.best_iteration)
    print('Fold %2d AUC : %.3f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_pred.append(roc_auc_score(valid_y, oof_preds))
    n_fold = n_fold + 1
    
print('Full AUC score %.3f' % np.mean(auc_pred))   
print('RSKF_LGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))
y_pred = (y_pred >= 0.5).astype(bool)  
print('RSKF_LGB ACC: %.3f' % accuracy_score(y_test, y_pred))
print('RSKF_LGB F1: %.3f' % f1_score(y_test, y_pred))
print('RSKF_LGB GMEAN: %.3f' % geometric_mean_score(y_test, y_pred)) 


