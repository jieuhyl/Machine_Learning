#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 01:32:21 2020

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


# xgboost =====================================================================
import xgboost as xgb
clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic',
                            booster = 'gbtree',
                            tree_method='approx',
                            silent=1,
                            random_state= 1337)


scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)

acc = cross_val_score(estimator = clf_xgb, X = X_train, y = y_train, cv = rskf, scoring='roc_auc')
acc.mean(), acc.std()


# GridSearchCV needs a predefined plan of the experiments
param_grid = { 
             'learning_rate':[0.1],
              'gamma':[0,0.01,0.1],
              'max_depth':[3,4,5], 
              'min_child_weight':[1,2,3],
              'subsample':[0.5],
              'colsample_bytree':[0.5],
              'reg_alpha':[0,0.1,1,2],
              'reg_lambda':[1,2,3],    
              'n_estimators': [100, 400]}

fit_params = {'early_stopping_rounds': 20,
              'eval_metric': ['auc'],
              'eval_set': [(X_valid, y_valid)]}


grid_search = GridSearchCV(estimator = clf_xgb,
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
clf_xgb_grid = grid_search.best_estimator_

y_pred = clf_xgb_grid.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_xgb_grid.predict_proba(X_test)[:, 1]
print('XGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



# RandomizedSearchCV needs the distribution of the experiments to be tested
# If you can provide the right distribution, the sampling will lead to faster and better results.
param_distribution = {'learning_rate': loguniform(0.0001, 1),
                      'max_depth':randint(2, 20),
                      'min_child_weight':randint(1, 10),
                      'subsample': uniform(0.01, 1),
                      'colsample_bytree': uniform(0.01, 1),
                      'reg_alpha': loguniform(0.0001, 100),
                      'reg_lambda': loguniform(0.0001, 100),
                      'gamma': loguniform(0.0001, 100),
                      'max_leaf_nodes': randint(2, 20),
                      'max_bins': randint(10, 255),
                      'scale_pos_weight': loguniform(0.001, 100),
                      'n_estimators': randint(50, 500)
                      }

fit_params = {'early_stopping_rounds': 20,
              'eval_metric': ['auc'],
              'eval_set': [(X_valid, y_valid)]}

random_search = RandomizedSearchCV(estimator = clf_xgb, 
                                   param_distributions=param_distribution,
                                   n_iter=100,
                                   cv=rskf,
                                   scoring='roc_auc',
                                   verbose = False,
                                   n_jobs=-1,
                                   random_state=1337)

start_time = time.time()
random_search = random_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
random_search.best_params_, random_search.best_score_

# last step
clf_xgb_random = random_search.best_estimator_

y_pred = clf_xgb_random.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_xgb_random.predict_proba(X_test)[:, 1]
print('XGB_RANDOM AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))




# also BayesSearchCV needs to work on the distributions of the experiments but it is less sensible to them
bayes_space  = {'learning_rate': Real(0.0001, 1, 'log-uniform'),
                 'min_child_weight': Integer(0, 10),
                 'max_depth': Integer(1, 50),
                 'max_delta_step': Integer(0, 20),
                 'subsample': Real(0.01, 1, 'uniform'),
                 'colsample_bytree': Real(0.01, 1, 'uniform'),
                 'colsample_bylevel': Real(0.01, 1, 'uniform'),
                 'colsample_bynode': Real(0.01, 1, 'uniform'),
                 'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
                 'reg_alpha': Real(1e-9, 1000, 'log-uniform'),
                 'gamma': Real(1e-9, 1000, 'log-uniform'),
                 'n_estimators': Integer(50, 500),
                 'scale_pos_weight': Real(1e-3, 1000, 'log-uniform')}
    
    
    
bayes_search = BayesSearchCV(estimator = clf_xgb,
                             search_spaces = bayes_space,
                             n_iter=100,
                             cv=rskf,
                             scoring='roc_auc', 
                             optimizer_kwargs={'base_estimator': 'GP'},
                             verbose= False,
                             n_jobs=-1,                             
                             random_state=1337)

'''
for baseEstimator in ['GP', 'RF', 'ET', 'GBRT']:
    bayes_search = BayesSearchCV(clf_xgb,
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
bayes_search = bayes_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
bayes_search.best_params_, bayes_search.best_score_ 

# last step
clf_xgb_bayes = bayes_search.best_estimator_

y_pred = clf_xgb_bayes.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_xgb_bayes.predict_proba(X_test)[:, 1]
print('XGB_BAYES AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))


'''
clf_xgb_bayes = bayes_search.best_estimator_
#clf_xgb_bayes.set_params(n_estimators = 300)
clf_xgb_bayes.fit(X_train, y_train, **fit_params)
y_pred = clf_xgb_bayes.predict(X_test, ntree_limit = clf_xgb_bayes.best_ntree_limit)
print(classification_report(y_test, y_pred))

y_pred = clf_xgb_bayes.predict_proba(X_test, ntree_limit = clf_xgb_bayes.best_ntree_limit)[:, 1]
print('XGB_BAYES AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))
'''




dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_valid, y_valid)
dtest = xgb.DMatrix(X_test, y_test)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
evals_result = {}


params = {'objective': 'binary:logistic',
          'booster': 'gbtree',
          'tree_method': 'approx',
         'eval_metric': 'auc',
         'colsample_bylevel': 0.02476895005211128,
         'colsample_bynode': 0.16342755157178349,
         'colsample_bytree': 0.7630791429988074,
         'gamma': 1.0497968504427192e-07,
         'learning_rate': 0.04400797406544243,
         'max_delta_step': 12,
         'max_depth': 10,
         'min_child_weight': 0,
         'n_estimators': 313,
         'random_state': 1337,
         'reg_alpha': 7.270501741447155e-05,
         'reg_lambda': 2.8247881562808347e-07,
         'scale_pos_weight': 0.6804081351966208,
         'subsample': 0.6602998085156114,
         'silent': 1}


model = xgb.train(params, 
                    dtrain, 
                    evals=watchlist,
                    num_boost_round=500, 
                    early_stopping_rounds=50, 
                    verbose_eval=10,
                    evals_result=evals_result)


y_pred = model.predict(dtest, ntree_limit = model.best_ntree_limit)
print('XGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))

y_pred = (y_pred >= 0.5).astype(bool)  
print(classification_report(y_test, y_pred))





#==============================================================================
# 1 tts
auc_pred = []
y_pred = np.zeros(X_test.shape[0])
TTS = 0
for i in range(0, 10):
    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    
    dtrain = xgb.DMatrix(train_x, train_y)
    dvalid = xgb.DMatrix(valid_x, valid_y)
    dtest = xgb.DMatrix(X_test, y_test)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    evals_result = {}
      
    params = {'objective': 'binary:logistic',
              'booster': 'gbtree',
              'tree_method': 'approx',
             'eval_metric': 'auc',
             'colsample_bylevel': 0.02476895005211128,
             'colsample_bynode': 0.16342755157178349,
             'colsample_bytree': 0.7630791429988074,
             'gamma': 1.0497968504427192e-07,
             'learning_rate': 0.04400797406544243,
             'max_delta_step': 12,
             'max_depth': 10,
             'min_child_weight': 0,
             'n_estimators': 313,
             'random_state': 1337,
             'reg_alpha': 7.270501741447155e-05,
             'reg_lambda': 2.8247881562808347e-07,
             'scale_pos_weight': 0.6804081351966208,
             'subsample': 0.6602998085156114,
             'silent': 1}
    
    
    model = xgb.train(params, 
                        dtrain, 
                        evals=watchlist,
                        num_boost_round=500, 
                        early_stopping_rounds=50, 
                        verbose_eval=100,
                        evals_result=evals_result)


    y_pred += model.predict(dtest, ntree_limit = model.best_ntree_limit)/10
    
    oof_preds = model.predict(dvalid, ntree_limit = model.best_ntree_limit)
    print('TTS %2d AUC : %.6f' % (TTS + 1, roc_auc_score(valid_y, oof_preds)))
    auc_pred.append(roc_auc_score(valid_y, oof_preds))
    TTS += 1
    
print('Full AUC score %.3f' % np.mean(auc_pred))   
print('TTS_XGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))
y_pred = (y_pred >= 0.5).astype(bool)  
print('TTS_XGB ACC: %.3f' % accuracy_score(y_test, y_pred))
print('TTS_XGB F1: %.3f' % f1_score(y_test, y_pred))
print('TTS_XGB GMEAN: %.3f' % geometric_mean_score(y_test, y_pred))    



# 2 skf
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1337)
auc_pred = []
y_pred = np.zeros(X_test.shape[0])
n_fold = 0
for idx_train, idx_valid in skf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    dtrain = xgb.DMatrix(train_x, train_y)
    dvalid = xgb.DMatrix(valid_x, valid_y)
    dtest = xgb.DMatrix(X_test, y_test)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    evals_result = {}
      
    params = {'objective': 'binary:logistic',
              'booster': 'gbtree',
              'tree_method': 'approx',
             'eval_metric': 'auc',
             'colsample_bylevel': 0.02476895005211128,
             'colsample_bynode': 0.16342755157178349,
             'colsample_bytree': 0.7630791429988074,
             'gamma': 1.0497968504427192e-07,
             'learning_rate': 0.04400797406544243,
             'max_delta_step': 12,
             'max_depth': 10,
             'min_child_weight': 0,
             'n_estimators': 313,
             'random_state': 1337,
             'reg_alpha': 7.270501741447155e-05,
             'reg_lambda': 2.8247881562808347e-07,
             'scale_pos_weight': 0.6804081351966208,
             'subsample': 0.6602998085156114,
             'silent': 1}
    
    
    model = xgb.train(params, 
                        dtrain, 
                        evals=watchlist,
                        num_boost_round=500, 
                        early_stopping_rounds=50, 
                        verbose_eval=100,
                        evals_result=evals_result)


    y_pred += model.predict(dtest, ntree_limit = model.best_ntree_limit)/5
    
    oof_preds = model.predict(dvalid, ntree_limit = model.best_ntree_limit)
    print('Fold %2d AUC : %.3f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_pred.append(roc_auc_score(valid_y, oof_preds))
    n_fold = n_fold + 1

print('Full AUC score %.3f' % np.mean(auc_pred))   
print('SKF_XGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))
y_pred = (y_pred >= 0.5).astype(bool)  
print('SKF_XGB ACC: %.3f' % accuracy_score(y_test, y_pred))
print('SKF_XGB F1: %.3f' % f1_score(y_test, y_pred))
print('SKF_XGB GMEAN: %.3f' % geometric_mean_score(y_test, y_pred))    




# 3 rskf
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
auc_pred  = []
y_pred = np.zeros(X_test.shape[0])
n_fold = 0
for idx_train, idx_valid in rskf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    dtrain = xgb.DMatrix(train_x, train_y)
    dvalid = xgb.DMatrix(valid_x, valid_y)
    dtest = xgb.DMatrix(X_test, y_test)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    evals_result = {}
      
    params = {'objective': 'binary:logistic',
              'booster': 'gbtree',
              'tree_method': 'approx',
             'eval_metric': 'auc',
             'colsample_bylevel': 0.02476895005211128,
             'colsample_bynode': 0.16342755157178349,
             'colsample_bytree': 0.7630791429988074,
             'gamma': 1.0497968504427192e-07,
             'learning_rate': 0.04400797406544243,
             'max_delta_step': 12,
             'max_depth': 10,
             'min_child_weight': 0,
             'n_estimators': 313,
             'random_state': 1337,
             'reg_alpha': 7.270501741447155e-05,
             'reg_lambda': 2.8247881562808347e-07,
             'scale_pos_weight': 0.6804081351966208,
             'subsample': 0.6602998085156114,
             'silent': 1}
    
    
    model = xgb.train(params, 
                        dtrain, 
                        evals=watchlist,
                        num_boost_round=500, 
                        early_stopping_rounds=50, 
                        verbose_eval=100,
                        evals_result=evals_result)


    y_pred += model.predict(dtest, ntree_limit = model.best_ntree_limit)/15
    
    oof_preds = model.predict(dvalid, ntree_limit = model.best_ntree_limit)
    print('Fold %2d AUC : %.3f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_pred.append(roc_auc_score(valid_y, oof_preds))
    n_fold = n_fold + 1
    
print('Full AUC score %.3f' % np.mean(auc_pred))   
print('RSKF_XGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))
y_pred = (y_pred >= 0.5).astype(bool)  
print('RSKF_XGB ACC: %.3f' % accuracy_score(y_test, y_pred))
print('RSKF_XGB F1: %.3f' % f1_score(y_test, y_pred))
print('RSKF_XGB GMEAN: %.3f' % geometric_mean_score(y_test, y_pred))   
