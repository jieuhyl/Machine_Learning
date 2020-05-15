# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:46:28 2020

@author: Jie.Hu
"""



import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
import time

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, weights=[0.5], random_state=1337)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=1337)


''' 6: Random Forest '''
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state = 1337)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_rf, X = X_train, y = y_train, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()


# KF n GS
parameters = {'criterion':['gini'],#, 'entropy'],
              'max_depth':[5,7,9],
              'max_features':['auto'], #, 'sqrt', 'log2'],
              'max_samples':[0.7,0.8,0.9],
              'min_samples_leaf':[1,3,5],
              'min_samples_split':[2,4,6],
              #'class_weight':['balanced', 'balanced_subsample', None],
              'n_estimators':[200, 400],
              'ccp_alpha':[0.0001],
              'bootstrap':[True],
              'oob_score':[True]}
                                       
grid_search = GridSearchCV(estimator = clf_rf,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = 3,
                           verbose= 1,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_

# last step
clf_rf_grid = grid_search.best_estimator_

y_pred = clf_rf_grid.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_rf_grid.predict_proba(X_test)[:, 1]
print('RF AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))


# not the best the params
cv_hist = pd.DataFrame(grid_search.cv_results_)
cv_hist.sort_values('rank_test_score', inplace = True)
cv_hist.reset_index(inplace = True)

# n_best
params_best_n = cv_hist.loc[0, 'params']
#print(params_best_n)

clf_best_n = clf_rf
clf_best_n.set_params(**params_best_n)

clf_best_n.fit(X_train, y_train)

y_pred = clf_best_n.predict_proba(X_test)[:, 1]
print('RF AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))


res = []
for i in range(100):
    params_best_n = cv_hist.loc[i, 'params']

    clf_best_n = clf_rf
    clf_best_n.set_params(**params_best_n)

    clf_best_n.fit(X_train, y_train)

    y_pred = clf_best_n.predict_proba(X_test)[:, 1]
    res.append(roc_auc_score(y_test, y_pred))




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

